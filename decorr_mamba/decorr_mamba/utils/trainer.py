from torch.utils.data import DataLoader, TensorDataset
import math
import torch  
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 
import os
import json
import matplotlib.pyplot as plt
import wandb
import numpy as np

from ..utils.helpers import TrainingArgs
from ..model.decorrelation import DecorrMamba
from ..data.synthetics import InductionData
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from ..utils.SOAP.soap import SOAP
from typing import Iterator

from torch.nn.parallel import DistributedDataParallel as DDP


def generate_fixed_exponential_schedule(num_iterations, num_validations, base):
    """
    Generate exactly `num_validations` validation points exponentially spaced 
    from 0 to num_iterations - 1.
    """
    # Generate exponentially spaced points in the range [0, 1]
    exp_positions = np.logspace(0, 1, num=num_validations+1, base=base, endpoint=True) - 1
    exp_positions /= exp_positions[-1]  # Normalize to [0, 1]

    # Map these points to the iteration range [0, num_iterations - 1]
    validation_schedule = np.unique(
    	np.round(exp_positions * (num_iterations - 1))).astype(int)

    # Ensure the last validation is at the final iteration
    validation_schedule[-1] = num_iterations

    # We validate at 0 separately, exclude this from here (1 was added to length above,
    # to compensate for this)

    return validation_schedule[1:]

class DecorrLRScheduler():
	''' 
	Scheduler for the decorrelation learning rate. Can't directly use the
	logic of the PyTorch scheduler as it's tied to a particular optimizer.'''
	def __init__(self, train_args: TrainingArgs):
		self.train_args = train_args
		self.init_lr = train_args.decorr_lr
		self.lr_lambda = self.train_args.schedule_fn

	def step(self, step):
		self.train_args.decorr_lr = self.init_lr * self.lr_lambda(step)


class MambaTrainer:
	''' Trains a Mamba architecture according to a pre-specified configuration

		Args:
			mamba_args (MambaConfig): model specification
			train_args (TrainingArgs): training protocol specification
			model (MambaLMHeadModel): implementation of Mamba architecture as 
			per mamba_args

		Attributes:
			mamba_args (MambaConfig)
			train_args (TrainingArgs)
			model (MambaLMHeadModel)
			device (str)

		Methods:
			train_sequence_steps(): trains the architecture
				following the protocol in train_args, using provided 
				training and validation datasets
	'''
	def __init__(self, 
			mamba_args: MambaConfig, train_args: TrainingArgs, 
			model: MambaLMHeadModel, rank: int, local_rank: int):

		self.mamba_args = mamba_args
		self.train_args = train_args
		self.model = model
		self.local_rank = local_rank
		self.rank = rank
		self.is_main = rank == 0
		self.device = torch.device(f'cuda:{local_rank}')

		def _add_param_to_groups(module, param_groups):
			'''
			Adds the parameters of the module to the appropriate param_groups list
			based on the presence of the _no_weight_decay attribute on the parameters.
			
			Args:
				module: a submodule of the model.
				param_groups: a dictionary containing 'decay' and 'no_decay' lists.
			'''
			for name, param in module.named_parameters(recurse=False):
				# Check if the parameter has the _no_weight_decay attribute
				if hasattr(param, '_no_weight_decay') and param._no_weight_decay:
					param_groups['no_decay'].append(param)
				else:
					param_groups['decay'].append(param)


		# collect parts of the model we don't want weight decay for. Only use weight
		# decay with AdamW optimizer
		self._param_groups = {'decay': [], 'no_decay': []}
		if self.train_args.weight_decay is not None:
			self.model.apply(lambda module: _add_param_to_groups(module, self._param_groups))
			# weight tying causes the embedding and output weights to be the same,
			# but the logic above counts this parameter twice. Remove to fix.
			del self._param_groups["decay"][-1]

	def get_model(self):
		if isinstance(self.model, DDP):
			return self.model.module
		else:
			return self.model
		
	def sync_decorr(self):
		""" 
		Averages gradients and losses across all of the distributed 
		processes, for each decorrelation layer.

		"""
		world_size = torch.distributed.get_world_size()
		
		def _average(n):
			torch.distributed.all_reduce(
			n, op=torch.distributed.ReduceOp.SUM)
			n /= world_size

		def _sync_one_param(layer: torch.nn.Parameter):

			# averaging gradients
			assert layer.decorr_layer.grad is not None, "Gradients not computed!"
			_average(layer.decorr_layer.grad)

			# averaging losses
			if layer.whit_loss is not None:
				_average(layer.whit_loss)

			if layer.corr_loss is not None:
				_average(layer.corr_loss)
			
			# averaging batch means
			if self.train_args.demeaning:
				assert layer.batch_mean is not None, f"Batch mean not available for layer {layer}"
				_average(layer.batch_mean)
			
		with torch.no_grad():
			self.get_model().apply_to_decorr(lambda x: _sync_one_param(x))

	@staticmethod
	def sync_tensor(tensor):
		torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
		tensor /= torch.distributed.get_world_size()

		return tensor

	def sync_mean_losses(self):
		""" 
		Averages mean losses across all of the distributed models
		"""
		world_size = torch.distributed.get_world_size()
		
		if self.get_model().mean_corr_loss is not None:
			torch.distributed.all_reduce(
				self.get_model().mean_corr_loss, op=torch.distributed.ReduceOp.SUM)
			self.get_model().mean_corr_loss /= world_size	

		if self.get_model().mean_whit_loss is not None:
			torch.distributed.all_reduce(
				self.get_model().mean_whit_loss, op=torch.distributed.ReduceOp.SUM)
			self.get_model().mean_whit_loss /= world_size	

	def train_sequence_steps(self, train_loader, val_loader: DataLoader, test_loader:DataLoader,
		use_amp: bool, log_freq: int, n_val: int, train_backprop: bool=True, 
		train_decorr: bool=True, save_checkpoints: bool=True, pad_idx: int = None,
		skip_init_val: bool=False, datashuffle_seed: int=0, metric="ppl", val_sched_base: int=20,
		crop_frac: float = 1.0, decorr_update_freq: int = 1):

		''' 
		Trains the model with the protocol specified in train_args. Trains based
		on a fixed number of gradient descent steps, performs validation
		at pre-defined points within the training loop. 

		'''
		def maybe_tqdm(iterator):
			""" Useful for controlling console printouts during DDP training"""
			return tqdm(iterator) if self.is_main else iterator

		# Index used for sequence length padding in proteome modelling task
		if pad_idx:
			criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
		else:
			criterion = nn.CrossEntropyLoss()

		if self.is_main:
			if not train_backprop:
				print("Warning: not training backpropagation parameters!")
			if not train_decorr:
				print("Warning: not training decorrelation parameters!")
			assert train_backprop or train_decorr, "Specify something to train"

			if not isinstance(self.get_model(), DecorrMamba) and train_decorr:
				print("Warning: train_decorr set to True but model does not use decorrelation!")
			
			if isinstance(self.get_model(), DecorrMamba):
				if int(self.train_args.B*self.get_model().sample_frac) < 1:
					print("Warning: decorrelation sub-sampling too small, will use 1 batch instead.\n")
			
			assert metric == "ppl" or metric == "bpb", "Metric must be either perplexity (ppl) or bits per byte (bpb)"

		if self.train_args.weight_decay is not None:
			if self.train_args.optimizer == "adam":
				optimizer = torch.optim.AdamW(
					[{'params': self._param_groups['decay'],
					'weight_decay': self.train_args.weight_decay}, 

					{'params': self._param_groups['no_decay'], 
					'weight_decay': 0.0}], 

					lr=self.train_args.lr,
					betas=self.train_args.beta,
					eps=self.train_args.epsilon)
				
			elif self.train_args.optimizer == "soap":

				optimizer = SOAP(
					[{'params': self._param_groups['decay'],
					'weight_decay': self.train_args.weight_decay}, 

					{'params': self._param_groups['no_decay'], 
					'weight_decay': 0.0}], 

					lr=self.train_args.lr,
					betas=self.train_args.beta,
					eps=self.train_args.epsilon)	

				# optimizer = SOAP(
				# 	[{'params': self._param_groups['decay'],
				# 	'weight_decay': self.train_args.weight_decay}, 

				# 	{'params': self._param_groups['no_decay'], 
				# 	'weight_decay': 0.0}],
				# 	lr=self.train_args.lr)
			else:
				raise NotImplementedError			
			
		else:
			if self.train_args.optimizer == "adam":
				optimizer = torch.optim.Adam(self.model.parameters(), 
											lr=self.train_args.lr, 
											betas=self.train_args.beta,
											eps=self.train_args.epsilon) 
				
			elif self.train_args.optimizer == "soap":

				optimizer = SOAP(self.model.parameters(), 
					 weight_decay=0.0,
					lr=self.train_args.lr,
					betas=self.train_args.beta,
					eps=self.train_args.epsilon)	

				# optimizer = SOAP(self.model.parameters(), weight_decay=0.0, 
				# 	 lr=self.train_args.lr)
			else:
				raise NotImplementedError				   

		if self.train_args.use_lr_sched:
			scheduler = torch.optim.lr_scheduler.LambdaLR(
				optimizer, lr_lambda=self.train_args.schedule_fn)
			
			# create a separate scheduler for the decorrelation learning rate
			# too!
			if isinstance(self.get_model(), DecorrMamba):
				decorr_scheduler = DecorrLRScheduler(
					train_args=self.train_args)

			# visualize learning rate schedule for the backprop parameters
			# NOTE: DOES NOT WORK FOR THE DECORRELATION LR!! 

			self.train_args.show_lr_schedule()
			plt.savefig(os.path.join('.', "schedule.png"))
			
		
		save_path = os.path.join(".", "checkpoints")
		os.makedirs(save_path, exist_ok=True)

		scaler = torch.amp.GradScaler(self.device.type, enabled=use_amp)

		val_sched = generate_fixed_exponential_schedule(
			num_iterations=self.train_args.n_steps, num_validations=n_val, base=val_sched_base)
		
		decorr_update_sched = [i*decorr_update_freq for i in range(
			self.train_args.n_steps // decorr_update_freq + 1)]
		
		train_iterator = iter(train_loader)

		if train_backprop:
			self.model.train()
		else:
			self.model.eval()

		epoch = 0 # needed for shuffling of multi-gpu data samplers
		if hasattr(train_loader.sampler, "set_epoch"):
			train_loader.sampler.set_epoch(datashuffle_seed + epoch)

		for step in maybe_tqdm(range(1, self.train_args.n_steps+1)):
			# initial validation before training
			if step == 1 and not skip_init_val:
				self.model.eval()
				total_val_ce_loss = 0.0
				total_val_metric = 0.0

				with torch.no_grad():
					with torch.amp.autocast(self.device.type, enabled=use_amp):	

						if isinstance(self.get_model(), DecorrMamba):
							self.get_model().fuse_decorr()
						
						for next_batch in maybe_tqdm(val_loader):

							in_seq = next_batch.long().to(self.device, non_blocking=True)
							pred = self.model(in_seq[:,:-1]).logits

							target = in_seq[:,1:].contiguous()

							output_dim = pred.shape[-1]
							loss = criterion(pred.view(-1, output_dim)[:,:self.mamba_args.vocab_size],
											 target.view(-1))

							if self.train_args.ddp:
								loss = self.sync_tensor(loss)
									
							total_val_ce_loss += loss.item()
							if metric == "ppl":
								total_val_metric += torch.exp(loss).item()
							elif metric == "bpb":
								total_val_metric += (loss * torch.log2(
									torch.tensor(torch.e))).item()
							else:
								raise NotImplementedError

				total_val_ce_loss /= len(val_loader)
				total_val_metric /= len(val_loader)

				if self.is_main:
					print(f"Initial val {metric}: {total_val_metric:.4f}")				
					print(f"Initial val CE loss: {total_val_ce_loss:.4f}")
					wandb.log({
						"val_ce_loss": total_val_ce_loss,
						f"val_{metric}": total_val_metric}, step=step)
						
				self.model.train()	

			# an infinite loop for the fixed number of gradient descent 
			# steps
			try:
				next_batch = next(train_iterator)
			except StopIteration:
				epoch += 1
				if hasattr(train_loader.sampler, "set_epoch"):
					train_loader.sampler.set_epoch(datashuffle_seed + epoch)
				train_iterator = iter(train_loader)  # Reset the iterator
				next_batch = next(train_iterator)

			in_seq = next_batch.long().to(self.device, non_blocking=True)

			if train_backprop:
				optimizer.zero_grad()
			
			if isinstance(self.get_model(), DecorrMamba):
				if self.get_model().compute_loss or self.model.training:
					self.get_model().reset_decorr()

			with torch.amp.autocast(self.device.type, enabled=use_amp):
				# shift input sequence by one token and compare
				with torch.enable_grad() if train_backprop else torch.no_grad():
					pred = self.model(in_seq[:,:-1]).logits

				target = in_seq[:,1:].contiguous()
				# NB: ignore the irrelevant extra dimensions in the output,
				# those are just there for padding (GPU efficiency). Collapse
				# across batch and length before feeding to loss.
				output_dim = pred.shape[-1]
				loss = criterion(
					pred.view(-1, output_dim)[:,:self.mamba_args.vocab_size],
					target.view(-1))	
				
				# needed for synchronizing during ddp
				loss_tensor = loss.detach().clone()
				
			# average the loss values across models
			if self.train_args.ddp:
				loss_tensor = self.sync_tensor(loss_tensor)

			if metric == "ppl":
				batch_metric = torch.exp(loss_tensor).item()
			elif metric == "bpb":
				batch_metric = (loss_tensor * torch.log2(
					torch.tensor(torch.e))).item()
			else:
				raise NotImplementedError

			if step%log_freq == 0:
				wandb.log({"train_ce_loss": loss_tensor.item(), 
							f"train_{metric}": batch_metric}, step=step)					
			
			if train_backprop:
				scaler.scale(loss).backward()
			
			# gradient clipping
			if self.train_args.gradient_clip is not None:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
											self.train_args.gradient_clip)									

			if train_backprop:
				scaler.step(optimizer)
				scaler.update()
				if self.train_args.use_lr_sched:
					scheduler.step()
					if isinstance(self.get_model(), DecorrMamba):
						decorr_scheduler.step(step)
			
			# update the decorrelation matrices AFTER standard backprop, 
			# else training breaks! Second condition allows for sparse
			# matrix updating
			if isinstance(self.get_model(), DecorrMamba) and step in decorr_update_sched:
				if self.get_model().compute_loss or self.model.training:
					self.get_model().decorr_operations(crop_frac=crop_frac)
					# average decorrelation gradients and losses across each
					# copy of the layer before updating parameters

					if self.train_args.ddp:
						self.sync_decorr()	

					self.get_model().mean_decorr_losses()
					# average the losses across the parallel models
					if self.train_args.ddp:
						self.sync_mean_losses()
				
				train_corr_loss = self.get_model().mean_corr_loss
				train_whit_loss = self.get_model().mean_whit_loss

				if train_corr_loss is not None:
					train_corr_loss = train_corr_loss.item()
				if train_whit_loss is not None:
					train_whit_loss = train_whit_loss.item()
				
				if step%log_freq == 0:
					wandb.log({"train_corr_loss": train_corr_loss, 
							"train_whit_loss": train_whit_loss}, step=step)	
					
				if train_decorr:
					self.get_model().update_decorr_matrices(
						self.train_args.decorr_lr, self.train_args.demeaning_lr)

			# Condition checking if validate_every number of gradient descent
			# steps have happened
			
			if step in val_sched:

				# -------------------------------- validation -------------------------------------	

				# Don't want to compute whitening/correlation losses here because
				# it makes validation extremely slow, if cross entropy is lower 
				# that's all that really matters for the most part
				
				self.model.eval()

				total_val_ce_loss = 0.0
				total_val_metric = 0.0
				# total_val_corr_loss = 0.0
				# total_val_whit_loss = 0.0	

				with torch.no_grad():
					with torch.amp.autocast(self.device.type, enabled=use_amp):	

						# fuse decorrelation matrices again, just once.
						if isinstance(self.get_model(), DecorrMamba):
							self.get_model().fuse_decorr()
						
						for next_batch in maybe_tqdm(val_loader):
							# if isinstance(self.model, DecorrMamba):
							# 	# only resets the losses for the decorrelation layers
							# 	# and the model
							# 	self.model.reset_decorr()

							in_seq = next_batch.long().to(self.device, non_blocking=True)
							pred = self.model(in_seq[:,:-1]).logits
								
							target = in_seq[:,1:].contiguous()

							output_dim = pred.shape[-1]
							loss = criterion(pred.view(-1, output_dim)[:,:self.mamba_args.vocab_size],
											 target.view(-1))

							if self.train_args.ddp:
								loss = self.sync_tensor(loss)
									
							total_val_ce_loss += loss.item()

							if metric == "ppl":
								total_val_metric += torch.exp(loss).item()
							elif metric == "bpb":
								total_val_metric += (loss * torch.log2(
									torch.tensor(torch.e))).item()


							# if isinstance(self.model, DecorrMamba):
							# 	# only compute losses of decorr matrices,
							# 	# not the associated gradients
							# 	self.model.reshape_decorr_inputs(b=b)
							# 	self.model.compute_decorr_grad_loss(compute_grad=False, b=b)				
							# 	self.model.mean_decorr_losses()
							# 	val_corr_loss = self.model.mean_corr_loss
							# 	val_whit_loss = self.model.mean_whit_loss

							# 	if val_corr_loss is not None:
							# 		total_val_corr_loss += val_corr_loss.item()
							# 	if val_whit_loss is not None:
							# 		total_val_whit_loss += val_whit_loss.item()				

				total_val_ce_loss /= len(val_loader)
				total_val_metric /= len(val_loader)

				if self.is_main:
					print(f"\nStep {step} val {metric}: {total_val_metric:.4f}")				
					print(f"Step {step} val CE loss: {total_val_ce_loss:.4f}")
					wandb.log({
						"val_ce_loss": total_val_ce_loss,
						f"val_{metric}": total_val_metric}, step=step)
				
				# if isinstance(self.model, DecorrMamba):
				# 	total_val_corr_loss /= len(val_loader)
				# 	total_val_whit_loss /= len(val_loader)
				# 	print(f"\"Epoch\" val correlation loss: {total_val_corr_loss:.4f}")	
				# 	print(f"\"Epoch\" val whitening loss: {total_val_whit_loss:.4f}")				
				# 	wandb.log({
				# 		"val_corr_loss": total_val_corr_loss, 
				# 		"val_whit_loss": total_val_whit_loss}, step=step)

					if save_checkpoints:
						torch.save({
							"model_state": self.model.state_dict(),
							"optimizer_state": optimizer.state_dict(),}, 
							os.path.join(save_path, f"step_{step}.pth")) 
						
						wandb.save(os.path.join(save_path, f"step_{step}.pth"))
				
				self.model.train()

		# Testing!
		self.model.eval()

		total_test_ce_loss = 0.0
		total_test_metric = 0.0

		with torch.no_grad():
			with torch.amp.autocast(self.device.type, enabled=use_amp):	
				# fuse decorrelation matrices again, just once.
				if isinstance(self.get_model(), DecorrMamba):
					self.get_model().fuse_decorr()
				
				for next_batch in maybe_tqdm(test_loader):

					in_seq = next_batch.long().to(self.device, non_blocking=True)
					pred = self.model(in_seq[:,:-1]).logits
						
					target = in_seq[:,1:].contiguous()

					output_dim = pred.shape[-1]
					loss = criterion(pred.view(-1, output_dim)[:,:self.mamba_args.vocab_size],
										target.view(-1))

					if self.train_args.ddp:
						loss = self.sync_tensor(loss)
							
					total_test_ce_loss += loss.item()

					if metric == "ppl":
						total_test_metric += torch.exp(loss).item()
					elif metric == "bpb":
						total_test_metric += (loss * torch.log2(
							torch.tensor(torch.e))).item()			

		total_test_ce_loss /= len(test_loader)
		total_test_metric /= len(test_loader)

		if self.is_main:
			print(f"\nTest {metric}: {total_test_metric:.4f}")				
			print(f"Test CE loss: {total_test_ce_loss:.4f}")
			wandb.log({
				"test_ce_loss": total_test_ce_loss,
				f"test_{metric}": total_test_metric})
			
		torch.save({
			"model_state": self.model.state_dict(),
			"optimizer_state": optimizer.state_dict(),}, 
			os.path.join(save_path, f"final.pth")) 
		
		wandb.save(os.path.join(save_path, f"final.pth"))

	def train_synthetic(self, train_dataset: TensorDataset, 
		val_loaders: dict[DataLoader, str], test_loaders: dict[DataLoader, str],
		use_amp: bool, log_freq: int, n_val: int, task: str, train_backprop: bool=True, 
		train_decorr: bool=True, save_checkpoints: bool=True, skip_init_val: bool=False,
		crop_frac: float = 1.0):

		''' 
		Trains the model with the protocol specified in train_args. Trains based
		on a fixed number of gradient descent steps, performs validation
		at pre-defined points within the training loop. 

		'''
		criterion = nn.CrossEntropyLoss()

		if not train_backprop:
			print("Warning: not training backpropagation parameters!")
		if not train_decorr:
			print("Warning: not training decorrelation parameters!")
		assert train_backprop or train_decorr, "Specify something to train"

		if not isinstance(self.get_model(), DecorrMamba) and train_decorr:
			print("Warning: train_decorr set to True but model does not use decorrelation!")
		
		if isinstance(self.get_model(), DecorrMamba):
			if int(self.train_args.B*self.get_model().sample_frac) < 1:
				print("Warning: decorrelation sub-sampling too small, will use 1 batch instead.\n")

		if self.train_args.weight_decay is not None:
			if self.train_args.optimizer == "adam":
				optimizer = torch.optim.AdamW(
					[{'params': self._param_groups['decay'],
					'weight_decay': self.train_args.weight_decay}, 

					{'params': self._param_groups['no_decay'], 
					'weight_decay': 0.0}], 

					lr=self.train_args.lr,
					betas=self.train_args.beta,
					eps=self.train_args.epsilon)
				
			elif self.train_args.optimizer == "soap":

				optimizer = SOAP(
					[{'params': self._param_groups['decay'],
					'weight_decay': self.train_args.weight_decay}, 

					{'params': self._param_groups['no_decay'], 
					'weight_decay': 0.0}], 

					lr=self.train_args.lr,
					betas=self.train_args.beta,
					eps=self.train_args.epsilon)	
			else:
				raise NotImplementedError			
			
		else:
			if self.train_args.optimizer == "adam":
				optimizer = torch.optim.Adam(self.model.parameters(), 
											lr=self.train_args.lr,
											betas=self.train_args.beta,
											eps=self.train_args.epsilon) 
				
			elif self.train_args.optimizer == "soap":

				optimizer = SOAP(self.model.parameters(), 
					 weight_decay=0.0,
					lr=self.train_args.lr,
					betas=self.train_args.beta,
					eps=self.train_args.epsilon)
			else:
				raise NotImplementedError				   

		if self.train_args.use_lr_sched:
			scheduler = torch.optim.lr_scheduler.LambdaLR(
				optimizer, lr_lambda=self.train_args.schedule_fn)
			
			# create a separate scheduler for the decorrelation learning rate
			# too!
			if isinstance(self.get_model(), DecorrMamba):
				decorr_scheduler = DecorrLRScheduler(
					train_args=self.train_args)

			# visualize learning rate schedule for the backprop parameters
			# NOTE: DOES NOT WORK FOR THE DECORRELATION LR!! 

			self.train_args.show_lr_schedule()
			plt.savefig(os.path.join('.', "schedule.png"))
			
		
		save_path = os.path.join(".", "checkpoints")
		os.makedirs(save_path, exist_ok=True)

		scaler = torch.amp.GradScaler(self.device.type, enabled=use_amp)
		val_sched = [i*(self.train_args.n_steps//n_val) for i in range(1,n_val+1)]
		train_iterator = iter(train_dataset)

		if train_backprop:
			self.model.train()
		else:
			self.model.eval()


		for step in tqdm(range(1, self.train_args.n_steps+1)):
			# initial validation before training
			if step == 1 and not skip_init_val:
				self.model.eval()
				for val_loader, name in val_loaders.items():
					total_val_ce_loss = 0.0
					total_val_acc = 0.0
					total_tokens = 0

					with torch.no_grad():
						with torch.amp.autocast(self.device.type, enabled=use_amp):	

							if isinstance(self.get_model(), DecorrMamba):
								self.get_model().fuse_decorr()
							
							for next_batch in tqdm(val_loader):

								in_seq, target = next_batch
								in_seq = in_seq.long().to(self.device, non_blocking=True)
								target = target.long().to(self.device, non_blocking=True).contiguous()

								if task == "induction":
									# feed entire sequence including the 
									# retrieval cue at the very end, and only train 
									# the output predicted at the final token 
									pred = self.model(in_seq).logits[:,-1,:]
								elif task == "selective_copy":
									# feed entire sequence, and only train the
									# output predicted for the sequence of however
									# many data tokens we chose (at the very end)
									pred_len = target.shape[1]
									pred = self.model(in_seq).logits[:,-pred_len:,:]
								else:
									raise NotImplementedError
						
								# collapse across batch dimension, and ignore the
								# unused output head dimensions
								output_dim = pred.shape[-1]
								pred = pred.reshape(-1, output_dim)[:,:self.mamba_args.vocab_size]
								target = target.view(-1)

								loss = criterion(pred, target)
										
								total_val_ce_loss += loss.item()

								# calculate accuracy
								probs = F.softmax(pred, dim=1)
								preds = torch.argmax(probs, dim=1)
								n_correct = torch.sum(preds==target).item()
								total_val_acc += n_correct
								total_tokens += target.numel()


					total_val_ce_loss /= len(val_loader)
					total_val_acc = 100*(total_val_acc/total_tokens)

					print(f"Initial val accuracy {name}: {total_val_acc:.2f}%")				
					print(f"Initial val CE loss {name}: {total_val_ce_loss:.4f}")
					wandb.log({
						"val_ce_loss_{name}": total_val_ce_loss,
						f"val_acc_{name}": total_val_acc}, step=step)
						
				self.model.train()	

			# an infinite loop for the fixed number of gradient descent 
			# steps
			try:
				next_batch = next(train_iterator)
			except StopIteration:
				train_iterator = iter(train_dataset)  # Reset the iterator
				next_batch = next(train_iterator)

			in_seq, target = next_batch
			in_seq = in_seq.long().to(self.device, non_blocking=True)
			target = target.long().to(self.device, non_blocking=True).contiguous()

			if train_backprop:
				optimizer.zero_grad()
			
			if isinstance(self.get_model(), DecorrMamba):
				if self.get_model().compute_loss or self.model.training:
					self.get_model().reset_decorr()

			with torch.amp.autocast(self.device.type, enabled=use_amp):
				with torch.enable_grad() if train_backprop else torch.no_grad():
					if task == "induction":
						# feed entire sequence including the 
						# retrieval cue at the very end, and only train 
						# the output predicted at the final token 
						pred = self.model(in_seq).logits[:,-1,:]
					elif task == "selective_copy":
						# feed entire sequence, and only train the
						# output predicted for the sequence of however
						# many data tokens we chose (at the very end)
						pred_len = target.shape[1]
						pred = self.model(in_seq).logits[:,-pred_len:,:]
					else:
						raise NotImplementedError

				# NB: ignore the irrelevant extra dimensions in the output,
				# those are just there for padding (GPU efficiency). Collapse
				# across batch and length before feeding to loss.
				output_dim = pred.shape[-1]
				pred = pred.reshape(-1, output_dim)[:,:self.mamba_args.vocab_size]
				target = target.view(-1)
				loss = criterion(pred, target)	
			
			# calculate accuracy on the batch
			probs = F.softmax(pred, dim=1)
			preds = torch.argmax(probs, dim=1)
			batch_acc = 100*(torch.sum(preds==target).item()/target.shape[0])

			if step%log_freq == 0:
				wandb.log({"train_ce_loss": loss.item(), 
							f"train_acc": batch_acc}, step=step)					
			
			if train_backprop:
				scaler.scale(loss).backward()
			
			# gradient clipping
			if self.train_args.gradient_clip is not None:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
											self.train_args.gradient_clip)									

			if train_backprop:
				scaler.step(optimizer)
				scaler.update()
				if self.train_args.use_lr_sched:
					scheduler.step()
					if isinstance(self.get_model(), DecorrMamba):
						decorr_scheduler.step(step)
			
			# update the decorrelation matrices AFTER standard backprop, 
			# else training breaks! 
			if isinstance(self.get_model(), DecorrMamba):
				if self.get_model().compute_loss or self.model.training:
					self.get_model().decorr_operations(crop_frac=crop_frac)
					self.get_model().mean_decorr_losses()

				train_corr_loss = self.get_model().mean_corr_loss
				train_whit_loss = self.get_model().mean_whit_loss

				if train_corr_loss is not None:
					train_corr_loss = train_corr_loss.item()
				if train_whit_loss is not None:
					train_whit_loss = train_whit_loss.item()
				
				if step%log_freq == 0:
					wandb.log({"train_corr_loss": train_corr_loss, 
							"train_whit_loss": train_whit_loss}, step=step)	
					
				if train_decorr:
					self.get_model().update_decorr_matrices(
						self.train_args.decorr_lr, self.train_args.demeaning_lr)

			# Condition checking if validate_every number of gradient descent
			# steps have happened
			
			if step in val_sched:

				# -------------------------------- validation -------------------------------------	

				# Don't want to compute whitening/correlation losses here because
				# it makes validation extremely slow, if cross entropy is lower 
				# that's all that really matters for the most part
				
				self.model.eval()
				for val_loader, name in val_loaders.items():
					total_val_ce_loss = 0.0
					total_val_acc = 0.0
					total_tokens = 0

					with torch.no_grad():
						with torch.amp.autocast(self.device.type, enabled=use_amp):	

							if isinstance(self.get_model(), DecorrMamba):
								self.get_model().fuse_decorr()
							
							for next_batch in tqdm(val_loader):

								in_seq, target = next_batch
								in_seq = in_seq.long().to(self.device, non_blocking=True)
								target = target.long().to(self.device, non_blocking=True).contiguous()

								if task == "induction":
									# feed entire sequence including the 
									# retrieval cue at the very end, and only train 
									# the output predicted at the final token 
									pred = self.model(in_seq).logits[:,-1,:]
								elif task == "selective_copy":
									# feed entire sequence, and only train the
									# output predicted for the sequence of however
									# many data tokens we chose (at the very end)
									pred_len = target.shape[1]
									pred = self.model(in_seq).logits[:,-pred_len:,:]
								else:
									raise NotImplementedError
						
								# collapse across batch dimension, and ignore the
								# unused output head dimensions
								output_dim = pred.shape[-1]
								pred = pred.reshape(-1, output_dim)[:,:self.mamba_args.vocab_size]
								target = target.view(-1)

								loss = criterion(pred, target)
										
								total_val_ce_loss += loss.item()

								# calculate accuracy
								probs = F.softmax(pred, dim=1)
								preds = torch.argmax(probs, dim=1)
								n_correct = torch.sum(preds==target).item()
								total_val_acc += n_correct
								total_tokens += target.numel()

					total_val_ce_loss /= len(val_loader)
					total_val_acc = 100*(total_val_acc / total_tokens)

					print(f"val accuracy {name}: {total_val_acc:.2f}%")				
					print(f"val CE loss {name}: {total_val_ce_loss:.4f}")
					wandb.log({
						f"val_ce_loss_{name}": total_val_ce_loss,
						f"val_acc_{name}": total_val_acc}, step=step)

					if save_checkpoints:
						torch.save({
							"model_state": self.model.state_dict(),
							"optimizer_state": optimizer.state_dict(),}, 
							os.path.join(save_path, f"step_{step}.pth")) 
						
						wandb.save(os.path.join(save_path, f"step_{step}.pth"))
				
				self.model.train()

		# Testing!
		self.model.eval()
		for test_loader, name in test_loaders.items():
			total_test_ce_loss = 0.0
			total_test_acc = 0.0
			total_tokens = 0

			with torch.no_grad():
				with torch.amp.autocast(self.device.type, enabled=use_amp):	

					if isinstance(self.get_model(), DecorrMamba):
						self.get_model().fuse_decorr()
					
					for next_batch in tqdm(test_loader):

						in_seq, target = next_batch
						in_seq = in_seq.long().to(self.device, non_blocking=True)
						target = target.long().to(self.device, non_blocking=True).contiguous()

						if task == "induction":
							# feed entire sequence including the 
							# retrieval cue at the very end, and only train 
							# the output predicted at the final token 
							pred = self.model(in_seq).logits[:,-1,:]
						elif task == "selective_copy":
							# feed entire sequence, and only train the
							# output predicted for the sequence of however
							# many data tokens we chose (at the very end)
							pred_len = target.shape[1]
							pred = self.model(in_seq).logits[:,-pred_len:,:]
						else:
							raise NotImplementedError
				
						# collapse across batch dimension, and ignore the
						# unused output head dimensions
						output_dim = pred.shape[-1]
						pred = pred.reshape(-1, output_dim)[:,:self.mamba_args.vocab_size]
						target = target.view(-1)

						loss = criterion(pred, target)
								
						total_test_ce_loss += loss.item()

						# calculate accuracy
						probs = F.softmax(pred, dim=1)
						preds = torch.argmax(probs, dim=1)

						n_correct = torch.sum(preds==target).item()
						total_test_acc += n_correct
						total_tokens += target.numel()


			total_test_ce_loss /= len(test_loader)
			total_test_acc  = 100*(total_test_acc/total_tokens)

			print(f"test accuracy {name}: {total_test_acc:.2f}%")				
			print(f"test CE loss {name}: {total_test_ce_loss:.4f}")
			wandb.log({
				f"test_ce_loss_{name}": total_test_ce_loss,
				f"test_acc_{name}": total_test_acc}, step=step)
			
		torch.save({
			"model_state": self.model.state_dict(),
			"optimizer_state": optimizer.state_dict(),}, 
			os.path.join(save_path, f"final.pth")) 
		
		wandb.save(os.path.join(save_path, f"final.pth"))