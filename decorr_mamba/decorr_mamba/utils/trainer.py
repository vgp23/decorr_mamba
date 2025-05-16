from torch.utils.data import DataLoader
import math
import torch  
import torch.nn as nn
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

from torch.nn.parallel import DistributedDataParallel as DDP


def generate_fixed_exponential_schedule(num_iterations, num_validations):
    """
    Generate exactly `num_validations` validation points exponentially spaced 
    from 0 to num_iterations - 1.
    """
    # Generate exponentially spaced points in the range [0, 1]
    exp_positions = np.logspace(0, 1, num=num_validations+1, base=20, endpoint=True) - 1
    exp_positions /= exp_positions[-1]  # Normalize to [0, 1]

    # Map these points to the iteration range [0, num_iterations - 1]
    validation_schedule = np.unique(
    	np.round(exp_positions * (num_iterations - 1))).astype(int)

    # Ensure the last validation is at the final iteration
    validation_schedule[-1] = num_iterations

    # We validate at 0 separately, exclude this from here (1 was added to length above,
    # to compensate for this)

    return validation_schedule[1:]

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

		def _sync_one_param(layer: torch.nn.Parameter):
			world_size = torch.distributed.get_world_size()
			# averaging gradients
			assert layer.decorr_layer.grad is not None, "Gradients not computed!"

			torch.distributed.all_reduce(
				layer.decorr_layer.grad, op=torch.distributed.ReduceOp.SUM)
			layer.decorr_layer.grad /= world_size

			# averaging losses
			if layer.whit_loss is not None:
				torch.distributed.all_reduce(
					layer.whit_loss, op=torch.distributed.ReduceOp.SUM)
				layer.whit_loss /= world_size

			if layer.corr_loss is not None:
				torch.distributed.all_reduce(
					layer.corr_loss, op=torch.distributed.ReduceOp.SUM)
				layer.corr_loss /= world_size

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

	def train_sequence_steps(self, train_loader: DataLoader, val_loader: DataLoader, 
		use_amp: bool, log_freq: int, n_val: int, train_backprop: bool=True, 
		train_decorr: bool=True, save_checkpoints: bool=True, pad_idx: int = None,
		skip_init_val: bool=False, datashuffle_seed: int=0, metric="ppl"):

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
					betas=self.train_args.adam_beta,
					eps=self.train_args.adam_epsilon)
				
			elif self.train_args.optimizer == "soap":
				
				if self.is_main:
					print("\nUsing SOAP optimizer!")

				# stick to default values for now

				# optimizer = SOAP(
				# 	[{'params': self._param_groups['decay'],
				# 	'weight_decay': self.train_args.weight_decay}, 

				# 	{'params': self._param_groups['no_decay'], 
				# 	'weight_decay': 0.0}], 

				# 	lr=self.train_args.lr,
				# 	betas=self.train_args.adam_beta,
				# 	eps=self.train_args.adam_epsilon)	

				optimizer = SOAP(
					[{'params': self._param_groups['decay'],
					'weight_decay': self.train_args.weight_decay}, 

					{'params': self._param_groups['no_decay'], 
					'weight_decay': 0.0}],
					lr=self.train_args.lr)
			else:
				raise NotImplementedError			
			
		else:
			if self.train_args.optimizer == "adam":
				optimizer = torch.optim.Adam(self.model.parameters(), 
											lr=self.train_args.lr, 
											betas=self.train_args.adam_beta,
											eps=self.train_args.adam_epsilon) 
			elif self.train_args.optimizer == "soap":
				
				if self.is_main:
					print("\nUsing SOAP optimizer!")

				# stick to default values for now

				# optimizer = SOAP(self.model.parameters(), 
				# 	 weight_decay=0.0,
				# 	lr=self.train_args.lr,
				# 	betas=self.train_args.adam_beta,
				# 	eps=self.train_args.adam_epsilon)	

				optimizer = SOAP(self.model.parameters(), weight_decay=0.0, 
					 lr=self.train_args.lr)
			else:
				raise NotImplementedError				   

		if self.train_args.use_lr_sched:
			scheduler = torch.optim.lr_scheduler.LambdaLR(
				optimizer, lr_lambda=self.train_args.schedule_fn)
			# visualize learning rate schedule
			self.train_args.show_lr_schedule()
			plt.savefig(os.path.join('.', "schedule.png"))
			
		
		save_path = os.path.join(".", "checkpoints")
		os.makedirs(save_path, exist_ok=True)

		scaler = torch.amp.GradScaler(self.device.type, enabled=use_amp)

		val_sched = generate_fixed_exponential_schedule(
			num_iterations=self.train_args.n_steps, num_validations=n_val)
		
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
			
			# # calculating update ratio information for the backprop-trained
			# # parameters
			# if step%log_freq == 0 and train_backprop:
				
			# 	n_pars = 0
				
			# 	mean_update_ratio = 0.0
			# 	min_update_ratio = float("inf")
			# 	max_update_ratio = -float("inf")

			# 	for _, param in self.model.named_parameters():
			# 		if param.grad is not None and not \
			# 			isinstance(param, DecorrLinear) and not \
			# 				isinstance(param, DecorrConv1d):

			# 			n_pars += 1
			# 			weight_norm = torch.norm(param).item()		
			# 			grad_norm = torch.norm(param.grad).item()
			# 			update_ratio = grad_norm / weight_norm
			# 			mean_update_ratio += update_ratio
			# 			if update_ratio < min_update_ratio:
			# 				min_update_ratio = update_ratio
			# 			if update_ratio > max_update_ratio:
			# 				max_update_ratio = update_ratio

			# 	mean_update_ratio /= n_pars

			# 	wandb.log({"update_ratio/mean": mean_update_ratio,
			# 				"update_ratio/min": min_update_ratio,
			# 				"update_ratio/max": max_update_ratio}, step=step)									

			if train_backprop:
				scaler.step(optimizer)
				scaler.update()
				if self.train_args.use_lr_sched:
					# doesn't affect decorrelation lr
					scheduler.step()
			
			# update the decorrelation matrices AFTER standard backprop, 
			# else training breaks!
			if isinstance(self.get_model(), DecorrMamba):
				if self.get_model().compute_loss or self.model.training:
					self.get_model().decorr_operations()
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
					self.get_model().update_decorr_matrices()

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