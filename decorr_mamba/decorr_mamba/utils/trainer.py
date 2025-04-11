from torch.utils.data import DataLoader
import math
import torch  
import torch.nn as nn
from tqdm import tqdm 
import os
import json
import wandb

from ..utils.helpers import TrainingArgs
from ..model.decorrelation import DecorrMamba
from ..data.synthetics import InductionData
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from ..utils.SOAP.soap import SOAP



# os.environ["WANDB_SILENT"] = "true"

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
			model: MambaLMHeadModel):

		self.mamba_args = mamba_args
		self.train_args = train_args
		self.model = model
		self.device = self.model.lm_head.weight.device

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


	def train_sequence_steps(self, train_loader: DataLoader, val_loader: DataLoader, 
		use_amp: bool, log_freq: int, n_val: int, train_backprop: bool=True, 
		train_decorr: bool=True, save_checkpoints: bool=True, pad_idx: int = None):

		''' 
		Trains the model with the protocol specified in train_args. Trains based
		on a fixed number of gradient descent steps, performs validation
		at pre-defined points within the training loop. 

		'''

		# Index used for sequence length padding in proteome modelling task
		if pad_idx:
			criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
		else:
			criterion = nn.CrossEntropyLoss()

		if not train_backprop:
			print("Warning: not training backpropagation parameters!")
		if not train_decorr:
			print("Warning: not training decorrelation parameters!")
		assert train_backprop or train_decorr, "Specify something to train"

		if not isinstance(self.model, DecorrMamba) and train_decorr:
			print("Warning: train_decorr set to True but model does not use decorrelation!")

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
		
		save_path = os.path.join(".", "checkpoints")
		os.makedirs(save_path, exist_ok=True)

		scaler = torch.amp.GradScaler(self.device.type, enabled=use_amp)

		validate_every = int(self.train_args.n_steps/n_val)
		train_iterator = iter(train_loader)

		# "epoch" isn't quite correct here, but alas
		epoch_train_ce_loss = 0.0
		epoch_train_ppl = 0.0
		epoch_train_corr_loss = 0.0
		epoch_train_whit_loss = 0.0

		if train_backprop:
			self.model.train()
		else:
			self.model.eval()

		for step in tqdm(range(1, self.train_args.n_steps+1)):
			# an infinite loop for the fixed number of gradient descent 
			# steps
			try:
				next_batch = next(train_iterator)
			except StopIteration:
				train_iterator = iter(train_loader)  # Reset the iterator
				next_batch = next(train_iterator)

			in_seq = next_batch.long().to(self.device, non_blocking=True)
			b = in_seq.shape[0] # Needed for decorr input reshaping 

			if train_backprop:
				optimizer.zero_grad()
			
			if isinstance(self.model, DecorrMamba):
				if self.model.compute_loss or self.model.training:
					self.model.reset_decorr()

			with torch.amp.autocast(self.device.type, enabled=use_amp):
				# shift input sequence by one token and compare
				with torch.enable_grad() if train_backprop else torch.no_grad():
					pred = self.model(in_seq[:,:-1]).logits

				target = in_seq[:,1:].contiguous()
				# NB: ignore the irrelevant extra dimensions in the output,
				# those are just there for padding (GPU efficiency). Collapse
				# across batch and length before feeding to model.
				output_dim = pred.shape[-1]
				loss = criterion(
					pred.view(-1, output_dim)[:,:self.mamba_args.vocab_size],
					target.view(-1))	

			epoch_train_ce_loss += loss.item()
			ppl = torch.exp(loss).item()
			epoch_train_ppl += ppl

			if step%log_freq == 0:
				wandb.log({"train_ce_loss": loss.item(), 
			   				"train_ppl": ppl}, step=step)					

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
			if isinstance(self.model, DecorrMamba):
				# torch.cuda.synchronize()  # ensure all async operations finish
				if self.model.compute_loss or self.model.training:
					self.model.decorr_operations(b=b)
					self.model.mean_decorr_losses()	
				
				train_corr_loss = self.model.mean_corr_loss
				train_whit_loss = self.model.mean_whit_loss

				if train_corr_loss is not None:
					train_corr_loss = train_corr_loss.item()
					epoch_train_corr_loss += train_corr_loss
				if train_whit_loss is not None:
					train_whit_loss = train_whit_loss.item()
					epoch_train_whit_loss += train_whit_loss
				if step%log_freq == 0:
					wandb.log({"train_corr_loss": train_corr_loss, 
							"train_whit_loss": train_whit_loss}, step=step)	
					

				if train_decorr:
					self.model.update_decorr_matrices()

			# Condition checking if validate_every number of gradient descent
			# steps have happened
			
			if step%validate_every == 0:
				
				epoch_train_ce_loss /= 	validate_every
				epoch_train_ppl /= validate_every			
				epoch_train_corr_loss /= validate_every
				epoch_train_whit_loss /= validate_every

				print(f"\"Epoch\" train CE loss: {epoch_train_ce_loss:.4f}")
				print(f"\"Epoch\" train perplexity: {epoch_train_ppl:.4f}")
				if isinstance(self.model, DecorrMamba):	
					if epoch_train_corr_loss > 0:		
						print(f"\"Epoch\" train correlation loss: {epoch_train_corr_loss:.4f}")
					if epoch_train_whit_loss > 0:						
						print(f"\"Epoch\" train whitening loss: {epoch_train_whit_loss:.4f}")	
				
				# reset these values for the next "epoch"
				epoch_train_ce_loss = 0.0
				epoch_train_corr_loss = 0.0
				epoch_train_whit_loss = 0.0
				epoch_train_ppl = 0.0

				# -------------------------------- validation -------------------------------------	

				# Don't want to compute whitening/correlation losses here because
				# it makes validation extremely slow, if cross entropy is lower 
				# that's all that really matters for the most part
				
				self.model.eval()

				total_val_ce_loss = 0.0
				total_val_ppl = 0.0
				# total_val_corr_loss = 0.0
				# total_val_whit_loss = 0.0	

				with torch.no_grad():
					with torch.amp.autocast(self.device.type, enabled=use_amp):	
						
						for next_batch in tqdm(val_loader):
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
									
							total_val_ce_loss += loss.item()
							ppl = torch.exp(loss).item()
							total_val_ppl += ppl

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
				total_val_ppl /= len(val_loader)
				print(f"\n\"Epoch\" val perplexity: {total_val_ppl:.4f}")				
				print(f"\"Epoch\" val CE loss: {total_val_ce_loss:.4f}")

				wandb.log({
					"val_ce_loss": total_val_ce_loss,
					"val_ppl": total_val_ppl}, step=step)
				
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

