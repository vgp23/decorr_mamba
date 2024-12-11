from torch.utils.data import DataLoader
import math
import torch  
import torch.nn as nn
from tqdm import tqdm 
import os
import json
import wandb

from ..utils.helpers import MambaArgs, TrainingArgs
from ..model.mamba import Mamba
from ..model.decorrelation import DecorrMamba

class MambaTrainer:
	''' Trains a Mamba architecture according to a pre-specified configuration

		Args:
			mamba_args (MambaArgs): model specification
			train_args (TrainingArgs): training protocol specification
			model (Mamba): implementation of Mamba architecture as per mamba_args
			device (str): device on which to train

		Attributes:
			mamba_args (MambaArgs)
			train_args (TrainingArgs)
			model (Mamba)

		Methods:
			train(self, train_loader, val_loader, backprop): trains the architecture
				following the protocol in train_args, using provided 
				training and validation datasets
	'''
	def __init__(self, 
			mamba_args: MambaArgs, train_args: TrainingArgs, model: Mamba, device: str):

		self.mamba_args = mamba_args
		self.train_args = train_args
		self.model = model
		self.device = device


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

	def train(self, train_loader: DataLoader, val_loader: DataLoader, 
		backprop: bool=True, save_checkpoints: bool=True, save_all_checkpoints: bool=False):

		''' 
		Trains the model with the protocol specified in train_args.

		Args:
			train_loader (DataLoader): PyTorch-compatible training dataloader
			val_loader (DataLoader): PyTorch-compatible validation dataloader
			backprop (bool, optional): turns off parameter updating for everything
				other than decorrelation matrices, allows for sanity check
				of decorrelation learning rule. Defaults to 'True'
			save_checkpoints (bool, optional): controls whether checkpoints are saved
				during epochs. Defaults to 'True'
			save_all_checkpoints (bool, optional): if saving checkpoints, controls
				whether all epoch checkpoints are saved or just those where the loss
				is better than the previous minimum. Defaults to 'False'	

		'''

		criterion = nn.CrossEntropyLoss()
		if not backprop:
			print("Warning: only training decorrelation matrices!")

		if not save_checkpoints:
			assert not save_all_checkpoints, \
			"Cannot save all checkpoints, as save_checkpoints is set to False." 

		# used in language modelling, usually
		if self.train_args.weight_decay is not None:
			# only apply decay to specific parameters
			optimizer = torch.optim.AdamW(
				[{'params': self._param_groups['decay'],
				  'weight_decay': self.train_args.weight_decay}, 

				 {'params': self._param_groups['no_decay'], 
				  'weight_decay': 0.0}], 

				  lr=self.train_args.lr,
				  betas=self.train_args.adam_beta,
				  eps=self.train_args.adam_epsilon)


		else: # used in synthetic tasks
			optimizer = torch.optim.Adam(self.model.parameters(), 
										lr=self.train_args.lr, 
										betas=self.train_args.adam_beta,
										eps=self.train_args.adam_epsilon)    



		if self.train_args.use_lr_sched:
			scheduler = torch.optim.lr_scheduler.LambdaLR(
				optimizer, lr_lambda=self.train_args.schedule_fn)

		min_loss = float("inf")
		
		save_path = os.path.join(".", "checkpoints")
		os.makedirs(save_path, exist_ok=True)

		# tracks losses across each epoch
		train_ce_losses = []
		train_perplexities = []
		val_ce_losses = []	
		val_perplexities = []

		if isinstance(self.model, DecorrMamba):	
			train_corr_losses = []
			train_whit_losses = []

			val_corr_losses = []
			val_whit_losses = []

		# needed for mixed precision training
		scaler = torch.amp.GradScaler(self.device.type)

		for epoch in range(self.train_args.n_epochs):
			print(f"Epoch: {epoch + 1}/{self.train_args.n_epochs}")

			self.model.train()

			if isinstance(self.model, DecorrMamba):
				# resets gradients and losses of decorrelation matrices
				self.model.reset_decorr_layers()

			train_loss = 0.0
		
			for in_seq, target_seq in tqdm(train_loader):

				if isinstance(self.model, DecorrMamba):
					# sets decorrelation matrix gradients to 0
					self.model.reset_decorr_grad()

				# automatic mixed precision
				with torch.amp.autocast(self.device.type):
					out_seq = self.model(in_seq.to(self.device, non_blocking=True))

					loss = criterion(
						out_seq.view(-1, self.mamba_args.vocab_size), 
						target_seq.to(self.device, non_blocking=True).view(-1))				
				
				train_loss += loss.item()

				if backprop:
					optimizer.zero_grad()
					scaler.scale(loss).backward()

				# gradient clipping
				if self.train_args.gradient_clip is not None:
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
												   self.train_args.gradient_clip)

				if backprop:
					scaler.step(optimizer)
					scaler.update()

				# update the decorrelation matrices AFTER standard backprop, else training breaks!
				if isinstance(self.model, DecorrMamba):
					# gradients internally computed during forward pass
					self.model.update_decorr_matrices()		
 
				if self.train_args.use_lr_sched:
					# doesn't affect decorrelation lr
					scheduler.step()

			train_loss /= len(train_loader)
			train_perplexity = math.exp(train_loss)
			print(f"Train loss: {train_loss:.4f}, Train perplexity: {train_perplexity:.4f}")

			train_ce_losses.append(train_loss)
			train_perplexities.append(train_perplexity)
			wandb.log(
				{"train_ce_loss": train_loss, 
	 			 "train_perplex": train_perplexity,
				 "epoch": epoch + 1})

			if isinstance(self.model, DecorrMamba):
				# epoch losses are summed automatically for each decorrelation layer within the model.
				# this just sums all of these sums across every decorrelation layer and stores
				# them within the parent model
				self.model.sum_decorr_losses()			
				total_correlation_loss = self.model.total_correlation_loss.item()
				total_whitening_loss = self.model.total_whitening_loss.item()

				correlation_loss = total_correlation_loss / len(train_loader)
				whitening_loss = total_whitening_loss / len(train_loader)
				print(
					f"Train correlation loss: {correlation_loss:.4f}, Train whitening loss: {whitening_loss:.4f}")

				train_corr_losses.append(correlation_loss)
				train_whit_losses.append(whitening_loss)
				wandb.log({"train_corr_loss": correlation_loss, 
			   			   "train_whit_loss": whitening_loss,
						   "epoch": epoch + 1})


			# -------------------------------- validation -------------------------------------	

			# apply_to_decorr(self.model, lambda module: print(getattr(module, "decorr_layer")))

			self.model.eval()

			if isinstance(self.model, DecorrMamba):
				# resets gradients and losses of decorrelation matrices
				self.model.reset_decorr_layers()	

			val_loss = 0.0

			with torch.no_grad():
				for in_seq, target_seq in val_loader:
					out_seq = self.model(in_seq.to(self.device, non_blocking=True))

					loss = criterion(
						out_seq.view(-1, self.mamba_args.vocab_size), 
						target_seq.to(self.device, non_blocking=True).view(-1))
					
					val_loss += loss.item()


			val_loss /= len(val_loader)
			val_perplexity = math.exp(val_loss)
			print(f"Val loss: {val_loss:.4f}, Val perplexity: {val_perplexity:.4f}")

			val_ce_losses.append(val_loss)
			val_perplexities.append(val_perplexity)
			wandb.log({
				"val_ce_loss": val_loss, 
				"val_perplex": val_perplexity,
				"epoch": epoch + 1})

			if isinstance(self.model, DecorrMamba):
				self.model.sum_decorr_losses()			
				total_correlation_loss = self.model.total_correlation_loss.item()
				total_whitening_loss = self.model.total_whitening_loss.item()

				correlation_loss = total_correlation_loss / len(train_loader)
				whitening_loss = total_whitening_loss / len(train_loader)
				print(f"Val correlation loss: {correlation_loss:.4f}, Val whitening loss: {whitening_loss:.4f}")	

				val_corr_losses.append(correlation_loss)
				val_whit_losses.append(whitening_loss)	
				wandb.log({
					"val_corr_loss": correlation_loss, 
					"val_whit_loss": whitening_loss,
					"epoch": epoch + 1})

			# saving model checkpoints and performance info

			if save_checkpoints and save_all_checkpoints:
				torch.save(
					self.model.state_dict(), os.path.join(save_path, f"epoch_{epoch}.pt"))

			if val_loss < min_loss:
				min_loss = val_loss
				if save_checkpoints and not save_all_checkpoints:
						torch.save(
							self.model.state_dict(), os.path.join(save_path, f"epoch_{epoch}.pt"))
						
			if isinstance(self.model, DecorrMamba):
				metrics = {
				
					"train_perplex": train_perplexities,
					"val_perplex": val_perplexities,
					"train_ce_loss": train_ce_losses,
					"val_ce_loss": val_ce_losses,

					"train_corr_loss": train_corr_losses,
					"val_corr_loss": val_corr_losses,
					"train_whit_loss": train_whit_losses,
					"val_whit_loss": val_whit_losses
				}
			else:
				metrics = {
				
					"train_perplex": train_perplexities,
					"val_perplex": val_perplexities,
					"train_ce_loss": train_ce_losses,
					"val_ce_loss": val_ce_losses,
				}				

			with open(
				os.path.join(save_path, "metrics.json"), "w") as json_file:
				json.dump(metrics, json_file)


		return self.model



