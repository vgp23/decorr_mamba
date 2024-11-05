from torch.utils.data import DataLoader
import math
import torch  
import torch.nn as nn
from tqdm import tqdm 
import os

# from einops import rearrange, repeat # might be useful later
from utils.helpers import MambaArgs, TrainingArgs
from model.mamba import Mamba

class MambaTrainer:
	''' Trains a Mamba architecture according to a pre-specified configuration

		Args:
			mamba_args (MambaArgs): model specification
			train_args (TrainingArgs): training protocol specification
			model (Mamba): implementation of Mamba architecture as per mamba_args

		Attributes:
			mamba_args (MambaArgs)
			train_args (TrainingArgs)
			model (Mamba)

		Methods:
			train(self, train_loader, val_loader): trains the architecture
				following the protocol in train_args, using provided 
				training and validation datasets
	'''
	def __init__(self, 
			mamba_args: MambaArgs, train_args: TrainingArgs, model: Mamba):

		self.mamba_args = mamba_args
		self.train_args = train_args
		self.model = model


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

	def train(self, train_loader: DataLoader, val_loader: DataLoader):
		''' Trains the model with the protocol specified in train_args.

		Args:
			train_loader (DataLoader): PyTorch-compatible training dataloader
			val_loader (DataLoader): PyTorch-compatible validation dataloader

		'''
		
		criterion = nn.CrossEntropyLoss()

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

		for epoch in range(self.train_args.n_epochs):
		    print(f"Epoch: {epoch + 1}/{self.train_args.n_epochs}")

		    self.model.train()
		    train_loss = 0.0
		    
		    for in_seq, target_seq in tqdm(train_loader):
		        out_seq = self.model(in_seq)
		        loss = criterion(out_seq.view(-1, self.mamba_args.vocab_size), target_seq.view(-1))
		        train_loss += loss.item()
		        optimizer.zero_grad()
		        loss.backward()

		        # gradient clipping
		        if self.train_args.gradient_clip is not None:
		        	torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
		        								   self.train_args.gradient_clip)


		        optimizer.step()
		        if self.train_args.use_lr_sched:
		        	scheduler.step()

		    train_loss /= len(train_loader)
		    train_perplexity = math.exp(train_loss)
		    print(f"Train loss: {train_loss:.4f}, Train perplexity: {train_perplexity:.4f}")

		    self.model.eval()
		    val_loss = 0.0

		    with torch.no_grad():
		        for in_seq, target_seq in val_loader:
		            out_seq = self.model(in_seq)
		            loss = criterion(out_seq.view(-1, self.mamba_args.vocab_size), target_seq.view(-1))
		            val_loss += loss.item()

		    val_loss /= len(val_loader)
		    val_perplexity = math.exp(val_loss)
		    print(f"Val loss: {val_loss:.4f}, Val perplexity: {val_perplexity:.4f}")

		    if val_loss < min_loss:
		        min_loss = val_loss

		        torch.save(self.model.state_dict(), os.path.join(save_path, f"epoch_{epoch}.pt"))

		return self.model



