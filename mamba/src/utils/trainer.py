from torch.utils.data import DataLoader
import math
import torch  
import torch.nn as nn
from tqdm import tqdm 

# from einops import rearrange, repeat # might be useful later
from utils.helpers import MambaArgs, LMTrainingArgs
from model.mamba import Mamba

class MambaTrainer:
	''' Trains a Mamba architecture according to a pre-specified configuration'''
	def __init__(self, 
			mamba_args: MambaArgs, train_args: LMTrainingArgs, model: Mamba):

		self.mamba_args = mamba_args
		self.train_args = train_args
		self.model = model


		def add_param_to_groups(module, param_groups):
		    """
		    Adds the parameters of the module to the appropriate param_groups list
		    based on the presence of the _no_weight_decay attribute on the parameters.
		    
		    Args:
		        module: a submodule of the model.
		        param_groups: a dictionary containing 'decay' and 'no_decay' lists.
		    """
		    for name, param in module.named_parameters(recurse=False):
		        # Check if the parameter has the _no_weight_decay attribute
		        if hasattr(param, '_no_weight_decay') and param._no_weight_decay:
		            param_groups['no_decay'].append(param)
		        else:
		            param_groups['decay'].append(param)


		# collect parts of the model we don't want weight decay for
		self._param_groups = {'decay': [], 'no_decay': []}
		self.model.apply(lambda module: add_param_to_groups(module, self._param_groups))
		# weight tying causes the embedding and output weights to be the same,
		# but the logic above counts this parameter twice. Remove to fix.
		del self._param_groups["decay"][-1]

	def train(self, 
			  train_loader: DataLoader, 
			  val_loader: DataLoader):
	
		# TODO: implement checkpoint saving functionality

		criterion = nn.CrossEntropyLoss()

		if self.train_args.optimizer == "AdamW":
		    optimizer = torch.optim.AdamW(
		    	[{'params': self._param_groups['decay'],
		    	  'weight_decay': self.train_args.weight_decay}, 

		    	 {'params': self._param_groups['no_decay'], 
		    	  'weight_decay': 0.0}], 

		    	  lr=1e-3,
		    	  betas=self.train_args.adam_beta,
		          eps=self.train_args.adam_epsilon)


		elif self.train_args.optimizer == "Adam": # used in synthetic tasks
		    optimizer = torch.optim.Adam(self.model.parameters(), 
		                                lr=self.train_args.peak_lr, 
		                                betas=self.train_args.adam_beta,
		                                eps=self.train_args.adam_epsilon)    

		scheduler = torch.optim.lr_scheduler.LambdaLR(
			optimizer, lr_lambda=self.train_args.schedule_fn)


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
		        	torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_args.gradient_clip)


		        optimizer.step()
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

		return self.model



