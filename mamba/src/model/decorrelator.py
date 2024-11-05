import torch
import torch.nn as nn
import torch.nn.functional as F 
from model.mamba import Mamba

class DecorrelationGradient(nn.Module):
	''' 
	Computes the gradient required for the decorrelation update
	'''

	def __init__(self, kappa = 0.5):
		super(DecorrelationGradient, self).__init__()
		self.kappa = kappa

	def forward(self, x, sample_indices=None):
		# reshape for sampling. Collapse across batch and sequence 
		# length dimensions.
		b, l, d = x.shape
		x = x.reshape(b*l, d)

		# sample only the values we wish to use
		if sample_indices is not None:
			x = x[sample_indices]

		# compute the gradient
		D = torch.diag_embed(x**2)
		V = D-1

		x = x.unsqueeze(-1)
		x_t = x.transpose(-2, -1)
		xx_t = torch.matmul(x, x_t)
		C = xx_t - D

		return torch.mean(((1-self.kappa)*C + self.kappa*V), dim=0)



class Decorrelator():
	def __init__(self):
		...

	@staticmethod
	def merge_AW(model: Mamba):
		''' 
		Merges the decorrelation and "main" model weights into a single matrix,
		used for efficient forward pass following training
		'''
		# TODO: implement this
		pass


	def add_decorrelation_layers(model: Mamba, kappa:float = 0.5):
		''' Modifies standard Mamba architecture to include input-decorrelating
			layers in the correct places. Modifies the forward pass of the model
			accordingly.
		'''

		# adding the required decorrelation matrices, without biases
		for layer in model.layers: # individual residual Mamba blocks

			# ---- mamba block modifications ----

			# input projections
			setattr(layer.block, "in_proj_decorr", 
				nn.Parameter(
					torch.randn((model.args.D, model.args.D))
					))
			# convolutional kernel. NB slightly different from what the decorrelation
			# paper proposes for convolutions, but for good reason
			setattr(layer.block, "conv1d_decorr", 
				nn.Parameter(
					torch.randn((model.args.D_inner, model.args.D_inner))
					))	
			# output projections
			setattr(layer.block, "out_proj_decorr", 
				nn.Parameter(
					torch.randn((model.args.D_inner, model.args.D_inner))
					))
			# used to compute gradient
			setattr(layer.block, "decorr_grad", DecorrelationGradient(kappa))

			# ---- s6 block modifications ----

			# parameter upscale projections
			setattr(layer.block.s6_block, "to_BCdelta_decorr", 
				nn.Parameter(
					torch.randn((model.args.D_inner, model.args.D_inner))
					))
			# delta "broadcast" projections
			setattr(layer.block.s6_block, "delta_upscale_decorr", 
				nn.Parameter(
					torch.randn((model.args.delta_rank, model.args.delta_rank))
					))
			# projecting hidden state through C to generate output
			setattr(layer.block.s6_block, "C_decorr", 
				nn.Parameter(
					torch.randn((model.args.D_inner*model.args.N, 
								 model.args.D_inner*model.args.N))
					))

			# used to compute gradient
			setattr(layer.block.s6_block, "decorr_grad", DecorrelationGradient(kappa))

		#TODO: INITIALIZE DECORRELATION WEIGHTS ACCORDING TO PAPER
		# IMPLEMENT MORE EFFICIENT MATRIX MULTIPLICATION FOR FORWARD PASS


		def s6_block_modified_forward(self, x):
			''' Forward pass function for s6 blocks with decorrelation '''

			b, l, d = x.shape 
			# generate all projected parameters and split them up
			x = x @ self.to_BCdelta_decorr.T
			to_BCdelta_decorr_grad = self.decorr_grad(x)

			BCdelta = self.to_BCdelta(x)

			# delta: (B, L, delta_rank). B, C: (B, L, N)
			(delta, B, C) = BCdelta.split(
			    split_size=[self.args.delta_rank, self.args.N, self.args.N], dim=-1)

			# "broadcasting" for delta and computing final parameters
			delta = delta @ self.delta_upscale_decorr.T
			delta_upscale_decorr_grad = self.decorr_grad(delta)

			delta = self.delta_upscale(delta) # (B,L,D)
			delta = F.softplus(delta)

			# discretization. NB that the discretized version of B is 
			# already applied to the input sequence here!
			A_bar, B_bar_x = self.discretize(delta, B, x) # (B, L, D, N)

			# scan through each individual token to compute hidden states
			hidden_states = torch.zeros(
			    b, l+1, self.args.D_inner, self.args.N).to(self.args.device)

			for i in range(0,l):
			    # because A is represented only through diagonal, Ah_t-1 is 
			    # equivalent to taking the elementwise product of the diagonal
			    # and the hidden state
			    hidden_states[:,i+1,:,:] = A_bar[:,i,:,:]*hidden_states[:,i,:,:].clone() + \
			        B_bar_x[:,i,:,:] # (B,D,N)

			# decorrelate hidden states before output transformation. First timepoint
			# is zeros, ignore
			hidden_states = hidden_states[:,1:,:,:].reshape(b, l, d*self.args.N)
			hidden_states = hidden_states @ self.C_decorr.T
			C_decorr_grad = self.decorr_grad(hidden_states)

			hidden_states = hidden_states.reshape(b,l,d,self.args.N)

			# compute outputs in parallel
			outputs = torch.einsum('bln, bldn -> bld', C, hidden_states)

			# throw in D as residual connections with no bias
			outputs = outputs + x * self.D.float()

			return outputs, (to_BCdelta_decorr_grad, delta_upscale_decorr_grad, C_decorr_grad)


		def mamba_block_modified_forward(self, x):
			''' 
			Forward pass function for Mamba blocks with decorrelation. 
			Wraps around the s6 block and returns the gradient updates for all
			decorrelation layers of the block
			'''

			b, l, _ = x.shape # used to avoid specifying these in the model args

			x =  x @ self.in_proj_decorr.T # equivalent to the "proper" multiplication
			in_proj_decorr_grad = self.decorr_grad(x)

			x = self.in_proj(x)
			# split the input into the two paths
			(x, res) = x.split(
			    split_size=[self.args.D_inner, self.args.D_inner], dim=-1)

			x = x @ self.conv1d_decorr.T
			conv1d_decorr_grad = self.decorr_grad(x)

			# input of shape (B,L,D), dimensions need switching for convolution
			x = torch.transpose(x, 1,2)
			x = self.conv1d(x)[:,:,:l] # the limit is needed because of the padding
			x = torch.transpose(x, 1,2)

			x = F.silu(x)
			x, s6_gradients = self.s6_block(x)
			x = x * F.silu(res)

			x = x @ self.out_proj_decorr.T 
			out_proj_decorr_grad = self.decorr_grad(x)

			y = self.out_proj(x)

			return y, (in_proj_decorr_grad, conv1d_decorr_grad, *s6_gradients, out_proj_decorr_grad)

		def residual_mamba_block_modified_forward(self, x):
			''' 
			Forward pass function for residual Mamba blocks with decorrelation.
			Modified to pass decorrelation gradients through
			'''
			y, block_decorr_grads = self.block(self.rms(x))
			return y + x, block_decorr_grads

		def full_mamba_modified_forward(self, x):
			''' 
			Forward pass function for complete Mamba model with decorrelation.
			Modified to pass decorrelation gradients through
			'''
			x = self.embedding(x)
			decorr_grads = []
			for layer in self.layers:
			    x, block_decorr_grads = layer(x)
			    decorr_grads.append(block_decorr_grads)
			    
			x = self.rms(x)
			logits = self.logits(x)

			return logits, decorr_grads

		# override all forward methods
		model.forward = full_mamba_modified_forward.__get__(model, model.__class__)

		for layer in model.layers:

			layer.forward = \
				residual_mamba_block_modified_forward.__get__(layer, layer.__class__)

			layer.block.forward = \
				mamba_block_modified_forward.__get__(layer.block, layer.block.__class__)

			layer.block.s6_block.forward = \
				s6_block_modified_forward.__get__(
					layer.block.s6_block, layer.block.s6_block.__class__)

		return model

