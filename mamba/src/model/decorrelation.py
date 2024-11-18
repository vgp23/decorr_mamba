import torch
import torch.nn as nn
import torch.nn.functional as F 
from model.mamba import Mamba
from einops import einsum
from utils.helpers import MambaArgs
from functools import partial
from copy import deepcopy

class DecorrLoss(nn.Module):
	''' 
	Computes the gradients and losses associated with the decorrelation update
	'''

	def __init__(self):
		super(DecorrLoss, self).__init__()


	def forward(self, x, kappa: float, compute_grad: bool = True, 
		compute_loss: bool = True):

		if not compute_grad and not compute_loss:
			# shouldn't ever be reached during normal use, but here for
			# completeness
			return None, None, None

		assert kappa is not None, "Specify kappa for loss and gradient computation"
		assert kappa <= 1.0 and kappa >= 0.0, "kappa must be between 0 and 1"

		# collapse across batch dimension
		b, l, d = x.shape
		x = x.reshape(b*l, d)

		# compute the individual loss elements
		D = torch.diag_embed(x**2)
		V = D - torch.eye(d)

		xx_t = einsum(x, x, 'b x, b x_t -> b x x_t')

		C = xx_t - D

		if compute_loss:
			# sum of squared covariances
			correlation_loss = \
				torch.mean(
					torch.mean(C**2, dim=(1,2)))

 			# sum of squared variances
			whitening_loss = \
				torch.mean(
					torch.mean(V**2, dim=(1,2)))

		else:
			correlation_loss = None 
			whitening_loss = None

		# compute the actual gradient, if applicable
		if compute_grad:
			grad = torch.mean(((1-kappa)*C + kappa*V), dim=0)
		else:
			grad = None

		return grad, correlation_loss, whitening_loss
	

class DecorrLinear(nn.Module):
	''' 
	Prefaces original nn.Linear layers in a network
	with a trainable matrix multiplication initialized
	at identity.
	'''

	def __init__(self, original_layer: nn.Module, **kwargs):

		super(DecorrLinear, self).__init__()

		self.original_layer = original_layer
		# a layer of the same dimensions as the original, with no biases.
		# gradients are computed outside of the backward pass
		self.decorr_layer = nn.Parameter(
			torch.eye(original_layer.weight.shape[1]), requires_grad=False)

		self.compute_grad = True # controlled by self.train()
		self.compute_loss = True 

		self.sample_frac = kwargs.get("sample_frac")
		self.kappa = kwargs.get("kappa")

		self.correlation_loss = 0
		self.whitening_loss = 0
		self.grad = None

		self.loss = DecorrLoss()


	def forward(self, x):

		# for efficiency, multiply decorrelation and standard weight matrices together before
		# passing input through the resulting single matrix
		y = x @ (self.original_layer.weight @ self.decorr_layer).T

		if self.original_layer.bias is not None:
			y += self.original_layer.bias

		# caveat: optimization above requires another forward pass through just the
		# decorrelation matrix to compute the loss and gradients
		if self.compute_grad or self.compute_loss:

			# sample only a fraction of each batch
			assert self.sample_frac is not None , \
				"Specify sample_frac for loss and gradient computation"

			assert self.sample_frac > 0 and self.sample_frac <= 1.0, \
				"sample_frac must be between 0 and 1"   

			with torch.no_grad():
				b, l, d = x.shape
				n_samples = int(self.sample_frac*l)

				sample_idx = torch.multinomial(
					torch.ones(b, l), n_samples, replacement=False)
				batch_idx = torch.arange(b).unsqueeze(1).expand(b, n_samples)

				selected = x[batch_idx, sample_idx]

				selected_decorr = selected @ self.decorr_layer.T

				grad, correlation_loss, whitening_loss = self.loss(
					selected_decorr, self.kappa, self.compute_grad, self.compute_loss)

				self.correlation_loss += correlation_loss
				self.whitening_loss += whitening_loss
				# updates happen on entire network at once in training loop
				self.grad = grad 

		return y

	def reset(self):
		self.correlation_loss = 0
		self.whitening_loss = 0
		self.grad = None

	def reset_grad(self):
		self.grad = None

	def train(self, mode: bool = True):
		super(DecorrLinear, self).train(mode)
		# enable computing of decorrelation gradients during forward pass
		self.compute_grad = mode

		if mode:
			# allows loss to also be computed in eval mode      
			self.compute_loss = True



class DecorrConv1d(DecorrLinear):
	''' 
	Prefaces original nn.Conv1d layers in a network with a trainable matrix multiplication 
	initialized at identity. Works in two modes, either decorrelating features of each 
	token independently, or all of the features within each convolutional patch.
	''' 
	def __init__(self, original_layer: nn.Module, model_args: MambaArgs, mode: str = "patch", 
		**kwargs):

		super(DecorrConv1d, self).__init__(original_layer, **kwargs)

		self.model_args = model_args
		self.mode = mode

		# determines how the decorrelation is applied.
		assert mode == "token" or mode == "patch", "conv_1d_mode must be \"token\" or \"patch\""
		if mode == "token":
			# decorrelate each token's features independently
			self.decorr_layer = nn.Parameter(torch.eye(model_args.D_inner), requires_grad=False)
		else:
			# decorrelate all input features to the convolutional kernel at once
			self.decorr_layer = nn.Parameter(
				torch.eye(model_args.D_inner*model_args.conv_1d_size), requires_grad=False)


	def forward(self, x):

		if self.compute_grad or self.compute_loss:
			assert self.sample_frac is not None , \
				"Specify sample_frac for loss and gradient computation"

			assert self.sample_frac > 0 and self.sample_frac <= 1.0, \
				"sample_frac must be between 0 and 1"   

		b, d, l = x.shape
		# (B, n_patches, conv_1d_size*D). All data in each convolution patch
		# is represented in a single vector
		x_unfolded = F.unfold(
			x.unsqueeze(1), (d, self.model_args.conv_1d_size), 
			stride=1, padding=(0, self.model_args.conv_1d_size-1)).transpose(1,2)

		# decorrelates each token's features independently. As in DecorrLinear, decorrelation
		# and convolutional operations are fused together into one matrix. Convolutions
		# in Mamba only mix information across the same token dimension,
		# (groups = D_inner), making the convolution operation a series of dot 
		# products within the unfolded patch, not a regular matrix multiplication.

		if self.mode == "token":
			# currently each patch's features are represented in
			# a single vector. Reshape these vectors into matrices, such
			# that the 0th dimension of each matrix contains all features
			# for the 0th dimension across all tokens, etc. 

			#( B, n_patches, conv_1d_size, D)
			patch_matrices = x_unfolded.reshape(
				b, -1, d, self.model_args.conv_1d_size).transpose(2,3)

			# decorr + conv1d for one patch is a matrix multiplication
			# of the matrix representation of the patch with the decorrelation
			# matrix, followed by row-wise dot products of the output with the 
			# convolution weights. This can be fused into one operation by defining
			# a separate scaled version of the decorrelation matrix for each token
			# in a convolutional patch, passing the tokens through their respective
			# matrices, and then summing over the token dimension

			token_specific_decorr = self.decorr_layer.unsqueeze(0).repeat(
				self.model_args.conv_1d_size, 1, 1)

			token_specific_decorr = einsum(token_specific_decorr, 
				torch.squeeze(self.original_layer.weight).T,

				'conv_1d_size out_dim in_dim, conv_1d_size out_dim' +
				' -> conv_1d_size out_dim in_dim')

			decorr_conv_outputs = einsum(patch_matrices, token_specific_decorr,
				'b n_patches conv_1d_size in_dim, conv_1d_size out_dim in_dim' + 
				' -> b n_patches conv_1d_size out_dim')

			# (B, n_patches, D). Summing over the decorrelated + scaled tokens
			# in each patch completes the dot product operation. 
			y = torch.sum(decorr_conv_outputs, dim=2)
			
			if self.original_layer.bias is not None:
				y += self.original_layer.bias 

			# like in DecorrLinear, fusion of decorrelation and layer operation requires
			# separate forward pass for gradient computation   
			if self.compute_grad or self.compute_loss:     
				with torch.no_grad():
					# sample fraction of each batch from which to compute loss + update
					x_t = x.transpose(1,2)

					n_samples = int(self.sample_frac*l)

					sample_idx = torch.multinomial(
						torch.ones(b, l), n_samples, replacement=False)
					batch_idx = torch.arange(b).unsqueeze(1).expand(b, n_samples)

					selected = x_t[batch_idx, sample_idx]

					selected_decorr = selected @ self.decorr_layer.T

					grad, correlation_loss, whitening_loss = self.loss(
						selected_decorr, self.kappa, self.compute_grad, self.compute_loss)

					self.correlation_loss += correlation_loss
					self.whitening_loss += whitening_loss
					# updates happen on entire network at once in training loop
					self.grad = grad   				      

		# decorrelates all features in each input patch to the convolution
		else:
			# split up decorrelation matrix to pre-multiply with each channel's kernel
			# (D, D*conv_1d_size (= n_patch_features), conv_1d_size)
			decorr_reshape = self.decorr_layer.reshape(
				d, self.model_args.conv_1d_size, d*self.model_args.conv_1d_size).transpose(1,2)

			fused_matrix = einsum(torch.squeeze(self.original_layer.weight), decorr_reshape,
				'd conv_1d_size, d n_patch_features conv_1d_size -> d n_patch_features')

			y = x_unfolded @ fused_matrix.T

			if self.original_layer.bias is not None:
				y += self.original_layer.bias

			if self.compute_grad or self.compute_loss:       
				with torch.no_grad():
					# sample fraction of each batch from which to compute loss + update
					b, n_patches, n_patch_features = x_unfolded.shape

					n_samples = int(self.sample_frac*n_patches)

					sample_idx = torch.multinomial(
						torch.ones(b, n_patches), n_samples, replacement=False)
					batch_idx = torch.arange(b).unsqueeze(1).expand(b, n_samples)

					selected_decorr = x_unfolded[batch_idx, sample_idx] @ self.decorr_layer.T

					grad, correlation_loss, whitening_loss = self.loss(
						selected_decorr, self.kappa, self.compute_grad, self.compute_loss)

					self.correlation_loss += correlation_loss
					self.whitening_loss += whitening_loss
					# updates happen on entire network at once in training loop
					self.grad = grad                

		return y.transpose(1,2)

def apply_to_decorr(model, f):
	''' 
	Recursively traverses a model's structure and applies an arbitrary
	function to the modules containing decorrelation layers. Used
	for printing things out, performing simple operations, etc.
	'''
	def _apply_to_decorr(module):
		for child in module.children():
			if isinstance(child, DecorrLinear):
				f(child)

	model.apply(_apply_to_decorr)


class DecorrMamba(Mamba):
	''' 
	Mamba architecture with built-in decorrelation layers and additional functionality
	for training/monitoring them

	'''
	def __init__(self,  conv_1d_mode: str, model_args: MambaArgs = None, 
		existing_model: Mamba = None, **kwargs):

		# creates a standard Mamba model according to args, and then inserts 
		# decorrelation matrices where appropriate. If a Mamba model is already
		# specified, just extends the existing model with added functionality and
		# decorrelation layers initialized at identity

		assert existing_model is not None or model_args is not None, \
			"Specify either a MambaArgs object to create a new model," +\
			" or a pre-made Mamba model to modify"

		if existing_model is not None:
			self.__dict__.update(deepcopy(existing_model).__dict__)
			if model_args is not None:
				print(
					"Warning: supplied args overwritten by the args of the existing model")
		else:
			super(DecorrMamba, self).__init__(model_args)


		def _create_decorr_matrices(module, kappa, sample_frac, conv_1d_mode, model_args):
			''' 
			Used to recursively traverse model and create decorrelation matrices 
			in pre-defined places
			'''
			for name, child in module.named_children():
				if name == "in_proj" or name == "out_proj" or name == "to_BCdelta":
					setattr(module, name, DecorrLinear(child, kappa=kappa, sample_frac=sample_frac))

				if name == "conv1d":
					setattr(module, name, DecorrConv1d(
						child, model_args, conv_1d_mode, kappa=kappa, sample_frac=sample_frac))


		self.apply(partial(_create_decorr_matrices, 
			kappa=kwargs.get("kappa"), sample_frac=kwargs.get("sample_frac"), 
			conv_1d_mode=conv_1d_mode, model_args=self.model_args))

		# remove weight decay for decorrelation layers
		apply_to_decorr(self, 
			lambda decorr_module: setattr(
				getattr(decorr_module, "decorr_layer"), "_no_weight_decay", True))


		self.total_correlation_loss = 0
		self.total_whitening_loss = 0
		self.decorr_lr = kwargs.get("decorr_lr")

		# these are just here for reference, individual decorrelation layers
		# take care of using these values automatically
		self.conv_1d_mode = conv_1d_mode
		self.sample_frac = kwargs.get("sample_frac")
		self.kappa = kwargs.get("kappa")

	def sum_decorr_losses(self):
		''' 
		Calculates the summed total of correlation and whitening losses across all 
		layers implementing decorrelation, for a Mamba model
		'''

		def _sum_losses(module):

			for child in module.children():
				# DecorrConv1d extends DecorrLinear, should account for 
				# convolutional layers too
				if isinstance(child, DecorrLinear):
					self.total_correlation_loss += child.correlation_loss
					self.total_whitening_loss += child.whitening_loss

		self.apply(_sum_losses)

	def update_decorr_matrices(self):
		''' 
		Updates the decorrelation matrices for all decorrelation layers
		within the model
		'''
		assert self.training, "Model must be in training mode"
		assert self.decorr_lr is not None, "No decorr_lr specified"

		# gradients should already have been computed during the forward pass
		def _update_decorr_matrices(module):
			for child in module.children():
				if isinstance(child, DecorrLinear):
					assert child.grad is not None, "Gradient not computed"
					child.decorr_layer -= self.decorr_lr * child.grad @ child.decorr_layer

		self.apply(_update_decorr_matrices)

	def reset_decorr_grad(self):
		''' Resets gradients of decorrelation matrices after forward pass'''
		apply_to_decorr(self, lambda x: x.reset_grad())

	def reset_decorr_layers(self):
		''' 
		Resets gradients and losses of decorrelation layers after parameter
		updates. Also resets the summed total losses across all decorrelation
		layers
		'''
		apply_to_decorr(self, lambda x: x.reset())
		self.total_correlation_loss = 0
		self.total_whitening_loss = 0

	def compute_decorr_losses(self, mode: bool=True):
		'''
		Turns the whitening and correlation loss computation during the 
		forward pass on and off.
		'''
		apply_to_decorr(self, lambda x: setattr(x, "compute_loss", mode))

			





	
