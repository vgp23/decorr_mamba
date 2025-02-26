import torch
import torch.nn as nn
from einops import einsum
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from functools import partial
from copy import deepcopy
import torch.nn.functional as F

class DecorrLoss(nn.Module):
	"""
	Computes the gradients and losses associated with the decorrelation update.

	This module calculates two types of losses:
	- **Correlation Loss**: Measures the sum of squared covariances between features.
	- **Whitening Loss**: Measures the sum of squared variances deviating from identity.
	
	It also computes the gradient used to update the decorrelation matrices in decorrelation layers.

	"""

	def __init__(self):
		super(DecorrLoss, self).__init__()

	def forward(self, x, kappa: float, compute_grad: bool, 
		compute_loss: bool, batched: bool):

		if not compute_grad and not compute_loss:
			# shouldn't ever be reached during normal use, but here for
			# completeness
			return None, None, None

		assert kappa is not None, "Specify kappa for loss and gradient computation"
		assert kappa <= 1.0 and kappa >= 0.0, "kappa must be between 0 and 1"



		# used for all modes where the decorrelation layer has only a single
		# matrix to train
		if not batched:
			# collapse input across the batch and length dimensions
			_, _, d = x.shape
			x = x.reshape(-1, d)
			# compute the individual loss elements
			D = torch.diag_embed(x**2)
			V = D - torch.eye(d, device=x.device)

			xx_t = einsum(x, x, 
				'n_samples x, n_samples x_t -> n_samples x x_t')

			C = xx_t - D

			if compute_loss:
				# mean of squared covariances across matrix, then
				# across the batch
				corr_loss = \
					torch.mean(
						torch.mean(C**2, dim=(1,2)))

				# mean of squared variances across matrix, then
				# across the batch
				whit_loss = \
					torch.mean(
						torch.mean(V**2, dim=(1,2)))

			else:
				corr_loss = None 
				whit_loss = None

			# compute the actual gradient, if applicable
			if compute_grad:
				grad = torch.mean(((1-kappa)*C + kappa*V), dim=0)
			else:
				grad = None

			return grad, corr_loss, whit_loss

		# used where decorrelation layer has multiple matrices to train
		# (this is the case for "channel_independent")
		else:
			# collapse across batch and n_patches dimension
			_, _, d, decorr_matrix_size = x.shape
			x = x.reshape(-1, d, decorr_matrix_size)
			# in this case we're updating d matrices, each with info
			# from one embedding dimension channel.

			# (D, all_samples, decorr_matrix_size)
			x = x.transpose(0, 1)

			# compute the individual loss elements
			D = torch.diag_embed(x**2)
			V = D - torch.eye(decorr_matrix_size, device=x.device)

			xx_t = einsum(x, x, 
				'd n_samples x, d n_samples x_t -> d n_samples x x_t')

			# (D, n_samples, decorr_matrix_size, decorr_matrix_size)
			C = xx_t - D

			if compute_loss:
				# mean of squared covariances, averaged across each matrix,
				# then across all samples, then across all parallel channels
				corr_loss = \
					torch.mean(
						torch.mean(
							torch.mean(C**2, dim=(2,3)), dim=1))

				# mean of squared variances, averaged across each matrix,
				# then across all samples, then across all parallel channels
				whit_loss = \
					torch.mean(
						torch.mean(
							torch.mean(V**2, dim=(2,3)), dim=1))

			else:
				corr_loss = None 
				whit_loss = None

			# compute the actual gradients, if applicable
			if compute_grad:
				grad = torch.mean(((1-kappa)*C + kappa*V), dim=1)
			else:
				grad = None

			return grad, corr_loss, whit_loss

class TrackedParameter(nn.Parameter):
	"""A wrapper around nn.Parameter that logs inputs during matrix multiplication."""
	
	def __new__(cls, data, requires_grad=True):
		instance = super(TrackedParameter, cls).__new__(cls, data)
		instance.requires_grad = requires_grad
		instance.inputs = []
		return instance
		

	def __matmul__(self, other):
		"""Intercepts matrix multiplication to log the input."""
		self.inputs.append(other.clone().detach())
		return super().__matmul__(self, other)

	def __rmatmul__(self, other):
		"""Tracks all tensors that multiplied self: x @ W"""
		self.inputs.append(other.clone().detach()) 
		return super().__rmatmul__(other)	

	def transpose(self, dim0, dim1):
		""" Ensures transposition is also tracked """
		# Make a new parameter, morifying the same inputs list as the original
		transposed_parameter = TrackedParameter(super().transpose(dim0, dim1))
		transposed_parameter.inputs = self.inputs
		return transposed_parameter

	@property
	def T(self):
		""" Handles W.T so it retains tracking """
		return self.transpose(0, 1)		

class DecorrLinear(nn.Linear):
	def __init__(self, original_layer: nn.Linear = None, **factory_kwargs):

		# allows layer to be created on top of existing one, or made from
		# scratch
		if original_layer is not None:
			self.__dict__.update(original_layer.__dict__)
		else:
			super(DecorrLinear, self).__init__(**factory_kwargs)

		self.decorr_layer = nn.Parameter(
			torch.eye(self.in_features).to(self.weight.device), requires_grad=False)

		self.register_parameter("original_weight", nn.Parameter(self.weight))
		print(self.original_weight.requires_grad)
		# delete self.weight, because we'll be overwriting it later
		del self._parameters["weight"]

		# fuse decorrelation and original weights into a single
		# weight matrix, then track its input to allow decorrelation
		# loss to be computed later. requires_grad is false, because we only want
		# to update original_weight with regular backprop. 

		# this tracks multiplications of form W @ X, W.T @ X, X @ W and X @ W.T
		fused_weight = TrackedParameter(
			self.original_weight @ self.decorr_layer, requires_grad=False)
		self.register_parameter("weight", fused_weight)

		# this tracks inputs during the standard forward pass
		self.register_forward_hook(self.track_weight_input)

		self.corr_loss = None
		self.whit_loss = None
		self.decorr_grad = None

	def track_weight_input(self, module, input, output):
		self.weight.inputs.append(input[0].clone().detach())

	def reset(self):
		self.corr_loss = None
		self.whit_loss = None
		self.decorr_grad = None
		self.weight.inputs = []


class DecorrConv1d(nn.Conv1d):
	def __init__(self, original_layer: nn.Conv1d = None, **factory_kwargs):

		if original_layer is not None:
			self.__dict__.update(original_layer.__dict__)
		else:
			super(DecorrConv1D, self).__init__(**factory_kwargs)

		# (in_channels, kernel_size, kernel_size)
		all_matrices = torch.eye(self.kernel_size[0]).unsqueeze(0).repeat(
					self.in_channels, 1, 1).to(self.weight.device)

		self.decorr_layer = nn.Parameter(all_matrices, requires_grad=False)

		self.register_parameter("original_weight", nn.Parameter(self.weight))
		# delete self.weight, because we'll be overwriting it later
		del self._parameters["weight"]

		# decorrelate all channels independently, for now. B dot (W @ A) is equivalent to
		# A dot (W.T @ B), using this property here.
		fused_weight = torch.unsqueeze(
			einsum(
				self.decorr_layer.data, torch.squeeze(self.original_weight.data),
				'd dummy conv_1d_size, d dummy -> d conv_1d_size'), 1)
		fused_weight = TrackedParameter(fused_weight, requires_grad=False)

		self.register_parameter("weight", fused_weight)

		# this tracks inputs during the standard forward pass
		self.register_forward_hook(self.track_weight_input)

		self.corr_loss = None
		self.whit_loss = None
		self.decorr_grad = None

	def track_weight_input(self, module, input, output):
		self.weight.inputs.append(input[0].clone().detach())

	def reset(self):
		self.corr_loss = None
		self.whit_loss = None
		self.decorr_grad = None
		self.weight.inputs = []

class DecorrMamba(MambaLMHeadModel):

	def __init__(self, existing_model: MambaLMHeadModel = None, copy: bool = False, 
			  kappa: float = 0.5, sample_frac: float = 0.1, decorr_lr = None,
			  **factory_kwargs):

		if existing_model is not None and copy:
			self.__dict__.update(deepcopy(existing_model).__dict__)
			if factory_kwargs.get("config") is not None:
				print(
					"Warning: supplied config overwritten by the config of the existing model")
		elif existing_model:
			self.__dict__.update(existing_model.__dict__)
			if factory_kwargs.get("config") is not None:
				print(
					"Warning: supplied config overwritten by the config of the existing model")			
		else:
			super(DecorrMamba, self).__init__(**factory_kwargs)

		self.n_decorr_layers = 0 # used for averaging the decorr losses later
		# used for referencing decorrelation layers directly without looping
		# over complete model structure
		self.decorr_layers = []

		def _create_decorr_matrices(module):
			''' 
			Used to recursively traverse model and create decorrelation matrices 
			in pre-defined places
			'''

			for name, child in module.named_children():
				if name == "in_proj" or name == "out_proj" or name == "x_proj":
					self.n_decorr_layers += 1
					setattr(module, name, DecorrLinear(original_layer=child))
					self.decorr_layers.append(getattr(module, name))

				if name == "conv1d":
					self.n_decorr_layers += 1 				
					setattr(module, name, DecorrConv1d(original_layer=child))
					self.decorr_layers.append(getattr(module, name))

		self.apply(_create_decorr_matrices)

		self.mean_corr_loss = 0
		self.mean_whit_loss = 0
		self.decorr_lr = decorr_lr
		self.kappa = kappa
		self.sample_frac = sample_frac
		self.loss_module = DecorrLoss()
	
	def reshape_decorr_inputs(self, b):
		''' 
		Tracking of inputs to decorrelation layers is complicated by the
		hardware-aware forward pass. The default implementation of tracking
		returns input tensors of shapes which aren't always correct for 
		decorrelation update computation. This fixes it.
		'''
		def _fix(module):
			for name, child in module.named_children():
				if isinstance(child, DecorrLinear):
					if name == "in_proj" and len(child.weight.inputs) > 0:
						child.weight.inputs[0] = child.weight.inputs[0].reshape(
							(self.config.d_model, b, -1)).permute(1, 2, 0)
						
					elif name == "x_proj" and len(child.weight.inputs) > 0:
						child.weight.inputs[0] = child.weight.inputs[0].reshape(
							(b, -1, self.backbone.layers[0].mixer.d_inner))	

		self.apply(_fix)
	
	def reset_decorr(self, re_fuse: bool = True):
		''' 
		Resets gradients and losses of decorrelation layers after parameter
		updates. Also resets the mean losses computed across all decorrelation
		layers, and re-fuses the weight + decorrelation matrices for the forward
		pass
		'''

		self.apply_to_decorr(lambda x: x.reset())
		if re_fuse:
			self.fuse_decorr()
		self.mean_corr_loss = 0
		self.mean_whit_loss = 0

	def compute_decorr_grad_loss(self, compute_grad: bool=True, compute_loss: bool = True):
		'''
		Computes gradient updates and/or losses for decorrelation layers across 
		the entire architecture, based on the layers' recorded inputs. 
		'''
		def _compute_decorr_grad_loss(module):
			assert len(module.weight.inputs) == 1, \
			f"No/too many inputs found for layer {module}"

			layer_inputs = module.weight.inputs[0]
			# randomly sample a certain percent of the input batch				
			b = layer_inputs.shape[0]	
			num_samples = int(self.sample_frac * b)
			indices = torch.randperm(b)[:num_samples]
			input_samples = layer_inputs[indices]

			# forward pass this through the decorrelation matrix
			if isinstance(module, DecorrLinear):
				batched=False
				decorr_out = input_samples @ module.decorr_layer

			# Conv1d layers use channel_independent decorrelation, forward
			# passing through the decorrelation matrices is more involved
			else:
				# all data in each convolution patch is represented in a single vector
				# (B, n_patches, conv_1d_size*D)
				batched=True
				x_unfolded = F.unfold(
					input_samples.unsqueeze(1), 
					(self.backbone.layers[0].mixer.d_inner, module.kernel_size[0]), 
					stride=1, padding=(0, module.kernel_size[0]-1)).transpose(1,2)
				# reshapes all inputs as corresponding convolutional "patches"
				patch_matrices = x_unfolded.reshape(
					num_samples, -1, self.backbone.layers[0].mixer.d_inner, module.kernel_size[0])
				
				# perform decorrelation operation
				decorr_out = einsum(module.decorr_layer, patch_matrices,
					'd conv_1d_size dummy, n_samples n_patches d dummy -> n_samples n_patches d conv_1d_size')

			grad, corr_loss, whit_loss = self.loss_module(
				decorr_out, self.kappa,
				compute_grad, compute_loss, batched=batched)
				
			module.grad = grad
			module.corr_loss = corr_loss
			module.whit_loss = whit_loss					

		self.apply_to_decorr(_compute_decorr_grad_loss)


	def mean_decorr_losses(self):
		''' 
		Calculates the mean correlation and whitening losses across all 
		layers implementing decorrelation, for a Mamba model
		'''

		def _sum_losses(module):
			self.mean_corr_loss += module.corr_loss
			self.mean_whit_loss += module.whit_loss

		self.apply_to_decorr(_sum_losses)
		
		self.mean_corr_loss /= self.n_decorr_layers
		self.mean_whit_loss /= self.n_decorr_layers

	def update_decorr_matrices(self):
		''' 
		Updates the decorrelation matrices for all decorrelation layers
		within the Mamba model
		'''
		assert self.decorr_lr is not None, "No decorr_lr specified"

		# gradients should already have been computed
		def _update_decorr_matrices(module):
			assert module.grad is not None, "Gradient not computed"

			unscaled_update = \
				module.decorr_layer.data - \
				self.decorr_lr * module.grad @ module.decorr_layer.data

			# IMPLEMENT GAIN SCALING HERE

			module.decorr_layer.data = unscaled_update

		self.apply_to_decorr(_update_decorr_matrices)
	
	def fuse_decorr(self):
		''' Creates the fused backprop weight + decorrelation matrix after 
		the update of decorrelation matrices'''
		def _fuse_decorr(module):
			if isinstance(module, DecorrLinear):
				fused_weight = TrackedParameter(
					module.original_weight @ module.decorr_layer, 
					requires_grad=False)
				module.register_parameter("weight", fused_weight)
		
			else: # Conv1d
				fused_weight = torch.unsqueeze(
					einsum(
						module.decorr_layer.data, torch.squeeze(module.original_weight.data),
						'd dummy conv_1d_size, d dummy -> d conv_1d_size'), 1)
				fused_weight = TrackedParameter(fused_weight, requires_grad=False)
				module.register_parameter("weight", fused_weight)

		self.apply_to_decorr(_fuse_decorr)


	def apply_to_decorr(self, f):
		"Used for applying simple functions to all of a model's decorrelated layers"
		for layer in self.decorr_layers:
			f(layer)


	
