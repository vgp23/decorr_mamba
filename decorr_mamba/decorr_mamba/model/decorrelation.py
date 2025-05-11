import torch
import time
import torch.nn as nn
from einops import einsum, rearrange, repeat
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from functools import partial
from copy import deepcopy
import torch.nn.functional as F
from torch.nn.functional import linear, conv1d
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.torch import custom_fwd, custom_bwd
import math
from decorr_mamba.model.sashimi_mamba import SaShiMiMamba

import selective_scan_cuda

try:
	from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
	import causal_conv1d_cuda
except ImportError:
	causal_conv1d_fn, causal_conv1d_update, causal_conv1d_cuda = None, None, None

try:
	from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
	selective_state_update = None

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, MambaInnerFn, rms_norm_forward, mamba_inner_fn

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

		with torch.no_grad():

			assert kappa is not None, "Specify kappa for loss and gradient computation"
			assert kappa <= 1.0 and kappa >= 0.0, "kappa must be between 0 and 1"

			# used for all modes where the decorrelation layer has only a single
			# matrix to train
			if not batched:
				# collapse input across the batch and length dimensions
				_, _, d = x.shape
				x = x.reshape(-1, d)
				mean_dim = 0
			# used where decorrelation layer has multiple matrices to train
			# (this is the case for "channel_independent")
			else:
				# collapse across batch and n_patches dimension
				_, _, d, decorr_matrix_size = x.shape
				# in this case we're updating d matrices, each with info
				# from one embedding dimension channel.
				# (D, all_samples, decorr_matrix_size)
				x = x.permute(2, 0, 1, 3).reshape(d, -1, decorr_matrix_size)
				mean_dim=1

			# computing losses
			if kappa == 0:
				# compute covariance components only
				C = x.unsqueeze(-1) * x.unsqueeze(-2)
				C.diagonal(dim1=-2, dim2=-1).zero_()
			elif kappa == 1:
				# compute variance components only
				V = x * x - 1
			else:
				# compute both
				C = x.unsqueeze(-1) * x.unsqueeze(-2)
				V = C.diagonal(dim1=-2, dim2=-1) - 1
				C.diagonal(dim1=-2, dim2=-1).zero_()

			# compute the actual gradient, if applicable
			if compute_grad:
				if kappa == 0:
					grad = torch.mean(C, dim=mean_dim)
				elif kappa == 1:
					# TODO: implement a nicer way of dealing with this to
					# avoid unnecessary operations in the gradient update
					grad = torch.diag_embed(torch.mean(V, dim=mean_dim))
				else:
					# implemented this way to remove unnecessary addition
					# of zeros
					unaveraged = (1-kappa) * C
					unaveraged.diagonal(dim1=-2, dim2=-1).add_(kappa*V)
					grad = torch.mean(unaveraged, dim=mean_dim)
			else:
				grad = None
		
			if compute_loss:
				# mean of squared covariances
				corr_loss = (C*C).mean() if kappa < 1 else None
				# mean of squared variances
				whit_loss = (V*V).mean() if kappa > 0 else None
			else:
				corr_loss = whit_loss = None 

			return grad, corr_loss, whit_loss

# class TrackingTensor:
# 	""" Wrapper around Tensor, used to capture the tensors which a 
# 		parameter tensor is multiplied by during the forward pass. Necessary
# 		because the selective scan algorithm does not use typical forward pass
# 		logic (forward pass of constituent layers is not necessarily called,
# 		making it impossible to track layer inputs for decorrelation gradient
# 		updates)"""
# 	def __init__(self, tensor, parent_layer):
# 		if not isinstance(tensor, torch.Tensor):
# 			raise TypeError("Expected a Tensor")
# 		self.tensor = tensor
# 		# set a reference to the decorr layer containing this tensor, for
# 		# tracking layer inputs
# 		self.parent_layer = parent_layer

# 	def __matmul__(self, other):
# 		"""Intercepts matrix multiplication to log the input."""
# 		self.parent_layer.inputs = other.detach()
# 		return self.tensor @ other

# 	def __rmatmul__(self, other):
# 		"""Tracks all tensors that multiplied self: x @ W"""
# 		self.parent_layer.inputs = other.detach()
# 		return other @ self.tensor
		
# 	def transpose(self, dim0, dim1):
# 		""" Ensures transposition is also tracked """
# 		# Make a new tensor, referencing the same parent layer
# 		transposed_parameter = TrackingTensor(
# 			self.tensor.transpose(dim0, dim1), self.parent_layer)
# 		return transposed_parameter

# 	@property
# 	def T(self):
# 		""" Handles W.T so it retains tracking """
# 		return self.transpose(0, 1)	

# 	def __getattr__(self, name):
# 		# Forward everything else to the underlying tensor
# 		return getattr(self.tensor, name)

# 	def __repr__(self):
# 		return f"TrackingTensor({self.tensor})"
			
class DecorrMixin:
	""" Wrapper class providing simple functionality to all decorrelation
		layers. """
	def __init__(self, compute_loss: bool, 
			  kappa: float, sample_frac: float):

		self.corr_loss = None
		self.whit_loss = None
		self.kappa = kappa
		self.sample_frac = sample_frac
		self.compute_loss = compute_loss
		self.loss_module = DecorrLoss()
		self.inputs = None

	def reset(self):
		""" Resets gradients and losses of decorrelation layers, used 
			before/after gradient descent steps."""
		
		self.corr_loss = None
		self.whit_loss = None
		self.decorr_layer.grad = None

	def update_decorr_matrices(self, decorr_lr: float):
		""" Updates decorrelation parameters. 
		
		Args:
			decorr_lr (float): decorrelation layer learning rate

		"""

		assert self.decorr_layer.grad is not None, "Gradient not computed"
		with torch.no_grad():
			self.decorr_layer -= decorr_lr * (self.decorr_layer.grad @ self.decorr_layer)

		
class DecorrLinear(DecorrMixin, nn.Linear):
	"""A linear layer with decorrelation applied to its weight matrix.

	Inherits from `nn.Linear` and `DecorrMixin` to introduce decorrelation
	of features during training.

	Args:
		compute_loss (bool, optional): Whether to compute the decorrelation loss.
			Defaults to True.
		kappa (float, optional): Controls the balance between decorrelation and
			whitening during training. Defaults to None.
		sample_frac (float, optional): Fraction of samples used for decorrelation
			computation. Defaults to None.
		**factory_kwargs: Additional arguments for `nn.Linear`.
	"""

	def __init__(self, compute_loss: bool = True,
				kappa: float = None, sample_frac: float = None,  **factory_kwargs):
		
		DecorrMixin.__init__(compute_loss, kappa, sample_frac)
		nn.Linear.__init__(**factory_kwargs)

		self.decorr_layer = nn.Parameter(
			torch.eye(self.in_features).to(self.weight.device), requires_grad=False)
		

	@classmethod
	def from_existing_layer(cls, original_layer: nn.Module,
						 compute_loss: bool = True, kappa: float = None, 
						 sample_frac: float = None):
		"""Creates a `DecorrLinear` instance from an existing `nn.Linear` layer.

		Args:
			original_layer (nn.Module): An existing `nn.Linear` layer.
			compute_loss (bool, optional): Whether to compute the decorrelation loss.
				Defaults to True.
			kappa (float, optional): Controls the balance between decorrelation and
				whitening during training. Defaults to None.
			sample_frac (float, optional): Fraction of samples used for decorrelation
				computation. Defaults to None.

		Returns:
			DecorrLinear: A new `DecorrLinear` instance with properties copied from
			the original `nn.Linear` layer.
		"""		

		assert isinstance(original_layer, nn.Linear), "Expected an instance of nn.Linear"

		# create a new DecorrLinear instance without calling __init__
		new_layer = cls.__new__(cls)
		# copy all attributes from the original layer
		new_layer.__dict__.update(original_layer.__dict__)

		# initialize DecorrMixin manually
		DecorrMixin.__init__(new_layer, compute_loss, kappa, sample_frac)

		# initialize DecorrLinear-specific attributes
		new_layer.decorr_layer = nn.Parameter(
			torch.eye(original_layer.in_features, 
			 device=original_layer.weight.device), requires_grad=False
		)

		return new_layer

	def forward(self, x):
		return linear(x, self.fused_weight, self.bias)

	def fuse_decorr(self):
		""" 
		Pre-multiplies the decorrelation and standard weight parameters
		into a single parameter matrix, numerically equivalent to passing
		inputs through both matrices separately."""

		self.fused_weight = self.weight @ self.decorr_layer
		
		
	# def reshape_decorr_inputs(self, x, b):
	# 	"""Reshapes the captured inputs for gradient computation.

	# 	Args:
	# 		x (torch.Tensor): Input tensor.
	# 		b (int): Batch size.

	# 	Returns:
	# 		torch.Tensor: Reshaped input tensor.
	# 	"""
	# 	# Can't get the batch size dynamically, because the batch and length 
	# 	# dimensions are collapsed together in the in_proj layers during the 
	# 	# selective scan algorithm. 
	# 	if self.reshape_type == "in_proj":
	# 		return x.reshape((self.in_features, b, -1)).permute(1, 2, 0)
			
	# 	elif self.reshape_type == "x_proj":
	# 		return x.reshape((b, -1, self.in_features))	
		
	# 	else:
	# 		return x
			
	def compute_decorr_grad_loss(self, x):
		"""Computes decorrelation losses and gradients.

		Args:
			x (torch.Tensor): Input tensor.
		"""
		with torch.no_grad():
			# sample a subset of the logged inputs
			b = x.shape[0]
			num_samples = int(self.sample_frac * b)
			if num_samples < 1:
				num_samples = 1
			
			subset = x[torch.randperm(b, device=x.device)[:num_samples]]

			# forward pass this through the decorrelation matrix
			decorr_out = subset @ self.decorr_layer.T

			grad, corr_loss, whit_loss = self.loss_module(
				decorr_out, self.kappa, self.training, self.compute_loss, batched=False)		

			self.decorr_layer.grad = grad
			self.corr_loss = corr_loss
			self.whit_loss = whit_loss	

class DecorrConv1d(DecorrMixin, nn.Conv1d):
	def __init__(self, compute_loss: bool = True, kappa: float = None, 
			  sample_frac: float = None, **factory_kwargs):
		"""A 1D convolutional layer with decorrelation applied to its weight matrix.

		Inherits from `nn.Conv1d` and `DecorrMixin` to introduce decorrelation
		updates during training.

		Args:
			compute_loss (bool, optional): Whether to compute the decorrelation loss.
				Defaults to True.
			kappa (float, optional): Controls the balance between decorrelation and
				whitening during training. Defaults to None.
			sample_frac (float, optional): Fraction of samples used for decorrelation
				computation. Defaults to None.
			**factory_kwargs: Additional arguments for `nn.Conv1d`.
		"""
		
		DecorrMixin.__init__(compute_loss, kappa, sample_frac)
		nn.Conv1d.__init__(**factory_kwargs)

		# (in_channels, kernel_size, kernel_size)
		all_matrices = torch.eye(self.kernel_size[0]).unsqueeze(0).repeat(
					self.in_channels, 1, 1).to(self.weight.device)

		self.decorr_layer = nn.Parameter(all_matrices, requires_grad=False)

	@classmethod
	def from_existing_layer(cls, original_layer: nn.Module, 
						 compute_loss: bool = True, kappa: float = None, 
						 sample_frac: float = None):
		"""Creates a `DecorrConv1d` instance from an existing `nn.Conv1d` layer.

		Args:
			original_layer (nn.Module): An existing `nn.Conv1d` layer.
			compute_loss (bool, optional): Whether to compute decorrelation loss.
				Defaults to True.
			kappa (float, optional): Controls the balance between decorrelation and
				whitening during training. Defaults to None.
			sample_frac (float, optional): Fraction of samples for decorrelation.
				Defaults to None.

		Returns:
			DecorrConv1d: A new `DecorrConv1d` instance with properties copied from
			the original `nn.Conv1d` layer.
		"""

		assert isinstance(original_layer, nn.Conv1d), "Expected an instance of nn.Conv1d"
		# create a new DecorrConv1d instance without calling __init__
		new_layer = cls.__new__(cls)
		# copy all attributes from the original layer
		new_layer.__dict__.update(original_layer.__dict__)

		# initialize DecorrMixin manually
		DecorrMixin.__init__(new_layer, compute_loss, kappa, sample_frac)

		# initialize DecorrConv1d-specific attributes
		# (in_channels, kernel_size, kernel_size)
		all_matrices = torch.eye(new_layer.kernel_size[0]).unsqueeze(0).repeat(
					new_layer.in_channels, 1, 1).to(new_layer.weight.device)

		new_layer.decorr_layer = nn.Parameter(all_matrices, requires_grad=False)

		return new_layer

	def forward(self, x):
		return conv1d(x, self.fused_weight, self.bias, 
				self.stride, self.padding, self.dilation, self.groups)

	def fuse_decorr(self):
		""" 
		Pre-multiplies the decorrelation and standard weight parameters
		into a single parameter matrix, numerically equivalent to passing
		inputs through both matrices separately."""

		self.fused_weight = torch.unsqueeze(
					einsum(
						self.decorr_layer, torch.squeeze(self.weight),
						'd dummy conv_1d_size, d dummy -> d conv_1d_size'), 
						1)
		
	def compute_decorr_grad_loss(self, x):
		"""Computes decorrelation losses and gradients.

		Args:
			x (torch.Tensor): Input tensor.
		"""

		with torch.no_grad():

			b, d_inner, _ = x.shape
			num_samples = int(self.sample_frac * b)
			if num_samples < 1:
				num_samples = 1

			# select a subset of the logged inputs
			subset = x[torch.randperm(b, device=x.device)[:num_samples]]

			# get patches that the convolutional kernel would see
			# all data in each convolution patch is represented in a single vector
			# (B, conv_1d_size*D, n_patches)
			x_unfolded = F.unfold(
				subset.unsqueeze(1), 
				(d_inner, self.kernel_size[0]), 
				stride=1, padding=(0, self.kernel_size[0]-1))
			
			# reshapes all inputs as corresponding convolutional "patches"
			patch_matrices = x_unfolded.reshape(
				num_samples, d_inner, self.kernel_size[0], -1)
			
			# perform decorrelation operation
			decorr_out = einsum(self.decorr_layer, patch_matrices,
				'd conv_1d_size dummy, n_samples d dummy n_patches -> n_samples n_patches d conv_1d_size')
			
			grad, corr_loss, whit_loss = self.loss_module(
				decorr_out, self.kappa, self.training, self.compute_loss, batched=True)

			self.decorr_layer.grad = grad
			self.corr_loss = corr_loss
			self.whit_loss = whit_loss	

class DecorrMambaInnerFn(MambaInnerFn):

	@staticmethod
	@custom_fwd
	def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
				out_proj_weight, out_proj_bias,
				A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
				C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, b_rms_weight=None, 
				c_rms_weight= None, dt_rms_weight= None, b_c_dt_rms_eps=1e-6):
		"""
			 xz: (batch, dim, seqlen)
		"""
		assert causal_conv1d_cuda is not None, \
			"causal_conv1d_cuda is not available. Please install causal-conv1d."
		assert checkpoint_lvl in [0, 1]
		L = xz.shape[-1]
		delta_rank = delta_proj_weight.shape[1]
		d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
		if torch.is_autocast_enabled():
			x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
			delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
			out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
			out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
							 if out_proj_bias is not None else None)
		if xz.stride(-1) != 1:
			xz = xz.contiguous()
		conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
		x, z = xz.chunk(2, dim=1)
		conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None

		conv1d_inputs = x.detach()

		conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
			x, conv1d_weight, conv1d_bias, None, None, None, True
		)
		# We're being very careful here about the layout, to avoid extra transposes.
		# We want delta to have d as the slowest moving dimension
		# and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
		x_proj_inputs = rearrange(conv1d_out.detach(), 'b d l -> b l d')

		x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
		delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
		ctx.is_variable_B = B is None
		ctx.is_variable_C = C is None
		ctx.B_proj_bias_is_None = B_proj_bias is None
		ctx.C_proj_bias_is_None = C_proj_bias is None
		if B is None:  # variable B
			B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
			if B_proj_bias is not None:
				B = B + B_proj_bias.to(dtype=B.dtype)
			if not A.is_complex():
				# B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
				B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
			else:
				B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
		else:
			if B.stride(-1) != 1:
				B = B.contiguous()
		if C is None:  # variable C
			C = x_dbl[:, -d_state:]  # (bl dstate)
			if C_proj_bias is not None:
				C = C + C_proj_bias.to(dtype=C.dtype)
			if not A.is_complex():
				# C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
				C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
			else:
				C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
		else:
			if C.stride(-1) != 1:
				C = C.contiguous()
		if D is not None:
			D = D.contiguous()
			
		if b_rms_weight is not None:
			B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
			B = rms_norm_forward(B, b_rms_weight, bias=None, eps=b_c_dt_rms_eps)
			B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
		if c_rms_weight is not None:
			C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
			C = rms_norm_forward(C, c_rms_weight, bias=None, eps=b_c_dt_rms_eps)
			C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
		if dt_rms_weight is not None:
			delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
			delta = rms_norm_forward(delta, dt_rms_weight, bias=None, eps=b_c_dt_rms_eps)
			delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
		
		out, scan_intermediates, out_z = selective_scan_cuda.fwd(
			conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
		)
		ctx.delta_softplus = delta_softplus
		ctx.out_proj_bias_is_None = out_proj_bias is None
		ctx.checkpoint_lvl = checkpoint_lvl
		ctx.b_rms_weight = b_rms_weight
		ctx.c_rms_weight = c_rms_weight
		ctx.dt_rms_weight = dt_rms_weight
		ctx.b_c_dt_rms_eps = b_c_dt_rms_eps
		if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
			conv1d_out, delta = None, None
		ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
							  delta_proj_weight, out_proj_weight, conv1d_out, delta,
							  A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out)
		out_proj_inputs = rearrange(out_z, "b d l -> b l d")

		return F.linear(out_proj_inputs, out_proj_weight, out_proj_bias), \
			{"conv1d": conv1d_inputs, "x_proj": x_proj_inputs,
			"out_proj": out_proj_inputs.detach()}
	
	@staticmethod
	def backward(ctx, dout, layer_inputs):
		grad_input = super(DecorrMambaInnerFn, DecorrMambaInnerFn).backward(ctx, dout)
		return grad_input
	
def decorr_mamba_inner_fn(
	xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
	out_proj_weight, out_proj_bias,
	A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
	C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, b_rms_weight= None, c_rms_weight= None, dt_rms_weight= None, b_c_dt_rms_eps=1e-6
):
	return DecorrMambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, 
						delta_proj_weight,
							out_proj_weight, out_proj_bias,
							A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, 
							delta_softplus, checkpoint_lvl, b_rms_weight, 
							c_rms_weight, dt_rms_weight, b_c_dt_rms_eps)

class DecorrMamba(MambaLMHeadModel):
	"""Extends MambaLMHeadModel by integrating decorrelation layers into the architecture."""

	def __init__(self, existing_model: MambaLMHeadModel = None, copy: bool = False, 
			  kappa: float = 0.5, sample_frac: float = 0.1, decorr_lr: float = None,
			  compute_loss: bool = True, **factory_kwargs):
		"""
		Initializes a DecorrMamba model by adding decorrelation layers to the existing model.

		Args:
			existing_model (MambaLMHeadModel, optional): Pre-existing model to copy or extend.
			copy (bool): Whether to deep copy the existing model.
			kappa (float): Scaling factor for decorrelation loss.
			sample_frac (float): Fraction of samples used for decorrelation computation.
			decorr_lr (float, optional): Learning rate for decorrelation matrix updates.
			compute_loss (bool): Whether to compute decorrelation loss.
			**factory_kwargs: Additional arguments for model initialization.
		"""

		if existing_model and copy:
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
			""" 
			Used to recursively traverse model and create decorrelation matrices 
			in pre-defined places

			Args:
				module (nn.Module): model layers to add decorrelation into 
			"""

			for name, child in module.named_children():
				if name == "in_proj" or name == "out_proj" or name == "x_proj":
					self.n_decorr_layers += 1
					setattr(module, name, DecorrLinear.from_existing_layer(
						original_layer=child, compute_loss=compute_loss, 
						kappa=kappa, sample_frac=sample_frac))
					
					self.decorr_layers.append(getattr(module, name))

				if name == "conv1d":
					self.n_decorr_layers += 1 				
					setattr(module, name, DecorrConv1d.from_existing_layer(
						original_layer=child, compute_loss=compute_loss,
						kappa=kappa, sample_frac=sample_frac))
					self.decorr_layers.append(getattr(module, name))

		self.apply(_create_decorr_matrices)

		self.mean_corr_loss = None
		self.mean_whit_loss = None
		self.decorr_lr = decorr_lr

		# These are here for reference only, these have been passed to 
		# the decorrelation modules and they work inside there. 
		self.kappa = kappa
		self.sample_frac = sample_frac
		self.compute_loss = compute_loss
		
		def _modify_mamba_block_functions(module):
			for _, child in module.named_children():
				if type(child) is Mamba:
					child.forward = self._mamba_block_forward.__get__(child)
					child.step = self._mamba_block_step.__get__(child)	

		self.apply(_modify_mamba_block_functions)

	def _mamba_block_forward(self, hidden_states, inference_params=None):
		"""
		hidden_states: (B, L, D)
		Returns: same shape as hidden_states
		"""
		batch, seqlen, dim = hidden_states.shape

		conv_state, ssm_state = None, None
		if inference_params is not None:
			conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
			if inference_params.seqlen_offset > 0:
				# The states are updated inplace
				out, _, _ = self.step(hidden_states, conv_state, ssm_state)
				return out

		# We do matmul and transpose BLH -> HBL at the same time

		self.in_proj.inputs = hidden_states.detach()

		xz = rearrange(
			self.in_proj.fused_weight @ rearrange(hidden_states, "b l d -> d (b l)"),
			"d (b l) -> b d l",
			l=seqlen,
		)

		if self.in_proj.bias is not None:
			xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

		if not self.complex:
			A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
		else:
			A = -torch.exp(self.log_A_real) + 1j * self.A_imag
			
		# In the backward pass we write dx and dz next to each other to avoid torch.cat
		if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
			out, layer_inputs = decorr_mamba_inner_fn(
				xz,
				self.conv1d.fused_weight,
				self.conv1d.bias,
				self.x_proj.fused_weight,
				self.dt_proj.weight,
				self.out_proj.fused_weight,
				self.out_proj.bias,
				A,
				None,  # input-dependent B
				None,  # input-dependent C
				self.D.float(),
				delta_bias=self.dt_proj.bias.float(),
				delta_softplus=True,
			)
			self.conv1d.inputs = layer_inputs["conv1d"]
			self.x_proj.inputs = layer_inputs["x_proj"]
			self.out_proj.inputs = layer_inputs["out_proj"]
		else: 
			x, z = xz.chunk(2, dim=1)
			# Compute short convolution
			if conv_state is not None:
				# If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
				# Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
				conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)

			self.conv1d.inputs = x.detach()
			if causal_conv1d_fn is None:
				x = self.act(self.conv1d(x)[..., :seqlen])
			else:
				assert self.activation in ["silu", "swish"]
				x = causal_conv1d_fn(
					x=x,
					weight=rearrange(self.conv1d.fused_weight, "d 1 w -> d w"),
					bias=self.conv1d.bias,
					activation=self.activation,
				)

			# We're careful here about the layout, to avoid extra transposes.
			# We want dt to have d as the slowest moving dimension
			# and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
			self.x_proj.inputs = rearrange(x, "b d l -> b l d").detach()
			x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
			dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
			dt = self.dt_proj.weight @ dt.t()
			dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
			B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
			C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
			assert self.activation in ["silu", "swish"]
			y = selective_scan_fn(
				x,
				dt,
				A,
				B,
				C,
				self.D.float(),
				z=z,
				delta_bias=self.dt_proj.bias.float(),
				delta_softplus=True,
				return_last_state=ssm_state is not None,
			)
			if ssm_state is not None:
				y, last_state = y
				ssm_state.copy_(last_state)
			y = rearrange(y, "b d l -> b l d")
			self.out_proj.inputs = y.detach()
			out = self.out_proj(y)

		return out

	def _mamba_block_step(self, hidden_states, conv_state, ssm_state):
		dtype = hidden_states.dtype
		assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
		xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
		x, z = xz.chunk(2, dim=-1)  # (B D)

		# Conv step
		if causal_conv1d_update is None:
			conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
			conv_state[:, :, -1] = x

			self.conv1d.inputs = conv_state.detach()

			x = torch.sum(conv_state * rearrange(self.conv1d.fused_weight, "d 1 w -> d w"), dim=-1)  # (B D)
			if self.conv1d.bias is not None:
				x = x + self.conv1d.bias
			x = self.act(x).to(dtype=dtype)
		else:

			self.conv1d.inputs = x.detach()

			x = causal_conv1d_update(
				x,
				conv_state,
				rearrange(self.conv1d.fused_weight, "d 1 w -> d w"),
				self.conv1d.bias,
				self.activation,
			)

		x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
		dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
		# Don't add dt_bias here

		dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
		A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

		# SSM step
		if selective_state_update is None:
			# Discretize A and B
			dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
			dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
			dB = torch.einsum("bd,bn->bdn", dt, B)
			ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
			y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
			y = y + self.D.to(dtype) * x
			y = y * self.act(z)  # (B D)
		else:
			y = selective_state_update(
				ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
			)

		out = self.out_proj(y)
		return out.unsqueeze(1), conv_state, ssm_state

	def forward(self, x):
		# fuse decorrelation + main model parameters, then let forward
		# pass proceed as normal
		print("NEW FORWARD")
		if self.training:
			self.fuse_decorr()
		return super().forward(x)


	def reset_decorr(self):
		""" 
		Resets gradients and losses of decorrelation layers after parameter
		updates. Also resets the mean losses computed across all decorrelation
		layers
		"""

		self.apply_to_decorr(lambda x: x.reset())
		self.mean_corr_loss = None
		self.mean_whit_loss = None

	def mean_decorr_losses(self):
		""" 
		Calculates the mean correlation and whitening losses across all 
		layers implementing decorrelation. 
		"""

		def _sum_losses(module):
			if module.corr_loss is not None:
				self.mean_corr_loss += module.corr_loss
			else:
				self.mean_corr_loss = None

			if module.whit_loss is not None:
				self.mean_whit_loss += module.whit_loss
			else:
				self.mean_whit_loss = None
		
		self.mean_corr_loss = 0.0
		self.mean_whit_loss = 0.0
		self.apply_to_decorr(_sum_losses)
		
		if self.mean_corr_loss is not None:
			self.mean_corr_loss /= self.n_decorr_layers
		
		if self.mean_whit_loss is not None:
			self.mean_whit_loss /= self.n_decorr_layers
	
	def fuse_decorr(self):
		"""
		Pre-computes multiplication of weight and decorrelation matrices for
		more efficient forward pass
		"""
		self.apply_to_decorr(lambda x: x.fuse_decorr())

	def decorr_operations(self):
		""" 
		Performs all decorrelation operations (loss and/or gradient computation, 
		depending on configuration)."""

		self.apply_to_decorr(lambda x: x.compute_decorr_grad_loss(x.inputs))

	def update_decorr_matrices(self):
		""" 
		Updates the decorrelation matrices for all decorrelation layers
		within the model
		"""
		assert self.decorr_lr is not None, "No decorr_lr specified"
		self.apply_to_decorr(lambda x: x.update_decorr_matrices(self.decorr_lr))

	def apply_to_decorr(self, f: callable):
		"""
		Used for applying simple functions to all of a model's decorrelation layers
		
		Args:
			f (callable): the function to be applied to the decorrelation 
				layers. 
		"""

		for layer in self.decorr_layers:
			f(layer)
	
	def compute_losses(self, compute_loss: bool):
		"""
		Enables or disables decorrelation loss computation across all layers.

		Args:
			compute_loss (bool): Flag to enable or disable decorrelation loss computation.
		"""		
		self.compute_loss = compute_loss
		self.apply_to_decorr(
			lambda x: setattr(x, 'compute_loss', compute_loss))

class DecorrSaShiMiMamba(DecorrMamba, SaShiMiMamba):
	""" Decorrelated version of SaShiMiMamba."""
	def __init__(self, existing_model: SaShiMiMamba = None, copy: bool = False, 
			  kappa: float = 0.5, sample_frac: float = 0.1, decorr_lr: float = None,
			  compute_loss: bool = True, **factory_kwargs):
		
		if existing_model and copy:
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
			SaShiMiMamba.__init__(self, **factory_kwargs)

		self.n_decorr_layers = 0 # used for averaging the decorr losses later
		# used for referencing decorrelation layers directly without looping
		# over complete model structure
		self.decorr_layers = []

		def _create_decorr_matrices(module):
			""" 
			Used to recursively traverse model and create decorrelation matrices 
			in pre-defined places

			Args:
				module (nn.Module): model layers to add decorrelation into 
			"""
			decorr_linear_names = ["in_proj", "out_proj", "x_proj", "up_pool", "down_pool"]

			for name, child in module.named_children():
				if name in decorr_linear_names:
					self.n_decorr_layers += 1
					setattr(module, name, DecorrLinear.from_existing_layer(
						original_layer=child, compute_loss=compute_loss, 
						kappa=kappa, sample_frac=sample_frac))
					
					self.decorr_layers.append(getattr(module, name))

				if name == "conv1d":
					self.n_decorr_layers += 1 				
					setattr(module, name, DecorrConv1d.from_existing_layer(
						original_layer=child, compute_loss=compute_loss,
						kappa=kappa, sample_frac=sample_frac))
					self.decorr_layers.append(getattr(module, name))

		self.apply(_create_decorr_matrices)

		self.mean_corr_loss = None
		self.mean_whit_loss = None
		self.decorr_lr = decorr_lr

		# These are here for reference only, these have been passed to 
		# the decorrelation modules and they work inside there. 
		self.kappa = kappa
		self.sample_frac = sample_frac
		self.compute_loss = compute_loss

		def _modify_mamba_block_functions(module):
			for _, child in module.named_children():
				if type(child) is Mamba:
					child.forward = self._mamba_block_forward.__get__(child)
					child.step = self._mamba_block_step.__get__(child)	

		self.apply(_modify_mamba_block_functions)

	def forward(self, x):
		# fuse decorrelation + main model parameters, then let forward
		# pass proceed as normal
		if self.training:
			self.fuse_decorr()

		# down sample and keep residuals
		x = self.embedding(x) 
		residuals = []
		for dp, blocks in zip(self.down_pooling, 
			self.mamba_stages_down[:-1]):
			residuals.append(x)
			x = blocks(x)
			# inputs are captured by Mamba blocks separately
			if isinstance(dp, DecorrLinear):
				dp.inputs = x
			x = dp(x)

		residuals.append(x)

		# get through the bend in the U
		x = self.mamba_stages_down[-1](x)
		x = x + residuals.pop()

		# up-sampling!
		for up, blocks in zip(
			reversed(self.up_pooling), reversed(self.mamba_stages_up)):
			if isinstance(up, DecorrLinear):
				up.inputs = x
			u = up(x)
			x = u + residuals.pop()
			x = blocks(x)
	
		return x

