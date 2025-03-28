import torch
import time
import torch.nn as nn
from einops import einsum, rearrange
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from functools import partial
from copy import deepcopy
import torch.nn.functional as F
from torch.nn.functional import linear, conv1d
from mamba_ssm.modules.mamba_simple import Mamba
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

loss_stream = torch.cuda.Stream()

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

			# compute the individual loss elements
			D = x * x
			V = D - 1 if kappa > 0 else None 

			if kappa < 1:
				C = x.unsqueeze(-1) * x.unsqueeze(-2)
				C.diagonal(dim1=-2, dim2=-1).sub_(D)
			else:
				C = None

			# compute the actual gradient, if applicable
			if compute_grad:
				if kappa == 0:
					grad = torch.mean(C, dim=mean_dim)
				elif kappa == 1:
					# TODO: implement a nicer way of dealing with this to
					# avoid unnecessary operations
					grad = torch.diag_embed(torch.mean(V, dim=mean_dim))
				else:
					unaveraged = (1-kappa) * C.clone()
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

		
class DecorrTensor(torch.Tensor):

	def __new__(cls, tensor, parent_layer):
		if not isinstance(tensor, torch.Tensor):
			raise TypeError("Expected a torch.Tensor")
		# Create a new instance that shares the same storage as the original tensor
		instance = tensor.as_subclass(cls)
		instance.parent_layer = parent_layer
		return instance

	def __matmul__(self, other):
		"""Intercepts matrix multiplication to log the input."""

		with torch.cuda.stream(loss_stream):
			self.parent_layer.decorr_operations(other.detach())

		result = super().__matmul__(other)
		return result.as_subclass(torch.Tensor)
	

	def __rmatmul__(self, other):
		"""Tracks all tensors that multiplied self: x @ W"""

		with torch.cuda.stream(loss_stream):
			self.parent_layer.decorr_operations(other.detach())

		result = super().__rmatmul__(other)	
		return result.as_subclass(torch.Tensor)

	def transpose(self, dim0, dim1):
		""" Ensures transposition is also tracked """
		# Make a new parameter, morifying the same inputs list as the original
		transposed_parameter = DecorrTensor(
			super().transpose(dim0, dim1), self.parent_layer)
		return transposed_parameter

	@property
	def T(self):
		""" Handles W.T so it retains tracking """
		return self.transpose(0, 1)	
			
class DecorrMixin:
	''' Wrapper class providing simple functionality to all decorrelation
		layers. '''
	def __init__(self, compute_loss: bool, 
			  kappa: float, sample_frac: float):

		self.corr_loss = None
		self.whit_loss = None
		self.decorr_grad = None
		self.kappa = kappa
		self.sample_frac = sample_frac
		self.compute_loss = compute_loss
		self.loss_module = DecorrLoss()
	
	def decorr_hook(self, module, input, output):
		# performs all of the additional decorrelation 
		# loss/gradient computation operations with the inputs
		# captured at the beginning of a module's forward pass
		with torch.cuda.stream(loss_stream):
			self.decorr_operations(input[0].detach())

	def reset(self, re_fuse=True):
		self.corr_loss = None
		self.whit_loss = None
		self.decorr_grad = None

		if re_fuse:
			self.fuse_decorr()

	def decorr_operations(self, x):

		if self.training or self.compute_loss:
			# Reshaping
			if isinstance(self, DecorrLinear):
				x = self.reshape_decorr_inputs(x)
			# Gradient/loss computation
			self.compute_decorr_grad_loss(x)

	def update_decorr_matrices(self, decorr_lr):
		assert self.grad is not None, "Gradient not computed"
		with torch.no_grad():
			self.decorr_layer -= decorr_lr * (self.grad @ self.decorr_layer)
		
	
class DecorrLinear(DecorrMixin, nn.Linear):
	def __init__(self, reshape_type: str = None, compute_loss: bool = True,
			    kappa: float = 0.0, sample_frac=0.1,  **factory_kwargs):
		
		DecorrMixin.__init__(compute_loss, kappa, sample_frac)
		nn.Linear.__init__(**factory_kwargs)

		self.decorr_layer = nn.Parameter(
			torch.eye(self.in_features).to(self.weight.device), requires_grad=False)
		
		self.register_forward_hook(self.decorr_hook)
		
		# Fuses decorrelation + weight matrix, made available for manual
		# matrix multiplications in forward pass
		self.fuse_decorr()
		# dictates how the reshaping of captured input tensors happens
		self.reshape_type = reshape_type

	@classmethod
	def from_existing_layer(cls, original_layer, reshape_type=None, 
						 compute_loss=True, kappa=0.0, sample_frac=0.1):

		assert isinstance(original_layer, nn.Linear), "Expected an instance of nn.Linear"

		# create a new DecorrLinear instance without calling __init__
		new_layer = cls.__new__(cls)
		# copy all attributes from the original layer
		new_layer.__dict__.update(original_layer.__dict__)

		# initialize DecorrMixin manually
		DecorrMixin.__init__(new_layer, compute_loss, kappa, sample_frac)

		# initialize DecorrLinear-specific attributes
		new_layer.reshape_type = reshape_type
		new_layer.decorr_layer = nn.Parameter(
			torch.eye(original_layer.in_features, 
			 device=original_layer.weight.device), requires_grad=False
		)

		new_layer.register_forward_hook(new_layer.decorr_hook)

		# Fuse decorrelation + weight matrix
		new_layer.fuse_decorr()

		return new_layer

	def forward(self, x):
		# forward hook takes care of input tracking, don't want a TrackedTensor
		# here. 
		return linear(x, self.fused_weight.as_subclass(torch.Tensor), self.bias)

	def fuse_decorr(self):
		self.fused_weight = DecorrTensor(self.weight @ self.decorr_layer, 
								parent_layer=self)
		
	def reshape_decorr_inputs(self, x):
		# reshapes the captured inputs to these layers so that gradient
		# computation can be performed
		if self.reshape_type == "in_proj":
			b = int(x.shape[1] / self.in_features)
			return x.reshape((self.in_features, b, -1)).permute(1, 2, 0)
			
		elif self.reshape_type == "x_proj":
			b = x.shape[1]
			return x.reshape((b, -1, self.in_features))	
		
		else:
			return x
			
	def compute_decorr_grad_loss(self, x):
		
		with torch.no_grad():
			# sample a subset of the logged inputs
			b = x.shape[0]
			num_samples = int(self.sample_frac * b)
			# indices = torch.randint(0, b, (num_samples,)).to(
			# 	next(self.parameters()).device)		
			
			subset = x[torch.randperm(b, device=x.device)[:num_samples]]

			# forward pass this through the decorrelation matrix
			decorr_out = subset @ self.decorr_layer.T

			grad, corr_loss, whit_loss = self.loss_module(
				decorr_out, self.kappa, self.training, self.compute_loss, batched=False)		

			self.grad = grad
			self.corr_loss = corr_loss
			self.whit_loss = whit_loss	

class DecorrConv1d(DecorrMixin, nn.Conv1d):
	def __init__(self, compute_loss: bool = True, kappa: float=0.0, 
			  sample_frac: float=0.1, **factory_kwargs):
		
		DecorrMixin.__init__(compute_loss, kappa, sample_frac)
		nn.Conv1d.__init__(**factory_kwargs)

		# (in_channels, kernel_size, kernel_size)
		all_matrices = torch.eye(self.kernel_size[0]).unsqueeze(0).repeat(
					self.in_channels, 1, 1).to(self.weight.device)

		self.decorr_layer = nn.Parameter(all_matrices, requires_grad=False)

		self.register_forward_hook(self.decorr_hook)
		# Fuses decorrelation + weight matrix, made available for manual
		# matrix multiplications in forward pass
		self.fuse_decorr()

	@classmethod
	def from_existing_layer(cls, original_layer, compute_loss=True, 
						 kappa=0.0, sample_frac=0.1):

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

		new_layer.register_forward_hook(new_layer.decorr_hook)

		# Fuse decorrelation + weight matrix
		new_layer.fuse_decorr()

		return new_layer

	def forward(self, x):
		# forward hook takes care of input tracking, don't want a TrackedTensor
		# here. 	
		return conv1d(x, self.fused_weight.as_subclass(torch.Tensor), self.bias, 
				self.stride, self.padding, self.dilation, self.groups)

	def fuse_decorr(self):
		self.fused_weight = DecorrTensor(
			torch.unsqueeze(
					einsum(
						self.decorr_layer.data, torch.squeeze(self.weight.data),
						'd dummy conv_1d_size, d dummy -> d conv_1d_size'), 
						1), 
						parent_layer=self)
		
	def compute_decorr_grad_loss(self, x):
		
		with torch.no_grad():

			b = x.shape[0]
			d_inner = x.shape[1]
			num_samples = int(self.sample_frac * b)

			# select a subset of the logged inputs
			subset = x[torch.randperm(b, device=x.device)[:num_samples]]

			# forward pass this through the decorrelation matrix
			# all data in each convolution patch is represented in a single vector
			# (B, n_patches, conv_1d_size*D)
			x_unfolded = F.unfold(
				subset.unsqueeze(1), 
				(d_inner, self.kernel_size[0]), 
				stride=1, padding=(0, self.kernel_size[0]-1))
			
			# reshapes all inputs as corresponding convolutional "patches"
			patch_matrices = x_unfolded.reshape(
				num_samples, d_inner, -1, self.kernel_size[0])
			
			# perform decorrelation operation
			decorr_out = einsum(self.decorr_layer, patch_matrices,
				'd conv_1d_size dummy, n_samples d n_patches dummy -> n_samples n_patches d conv_1d_size')
			# with torch.profiler.profile(
			# 	activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
			# 	record_shapes=True,
			# 	with_stack=True
			# ) as prof:
			grad, corr_loss, whit_loss = self.loss_module(
				decorr_out, self.kappa, self.training, self.compute_loss, batched=True)

			# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

			self.grad = grad
			self.corr_loss = corr_loss
			self.whit_loss = whit_loss	

class DecorrMamba(MambaLMHeadModel):

	def __init__(self, existing_model: MambaLMHeadModel = None, copy: bool = False, 
			  kappa: float = 0.5, sample_frac: float = 0.1, decorr_lr = None,
			  compute_loss: bool = True, **factory_kwargs):

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
					setattr(module, name, DecorrLinear.from_existing_layer(
						original_layer=child, reshape_type=name,
						compute_loss=compute_loss, kappa=kappa))
					
					self.decorr_layers.append(getattr(module, name))

				if name == "conv1d":
					self.n_decorr_layers += 1 				
					setattr(module, name, DecorrConv1d.from_existing_layer(
						original_layer=child, compute_loss=compute_loss,
						kappa=kappa))
					self.decorr_layers.append(getattr(module, name))

		self.apply(_create_decorr_matrices)

		self.mean_corr_loss = 0
		self.mean_whit_loss = 0
		self.decorr_lr = decorr_lr

		# These are here for reference only, these have been passed to 
		# the decorrelation modules and they work inside there. 
		self.kappa = kappa
		self.sample_frac = sample_frac
		self.compute_loss = compute_loss


		# The two following functions replace the functions of the Mamba blocks
		# within the model, to make it work with the fused + tracked weight matrices.
		# This was easier than extending the Mamba class. 
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
			xz = rearrange(
				self.in_proj.fused_weight @ rearrange(hidden_states, "b l d -> d (b l)"),
				"d (b l) -> b d l",
				l=seqlen,
			)
			if self.in_proj.bias is not None:
				xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

			A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
			# In the backward pass we write dx and dz next to each other to avoid torch.cat
			if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
				out = mamba_inner_fn(
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
			else:
				x, z = xz.chunk(2, dim=1)
				# Compute short convolution
				if conv_state is not None:
					# If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
					# Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
					conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
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
				x = torch.sum(conv_state * rearrange(self.conv1d.fused_weight, "d 1 w -> d w"), dim=-1)  # (B D)
				if self.conv1d.bias is not None:
					x = x + self.conv1d.bias
				x = self.act(x).to(dtype=dtype)
			else:
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
		
		def _modify_mamba_functions(module):
			for _, child in module.named_children():
				if type(child) is Mamba:
					child.forward = _mamba_block_forward.__get__(child)
					child.step = _mamba_block_step.__get__(child)	

		self.apply(_modify_mamba_functions)
	
	def reset_decorr(self, re_fuse: bool = True):
		''' 
		Resets gradients and losses of decorrelation layers after parameter
		updates. Also resets the mean losses computed across all decorrelation
		layers, and re-fuses the weight + decorrelation matrices for the forward
		pass
		'''

		self.apply_to_decorr(lambda x: x.reset(re_fuse=re_fuse))
		self.mean_corr_loss = 0
		self.mean_whit_loss = 0

	def mean_decorr_losses(self):
		''' 
		Calculates the mean correlation and whitening losses across all 
		layers implementing decorrelation, for a Mamba model
		'''

		def _sum_losses(module):
			if module.corr_loss is not None:
				self.mean_corr_loss += module.corr_loss
			else:
				self.mean_corr_loss = None

			if module.whit_loss is not None:
				self.mean_whit_loss += module.whit_loss
			else:
				self.mean_whit_loss = None

		self.apply_to_decorr(_sum_losses)
		
		if self.mean_corr_loss is not None:
			self.mean_corr_loss /= self.n_decorr_layers
		
		if self.mean_whit_loss is not None:
			self.mean_whit_loss /= self.n_decorr_layers

	def update_decorr_matrices(self):
		''' 
		Updates the decorrelation matrices for all decorrelation layers
		within the Mamba model
		'''
		assert self.decorr_lr is not None, "No decorr_lr specified"
		self.apply_to_decorr(lambda x: x.update_decorr_matrices(self.decorr_lr))

	def apply_to_decorr(self, f):
		"Used for applying simple functions to all of a model's decorrelated layers"
		for layer in self.decorr_layers:
			f(layer)
	
	def compute_losses(self, compute_loss):
		self.compute_loss = compute_loss
		self.apply_to_decorr(
			lambda x: setattr(x, 'compute_loss', compute_loss))
