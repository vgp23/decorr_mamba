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

			# No point in calculating these if they're note used
			if kappa == 0.0:
				V = 0
			else:
				V = D - torch.eye(d, device=x.device)

			xx_t = einsum(x, x, 
				'n_samples x, n_samples x_t -> n_samples x x_t')

			C = xx_t - D

			# compute the actual gradient, if applicable
			if compute_grad:
				grad = torch.mean(((1-kappa)*C + kappa*V), dim=0)
			else:
				grad = None

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

			if kappa == 0.0:
				V = 0
			else:
				V = D - torch.eye(decorr_matrix_size, device=x.device)

			xx_t = einsum(x, x, 
				'd n_samples x, d n_samples x_t -> d n_samples x x_t')

			# (D, n_samples, decorr_matrix_size, decorr_matrix_size)
			C = xx_t - D

			# compute the actual gradients, if applicable
			if compute_grad:
				grad = torch.mean(((1-kappa)*C + kappa*V), dim=1)
			else:
				grad = None

		if compute_loss:
			# mean of squared covariances, averaged across each matrix,
			# then across all samples, then across all parallel channels
			corr_loss = torch.mean(C**2)

			if kappa == 0.0:
				whit_loss = None
			else:
				# mean of squared variances, averaged across each matrix,
				# then across all samples, then across all parallel channels
				whit_loss = torch.mean(V**2)

		else:
			corr_loss = None 
			whit_loss = None

		return grad, corr_loss, whit_loss

		
class TrackingTensor(torch.Tensor):

	def __new__(cls, tensor, parent_layer):
		if not isinstance(tensor, torch.Tensor):
			raise TypeError("Expected a torch.Tensor")
		# Create a new instance that shares the same storage as the original tensor
		instance = tensor.as_subclass(cls)
		instance.parent_layer = parent_layer
		return instance

	def __matmul__(self, other):
		"""Intercepts matrix multiplication to log the input."""
		self.parent_layer.inputs.append(other.clone().detach())
		result = super().__matmul__(other)
		return result.as_subclass(torch.Tensor)

	def __rmatmul__(self, other):
		"""Tracks all tensors that multiplied self: x @ W"""
		self.parent_layer.inputs.append(other.clone().detach()) 
		result = super().__rmatmul__(other)	
		return result.as_subclass(torch.Tensor)

	def transpose(self, dim0, dim1):
		""" Ensures transposition is also tracked """
		# Make a new parameter, morifying the same inputs list as the original
		transposed_parameter = TrackingTensor(
			super().transpose(dim0, dim1), self.parent_layer)
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
		
		# Fuses decorrelation + weight matrix, made available for manual
		# matrix multiplications in forward pass
		self.fuse_decorr()

		# this tracks inputs during the standard forward pass
		self.register_forward_hook(self.track_weight_input)
		self.inputs = []

		self.corr_loss = None
		self.whit_loss = None
		self.decorr_grad = None

	def track_weight_input(self, module, input, output):
		self.inputs.append(input[0].clone().detach())
	
	def forward(self, x):
		# forward hook takes care of input tracking, don't want a TrackedTensor
		# here. 
		return linear(x, self.fused_weight.as_subclass(torch.Tensor), self.bias)

	def reset(self):
		self.corr_loss = None
		self.whit_loss = None
		self.decorr_grad = None
		self.inputs = []

	def fuse_decorr(self):
		self.fused_weight = TrackingTensor(self.weight @ self.decorr_layer, 
								parent_layer=self)

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

		# Fuses decorrelation + weight matrix, made available for manual
		# matrix multiplications in forward pass
		self.fuse_decorr()

		# this tracks inputs during the standard forward pass
		self.register_forward_hook(self.track_weight_input)
		self.inputs = []

		self.corr_loss = None
		self.whit_loss = None
		self.decorr_grad = None

	def track_weight_input(self, module, input, output):
		self.inputs.append(input[0].clone().detach())
	
	def forward(self, x):
		# forward hook takes care of input tracking, don't want a TrackedTensor
		# here. 	
		return conv1d(x, self.fused_weight.as_subclass(torch.Tensor), self.bias, 
				self.stride, self.padding, self.dilation, self.groups)

	def reset(self):
		self.corr_loss = None
		self.whit_loss = None
		self.decorr_grad = None
		self.inputs = []

	def fuse_decorr(self):
		self.fused_weight = TrackingTensor(
			torch.unsqueeze(
					einsum(
						self.decorr_layer.data, torch.squeeze(self.weight.data),
						'd dummy conv_1d_size, d dummy -> d conv_1d_size'), 
						1), 
						parent_layer=self)

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
			for name, child in module.named_children():
				if type(child) is Mamba:
					child.forward = _mamba_block_forward.__get__(child)
					child.step = _mamba_block_step.__get__(child)	

		self.apply(_modify_mamba_functions)

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
					if name == "in_proj" and len(child.inputs) > 0:
						child.inputs[0] = child.inputs[0].reshape(
							(self.config.d_model, b, -1)).permute(1, 2, 0)
						
					elif name == "x_proj" and len(child.inputs) > 0:
						child.inputs[0] = child.inputs[0].reshape(
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
			self.apply_to_decorr(lambda x: x.fuse_decorr())

		self.mean_corr_loss = 0
		self.mean_whit_loss = 0

	def compute_decorr_grad_loss(self, b, compute_grad: bool=True, 
							  compute_loss: bool = True):
		'''
		Computes gradient updates and/or losses for decorrelation layers across 
		the entire architecture, based on the layers' recorded inputs. 
		'''
		# sampling is expensive time-wise, so we'll use the same batch
		# samples for all layers. 
		# pre_sampling_time = time.time()
		num_samples = int(self.sample_frac * b)
		indices = torch.randint(0, b, (num_samples,)).to(
			next(self.parameters()).device)
		# post_sampling_time = time.time() - pre_sampling_time

		def _compute_decorr_grad_loss(module, indices):
			with torch.no_grad():
				assert len(module.inputs) == 1, \
				f"No/too many inputs found for layer {module}"
				# pre_index = time.time()
				subset = torch.index_select(module.inputs[0], 0, indices)
				# post_index = time.time() - pre_index
				# pre_forward_time = time.time()
				# forward pass this through the decorrelation matrix
				if isinstance(module, DecorrLinear):
					# print("Linear layer")
					batched=False
					decorr_out = F.linear(subset, module.decorr_layer)

				# Conv1d layers use channel_independent decorrelation, forward
				# passing through the decorrelation matrices is more involved
				else:
					# all data in each convolution patch is represented in a single vector
					# (B, n_patches, conv_1d_size*D)
					# print("Conv1d layer")
					batched=True
					x_unfolded = F.unfold(
						subset.unsqueeze(1), 
						(self.backbone.layers[0].mixer.d_inner, module.kernel_size[0]), 
						stride=1, padding=(0, module.kernel_size[0]-1))
					
					# reshapes all inputs as corresponding convolutional "patches"
					patch_matrices = x_unfolded.reshape(
						num_samples, self.backbone.layers[0].mixer.d_inner, -1, module.kernel_size[0])
					
					# perform decorrelation operation
					decorr_out = einsum(module.decorr_layer, patch_matrices,
						'd conv_1d_size dummy, n_samples d n_patches dummy -> n_samples n_patches d conv_1d_size')

				# post_forward_time = time.time() - pre_forward_time

				# pre_loss_time = time.time()
				grad, corr_loss, whit_loss = self.loss_module(
					decorr_out, self.kappa,
					compute_grad, compute_loss, batched=batched)
				# post_loss_time = time.time() - pre_loss_time
				
				# pre_set_time = time.time()
				module.grad = grad
				module.corr_loss = corr_loss
				module.whit_loss = whit_loss	
				# post_set_time = time.time()	- pre_set_time


		# 		print(f"Indexing: {post_index}")
		# 		print(f"Forward: {post_forward_time}")
		# 		print(f"Loss/grad computation: {post_loss_time}")
		# 		print(f"Matrix changing time: {post_set_time}")			
		# print(f"Sampling: {post_sampling_time}")

		# total_time = time.time()
		self.apply_to_decorr(partial(_compute_decorr_grad_loss, indices=indices))
		# post_total_time = time.time() - total_time
		# print(f"Everything in apply: {post_total_time}")


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

		# gradients should already have been computed
		def _update_decorr_matrices(module):
			assert module.grad is not None, "Gradient not computed"
			with torch.no_grad():
				# PLACE TO IMPLEMENT GAIN SCALING?
				module.decorr_layer -= self.decorr_lr * (module.grad @ module.decorr_layer)

		self.apply_to_decorr(_update_decorr_matrices)

	def apply_to_decorr(self, f):
		"Used for applying simple functions to all of a model's decorrelated layers"
		for layer in self.decorr_layers:
			f(layer)

	
