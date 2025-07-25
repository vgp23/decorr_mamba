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
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.utils.torch import custom_fwd, custom_bwd
import math
from decorr_mamba.model.sashimi_mamba import SaShiMiMamba
from decorr_mamba.model.decorrelation_functions import decorr_mamba_inner_fn #decorr_mamba_split_conv1d_scan_combined
from collections import namedtuple


try:
	from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
	from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
	import causal_conv1d_cuda
except ImportError:
	causal_conv1d_fn, causal_conv1d_update, causal_conv1d_cuda, causal_conv1d_varlen_states = None, None, None, None

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
				x = rearrange(x, 'b l d -> (b l) d')
				mean_dim = 0
			# used where decorrelation layer has multiple matrices to train
			# (this is the case for Conv1d)
			else:
				# for conv1d! 
				# collapse across batch and n_patches dimension
				# in this case we're updating d matrices, each with info
				# from one embedding dimension channel.
				# (D, all_samples, decorr_matrix_size)
				x = rearrange(x, 
					'b n_patches d decorr_matrix_size -> d (b n_patches) decorr_matrix_size')
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
			
class DecorrMixin:
	""" Wrapper class providing simple functionality to all decorrelation
		layers. """
	def __init__(self, compute_loss: bool, 
			  kappa: float, sample_frac: float, demeaning: bool = False):

		self.corr_loss = None
		self.whit_loss = None
		self.kappa = kappa
		self.sample_frac = sample_frac
		self.compute_loss = compute_loss
		self.loss_module = DecorrLoss()
		self.inputs = None
		self.demeaning = demeaning 

	def reset(self):
		""" Resets gradients and losses of decorrelation layers, used 
			before/after gradient descent steps."""
		
		self.corr_loss = None
		self.whit_loss = None
		self.decorr_layer.grad = None

	def update_decorr_matrix(self, decorr_lr: float):
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
				kappa: float = None, sample_frac: float = None, 
				demeaning: bool = False, **factory_kwargs):
		
		DecorrMixin.__init__(compute_loss, kappa, sample_frac, demeaning)
		nn.Linear.__init__(**factory_kwargs)

		self.decorr_layer = nn.Parameter(
			torch.eye(self.in_features).to(self.weight.device), requires_grad=False)

		if self.demeaning:
			self.register_buffer("batch_mean", None)
			self.register_buffer("running_mean", 
				torch.zeros(self.in_features,
				device=self.weight.device))
		

	@classmethod
	def from_existing_layer(cls, original_layer: nn.Module,
						 compute_loss: bool = True, kappa: float = None, 
						 sample_frac: float = None, demeaning: bool = False):
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
		DecorrMixin.__init__(new_layer, compute_loss, kappa, sample_frac, demeaning)

		# initialize DecorrLinear-specific attributes
		new_layer.decorr_layer = nn.Parameter(
			torch.eye(original_layer.in_features, 
			 device=original_layer.weight.device), requires_grad=False
		)

		if new_layer.demeaning:
			new_layer.register_buffer("batch_mean", None)
			new_layer.register_buffer("running_mean", 
				torch.zeros(new_layer.in_features,
				device=new_layer.weight.device))

		return new_layer

	def forward(self, x):
		# all demeaning and saving decorrelation layer inputs happens in the
		# main forward pass function for Mamba, check that there!
		return linear(x, self.fused_weight, self.bias)

	def fuse_decorr(self):
		""" 
		Pre-multiplies the decorrelation and standard weight parameters
		into a single parameter matrix, numerically equivalent to passing
		inputs through both matrices separately."""

		self.fused_weight = self.weight @ self.decorr_layer
			
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
			with torch.no_grad():
				decorr_out = subset @ self.decorr_layer.T

			grad, corr_loss, whit_loss = self.loss_module(
				decorr_out, self.kappa, self.training, self.compute_loss, batched=False)		


			self.decorr_layer.grad = grad
			self.corr_loss = corr_loss
			self.whit_loss = whit_loss	

class DecorrConv1d(DecorrMixin, nn.Conv1d):
	def __init__(self, compute_loss: bool = True, kappa: float = None, 
			  sample_frac: float = None, demeaning: bool=False, **factory_kwargs):
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
		
		DecorrMixin.__init__(compute_loss, kappa, sample_frac, demeaning)
		nn.Conv1d.__init__(**factory_kwargs)

		# (in_channels, kernel_size, kernel_size)
		all_matrices = torch.eye(self.kernel_size[0]).unsqueeze(0).repeat(
					self.in_channels, 1, 1).to(self.weight.device)
		
		self.decorr_layer = nn.Parameter(all_matrices, requires_grad=False)

		if self.demeaning:
			self.register_buffer("batch_mean", None)
			self.register_buffer("running_mean", 
				torch.zeros(self.in_channels,
				device=self.weight.device))
	

	@classmethod
	def from_existing_layer(cls, original_layer: nn.Module, 
						 compute_loss: bool = True, kappa: float = None, 
						 sample_frac: float = None, demeaning: bool=False):
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
		DecorrMixin.__init__(new_layer, compute_loss, kappa, sample_frac, demeaning)

		# initialize DecorrConv1d-specific attributes
		# (in_channels, kernel_size, kernel_size)
		all_matrices = torch.eye(new_layer.kernel_size[0]).unsqueeze(0).repeat(
					new_layer.in_channels, 1, 1).to(new_layer.weight.device)

		new_layer.decorr_layer = nn.Parameter(all_matrices, requires_grad=False)

		if new_layer.demeaning:
			new_layer.register_buffer("batch_mean", None)
			new_layer.register_buffer("running_mean", 
				torch.zeros(new_layer.in_channels, 
				device=new_layer.weight.device))

		return new_layer

	def forward(self, x):
		# all demeaning and saving decorrelation layer inputs happens in the
		# main forward pass function for Mamba, check that there!
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



class DecorrMamba(MambaLMHeadModel):
	"""Extends MambaLMHeadModel by integrating decorrelation layers into the architecture."""

	def __init__(self, existing_model: MambaLMHeadModel = None, copy: bool = False, 
			  kappa: float = 0.5, sample_frac: float = 0.1,
			  compute_loss: bool = True, demeaning: bool = False, **factory_kwargs):
		"""
		Initializes a DecorrMamba model by adding decorrelation layers to the existing model.

		Args:
			existing_model (MambaLMHeadModel, optional): Pre-existing model to copy or extend.
			copy (bool): Whether to deep copy the existing model.
			kappa (float): Scaling factor for decorrelation loss.
			sample_frac (float): Fraction of samples used for decorrelation computation.
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

		self.demeaning = demeaning
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
				# works for Mamba2 blocks as well! In those cases there's no
				# parameter named x_proj so it's just skipped. NB: this is NOT
				# designed for Mamba2 with tensor parallelism, that would 
				# require implementation of new decorrelation classes for the
				# input and output projections. 
				if name == "in_proj" or name == "out_proj" or name == "x_proj":
					self.n_decorr_layers += 1
					setattr(module, name, DecorrLinear.from_existing_layer(
						original_layer=child, compute_loss=compute_loss, 
						kappa=kappa, sample_frac=sample_frac, 
						demeaning=demeaning))
					
					self.decorr_layers.append(getattr(module, name))

				if name == "conv1d":
					self.n_decorr_layers += 1 				
					setattr(module, name, DecorrConv1d.from_existing_layer(
						original_layer=child, compute_loss=compute_loss,
						kappa=kappa, sample_frac=sample_frac,
						demeaning=demeaning))
					
					self.decorr_layers.append(getattr(module, name))

		self.apply(_create_decorr_matrices)

		self.mean_corr_loss = None
		self.mean_whit_loss = None

		# These are here for reference only, these have been passed to 
		# the decorrelation modules and they work inside there. 
		self.kappa = kappa
		self.sample_frac = sample_frac
		self.compute_loss = compute_loss
		
		def _modify_mamba_block_functions(module):
			for _, child in module.named_children():
				if type(child) is Mamba:
					child.forward = partial(self._mamba_block_forward.__get__(child), demeaning=demeaning)
					child.step = partial(self._mamba_block_step.__get__(child), demeaning=demeaning)	
				# elif type(child) is Mamba2:
				# 	child.forward = self._mamba2_block_forward.__get__(child)
				# 	child.step = self._mamba2_block_step.__get__(child)		
			

		self.apply(_modify_mamba_block_functions)

	def _mamba_block_forward(self, hidden_states, inference_params=None, to_remove=[], demeaning=False):
		"""
		hidden_states: (B, L, D)
		Returns: same shape as hidden_states
		"""

		# set the function to reference fused weights when decorrelation layers
		# exist, otherwise the regular weights. This only applies to the 
		# weights where decorrelation would be active, and is used for the 
		# ablation study. 
		in_proj_weight = self.in_proj.fused_weight if not "in_proj" in to_remove else self.in_proj.weight
		out_proj_weight = self.out_proj.fused_weight if not "out_proj" in to_remove else self.out_proj.weight
		conv1d_weight = self.conv1d.fused_weight if not "conv1d" in to_remove else self.conv1d.weight
		x_proj_weight = self.x_proj.fused_weight if not "x_proj" in to_remove else self.x_proj.weight

		assert self.in_proj.demeaning == self.x_proj.demeaning == \
			self.conv1d.demeaning == self.out_proj.demeaning, "Must have demeaning set the same for all layers"
		
		if not "in_proj" in to_remove:
			if demeaning:
				if self.training:
					# save this mean for updating the running average, and use
					# it to de-mean the current batch
					with torch.no_grad():
						self.in_proj.batch_mean = \
							torch.mean(
								hidden_states.detach(), axis=[0,1])
					hidden_states = hidden_states - self.in_proj.batch_mean[None, None, :]
				else:
					# use the running average to de-mean
					hidden_states = hidden_states - self.in_proj.running_mean[None, None, :]

			# save inputs for decorrelation layer update
			if self.training:
				self.in_proj.inputs = hidden_states.detach()

		batch, seqlen, dim = hidden_states.shape

		conv_state, ssm_state = None, None
		if inference_params is not None:
			conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
			if inference_params.seqlen_offset > 0:
				# The states are updated inplace
				out, _, _ = self.step(hidden_states, conv_state, ssm_state, to_remove)
				return out

		# We do matmul and transpose BLH -> HBL at the same time
		xz = rearrange(
			in_proj_weight @ rearrange(hidden_states, "b l d -> d (b l)"),
			"d (b l) -> b d l",
			l=seqlen,
		)

		if self.in_proj.bias is not None:
			xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

		if not self.complex:
			A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
		else:
			A = torch.complex(-torch.exp(self.log_A_real), self.A_imag)
			
		# In the backward pass we write dx and dz next to each other to avoid torch.cat
		if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
			running_means = None
			if not self.training and demeaning:
				running_means = {
					"conv1d": self.conv1d.running_mean,
					"x_proj": self.x_proj.running_mean,
					"out_proj": self.out_proj.running_mean
				}

			out, decorr_train_info = decorr_mamba_inner_fn(
				xz,
				conv1d_weight,
				self.conv1d.bias,
				x_proj_weight,
				self.dt_proj.weight,
				out_proj_weight,
				self.out_proj.bias,
				A,
				None,  # input-dependent B
				None,  # input-dependent C
				self.D.float(),
				delta_bias=self.dt_proj.bias.float(),
				delta_softplus=True,
				demeaning=demeaning,
				running_means=running_means
			)

			# setting the necessary attributes to perform a training step
			if self.training:
				for layer_name in ["conv1d", "x_proj", "out_proj"]:
					if layer_name not in to_remove:
						layer = getattr(self, layer_name)
						layer.inputs = decorr_train_info["inputs"][layer_name]
						if layer.demeaning:
							layer.batch_mean = decorr_train_info["means"][layer_name]

		else: 
			x, z = xz.chunk(2, dim=1)
			# Compute short convolution
			if conv_state is not None:
				# If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
				# Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
				conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)

			if not "conv1d" in to_remove:
				if demeaning:
					if self.training:
						with torch.no_grad():
							self.conv1d.batch_mean = \
								torch.mean(
									x.detach(), axis=[0,2])
							
						x = x - self.conv1d.batch_mean[None, :, None]
					else:
						x = x - self.conv1d.running_mean[None, :, None]
				if self.training:
					self.conv1d.inputs = x.detach()
				
			if causal_conv1d_fn is None:
				x = self.act(self.conv1d(x)[..., :seqlen])
			else:
				assert self.activation in ["silu", "swish"]
				x = causal_conv1d_fn(
					x=x,
					weight=rearrange(conv1d_weight, "d 1 w -> d w"),
					bias=self.conv1d.bias,
					activation=self.activation,
				)

			# We're careful here about the layout, to avoid extra transposes.
			# We want dt to have d as the slowest moving dimension
			# and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.

			if not "x_proj" in to_remove:
				if self.x_proj.demeaning:
					if self.training:
						with torch.no_grad():
							self.x_proj.batch_mean = torch.mean(
								x, axis=[0,2])
							
						x = x - self.x_proj.batch_mean[None, :, None]
					else:
						x = x - self.x_proj.running_mean[None, :, None]
				if self.training:
					self.x_proj.inputs = rearrange(x.detach(), "b d l -> b l d")
			
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

			if not "out_proj" in self.to_remove:
				if self.out_proj.demeaning:
					if self.training:
						with torch.no_grad():
							self.out_proj.batch_mean = torch.mean(
								y, axis=[0,1])
						y = y - self.out_proj.batch_mean[None, None, :]
					else:
						y = y - self.out_proj.running_mean[None, None, :]
				
				if self.training:
					self.out_proj.inputs = y.detach()

			out = self.out_proj(y)

		return out

	def _mamba_block_step(self, hidden_states, conv_state, ssm_state, to_remove=[], demeaning=False):

		conv1d_weight = self.conv1d.fused_weight if not "conv1d" in to_remove else self.conv1d.weight

		dtype = hidden_states.dtype
		assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
		
		# de-meaning of the input here has already happened in _mamba_block_forward!

		xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
		x, z = xz.chunk(2, dim=-1)  # (B D)

		if demeaning:
			x = x - self.conv1d.running_mean[None, :]

		# Conv step
		if causal_conv1d_update is None:
			conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
			conv_state[:, :, -1] = x

			x = torch.sum(conv_state * rearrange(conv1d_weight, "d 1 w -> d w"), dim=-1)  # (B D)
			if self.conv1d.bias is not None:
				x = x + self.conv1d.bias
			x = self.act(x).to(dtype=dtype)
		else:
			
			x = causal_conv1d_update(
				x,
				conv_state,
				rearrange(conv1d_weight, "d 1 w -> d w"),
				self.conv1d.bias,
				self.activation,
			)

		if demeaning:
			x = x - self.x_proj.running_mean[None, :]

		x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
		dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
		# Don't add dt_bias here

		dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)

		if not self.is_complex:
			A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
		else:
			A = torch.complex(-torch.exp(self.log_A_real), self.A_imag)

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

		if demeaning:
			y = y - self.out_proj.running_mean[None, :]

		out = self.out_proj(y)
		return out.unsqueeze(1), conv_state, ssm_state

	# def _mamba2_block_forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
	# 	"""
	# 	u: (batch, seqlen, hidden_dim) if seqlen=None.
	# 		If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
	# 		split u during sequence parallel, we split the batch * seqlen dimension
	# 		(in case batch is small).
	# 	Returns: same shape as u
	# 	"""
	# 	seqlen_og = seqlen
	# 	if seqlen is None:
	# 		batch, seqlen, dim = u.shape
	# 	else:
	# 		batch_seqlen, dim = u.shape
	# 		batch = batch_seqlen // seqlen

	# 	conv_state, ssm_state = None, None
	# 	if inference_params is not None:
	# 		inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
	# 		conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
	# 		if inference_params.seqlen_offset > 0:
	# 			# The states are updated inplace
	# 			out, _, _ = self.step(u, conv_state, ssm_state)
	# 			return out
			
	# 	self.in_proj.inputs = u.detach()
	# 	zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
	# 	if seqlen_og is not None:
	# 		zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
	# 	# If the model is loaded in fp16, without the .float() here, A might be -inf
	# 	A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
	# 	dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
	# 	if self.use_mem_eff_path and inference_params is None:
	# 		out, layer_inputs = decorr_mamba_split_conv1d_scan_combined(
	# 			zxbcdt,
	# 			rearrange(self.conv1d.fused_weight, "d 1 w -> d w"),
	# 			self.conv1d.bias,
	# 			self.dt_bias,
	# 			A,
	# 			D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
	# 			chunk_size=self.chunk_size,
	# 			seq_idx=seq_idx,
	# 			activation=self.activation,
	# 			rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
	# 			rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
	# 			outproj_weight=self.out_proj.fused_weight,
	# 			outproj_bias=self.out_proj.bias,
	# 			headdim=None if self.D_has_hdim else self.headdim,
	# 			ngroups=self.ngroups,
	# 			norm_before_gate=self.norm_before_gate,
	# 			**dt_limit_kwargs,
	# 		)

	# 		self.conv1d.inputs = layer_inputs["conv1d"]
	# 		self.out_proj.inputs = layer_inputs["out_proj"]

	# 		if seqlen_og is not None:
	# 			out = rearrange(out, "b l d -> (b l) d")
	# 		if self.process_group is not None:
	# 			reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
	# 			out = reduce_fn(out, self.process_group)
	# 	else:
	# 		d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
	# 		z0, x0, z, xBC, dt = torch.split(
	# 			zxbcdt,
	# 			[d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
	# 			dim=-1
	# 		)
	# 		if conv_state is not None:
	# 			if cu_seqlens is None:
	# 				# If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
	# 				# Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
	# 				xBC_t = rearrange(xBC, "b l d -> b d l")
	# 				conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
	# 			else:
	# 				assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
	# 				assert batch == 1, "varlen inference only supports batch dimension 1"
	# 				conv_varlen_states = causal_conv1d_varlen_states(
	# 					xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
	# 				)
	# 				conv_state.copy_(conv_varlen_states)
	# 		assert self.activation in ["silu", "swish"]

	# 		self.conv1d.inputs = xBC.detach().transpose(1,2)

	# 		if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
	# 			assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
	# 			xBC = self.act(
	# 				self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
	# 			)  # (B, L, self.d_ssm + 2 * ngroups * d_state)
	# 		else:
	# 			xBC = causal_conv1d_fn(
	# 				xBC.transpose(1, 2),
	# 				rearrange(self.conv1d.fused_weight, "d 1 w -> d w"),
	# 				bias=self.conv1d.bias,
	# 				activation=self.activation,
	# 				seq_idx=seq_idx,
	# 			).transpose(1, 2)
	# 		x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
	# 		y = mamba_chunk_scan_combined(
	# 			rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
	# 			dt,
	# 			A,
	# 			rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
	# 			rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
	# 			chunk_size=self.chunk_size,
	# 			D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
	# 			z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
	# 			dt_bias=self.dt_bias,
	# 			dt_softplus=True,
	# 			seq_idx=seq_idx,
	# 			cu_seqlens=cu_seqlens,
	# 			**dt_limit_kwargs,
	# 			return_final_states=ssm_state is not None,
	# 			return_varlen_states=cu_seqlens is not None and inference_params is not None,
	# 		)
	# 		if ssm_state is not None:
	# 			y, last_state, *rest = y
	# 			if cu_seqlens is None:
	# 				ssm_state.copy_(last_state)
	# 			else:
	# 				varlen_states = rest[0]
	# 				ssm_state.copy_(varlen_states)
	# 		y = rearrange(y, "b l h p -> b l (h p)")
	# 		if self.rmsnorm:
	# 			y = self.norm(y, z)
	# 		if d_mlp > 0:
	# 			y = torch.cat([F.silu(z0) * x0, y], dim=-1)
	# 		if seqlen_og is not None:
	# 			y = rearrange(y, "b l d -> (b l) d")

	# 		self.out_proj.inputs = y.detach()
			
	# 		out = self.out_proj(y)
	# 	return out

	# def _mamba2_block_step(self, hidden_states, conv_state, ssm_state):
	# 	dtype = hidden_states.dtype
	# 	assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"

	# 	self.in_proj.inputs = hidden_states.detach().squeeze(1)

	# 	zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
	# 	d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
	# 	z0, x0, z, xBC, dt = torch.split(
	# 		zxbcdt,
	# 		[d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
	# 		dim=-1
	# 	)

	# 	self.conv1d.inputs = xBC.detach()

	# 	# Conv step
	# 	if causal_conv1d_update is None:
	# 		conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
	# 		conv_state[:, :, -1] = xBC
	# 		xBC = torch.sum(conv_state * rearrange(self.conv1d.fused_weight, "d 1 w -> d w"), dim=-1)  # (B D)
	# 		if self.conv1d.bias is not None:
	# 			xBC = xBC + self.conv1d.bias
	# 		xBC = self.act(xBC).to(dtype=dtype)
	# 	else:
	# 		xBC = causal_conv1d_update(
	# 			xBC,
	# 			conv_state,
	# 			rearrange(self.conv1d.fused_weight, "d 1 w -> d w"),
	# 			self.conv1d.bias,
	# 			self.activation,
	# 		)

	# 	x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
	# 	A = -torch.exp(self.A_log.float())  # (nheads,)

	# 	# SSM step
	# 	if selective_state_update is None:
	# 		assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
	# 		# Discretize A and B
	# 		dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
	# 		dA = torch.exp(dt * A)  # (batch, nheads)
	# 		x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
	# 		dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
	# 		ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
	# 		y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
	# 		y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
	# 		y = rearrange(y, "b h p -> b (h p)")
	# 		if not self.rmsnorm:
	# 			y = y * self.act(z)  # (B D)
	# 	else:
	# 		A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
	# 		dt = repeat(dt, "b h -> b h p", p=self.headdim)
	# 		dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
	# 		D = repeat(self.D, "h -> h p", p=self.headdim)
	# 		B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
	# 		C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
	# 		x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
	# 		if not self.rmsnorm:
	# 			z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
	# 		y = selective_state_update(
	# 			ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
	# 			dt_bias=dt_bias, dt_softplus=True
	# 		)
	# 		y = rearrange(y, "b h p -> b (h p)")
	# 	if self.rmsnorm:
	# 		y = self.norm(y, z)
	# 	if d_mlp > 0:
	# 		y = torch.cat([F.silu(z0) * x0, y], dim=-1)

	# 	self.out_proj.inputs = y.detach()

	# 	out = self.out_proj(y)
	# 	return out.unsqueeze(1), conv_state, ssm_state

	def forward(self, x):
		# fuse decorrelation + main model parameters, then let forward
		# pass proceed as normal
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

	def decorr_operations(self, crop_frac: float = 1.0):
		""" 
		Performs all decorrelation operations (loss and/or gradient computation, 
		depending on configuration)."""
		# crop_frac is used in case we want to sample even less than one 
		# sequence. We shorten the sequence length to accomplish this!
		def cropped_decorr_op(module):

			if isinstance(module, DecorrLinear):
				# inputs of shape (B,L,D)
				_, L, _ = module.inputs.shape
				cropped_len = int(crop_frac*L)
				module.compute_decorr_grad_loss(
					module.inputs[:, :cropped_len, :])
			elif isinstance(module, DecorrConv1d):
				# inputs of shape (B, D, L)
				_, _, L = module.inputs.shape
				cropped_len = int(crop_frac*L)
				module.compute_decorr_grad_loss(
					module.inputs[:, :, :cropped_len])	
			else:
				cropped_len = None
				return NotImplementedError
			
			assert cropped_len >= 1, "Cropping length too small!"
			
		self.apply_to_decorr(cropped_decorr_op)

	def update_decorr_matrices(self, decorr_lr: float, demeaning_lr: float = None):
		""" 
		Updates the decorrelation matrices for all decorrelation layers
		within the model
		"""
		def _update(module):
			module.update_decorr_matrix(decorr_lr)
			if self.demeaning:
				assert demeaning_lr is not None, "Must supply demeaning_lr for demeaning!"
				with torch.no_grad():
					module.running_mean.add_(
						demeaning_lr * (module.batch_mean - module.running_mean))
												 
		self.apply_to_decorr(_update)

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
			  kappa: float = 0.5, sample_frac: float = 0.1,
			  compute_loss: bool = True, to_remove: list[str] = [], 
			  demeaning: bool = False, **factory_kwargs):
		
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
		self.to_remove = to_remove
		self.demeaning=demeaning

		def _create_decorr_matrices(module):
			""" 
			Used to recursively traverse model and create decorrelation matrices 
			in pre-defined places

			Args:
				module (nn.Module): model layers to add decorrelation into 
			"""
			if self.config.ssm_cfg.get("layer") == "Mamba2":
				decorr_linear_names = ["in_proj", "out_proj", "up_pool", "down_pool"]
			else:
				decorr_linear_names = ["in_proj", "out_proj", "x_proj", "up_pool", "down_pool"]

			for name, child in module.named_children():
				if name in decorr_linear_names and not name in self.to_remove:
					self.n_decorr_layers += 1
					setattr(module, name, DecorrLinear.from_existing_layer(
						original_layer=child, compute_loss=compute_loss, 
						kappa=kappa, sample_frac=sample_frac, demeaning=demeaning))
					
					self.decorr_layers.append(getattr(module, name))

				if name == "conv1d" and not "conv1d" in self.to_remove:
					self.n_decorr_layers += 1 				
					setattr(module, name, DecorrConv1d.from_existing_layer(
						original_layer=child, compute_loss=compute_loss,
						kappa=kappa, sample_frac=sample_frac, demeaning=demeaning))
					self.decorr_layers.append(getattr(module, name))

		self.apply(_create_decorr_matrices)

		self.mean_corr_loss = None
		self.mean_whit_loss = None

		# These are here for reference only, these have been passed to 
		# the decorrelation modules and they work inside there. 
		self.kappa = kappa
		self.sample_frac = sample_frac
		self.compute_loss = compute_loss

		def _modify_mamba_block_functions(module):
			for _, child in module.named_children():
				if type(child) is Mamba:
					child.forward = partial(self._mamba_block_forward.__get__(child), to_remove=to_remove, demeaning=demeaning)
					child.step = partial(self._mamba_block_step.__get__(child), to_remove=to_remove, demeaning=demeaning)
				# elif type(child) is Mamba2:
				# 	child.forward = self._mamba2_block_forward.__get__(child)
				# 	child.step = self._mamba2_block_step.__get__(child)				

		self.apply(_modify_mamba_block_functions)

	def forward(self, x):
		# fuse decorrelation + main model parameters, then let forward
		# pass proceed as normal
		if self.training:
			self.fuse_decorr()

		# down sample and keep residuals
		# NB demeaning is already handled in the Mamba blocks, we just need to
		# perform demeaning manually for the up and down-pooling layers.

		x = self.embedding(x) 
		residuals = []
		for dp, blocks in zip(self.down_pooling, 
			self.mamba_stages_down[:-1]):
			residuals.append(x)
			x = blocks(x)
			# inputs are captured (and optionally demeaned) by Mamba blocks separately
			if isinstance(dp.down_pool, DecorrLinear):
				dp.capture_inputs = self.training
				dp.demeaning = self.demeaning

			x = dp(x)

		residuals.append(x)

		# get through the bend in the U
		x = self.mamba_stages_down[-1](x)
		x = x + residuals.pop()

		# up-sampling!
		for up, blocks in zip(
			reversed(self.up_pooling), reversed(self.mamba_stages_up)):
			if isinstance(up.up_pool, DecorrLinear):
				up.capture_inputs = self.training
				up.demeaning = self.demeaning
			u = up(x)
			x = u + residuals.pop()
			x = blocks(x)
	
		lm_logits = self.lm_head(x)
		CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
		return CausalLMOutput(logits=lm_logits)

