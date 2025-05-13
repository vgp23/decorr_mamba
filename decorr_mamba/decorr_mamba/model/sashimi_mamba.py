import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from copy import deepcopy
from collections import namedtuple

from mamba_ssm.models.mixer_seq_simple import create_block, _init_weights
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba

try:
	from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
	RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class MambaBlocks(nn.Module):
	''' Series of Mamba blocks interleaved with residual connections
	  	and normalization, to be used at each pooling stage in the
		SaShiMi architecture'''
	def __init__(
		self,
		d_model: int,
		n_layer: int,
		d_intermediate: int,
		ssm_cfg=None,
		attn_layer_idx=None,
		attn_cfg=None,
		norm_epsilon: float = 1e-5,
		rms_norm: bool = False,
		initializer_cfg=None,
		fused_add_norm=False,
		residual_in_fp32=False,
		device=None,
		dtype=None,
	) -> None:
		factory_kwargs = {"device": device, "dtype": dtype}
		super().__init__()
		self.residual_in_fp32 = residual_in_fp32

		# We change the order of residual and layer norm:
		# Instead of LN -> Attn / MLP -> Add, we do:
		# Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
		# the main branch (output of MLP / Mixer). The model definition is unchanged.
		# This is for performance reason: we can fuse add + layer_norm.
		self.fused_add_norm = fused_add_norm
		if self.fused_add_norm:
			if layer_norm_fn is None or rms_norm_fn is None:
				raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

		self.layers = nn.ModuleList(
			[
				create_block(
					d_model,
					d_intermediate=d_intermediate,
					ssm_cfg=ssm_cfg,
					attn_layer_idx=attn_layer_idx,
					attn_cfg=attn_cfg,
					norm_epsilon=norm_epsilon,
					rms_norm=rms_norm,
					residual_in_fp32=residual_in_fp32,
					fused_add_norm=fused_add_norm,
					layer_idx=i,
					**factory_kwargs,
				)
				for i in range(n_layer)
			]
		)

		self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
			d_model, eps=norm_epsilon, **factory_kwargs
		)

		self.apply(
			partial(
				_init_weights,
				n_layer=n_layer,
				**(initializer_cfg if initializer_cfg is not None else {}),
				n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
			)
		)

	def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
		return {
			i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
			for i, layer in enumerate(self.layers)
		}

	def forward(self, hidden_states, inference_params=None, **mixer_kwargs):
		residual = None
		for layer in self.layers:
			hidden_states, residual = layer(
				hidden_states, residual, inference_params=inference_params, **mixer_kwargs
			)
		if not self.fused_add_norm:
			residual = (hidden_states + residual) if residual is not None else hidden_states
			hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
		else:
			# Set prenorm=False here since we don't need the residual
			hidden_states = layer_norm_fn(
				hidden_states,
				self.norm_f.weight,
				self.norm_f.bias,
				eps=self.norm_f.eps,
				residual=residual,
				prenorm=False,
				residual_in_fp32=self.residual_in_fp32,
				is_rms_norm=isinstance(self.norm_f, RMSNorm)
			)
		return hidden_states

class DownPool(nn.Module):
	""" SaShiMi down-pooling block """
	def __init__(self, H, q, p, **factory_kwargs):
		super().__init__()
		self.H = H
		self.q = q
		self.p = p
		device = factory_kwargs.get("device")
		dtype = factory_kwargs.get("dtype")
		self.down_pool = nn.Linear(p*H, q*H, dtype=dtype, device=device)
		self.capture_inputs = False # Captures linear layerinputs during decorr training

	def forward(self, x):
		B, L, _ = x.shape
		reshaped = x.reshape(B, -1, self.H*self.p)
		if self.capture_inputs:
			self.down_pool.inputs = reshaped.detach()
			self.capture_inputs = False
		down_sampled = self.down_pool(reshaped)
		return down_sampled 

class UpPool(nn.Module):
	""" SaShiMi up-pooling block """
	def __init__(self, H, q, p, **factory_kwargs):
		super().__init__()
		self.H = H
		self.q = q
		self.p = p
		device = factory_kwargs.get('device')
		dtype = factory_kwargs.get("dtype")
		self.up_pool = nn.Linear(H, int(p*H/q), dtype=dtype, device=device)
		self.capture_inputs = False # Captures linear layer inputs during decorr training

	def forward(self, x):
		B, _, _ = x.shape
		if self.capture_inputs:
			self.up_pool.inputs = x.detach()
			self.capture_inputs = False
		up_sampled = self.up_pool(x)
		# shift to maintain causality
		up_sampled = F.pad(up_sampled[:, :-1, :], (0,0,1,0)) 
		reshaped = up_sampled.reshape(B, -1, int(self.H/self.q))
		return reshaped 

class SaShiMiMamba(nn.Module):
	""" A simple implementation of SaShiMi with a Mamba backbone instead of
	the default S4 blocks"""
	def __init__(self, p:int, q:int, depth: int, config: MambaConfig, 
			  blocks_per_stage:int=5, device=None, dtype=None, 
			  initializer_cfg=None):
			
		super().__init__()
			
		self.config = config
		d_model = config.d_model
		d_intermediate = config.d_intermediate
		vocab_size = config.vocab_size
		ssm_cfg = config.ssm_cfg
		attn_layer_idx = config.attn_layer_idx
		attn_cfg = config.attn_cfg
		rms_norm = config.rms_norm
		residual_in_fp32 = config.residual_in_fp32
		fused_add_norm = config.fused_add_norm
		pad_vocab_size_multiple = config.pad_vocab_size_multiple
		factory_kwargs = {"device": device, "dtype": dtype}	

		if vocab_size % pad_vocab_size_multiple != 0:
			vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)	

		self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
		
		self.down_pooling = nn.ModuleList([
			DownPool(q**i*d_model, q, p, **factory_kwargs) for i in range(depth)
		])
		
		self.up_pooling = nn.ModuleList([
			UpPool(q**i*d_model, q, p, **factory_kwargs) for i in range(1, depth+1)
		])
		
		self.mamba_stages_down = nn.ModuleList([
			MambaBlocks(
			d_model=q**i*d_model,
			n_layer=blocks_per_stage,
			d_intermediate=d_intermediate,
			ssm_cfg=ssm_cfg,
			attn_layer_idx=attn_layer_idx,
			attn_cfg=attn_cfg,
			rms_norm=rms_norm,
			initializer_cfg=initializer_cfg,
			fused_add_norm=fused_add_norm,
			residual_in_fp32=residual_in_fp32,
			**factory_kwargs) for i in range(0,depth+1)
		])		
		self.mamba_stages_up = nn.ModuleList([
			MambaBlocks(
			d_model=q**i*d_model,
			n_layer=blocks_per_stage,
			d_intermediate=d_intermediate,
			ssm_cfg=ssm_cfg,
			attn_layer_idx=attn_layer_idx,
			attn_cfg=attn_cfg,
			rms_norm=rms_norm,
			initializer_cfg=initializer_cfg,
			fused_add_norm=fused_add_norm,
			residual_in_fp32=residual_in_fp32,
			**factory_kwargs) for i in range(0,depth)
		])

		self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

		# weight tying, remove the original weight parameter
		del self.lm_head.weight
		self.lm_head.weight = self.embedding.weight
	
	def forward(self, x):
		# down sample and keep residuals
		x = self.embedding(x) 
		residuals = []
		for dp, blocks in zip(self.down_pooling, 
			self.mamba_stages_down[:-1]):
			residuals.append(x)
			x = blocks(x)
			x = dp(x)

		residuals.append(x)

		# get through the bend in the U
		x = self.mamba_stages_down[-1](x)
		x = x + residuals.pop()

		# up-sampling!
		for up, blocks in zip(
			reversed(self.up_pooling), reversed(self.mamba_stages_up)):
			u = up(x)
			x = u + residuals.pop()
			x = blocks(x)
	
		lm_logits = self.lm_head(x)
		CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
		return CausalLMOutput(logits=lm_logits)
		
		
if __name__ == "__main__":
	B = 16
	L = 512
	batch = torch.randint(0, 256, (B, 512)).to("cuda:4")

	config = MambaConfig(d_model=64, vocab_size=256)

	# model = SaShiMiMamba(p=16, q=2, depth=2, 
	# 				  config=config, blocks_per_stage=3).to("cuda:3")

	model = SaShiMiMamba(config=config, p=16, q=2, depth=2, 
							blocks_per_stage=3).to("cuda:4")
	
	print(model)
	n_pars = sum(p.numel() for p in model.parameters())

	print(f"Number of parameters: {n_pars}")

	out = model(batch)
	print(out.shape)

