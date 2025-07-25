from typing import Optional

import math
from packaging import version

import torch
import torch.nn.functional as F
from torch import Tensor
from mamba_ssm.utils.torch import custom_bwd, custom_fwd
import selective_scan_cuda

import triton
import triton.language as tl

from einops import rearrange, repeat

try:
	from causal_conv1d import causal_conv1d_fn
	import causal_conv1d_cuda
except ImportError:
	causal_conv1d_fn, causal_conv1d_cuda = None, None

# from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd, _bmm_chunk_bwd
# from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd, _chunk_cumsum_bwd
# from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd, _chunk_state_bwd_db
# from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_bwd_ddAcs_stable
# from mamba_ssm.ops.triton.ssd_chunk_state import chunk_state, chunk_state_ref
# from mamba_ssm.ops.triton.ssd_chunk_state import chunk_state_varlen
# from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd, _state_passing_bwd
# from mamba_ssm.ops.triton.ssd_state_passing import state_passing, state_passing_ref
# from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd, _chunk_scan_bwd_dz, _chunk_scan_bwd_dstates
# from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dC, _chunk_scan_bwd_dcb
# from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_ddAcs_stable
# from mamba_ssm.ops.triton.ssd_chunk_scan import chunk_scan, chunk_scan_ref
# from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_ddAcs_prev
from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn, _layer_norm_fwd, _layer_norm_bwd
from mamba_ssm.ops.triton.k_activations import _swiglu_fwd, _swiglu_bwd
from mamba_ssm.ops.triton.ssd_combined import _mamba_chunk_scan_combined_fwd, _mamba_chunk_scan_combined_bwd
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, MambaInnerFn, rms_norm_forward, mamba_inner_fn
from mamba_ssm.ops.triton.ssd_combined import MambaSplitConv1dScanCombinedFn

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')

# Decorrelation training requires capturing inputs to each of the decorrelation
# layers, which requires re-writing some of the default Mamba functions. To
# avoid clutter in the main decorrelation.py, we'll do these here. 

# def decorr_mamba_split_conv1d_scan_combined(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu", rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None, ngroups=1, norm_before_gate=True):
# 	"""
# 	Argument:
# 		zxbcdt: (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads) where dim == nheads * headdim
# 		conv1d_weight: (dim + 2 * ngroups * dstate, width)
# 		conv1d_bias: (dim + 2 * ngroups * dstate,)
# 		dt_bias: (nheads,)
# 		A: (nheads)
# 		D: (nheads, headdim) or (nheads,)
# 		initial_states: (batch, nheads, headdim, dstate)
# 		seq_idx: (batch, seqlen), int32
# 		rmsnorm_weight: (dim,)
# 		outproj_weight: (out_dim, dim)
# 		outproj_bias: (out_dim,)
# 		headdim: if D is 1D, headdim must be passed in
# 		norm_before_gate: if True, we do RMSNorm(x) * F.silu(z). If False, we do RMSNorm(x * F.silu(z))
# 	Return:
# 		out: (batch, seqlen, dim)
# 	"""
# 	return DecorrMambaSplitConv1dScanCombinedFn.apply(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states, seq_idx, dt_limit, return_final_states, activation, rmsnorm_weight, rmsnorm_eps, outproj_weight, outproj_bias, headdim, ngroups, norm_before_gate)

# class DecorrMambaSplitConv1dScanCombinedFn(torch.autograd.Function):

# 	@staticmethod
# 	@custom_fwd
# 	def forward(ctx, zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu",
# 				rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None,
# 				ngroups=1, norm_before_gate=True):
# 		assert activation in [None, "silu", "swish"]
# 		if D.dim() == 1:
# 			assert headdim is not None
# 			nheads, = D.shape
# 		else:
# 			nheads, headdim = D.shape
# 		batch, seqlen, _ = zxbcdt.shape
# 		dim = nheads * headdim
# 		assert nheads % ngroups == 0
# 		dstate = (conv1d_weight.shape[0] - dim) // ngroups // 2
# 		d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ngroups * dstate - nheads) // 2
# 		assert d_nonssm >= 0
# 		assert zxbcdt.shape == (batch, seqlen, 2 * d_nonssm + 2 * dim + 2 * ngroups * dstate + nheads)
# 		assert dt_bias.shape == (nheads,)
# 		assert A.shape == (nheads,)
# 		zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + ngroups * dstate * 2, nheads], dim=-1)
# 		seq_idx = seq_idx.contiguous() if seq_idx is not None else None

# 		conv1d_inputs = rearrange(xBC.detach(), "b s d -> b d s")

# 		xBC_conv = rearrange(
# 			causal_conv1d_cuda.causal_conv1d_fwd(rearrange(xBC, "b s d -> b d s"),
# 												 conv1d_weight, conv1d_bias, seq_idx, None, None, activation in ["silu", "swish"]),
# 			"b d s -> b s d"
# 		)
# 		x, B, C = torch.split(xBC_conv, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
# 		x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
# 		B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
# 		C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
# 		z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None
# 		if rmsnorm_weight is None:
# 			out, out_x, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size=chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=dt_limit)
# 			out = rearrange(out, "b s h p -> b s (h p)")
# 			rstd = None
# 			if d_nonssm > 0:
# 				out = torch.cat([_swiglu_fwd(zx0), out], dim=-1)
# 		else:
# 			out_x, _, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size=chunk_size, D=D, z=None, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=dt_limit)
# 			# reshape input data into 2D tensor
# 			x_rms = rearrange(out_x, "b s h p -> (b s) (h p)")
# 			z_rms = rearrange(z, "b s h p -> (b s) (h p)")
# 			rmsnorm_weight = rmsnorm_weight.contiguous()
# 			if d_nonssm == 0:
# 				out = None
# 			else:
# 				out01 = torch.empty((batch, seqlen, d_nonssm + dim), dtype=x_rms.dtype, device=x_rms.device)
# 				out = rearrange(out01[..., d_nonssm:], "b s d -> (b s) d")
# 				_swiglu_fwd(zx0, out=out01[..., :d_nonssm])
# 			out, _, rstd = _layer_norm_fwd(x_rms, rmsnorm_weight, None, rmsnorm_eps, z_rms, out=out,
# 										   group_size=dim // ngroups,
# 										   norm_before_gate=norm_before_gate, is_rms_norm=True)
# 			if d_nonssm == 0:
# 				out = rearrange(out, "(b s) d -> b s d", b=batch)
# 			else:
# 				out = out01
# 		ctx.outproj_weight_dtype = outproj_weight.dtype if outproj_weight is not None else None
# 		if outproj_weight is not None:
# 			if torch.is_autocast_enabled():
# 				dtype = torch.get_autocast_gpu_dtype()
# 				out, outproj_weight = out.to(dtype), outproj_weight.to(dtype)
# 				outproj_bias = outproj_bias.to(dtype) if outproj_bias is not None else None

# 			out_proj_inputs = out.detach()
# 			out = F.linear(out, outproj_weight, outproj_bias)
# 		else:
# 			assert outproj_bias is None
# 			out_proj_inputs = None

# 		ctx.save_for_backward(zxbcdt, conv1d_weight, conv1d_bias,
# 							  out_x, A, D, dt_bias, initial_states, seq_idx, rmsnorm_weight, rstd, outproj_weight, outproj_bias)
# 		ctx.dt_limit = dt_limit
# 		ctx.return_final_states = return_final_states
# 		ctx.activation = activation
# 		ctx.rmsnorm_eps = rmsnorm_eps
# 		ctx.norm_before_gate = norm_before_gate
# 		ctx.chunk_size = chunk_size
# 		ctx.headdim = headdim
# 		ctx.ngroups = ngroups
# 		layer_inputs = {"conv1d": conv1d_inputs, "out_proj": out_proj_inputs}
# 		return (out, layer_inputs) if not return_final_states else (out, final_states, layer_inputs)

# 	@staticmethod
# 	@custom_bwd
# 	def backward(ctx, dout, *args):
# 		# by default, "args" is a single element containing the final states.
# 		# Here, it's either the layer inputs alone (for computing decorrelation
# 		# updates), or the layer inputs and the final states. We don't want 
# 		# to do anything for the final states, so we'll skip passing them to
# 		# the backward function. 
# 		if len(args) == 2:
# 			args = args[0]
# 		elif len(args) > 2:
# 			raise NotImplementedError("More args received than expected, don't know how to handle")
# 		# If we've only got one element, that's just the inputs, so ignore everything.
# 		else:
# 			args = () 
	
# 		grad_input = MambaSplitConv1dScanCombinedFn.backward(ctx, dout, *args)

# 		# Adding the extra None to account for the lack of gradients w.r.t the
# 		# saved layer inputs
# 		return grad_input + (None,)   

class DecorrMambaInnerFn(MambaInnerFn):

	@staticmethod
	@custom_fwd
	def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
				out_proj_weight, out_proj_bias,
				A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
				C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, b_rms_weight=None, 
				c_rms_weight= None, dt_rms_weight= None, b_c_dt_rms_eps=1e-6,
				demeaning=False, running_means=None):
		"""
			 xz: (batch, dim, seqlen)
		"""
		assert causal_conv1d_cuda is not None, \
			"causal_conv1d_cuda is not available. Please install causal-conv1d."
		
		ctx.demeaning = demeaning
		
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

		conv1d_mean, x_proj_mean, out_proj_mean = None, None, None

		if demeaning:
			# inference only, use running average computed so far
			if running_means:
				x = x - running_means["conv1d"][None, :, None]
			# training, de-mean using batch mean
			else:
				with torch.no_grad():
					conv1d_mean = torch.mean(
						x.detach(), axis=[0,2])
				x = x - conv1d_mean[None, :, None]

		conv1d_inputs = x.detach()

		conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
			x, conv1d_weight, conv1d_bias, None, None, None, True
		)
		# We're being very careful here about the layout, to avoid extra transposes.
		# We want delta to have d as the slowest moving dimension
		# and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.

		if demeaning:
			# inference only, use running average computed so far
			if running_means:
				conv1d_out = conv1d_out - running_means["x_proj"][None, :, None]
			# training, de-mean using batch mean
			else:
				with torch.no_grad():
					x_proj_mean = torch.mean(
						conv1d_out.detach(), axis=[0,2])
				conv1d_out = conv1d_out - x_proj_mean[None, :, None]

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
		
		out_z = rearrange(out_z, "b d l -> b l d")

		if demeaning:
			# inference only, use running average computed so far
			if running_means:
				out_z = out_z - running_means["out_proj"][None, None, :]
			# training, de-mean using batch mean
			else:
				with torch.no_grad():
					out_proj_mean = torch.mean(
						out_z.detach(), axis=[0,1])
				out_z = out_z - out_proj_mean[None, None, :]

		input_dict = {"conv1d": conv1d_inputs, "x_proj": x_proj_inputs,
			"out_proj": out_z.detach()}
		mean_dict = {"conv1d": conv1d_mean,
				"x_proj": x_proj_mean, "out_proj": out_proj_mean}

		ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
							  delta_proj_weight, out_proj_weight, conv1d_out, delta,
							  A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out,
							  conv1d_mean, x_proj_mean, out_proj_mean)

		return F.linear(out_z, out_proj_weight, out_proj_bias), \
			{"inputs": input_dict, "means": mean_dict}

	
	@staticmethod
	@custom_bwd
	def backward(ctx, dout, _):
		# dout: (batch, seqlen, dim)
		assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
		(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
		 conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out,
		 conv1d_mean, x_proj_mean, out_proj_mean) = ctx.saved_tensors
		
		L = xz.shape[-1]
		delta_rank = delta_proj_weight.shape[1]
		d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
		x, z = xz.chunk(2, dim=1)

		if ctx.demeaning:
			x = x - conv1d_mean[None, :, None]

		if dout.stride(-1) != 1:
			dout = dout.contiguous()
		if ctx.checkpoint_lvl == 1:
			conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
				x, conv1d_weight, conv1d_bias, None, None, None, True
			)
			delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
							  "d (b l) -> b d l", l = L)
			if dt_rms_weight is not None:
				delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
				delta = rms_norm_forward(delta, ctx.dt_rms_weight, None, ctx.b_c_dt_rms_eps)
				delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
			if b_rms_weight is not None:
				# Recompute & RMSNorm B
				B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
				B = rms_norm_forward(
					B, ctx.b_rms_weight, None, ctx.b_c_dt_rms_eps
				)
				B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
			if c_rms_weight is not None:
				# Recompute & RMSNorm C
				C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
				C = rms_norm_forward(
					C, ctx.c_rms_weight, None, ctx.b_c_dt_rms_eps
				)
				C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
		
		if ctx.demeaning:
			conv1d_out = conv1d_out - x_proj_mean[None, :, None]
			
		# The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
		# backward of selective_scan_cuda with the backward of chunk).
		dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
		dx, dz = dxz.chunk(2, dim=1)
		dout = rearrange(dout, "b l e -> e (b l)")
		dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
		dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
			conv1d_out, delta, A, B, C, D, z, delta_bias, dout_y, scan_intermediates, out, dz,
			ctx.delta_softplus,
			True  # option to recompute out_z
		)

		if ctx.demeaning:
			out_z = out_z - out_proj_mean[None, :, None]
			dconv1d_out = dconv1d_out - dconv1d_out.mean(dim=(0,2), keepdim=True)

		dout_proj_weight = torch.einsum("eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)"))
		dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
		dD = dD if D is not None else None
		dx_dbl = torch.empty_like(x_dbl)
		dB_proj_bias = None
		if ctx.is_variable_B:
			if not A.is_complex():
				dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
			else:
				dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
			dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
			dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
			dB = None
		dC_proj_bias = None
		if ctx.is_variable_C:
			if not A.is_complex():
				dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
			else:
				dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
			dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
			dx_dbl[:, -d_state:] = dC  # (bl d)
			dC = None
		ddelta = rearrange(ddelta, "b d l -> d (b l)")
		ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
		dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
		dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
		dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
		dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
		dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
		# The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
		# backward of conv1d with the backward of chunk).
		dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
			x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
		)

		if ctx.demeaning:
			dx = dx - dx.mean(dim=(0,2), keepdim=True)

		dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
		dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
		return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
				dout_proj_weight, dout_proj_bias,
				dA, dB, dC, dD,
				ddelta_bias if delta_bias is not None else None,
				# 6-None are delta_softplus, checkpoint_lvl, b_rms_weight, c_rms_weight, dt_rms_weight, b_c_dt_rms_eps
				# the other two are for demeaning (flag) and running_means (buffers)
				dB_proj_bias, dC_proj_bias, None, None, None, None, None, None, None, None)
	
def decorr_mamba_inner_fn(
	xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
	out_proj_weight, out_proj_bias,
	A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
	C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, b_rms_weight= None, c_rms_weight= None, dt_rms_weight= None, b_c_dt_rms_eps=1e-6,
	demeaning=False, running_means=None
):
	return DecorrMambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, 
						delta_proj_weight,
							out_proj_weight, out_proj_bias,
							A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, 
							delta_softplus, checkpoint_lvl, b_rms_weight, 
							c_rms_weight, dt_rms_weight, b_c_dt_rms_eps,
							demeaning, running_means)