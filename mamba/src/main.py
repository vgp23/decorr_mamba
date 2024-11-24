
from model.mamba import Mamba
from model.decorrelation import DecorrMamba, DecorrConv1d, apply_to_decorr, DecorrLoss
from utils.trainer import MambaTrainer
from torch.utils.data import DataLoader
from utils.helpers import MambaArgs, TrainingArgs, DefaultArgs, LanguageDatasetMaker
from einops import einsum
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import pickle
from copy import deepcopy

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def generate(model,
             tokenizer,
             prompt: str,
             n_tokens_to_gen: int = 50,
             sample: bool = True,
             top_k: int = 40):
    
    model.eval()
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to("mps")
    
    for token_n in range(n_tokens_to_gen):
        with torch.no_grad():
            indices_to_input = input_ids
            next_token_logits = model(indices_to_input)[:, -1]
        
        probs = F.softmax(next_token_logits, dim=-1)
        (batch, vocab_size) = probs.shape
        
        if top_k is not None:
            (values, indices) = torch.topk(probs, k=top_k)
            probs[probs < values[:, -1, None]] = 0
            probs = probs / probs.sum(axis=1, keepdims=True)
        
        if sample:
            next_indices = torch.multinomial(probs, num_samples=1)
        else:
            next_indices = torch.argmax(probs, dim=-1)[:, None]
        
        input_ids = torch.cat([input_ids, next_indices], dim=1)

    output_completions = [tokenizer.decode(output.tolist()) for output in input_ids][0]
    
    return output_completions


if __name__ == "__main__":

	torch.manual_seed(5)

	B = 10

	L = 64
	D = 512
	N = 32

	model_args = MambaArgs(N, D, n_layers=1)

	x = torch.randn(B,model_args.D_inner,L)
	# print(x.shape)
	# print(x[0,:,:])

	conv1d = nn.Conv1d(
		in_channels=model_args.D_inner,
		out_channels=model_args.D_inner,
		bias=True,
		kernel_size=model_args.conv_1d_size,
		groups=model_args.D_inner,
		padding=model_args.conv_1d_size - 1,
	) 


# ---------------------- forward pass code, fused --------------------

	mode = "channel_shared"
	fuse = False
	compute_grad = True
	compute_loss = True
	sample_frac = 0.1


	decorr_layer= nn.Parameter(
		torch.randn(model_args.conv_1d_size, model_args.conv_1d_size), requires_grad=False)

	b, d, l = x.shape
	# (B, n_patches, conv_1d_size*D). All data in each convolution patch
	# is represented in a single vector
	x_unfolded = F.unfold(
		x.unsqueeze(1), (d, model_args.conv_1d_size), 
		stride=1, padding=(0, model_args.conv_1d_size-1)).transpose(1,2)

	if mode == "channel_universal":

		# represent each patch as a matrix, with embedding channels grouped
		# in the last dimension

		#(B, n_patches, D, conv_1d_size)
		patch_matrices = x_unfolded.reshape(
			b, -1, d, model_args.conv_1d_size)

		if fuse: # fused decorrelation and convolution

			# pre-multiply each kernel by a copy of the decorrelation matrix, 
			# then perform channel-independent 1d convolution using the resulting kernels

			# (D_inner, conv_1d_size, conv_1d_size)
			decorr_repeat = decorr_layer.unsqueeze(0).repeat(
					model_args.D_inner, 1, 1)

			# (D_inner, conv_1d_size)
			decorr_kernels = einsum(torch.squeeze(conv1d.weight), decorr_repeat, 
				'd dummy, d dummy conv_1d_size -> d conv_1d_size')

			# apply fused transform
			y = einsum(decorr_kernels, patch_matrices,
				'd dummy, b n_patches d dummy -> b n_patches d')

		else: # unfused operation
			
			# perform decorrelation
			# (B, n_patches, conv_1d_size, D)
			decorrelated =  einsum(decorr_layer, patch_matrices,
				'conv_1d_size dummy, b n_patches d dummy -> b n_patches conv_1d_size d')

			# perform convolution on decorrelated inputs
			y = einsum(torch.squeeze(conv1d.weight), decorrelated, 
				'd dummy, b n_patches dummy d -> b n_patches d')


		if conv1d.bias is not None:
			y += conv1d.bias

		if compute_grad or compute_loss:
			# sample fraction of each batch from which to compute loss + update.
			# each channel counts as an independent input to potentially sample.
			b, n_patches, d, _ = patch_matrices.shape
			# collapse across the n_patches dimension, we want to sample 
			# individual embedding dimension channel information across
			# patches.	
			all_patch_channel_info = patch_matrices.reshape(b, n_patches*d, -1)
			print(all_patch_channel_info.shape)

			n_samples = int(sample_frac*n_patches*d)

			sample_idx = torch.multinomial(
				torch.ones(b, n_patches*d), n_samples, replacement=False)
			batch_idx = torch.arange(b).unsqueeze(1).expand(b, n_samples)

			selected = all_patch_channel_info[batch_idx, sample_idx]

			selected_decorr = selected @ decorr_layer.T

			grad, correlation_loss, whitening_loss = self.loss(
				selected_decorr, self.kappa, self.compute_grad, self.compute_loss)

			self.correlation_loss += correlation_loss
			self.whitening_loss += whitening_loss
			# updates happen on entire network at once in training loop
			self.grad = grad   				      			



		# print(y.shape)











	# print(x_unfolded[0,3,:5])




	# thing = torch.randn(B,mamba_args.D_inner,L)

   

	# decorr1 = DecorrConv1d(conv1d, mamba_args, "patch")
	# decorr1.train(False)
	# decorr1.compute_loss=False

	# out1 = decorr1(thing)


	










	