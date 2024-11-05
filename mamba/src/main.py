from model.decorrelator import Decorrelator, DecorrelationGradient
from model.mamba import Mamba
from utils.helpers import MambaArgs
import torch
import torch.nn as nn
import torch.nn.functional as F


if __name__ == "__main__":

	L = 32
	B = 8
	D = 16
	N = 8

	device = 'cpu'
	mamba_args = MambaArgs(N, D, n_layers=3)
	model = Mamba(mamba_args).to(mamba_args.device)
	dummy_data = torch.randint(0, 200, (B,L))

	new_model = Decorrelator.add_decorrelation_layers(model)

	out, grads = new_model(dummy_data)

	for i, layer in enumerate(grads):
		print(f"Layer {i+1}")
		for decorr in layer:
			print(decorr.shape)
	# test = torch.randn((B,L,D))
	# weights = torch.randn((D,D))

	# print(torch.matmul(test, weights)[0,0])
	# print(weights.T @ test[0,0])

	# print(new_model.layers[0].block.in_proj_decorr)

	# test = torch.randn((B,L,D))
	# grad = DecorrelationGradient()
	# sample_indices = torch.randint(low=0, high=B*L, size=(55,))

	# test_grad = grad(test, sample_indices)
	# print(test_grad.shape)
	# a = torch.randn(2, 2, 3)
	# print(a)

	# print(torch.diag_embed(a))


	