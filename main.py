from decorr_mamba.model.decorrelation import DecorrLoss
import torch

if __name__ == "__main__":
	B = 10
	L = 124
	D = 512
	kappa = 0.5

	loss = DecorrLoss("mps")
	x = torch.randn((B,L,D)).to('mps')

	t = loss(x, kappa=0.5)