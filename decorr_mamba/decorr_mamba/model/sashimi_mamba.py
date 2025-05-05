import torch
import torch.nn as nn
import torch.nn.functional as F


class DownPool(nn.Module):
	""" SaShiMi down-pooling block """
	def __init__(self, H, q, p):
		super().__init__()
		self.H = H
		self.q = q
		self.p = p
		self.down_pool = nn.Linear(p*H, q*H)

	def forward(self, x):
		B, L, _ = x.shape
		reshaped = x.reshape(B, -1, self.H*self.p)
		down_sampled = self.down_pool(reshaped)
		return down_sampled 

class UpPool(nn.Module):
	""" SaShiMi up-pooling block """
	def __init__(self, H, q, p):
		super().__init__()
		self.H = H
		self.q = q
		self.p = p
		self.up_pool = nn.Linear(H, int(p*H/q))

	def forward(self, x):
		B, _, _ = x.shape
		up_sampled = self.up_pool(x)
		reshaped = up_sampled.reshape(B, -1, int(self.H/self.q))
		return reshaped 


class MambaSaShiMi(nn.Module):
	""" A simple implementation of SaShiMi with a Mamba backbone instead of
	the default S4 blocks"""
	def __init__(self, d_outer, p, q, vocab_size, blocks_per_stage=5, n_pool_stages=2):
		super().__init__()

		self.p = p
		self.q = q
		self.d_outer = d_outer
		self.vocab_size = vocab_size

		self.embedding = nn.Embedding(self.vocab_size, self.d_outer)


if __name__ == "__main__":
    B = 8


    d_outer = 64
    p = 16
    q = 2


    # t = nn.Embedding(256, d_outer)

    # outs = t(ins.long())
    ins = torch.randn((B, 1024, d_outer))

    dp = DownPool(d_outer, q, p)
    up = UpPool(d_outer*q, q, p)

    print(ins.shape)
    down = dp(ins)
    print(down.shape)

    ups = up(down)
    print(ups.shape)