import torch
import torch.nn as nn
import torch.nn.functional as F 
from model.mamba import Mamba

class Decorrelator():
	def __init__(self):
		... 

	@staticmethod
	def add_decorrelation_layers(model: Mamba):
		''' Modifies standard Mamba architecture to include input-decorrelating
			layers in the correct places. Modifies the forward pass of the model
			accordingly.
		'''

		# we need decorrelation layers before the following places:
		
		# input upscale projections
		# output downscale projections
		# conv1d kernel
		# parameter upscale projections (B, C and delta)

		# THE ASSUMPTION: same decorrelation matrix for all token
		# positions in the sequence, I guess?

		...
