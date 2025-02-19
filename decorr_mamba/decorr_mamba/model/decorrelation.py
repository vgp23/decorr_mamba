import torch
import torch.nn as nn
import torch.nn.functional as F 
# from ..model.mamba import Mamba
from einops import einsum
from ..utils.helpers import MambaArgs
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from functools import partial
from copy import deepcopy

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

	def forward(self, x, kappa: float, device: str, compute_grad: bool, 
		compute_loss: bool, batched: bool):

		"""
		Computes the decorrelation gradient and/or losses for the given input tensor.

		Args:
			x (torch.Tensor): Input tensor of shape (n_samples, D) or 
				(n_samples, D, decorr_matrix_size), where:
				- n_samples: number of samples taken from the current batch
				- D: Feature dimension.
				- decorr_matrix_size: width/height of the decorrelation matrices
			kappa (float): Regularization parameter for blending correlation and whitening 
				losses. Must be between 0 and 1, where:
				- `kappa = 0` emphasizes decorrelation.
				- `kappa = 1` emphasizes whitening.
			compute_grad (bool): If `True`, computes the gradient for the 
				decorrelation matrix. 
			compute_loss (bool): If `True`, computes the decorrelation 
				losses (correlation and whitening).
			batched (bool): If 'True' computes losses and gradients
				for a batch of decorrelation matrices at once.
			device (str): model device

		Returns:
			Tuple[torch.Tensor or None, float or None, float or None]:
				- **grad** (torch.Tensor or None): Gradient tensor of shape (D, D). 
				  Returns `None` if `compute_grad` is `False`.
				- **corr_loss** (float or None): Correlation loss scalar. 
				  Returns `None` if `compute_loss` is `False`.
				- **whit_loss** (float or None): Whitening loss scalar. 
				  Returns `None` if `compute_loss` is `False`.

		Raises:
			AssertionError: If `kappa` is not specified or is not in the range [0, 1].
			AssertionError: If both `compute_grad` and `compute_loss` are `False` 
				(unexpected usage case).

		Notes:
			- The decorrelation gradient is calculated as a weighted combination of the 
			  covariance matrix (`C`) and the variance deviation matrix (`V`):
				`grad = mean((1-kappa)*C + kappa*V, dim=0)`
			- The losses are calculated as:
				- Correlation Loss: Mean of the squared covariances across batches.
				- Whitening Loss: Mean of the squared deviations of variances from identity.
			- The only use of "batched" mode is for computing decorrelation matrix updates
				for DecorrConv1d blocks with their decorrelation mode set to "channel_independent"

		"""	

		if not compute_grad and not compute_loss:
			# shouldn't ever be reached during normal use, but here for
			# completeness
			return None, None, None

		assert kappa is not None, "Specify kappa for loss and gradient computation"
		assert kappa <= 1.0 and kappa >= 0.0, "kappa must be between 0 and 1"



		# used for all modes where the decorrelation layer has only a single
		# matrix to train
		if not batched:
			assert len(x.shape) == 2, "Wrong input dimensionality for non-batched mode"
			# compute the individual loss elements
			n_samples, d = x.shape
			D = torch.diag_embed(x**2)
			V = D - torch.eye(d, device=device)

			xx_t = einsum(x, x, 
				'n_samples x, n_samples x_t -> n_samples x x_t')

			C = xx_t - D

			if compute_loss:
				# mean of squared covariances across matrix, then
				# across the batch
				corr_loss = \
					torch.mean(
						torch.mean(C**2, dim=(1,2)))

				# mean of squared variances across matrix, then
				# across the batch
				whit_loss = \
					torch.mean(
						torch.mean(V**2, dim=(1,2)))

			else:
				corr_loss = None 
				whit_loss = None

			# compute the actual gradient, if applicable
			if compute_grad:
				grad = torch.mean(((1-kappa)*C + kappa*V), dim=0)
			else:
				grad = None

			return grad, corr_loss, whit_loss

		# used where decorrelation layer has multiple matrices to train
		# (this is the case for "channel_universal")
		else:
			assert len(x.shape) == 3, "Wrong input dimensionality for batched mode"
			n_samples, d, decorr_matrix_size = x.shape
			# in this case we're updating d matrices, each with info
			# from one embedding dimension channel.

			# (D, all_samples, decorr_matrix_size)
			x = x.transpose(0, 1)

			# compute the individual loss elements
			D = torch.diag_embed(x**2)
			V = D - torch.eye(decorr_matrix_size, device=device)

			xx_t = einsum(x, x, 
				'd n_samples x, d n_samples x_t -> d n_samples x x_t')

			# (D, n_samples, decorr_matrix_size, decorr_matrix_size)
			C = xx_t - D

			if compute_loss:
				# mean of squared covariances, averaged across each matrix,
				# then across all samples, then across all parallel channels
				corr_loss = \
					torch.mean(
						torch.mean(
							torch.mean(C**2, dim=(2,3)), dim=1))

				# mean of squared variances, averaged across each matrix,
				# then across all samples, then across all parallel channels
				whit_loss = \
					torch.mean(
						torch.mean(
							torch.mean(V**2, dim=(2,3)), dim=1))

			else:
				corr_loss = None 
				whit_loss = None

			# compute the actual gradients, if applicable
			if compute_grad:
				grad = torch.mean(((1-kappa)*C + kappa*V), dim=1)
			else:
				grad = None

			return grad, corr_loss, whit_loss
			
	

class DecorrLinear(nn.Module):
	"""
	A modified linear layer that prefaces the original `nn.Linear` layer 
	with a trainable decorrelation matrix initialized at identity.

	This layer adds decorrelation functionality to the standard linear operation, 
	enabling the computation of decorrelation and whitening losses. Gradients for 
	the decorrelation matrix are computed inside the forward pass.

	Args:
		original_layer (nn.Module): The original `nn.Linear` layer to augment with 
			a decorrelation matrix.
		**kwargs: Additional arguments for configuring the decorrelation behavior.
			- `sample_frac` (float): Fraction of input samples to use for computing 
			  decorrelation loss and gradients. Must be between 0 and 1.
			- `kappa` (float): Hyperparameter used to compute decorrelation matrix gradients.

	Attributes:
		original_layer (nn.Module): The original linear layer being augmented.
		decorr_layer (nn.Parameter): Trainable decorrelation matrix initialized at identity.
		compute_grad (bool): Indicates whether decorrelation gradients are computed 
			during the forward pass.
		compute_loss (bool): Indicates whether decorrelation and whitening losses 
			are computed during the forward pass.
		sample_frac (float): Fraction of input samples to use for loss and gradient computation.
		kappa (float): Hyperparameter used to compute decorrelation matrix gradients.
		corr_loss (float): Accumulated correlation loss across batches.
		whit_loss (float): Accumulated whitening loss across batches.
		grad (torch.Tensor or None): Gradient of the decorrelation matrix, computed 
			during the forward pass.
		loss (DecorrLoss): Loss function instance for computing decorrelation and whitening losses.

	Methods:		
		reset():
			Resets losses and gradients to their initial state.
		
		train(mode: bool = True):
			Enables or disables training mode for the layer. Controls whether decorrelation 
			gradients are computed.

	"""
	def __init__(self, original_layer: nn.Module, fuse: bool, **kwargs):
		"""
		Initializes the DecorrLinear with a decorrelation matrix.

		Args:
			original_layer (nn.Module): The original linear layer to extend.				
			fuse (bool): Controls whether the decorrelation and main layer
				operations are fused into one transformation before being applied to the 
				input. More efficient. 	
			**kwargs: Additional arguments for decorrelation parameters, such as:
				- `kappa` (float): Decorrelation gradient hyperparameter.
				- `sample_frac` (float): Fraction of data sampled for decorrelation
					gradient calculation.
				- 'use_gain_scaling' (bool): Controls whether decorrelation update steps use  
					gain factor scaling as per Ahmad (2024)		
				- 'use_demeaning' (bool): Controls whether de-meaning is used prior to 
					decorrelation. Mean is computed iteratively as outlined in Ahmad (2024). 
				- 'demeaning_lr' (float): Learning rate for the exponential moving average
					algorithm used to compute the dataset mean, required for de-meaning.  				

		"""
		super(DecorrLinear, self).__init__()

		self.original_layer = original_layer
		# a layer of the same dimensions as the original, with no biases.
		# gradients are computed outside of the backward pass
		self.decorr_layer = nn.Parameter(
			torch.eye(original_layer.weight.shape[1]), requires_grad=False)

		self.fuse = fuse # used during the forward pass

		self.compute_grad = True # controlled by self.train()
		self.compute_loss = True 

		self.sample_frac = kwargs.get("sample_frac")
		self.kappa = kwargs.get("kappa")
		self.use_gain_scaling = kwargs.get("use_gain_scaling")
		self.use_demeaning = kwargs.get("use_demeaning")
		self.demeaning_lr = kwargs.get("demeaning_lr")

		if self.use_demeaning:
			self.register_buffer("running_mean", torch.zeros(original_layer.weight.shape[1]))	

		self.corr_loss = 0
		self.whit_loss = 0
		self.gain_factor = None

		self.loss = DecorrLoss()


	def forward(self, x):
		b, l, d = x.shape
		# idea with de-meaning: keep a running average of the training dataset for
		# use during inference, but de-mean during training using the batch mean

		if self.use_demeaning: # de-mean prior to decorrelation
			if self.training:
				assert self.demeaning_lr is not None, "Need a demeaning_lr for training"
				with torch.no_grad():
					# collapse across the batch and length dimensions, take
					# mean across the resulting dimension, and use this to de-mean
					# + update the mean estimate
					batch_mean = torch.mean(x.reshape((b*l, d)), axis=0)
					x -= batch_mean
					self.running_mean += self.demeaning_lr*batch_mean
			else:
				with torch.no_grad():
					x -= self.running_mean


		if self.fuse: # fused operation
			y = x @ (self.original_layer.weight @ self.decorr_layer).T

		else: # unfused operation
			decorr_out = x @ self.decorr_layer.T
			y = decorr_out @ (self.original_layer.weight).T

		if self.original_layer.bias is not None:
			y += self.original_layer.bias

		# the only time we're using the unfused operations is for debugging
		# purposes/sanity checks; loss and gradient computation assume 
		# the fused operation has been used, needing a sampled fraction of
		# training data to be passed through the decorrelation matrix again. 

		if self.compute_grad or self.compute_loss:

			# sample only a fraction of each batch
			assert self.sample_frac is not None , \
				"Specify sample_frac for loss and gradient computation"

			assert self.sample_frac > 0 and self.sample_frac <= 1.0, \
				"sample_frac must be between 0 and 1"   

			with torch.no_grad():
				# collapse across the batch and sequence length dimension, then sample
				# a pre-specified fraction from this
				x = x.reshape((b*l, d))
				n_samples = int(self.sample_frac*b*l)
				sample_idx = torch.multinomial(torch.ones(b*l), n_samples, replacement=False)

				selected = x[sample_idx]

				selected_decorr = selected @ self.decorr_layer.T

				grad, corr_loss, whit_loss = self.loss(
					selected_decorr, self.kappa, next(self.parameters()).device,
					compute_grad=self.compute_grad, compute_loss=self.compute_loss, batched=False)

				self.corr_loss = corr_loss
				self.whit_loss = whit_loss
				self.decorr_layer.grad = grad 

				if self.use_gain_scaling:
					# take expectations across all samples. Compute the gain vector according 
					# to equation in Appendix D of Ahmad et al. (2024)
					self.gain_vector = torch.sqrt(
						torch.mean((selected**2), axis=0) / \
						(torch.mean((selected_decorr**2), axis=0)) + 1e-08)

		return y

	def reset(self):
		self.corr_loss = 0
		self.whit_loss = 0
		self.decorr_layer.grad = None
		self.gain_factor = None

	def train(self, mode: bool = True):
		super(DecorrLinear, self).train(mode)
		# enable computing of decorrelation gradients during forward pass
		self.compute_grad = mode

		if mode:
			# allows loss to also be computed in eval mode      
			self.compute_loss = True 


class DecorrConv1d(DecorrLinear):
	"""
	Implements decorrelation for `nn.Conv1d` layers by prefacing them with a trainable 
	decorrelation matrix initialized at identity.

	The decorrelation matrix can operate in two modes:
	- "token": Each token's features are decorrelated independently.
	- "patch": Features across entire convolutional patches are decorrelated.

	Attributes:
		mode (str): Mode of operation for decorrelation (`"token"` or `"patch"`).
		decorr_layer (nn.Parameter): Trainable decorrelation matrix, initialized at identity.

	"""
	def __init__(self, original_layer: nn.Module, mode: str, 
		fuse: bool, **kwargs):
		"""
		Initializes the DecorrConv1d layer with a decorrelation matrix.

		Args:
			original_layer (nn.Module): The original convolutional layer to extend.	
			mode (str): Decorrelation mode. 
				- `"token"`: Decorrelate features of each token independently.
				- `"patch"`: Decorrelate features across convolutional patches.
				- `"channel_shared"`: Decorrelate features across each embedding channel
						within every convolutional patch, using the same decorrelation
						matrix across all channels.	
				- `"channel_independent"`: Decorrelate features across each embedding channel
						within every convolutional patch, using a different decorrelation
						matrix for each channel.							
			fuse (bool): Controls whether the decorrelation and main layer
				operations are fused into one transformation before being applied to the 
				input. More efficient. 								
			**kwargs: Additional arguments for decorrelation parameters, such as:
				- `kappa` (float): Decorrelation gradient hyperparameter.
				- `sample_frac` (float): Fraction of data sampled for decorrelation
					gradient calculation.
				- 'use_gain_scaling' (bool): Controls whether decorrelation update steps use  
					gain factor scaling as per Ahmad (2024)	
				- 'use_demeaning' (bool): Controls whether de-meaning is used prior to 
					decorrelation. Mean is computed iteratively as outlined in Ahmad (2024). 
		Raises:
			AssertionError: If `mode` is not `"token"`, `"patch"`, `"channel_independent"`
			or `"channel_shared"`.
		"""

		super(DecorrConv1d, self).__init__(
			original_layer=original_layer, fuse=fuse, **kwargs)

		self.mode = mode

		# determines how the decorrelation is applied.
		assert mode == "token" or mode == "patch" \
			or mode == "channel_shared" or mode == "channel_independent", \
			"conv_1d_mode must be \"token\", \"patch\", \"channel_shared\", or \"channel_independent\""

		if mode == "token":
			# decorrelate each token's features independently
			self.decorr_layer = nn.Parameter(torch.eye(
				self.original_layer.in_channels), requires_grad=False)
			if self.use_demeaning:
				self.register_buffer("running_mean", torch.zeros(
					self.original_layer.in_channels
				))					

		elif mode == "patch":
			# decorrelate all input features within each convolutional patch
			self.decorr_layer = nn.Parameter(
				torch.eye(self.original_layer.in_channels*self.original_layer.kernel_size[0]), 
				requires_grad=False)
			
			if self.use_demeaning:
				self.register_buffer("running_mean", 
					torch.zeros(self.original_layer.in_channels*self.original_layer.kernel_size[0]))					

		elif mode == "channel_shared":
			# decorrelate each patch channel's input features independently,
			# using the same decorrelation matrix for all channels
			self.decorr_layer = nn.Parameter(
				torch.eye(self.original_layer.kernel_size[0]), requires_grad=False)
			if self.use_demeaning:
				self.register_buffer("running_mean", 
						 torch.zeros(self.original_layer.kernel_size[0]))				

		elif mode == "channel_independent":
			# decorrelate each patch channel's input features independently,
			# using a separate decorrelation matrix for all channels
			all_matrices = torch.eye(self.original_layer.kernel_size[0]).unsqueeze(0).repeat(
					self.original_layer.in_channels, 1, 1)

			self.decorr_layer = nn.Parameter(all_matrices, requires_grad=False)
			if self.use_demeaning:
				self.register_buffer("running_mean", 
					torch.zeros(
						(self.original_layer.in_channels, self.original_layer.kernel_size[0])))	
		else:
			raise NotImplementedError						

	def forward(self, x):

		if self.compute_grad or self.compute_loss:
			assert self.sample_frac is not None , \
				"Specify sample_frac for loss and gradient computation"

			assert self.sample_frac > 0 and self.sample_frac <= 1.0, \
				"sample_frac must be between 0 and 1"   

		# NB: although the option to use unfused operations exists for all modes,
		# training assumes that the operations are fused and that the decorrelated
		# inputs are not directly available, needing another forward pass through
		# just the decorrelation matrices.

		b, d, l = x.shape

		# all data in each convolution patch is represented in a single vector
		# (B, n_patches, conv_1d_size*D)
		x_unfolded = F.unfold(
			x.unsqueeze(1), (d, self.original_layer.kernel_size[0]), 
			stride=1, padding=(0, self.original_layer.kernel_size[0]-1)).transpose(1,2)

		# NB for rest of implementation: Conv1d in Mamba defines a separate kernel for
		# each embedding dimension (n_groups = D_inner). This avoids mixing 
		# information across channels, which is important as SSM components operate on 
		# each embedding dimension independently. 

		# decorrelates each token's features across the embedding dimension
		if self.mode == "token":

			if self.fuse: # fused decorrelation and convolution
				# represent each patch as a matrix, with embedding channels grouped
				# in the last dimension				
				# (B, n_patches, D, conv_1d_size)
				patch_matrices = x_unfolded.reshape(
					b, -1, d, self.original_layer.kernel_size[0])

				# the unfused operation passes each token embedding through
				# the same decorrelation matrix, then performs convolution via
				# row-wise dot products of the kernels with the decorrelated 
				# patch matrix. We fuse these by defining a separate kernel-scaled version 
				# of the decorrelation matrix for each token in the patch, passing the 
				# tokens through their respective matrices, and then summing over the 
				# token dimension

				# (conv_1d_size, D, D)
				token_specific_decorr = self.decorr_layer.unsqueeze(0).repeat(
					self.original_layer.kernel_size[0], 1, 1)

				# d1 and d2 refer to the dimensions of each decorrelation matrix;
				# the kernel scaling operations make them lose their symmetry.

				# (conv_1d_size, D, D)
				token_specific_decorr = einsum(token_specific_decorr, 
					torch.squeeze(self.original_layer.weight),
					'conv_1d_size d1 d2, d1 conv_1d_size' +
					' -> conv_1d_size d1 d2')

				# pass inputs through token-specific decorrelation matrices.

				# (B, n_patches, conv_1d_size, D)
				decorr_conv_outputs = einsum(token_specific_decorr, patch_matrices, 
					'conv_1d_size d1 dummy, b n_patches dummy conv_1d_size' + 
					' -> b n_patches conv_1d_size d1')

				# summing over the decorrelated + scaled tokens
				# in each patch completes the convolutional operation

				# (B, n_patches, D)
				y = torch.sum(decorr_conv_outputs, dim=2)
			
			else: # unfused operation

				# avoid redundant decorrelation operations by operating on original input 
				# (B, L, D)
				decorr_inputs = x.transpose(1,2) @ self.decorr_layer.T

				# unfold and perform convolution as normal

				# (B, n_patches, conv_1d_size*D)
				x_decorr_unfolded = F.unfold(
					decorr_inputs.transpose(1,2).unsqueeze(1), (d, self.original_layer.kernel_size[0]), 
					stride=1, padding=(0, self.original_layer.kernel_size[0]-1)).transpose(1,2)

				# (B, n_patches, D, conv_1d_size)
				decorr_patch_matrices = x_decorr_unfolded.reshape(
					b, -1, d, self.original_layer.kernel_size[0])			

				y = einsum(torch.squeeze(self.original_layer.weight), decorr_patch_matrices,
					'd dummy, b n_patches d dummy -> b n_patches d')

			# forward pass for gradients & losses  
			if self.compute_grad or self.compute_loss:     
				with torch.no_grad():
					# sample fraction of each batch from which to compute loss + update
					x_t = x.transpose(1,2)

					n_samples = int(self.sample_frac*l)

					sample_idx = torch.multinomial(
						torch.ones(b, l), n_samples, replacement=False)
					batch_idx = torch.arange(b).unsqueeze(1).expand(b, n_samples)

					selected = x_t[batch_idx, sample_idx]

					selected_decorr = selected @ self.decorr_layer.T

					grad, corr_loss, whit_loss = self.loss(
						selected_decorr, self.kappa, next(self.parameters()).device,
						self.compute_grad, self.compute_loss, batched=False)

					self.corr_loss = corr_loss
					self.whit_loss = whit_loss
					# updates happen on entire network at once in training loop
					self.decorr_layer.grad = grad   				

					if self.use_gain_scaling:
						# collapse across batch and length dimensions, then
						# take expectations across the newly created dimension. Compute
						# the gain vector according to equation in Appendix D of Ahmad et al. (2024)
						self.gain_vector = torch.sqrt(
							torch.mean((selected**2).reshape((-1, d)), axis=0) / \
							(torch.mean((selected_decorr**2).reshape((-1, d)), axis=0)) + 1e-08)

		# decorrelates all features in each input patch to the convolution
		elif self.mode == "patch":

			if self.fuse: # fused operation
				# split up decorrelation matrix to pre-multiply with each channel's kernel

				# (D, conv_1d_size, D*conv_1d_size (= n_patch_features))
				decorr_split = self.decorr_layer.reshape(
					d, self.original_layer.kernel_size[0], d*self.original_layer.kernel_size[0])

				fused_matrix = einsum(
					decorr_split, torch.squeeze(self.original_layer.weight),
					'd dummy n_patch_features, d dummy -> d n_patch_features')

				y = x_unfolded @ fused_matrix.T

			else: # unfused operation

				# (B, n_patches, D*conv_1d_size)
				decorr = x_unfolded @ self.decorr_layer.T

				# fold patch vectors back up into matrices, then multiply
				# with convolutional kernels

				# (B, n_patches, D, conv_1d_size)
				decorr_patches = decorr.reshape(b,-1, d, self.original_layer.kernel_size[0])

				y = einsum(torch.squeeze(self.original_layer.weight), decorr_patches,
					"d dummy, b n_patches d dummy -> b n_patches d")

			# forward pass for gradients & losses
			if self.compute_grad or self.compute_loss:       
				with torch.no_grad():
					# sample fraction of each batch from which to compute loss + update
					b, n_patches, n_patch_features = x_unfolded.shape

					n_samples = int(self.sample_frac*n_patches)

					sample_idx = torch.multinomial(
						torch.ones(b, n_patches), n_samples, replacement=False)
					batch_idx = torch.arange(b).unsqueeze(1).expand(b, n_samples)

					selected = x_unfolded[batch_idx, sample_idx]

					selected_decorr = selected @ self.decorr_layer.T

					grad, corr_loss, whit_loss = self.loss(
						selected_decorr, self.kappa, next(self.parameters()).device,
						self.compute_grad, self.compute_loss, batched=False)

					self.corr_loss = corr_loss
					self.whit_loss = whit_loss
					# updates happen on entire network at once in training loop
					self.decorr_layer.grad = grad  

					if self.use_gain_scaling:
						# collapse across batch and patch dimensions, then
						# take expectations across the newly created dimension. Compute
						# the gain vector according to equation in Appendix D of Ahmad et al. (2024)
						self.gain_vector = torch.sqrt(
							torch.mean((selected**2).reshape((-1, n_patch_features)), axis=0) / \
							(torch.mean((selected_decorr**2).reshape((-1, n_patch_features)), axis=0)) + 1e-08)

		# decorrelates each patch channel's input features independently,
		# using the same decorrelation matrix for all channels					
		elif self.mode == "channel_shared":   
			# represent each patch as a matrix, with embedding channels grouped
			# in the last dimension

			#(B, n_patches, D, conv_1d_size)
			patch_matrices = x_unfolded.reshape(
				b, -1, d, self.original_layer.kernel_size[0])

			if self.fuse: # fused decorrelation and convolution

				# pre-multiply each kernel by a copy of the decorrelation matrix, 
				# then perform channel-independent 1d convolution using the resulting kernels

				# (D_inner, conv_1d_size, conv_1d_size)
				decorr_repeat = self.decorr_layer.unsqueeze(0).repeat(
						self.original_layer.in_channels, 1, 1)

				# (D_inner, conv_1d_size)
				decorr_kernels = einsum(
					torch.squeeze(self.original_layer.weight), decorr_repeat, 
					'd dummy, d dummy conv_1d_size -> d conv_1d_size')

				# apply fused transform
				y = einsum(decorr_kernels, patch_matrices,
					'd dummy, b n_patches d dummy -> b n_patches d')

			else: # unfused operation
				
				# perform decorrelation
				# (B, n_patches, conv_1d_size, D)
				decorrelated =  einsum(self.decorr_layer, patch_matrices,
					'conv_1d_size dummy, b n_patches d dummy -> b n_patches conv_1d_size d')

				# perform convolution on decorrelated inputs
				y = einsum(torch.squeeze(self.original_layer.weight), decorrelated, 
					'd dummy, b n_patches dummy d -> b n_patches d')

			# forward pass for gradients & losses
			if self.compute_grad or self.compute_loss:
				with torch.no_grad():				
					# sample fraction of each batch from which to compute loss + update.
					# each channel counts as an independent input to potentially sample.
					b, n_patches, d, _ = patch_matrices.shape
					# collapse across the n_patches dimension, we want to sample 
					# individual embedding dimension channel information across
					# patches.	
					all_patch_channel_info = patch_matrices.reshape(b, n_patches*d, -1)

					n_samples = int(self.sample_frac*n_patches*d)

					sample_idx = torch.multinomial(
						torch.ones(b, n_patches*d), n_samples, replacement=False)
					batch_idx = torch.arange(b).unsqueeze(1).expand(b, n_samples)

					selected = all_patch_channel_info[batch_idx, sample_idx]

					selected_decorr = selected @ self.decorr_layer.T

					grad, corr_loss, whit_loss = self.loss(
						selected_decorr, self.kappa, next(self.parameters()).device,
						self.compute_grad, self.compute_loss, batched=False)

					self.corr_loss = corr_loss
					self.whit_loss = whit_loss
					# updates happen on entire network at once in training loop
					self.decorr_layer.grad = grad

					if self.use_gain_scaling:
						# collapse across batch and embedding*n_patches dimensions, then
						# take expectations across the newly created dimension. Compute
						# the gain vector according to equation in Appendix D of Ahmad et al. (2024)
						self.gain_vector = torch.sqrt(
							torch.mean(
								(selected**2).reshape((-1, self.original_layer.kernel_size[0])), axis=0) / \
							(torch.mean(
								(selected_decorr**2).reshape((-1, self.original_layer.kernel_size[0])), axis=0)) + 1e-08)

		# decorrelates each patch channel's input features independently,
		# using a separate decorrelation matrix for all channels	
		elif self.mode == "channel_independent":
			# represent each patch as a matrix

			#(B, n_patches, D, conv_1d_size)
			patch_matrices = x_unfolded.reshape(
				b, -1, d, self.original_layer.kernel_size[0])

			if self.use_demeaning: # de-mean prior to decorrelation
				if self.training:
					assert self.demeaning_lr is not None, "Need a demeaning_lr for training"
					with torch.no_grad():
						# collapse across the batch and n_patches, take
						# mean across the resulting dimension, and use this to de-mean
						# + update the mean estimate
						batch_mean = torch.mean(x.reshape((-1, d, self.original_layer.kernel_size[0])), 
							axis=0)
						patch_matrices -= batch_mean.unsqueeze(0).unsqueeze(0)
						self.running_mean += self.demeaning_lr*batch_mean
				else:
					with torch.no_grad():
						patch_matrices -= self.running_mean.unsqueeze(0).unsqueeze(0)			

			if self.fuse: # fused decorrelation and convolution

				# pre-multiply each individual decorrelation matrix (one for each
				# channel) by its corresponding kernel, row-wise (each matrix row
				# gets multiplied by the corresponding scalar within the kernel)

				# "target" dimension just indicates where the products are happening,
				# it's inappropriate to call it "dummy" since it's not getting summed over

				# (D, conv_1d_size, conv_1d_size)
				decorr_kernels = einsum(
					torch.squeeze(self.original_layer.weight), self.decorr_layer, 
					'd target, d target conv_1d_size -> d target conv_1d_size')

				# pass each patch matrix embedding dimension vector through its
				# corresponding scaled decorrelation matrix

				# (B, n_patches, D, conv_1d_size)
				decorr_conv_outputs = einsum(decorr_kernels, patch_matrices,
					'd conv_1d_size dummy, b n_patches d dummy -> b n_patches d conv_1d_size')

				# summing across last dimension completes the convolutional operation
				y = torch.sum(decorr_conv_outputs, dim=3)

			else: # unfused operation
				
				# perform decorrelation
				# (B, n_patches, conv_1d_size, D)
				decorrelated =  einsum(self.decorr_layer, patch_matrices,
					'd conv_1d_size dummy, b n_patches d dummy -> b n_patches conv_1d_size d')

				# perform convolution on decorrelated inputs
				y = einsum(torch.squeeze(self.original_layer.weight), decorrelated, 
					'd dummy, b n_patches dummy d -> b n_patches d')

			if self.compute_grad or self.compute_loss:
				with torch.no_grad():				
					# sample fraction of the batch from which to compute loss + update.
					# first collapse across batch and n_patches dimension
					patch_matrices = patch_matrices.reshape(-1, d, self.original_layer.kernel_size[0])

					n_samples = int(self.sample_frac*patch_matrices.shape[0])

					sample_idx = torch.multinomial(torch.ones(patch_matrices.shape[0]), 
						n_samples, replacement=False)

					selected = patch_matrices[sample_idx]

					selected_decorr = einsum(self.decorr_layer, selected,
						'd conv_1d_size dummy, n_samples d dummy -> n_samples d conv_1d_size')

					grad, corr_loss, whit_loss = self.loss(
						selected_decorr, self.kappa, next(self.parameters()).device,
						compute_grad=self.compute_grad, compute_loss=self.compute_loss,
						batched=True)

					self.corr_loss = corr_loss
					self.whit_loss = whit_loss
					# updates happen on entire network at once in training loop
					self.decorr_layer.grad = grad 

					if self.use_gain_scaling:
						# take expectations across the sample dimension. Compute
						# the gain vector according to equation in Appendix D of Ahmad et al. (2024)

						# a special case; we're computing a gain_vector for each of the
						# D decorrelation matrices in parallel. We rearrange with the D dimension
						# first, just to be explicit about this.
						selected = selected.transpose(0,1)
						selected_decorr = selected_decorr.transpose(0,1)

						self.gain_vector = torch.sqrt(
							torch.mean(selected**2, axis=1) / (torch.mean(selected_decorr**2, axis=1)) + 1e-08)
		else:
			raise NotImplementedError
		
		if self.original_layer.bias is not None:
			y += self.original_layer.bias

		return y.transpose(1,2)


def apply_to_decorr(model, f):
	''' 
	Recursively traverses a model's structure and applies an arbitrary
	function to the modules containing decorrelation layers. Used
	for printing things out, performing simple operations, etc.
	
	Args:
		f (Callable[[DecorrLinear], None])
	
	'''
	def _apply_to_decorr(module):
		for child in module.children():
			if isinstance(child, DecorrLinear):
				f(child)

	model.apply(_apply_to_decorr)


class TrackedParameter(nn.Parameter):
	"""A wrapper around nn.Parameter that logs inputs during matrix multiplication."""
	
	def __new__(cls, data, requires_grad=True):
		instance = super(TrackedParameter, cls).__new__(cls, data)
		instance.requires_grad = requires_grad
		instance.inputs = []
		return instance
		

	def __matmul__(self, other):
		"""Intercepts matrix multiplication to log the input."""
		self.inputs.append(other.clone().detach())
		return super().__matmul__(self, other)

	def __rmatmul__(self, other):
		"""Tracks all tensors that multiplied self: x @ W"""
		self.inputs.append(other.clone().detach()) 
		return super().__rmatmul__(other)	

	def transpose(self, dim0, dim1):
		""" Ensures transposition is also tracked """
		# Make a new parameter, morifying the same inputs list as the original
		transposed_parameter = TrackedParameter(super().transpose(dim0, dim1))
		transposed_parameter.inputs = self.inputs
		return transposed_parameter

	@property
	def T(self):
		""" Handles W.T so it retains tracking """
		return self.transpose(0, 1)


class DecorrLinearNew(nn.Linear):
	def __init__(self, original_layer: nn.Linear = None, 
			sample_frac: float = 0.1, kappa: float = 0.5, **factory_kwargs):

		# allows layer to be created on top of existing one, or made from
		# scratch
		if original_layer is not None:
			self.__dict__.update(original_layer.__dict__)
		else:
			super(DecorrLinearNew, self).__init__(**factory_kwargs)

		self.decorr_layer = nn.Parameter(
			torch.eye(self.weight.shape[1]), requires_grad=False)

		self.register_parameter("original_weight", nn.Parameter(self.weight))
		# delete self.weight, because we'll be overwriting it later
		del self._parameters["weight"]

		# fuse decorrelation and original weights into a single
		# weight matrix, then track its input to allow decorrelation
		# loss to be computed later. requires_grad is false, because we only want
		# to update original_weight with regular backprop. 

		# this tracks multiplications of form W @ X, W.T @ X, X @ W and X @ W.T
		fused_weight = TrackedParameter(
			self.original_weight @ self.decorr_layer, requires_grad=False)
		self.register_parameter("weight", fused_weight)

		# this tracks inputs during the standard forward pass
		self.register_forward_hook(self.track_weight_input)


		# decorrelation training parameters
		self.sample_frac = sample_frac
		self.kappa = kappa
		self.corr_loss = 0
		self.whit_loss = 0

	def track_weight_input(self, module, input, output):
		self.weight.inputs = input[0].clone().detach()

	def reset(self):
		pass

class DecorrMamba(MambaLMHeadModel):
	"""
	Mamba architecture with built-in decorrelation layers and additional functionality
	for training and monitoring them.

	This class extends the `Mamba` architecture to add decorrelation layers, which 
	are initialized and managed dynamically. It supports the following functionalities:
	- Creation of decorrelation matrices in specific layers (e.g., `in_proj`, `out_proj`).
	- Loss monitoring for decorrelation layers (e.g., correlation loss, whitening loss).
	- Dynamic updates to decorrelation layers during training.

	Attributes:
		mean_corr_loss (float): The mean correlation loss across all decorrelation layers.
		mean_whit_loss (float): The mean whitening loss across all decorrelation layers.
		decorr_lr (float): Learning rate for updating decorrelation matrices.
		conv_1d_mode (str): The mode used for 1D convolution decorrelation layers. "patch"
			decorrelates all features seen by a convolutional kernel, "token" decorrelates
			all token features independently. 
		sample_frac (float): Fraction of the data sampled for decorrelation update calculations
		kappa (float): Hyperparameter controlling contribution of whitening and decorrelation
			loss terms in the gradient computation, for the decorrelation amtrices

	Methods:
		mean_decorr_losses():
			Calculates the mean correlation and whitening losses across all decorrelation layers.

		update_decorr_matrices():
			Updates decorrelation matrices for all decorrelation layers using their respective gradients.

		reset_decorr():
			Resets the gradients and losses of decorrelation layers, and the model mean losses.

		compute_decorr_losses(mode: bool=True):
			Enables or disables the computation of decorrelation losses during the forward pass.
	"""

	def __init__(self, model_args: MambaConfig = None, 
		existing_model: MambaLMHeadModel = None, fuse: bool = True, use_gain_scaling: bool = True,
		use_demeaning: bool=True, conv_1d_mode: str = "channel_independent", **kwargs):

		"""
		Initializes the DecorrMamba model.

		Either creates a new Mamba model with decorrelation layers or extends an 
		existing Mamba model with decorrelation functionality. In the latter case,
		decorrelation matrices are initialized at identity.

		Args:
			conv_1d_mode (str): The mode used for decorrelation in Conv1d layers.
			model_args (MambaConfig, optional): Arguments to configure a new Mamba model.
			existing_model (MambaLMHeadModel, optional): Pre-existing Mamba model to modify.
			fuse (bool, default=True): Controls whether the decorrelation and main layer
				operations are fused into one transformation before being applied to the 
				input. More efficient. 	
			use_gain_scaling (bool, default=True): Controls whether decorrelation
				update steps use gain factor scaling as per Ahmad (2024)	
			use_demeaning (bool, default=True): Controls whether data is de-meaned
				prior to decorrelation as per Ahmad (2024)					
			**kwargs: Additional keyword arguments for decorrelation parameters:
				- kappa (float): Hyperparameter for decorrelation gradient.
				- sample_frac (float): Fraction of the data sampled for decorrelation 
					matrix gradient calculations.
				- decorr_lr (float): Learning rate for decorrelation updates.	
				- demeaning_lr (float): Learning rate for EMA estimation of data means,
					required for demeaning					

		Raises:
			AssertionError: If neither `model_args` nor `existing_model` is provided.
		"""

		assert existing_model is not None or model_args is not None, \
			"Specify either a MambaArgs object to create a new model," +\
			" or a pre-made Mamba model to modify"


		if existing_model is not None:
			self.__dict__.update(deepcopy(existing_model).__dict__)
			if model_args is not None:
				print(
					"Warning: supplied args overwritten by the args of the existing model")
		else:
			super(DecorrMamba, self).__init__(model_args)

		self.n_decorr_layers = 0 # used for averaging the decorr losses later

		def _create_decorr_matrices(module, kappa, sample_frac):
			''' 
			Used to recursively traverse model and create decorrelation matrices 
			in pre-defined places
			'''
			for name, child in module.named_children():
				if name == "in_proj" or name == "out_proj" or name == "x_proj":
					self.n_decorr_layers += 1
					setattr(module, name, DecorrLinearNew(
						child, kappa=kappa, sample_frac=sample_frac))

				# if name == "conv1d":
				# 	self.n_decorr_layers += 1 				
				# 	setattr(module, name, DecorrConv1d(
				# 		original_layer=child, use_gain_scaling=use_gain_scaling,
				# 		use_demeaning=use_demeaning, mode=conv_1d_mode, fuse=fuse, kappa=kappa, 
				# 		sample_frac=sample_frac, demeaning_lr=kwargs.get("demeaning_lr")))


		self.apply(partial(_create_decorr_matrices, 
			kappa=kwargs.get("kappa"), sample_frac=kwargs.get("sample_frac")))

		# remove weight decay for decorrelation layers
		apply_to_decorr(self, 
			lambda decorr_module: setattr(
				getattr(decorr_module, "decorr_layer"), "_no_weight_decay", True))

		self.mean_corr_loss = 0
		self.mean_whit_loss = 0
		self.decorr_lr = kwargs.get("decorr_lr")
		
		self.conv_1d_mode = conv_1d_mode

	def mean_decorr_losses(self):
		''' 
		Calculates the mean correlation and whitening losses across all 
		layers implementing decorrelation, for a Mamba model
		'''

		def _sum_losses(module):

			for child in module.children():
				# DecorrConv1d extends DecorrLinear, should account for 
				# convolutional layers too
				if isinstance(child, DecorrLinear):
					self.mean_corr_loss += child.corr_loss
					self.mean_whit_loss += child.whit_loss

		self.apply(_sum_losses)

		self.mean_corr_loss /= self.n_decorr_layers
		self.mean_whit_loss /= self.n_decorr_layers

	def update_decorr_matrices(self):
		''' 
		Updates the decorrelation matrices for all decorrelation layers
		within the Mamba model
		'''
		assert self.training, "Model must be in training mode"
		assert self.decorr_lr is not None, "No decorr_lr specified"

		# gradients should already have been computed during the forward pass
		def _update_decorr_matrices(module):
			for child in module.children():
				if isinstance(child, DecorrLinear):

					assert child.decorr_layer.grad is not None, "Gradient not computed"

					unscaled_update = \
						child.decorr_layer.data - \
						self.decorr_lr * child.decorr_layer.grad @ child.decorr_layer.data

					# gain scaling as per Ahmad (2024)
					if child.use_gain_scaling:

						# all cases are handled the same way except for conv1d layers 
						# with "channel_independent" mode, as there are D independent
						# decorrelation matrices to update there. Handle this first.
						if isinstance(child, DecorrConv1d) and \
							self.conv_1d_mode=="channel_independent":

							scaled_update = einsum(child.gain_vector, unscaled_update,
								'd out_dim, d out_dim in_dim -> d out_dim in_dim')

						# all other cases are handled the following way
						else:
							scaled_update = einsum(child.gain_vector, unscaled_update,
								'out_dim, out_dim in_dim -> out_dim in_dim')

						child.decorr_layer.data = scaled_update

					# update without gain scaling
					else:
						child.decorr_layer.data = unscaled_update

		self.apply(_update_decorr_matrices)

	def reset_decorr(self):
		''' 
		Resets gradients and losses of decorrelation layers after parameter
		updates. Also resets the mean losses computed across all decorrelation
		layers.
		'''
		apply_to_decorr(self, lambda x: x.reset())
		self.mean_corr_loss = 0
		self.mean_whit_loss = 0

	def compute_decorr_losses(self, mode: bool=True):
		"""
		Enables or disables the computation of decorrelation losses during the 
		forward pass. Useful for switching between development and pure inference modes.

		Args:
			mode (bool, default=True): If True, computes decorrelation losses; 
									   if False, skips loss computation.
		"""
		apply_to_decorr(self, lambda x: setattr(x, "compute_loss", mode))

			





	
