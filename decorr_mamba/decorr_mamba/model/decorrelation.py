import torch
import torch.nn as nn
import torch.nn.functional as F 
from ..model.mamba import Mamba
from einops import einsum
from ..utils.helpers import MambaArgs
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

	def forward(self, x, kappa: float, model_args: MambaArgs, compute_grad: bool = True, 
		compute_loss: bool = True, batched: bool = False):

		"""
		Computes the decorrelation gradient and/or losses for the given input tensor.

		Args:
			x (torch.Tensor): Input tensor of shape (B, L, D), where:
				- B: Batch size.
				- L: Sequence length.
				- D: Feature dimension.
			kappa (float): Regularization parameter for blending correlation and whitening 
				losses. Must be between 0 and 1, where:
				- `kappa = 0` emphasizes decorrelation.
				- `kappa = 1` emphasizes whitening.
			compute_grad (bool, optional): If `True`, computes the gradient for the 
				decorrelation matrix. Defaults to `True`.
			compute_loss (bool, optional): If `True`, computes the decorrelation 
				losses (correlation and whitening). Defaults to `True`.
			batchde (bool, optional): If 'True' computes losses and gradients
				for a batch of decorrelation matrices at once. Defaults to 'False'
			model_args (MambaArgs): args used to define the model. Used to access
				the current model's device. 

		Returns:
			Tuple[torch.Tensor or None, float or None, float or None]:
				- **grad** (torch.Tensor or None): Gradient tensor of shape (D, D). 
				  Returns `None` if `compute_grad` is `False`.
				- **correlation_loss** (float or None): Correlation loss scalar. 
				  Returns `None` if `compute_loss` is `False`.
				- **whitening_loss** (float or None): Whitening loss scalar. 
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
			# collapse across batch dimension
			b, l, d = x.shape
			x = x.reshape(b*l, d)

			# compute the individual loss elements
			D = torch.diag_embed(x**2)
			V = D - torch.eye(d, device=model_args.device)

			xx_t = einsum(x, x, 'b x, b x_t -> b x x_t')

			C = xx_t - D

			if compute_loss:
				# sum of squared covariances
				correlation_loss = \
					torch.mean(
						torch.mean(C**2, dim=(1,2)))

				# sum of squared variances
				whitening_loss = \
					torch.mean(
						torch.mean(V**2, dim=(1,2)))

			else:
				correlation_loss = None 
				whitening_loss = None

			# compute the actual gradient, if applicable
			if compute_grad:
				grad = torch.mean(((1-kappa)*C + kappa*V), dim=0)
			else:
				grad = None

			return grad, correlation_loss, whitening_loss

		# used where decorrelation layer has multiple matrices to train
		# (this is the case for "channel_universal")
		else:
			b, n_samples, d, conv_1d_size = x.shape
			# in this case we're updating d matrices, each with info
			# from one embedding dimension channel. Rearrange dimensions
			# and collapse all samples into a single dimension. 

			# (D, all_samples, conv_1d_size)
			x = x.permute(2, 0, 1, 3).reshape(d, -1, conv_1d_size)

			# compute the individual loss elements
			D = torch.diag_embed(x**2)
			V = D - torch.eye(conv_1d_size, device=model_args.device)

			xx_t = einsum(x, x, 
				'd all_samples x, d all_samples x_t -> d all_samples x x_t')

			# (D, all_samples, conv_1d_size, conv_1d_size)
			C = xx_t - D

			if compute_loss:
				# sum of squared covariances, averaged across 
				# all samples, then averaged across all parallel channels
				correlation_loss = \
					torch.mean(
						torch.mean(
							torch.mean(C**2, dim=(2,3)), dim=1))

				# sum of squared variances, averaged across 
				# all samples, then averaged across all parallel channels
				whitening_loss = \
					torch.mean(
						torch.mean(
							torch.mean(V**2, dim=(2,3)), dim=1))

			else:
				correlation_loss = None 
				whitening_loss = None

			# compute the actual gradients, if applicable
			if compute_grad:
				grad = torch.mean(((1-kappa)*C + kappa*V), dim=1)
			else:
				grad = None

			return grad, correlation_loss, whitening_loss
			
	

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
		correlation_loss (float): Accumulated correlation loss across batches.
		whitening_loss (float): Accumulated whitening loss across batches.
		grad (torch.Tensor or None): Gradient of the decorrelation matrix, computed 
			during the forward pass.
		loss (DecorrLoss): Loss function instance for computing decorrelation and whitening losses.

	Methods:		
		reset():
			Resets accumulated losses and gradients to their initial state.
		
		reset_grad():
			Resets the gradient of the decorrelation matrix only.
		
		train(mode: bool = True):
			Enables or disables training mode for the layer. Controls whether decorrelation 
			gradients are computed.

	"""
	def __init__(self, original_layer: nn.Module, model_args: MambaArgs, 
		fuse: bool=True, **kwargs):
		"""
		Initializes the DecorrLinear with a decorrelation matrix.

		Args:
			original_layer (nn.Module): The original linear layer to extend.
			**kwargs: Additional arguments for decorrelation parameters, such as:
				- `kappa` (float): Decorrelation gradient hyperparameter.
				- `sample_frac` (float): Fraction of data sampled for decorrelation
					gradient calculation.
			fuse (bool, default=True): Controls whether the decorrelation and main layer
				operations are fused into one transformation before being applied to the 
				input. More efficient. 						
		"""
		super(DecorrLinear, self).__init__()

		self.original_layer = original_layer
		self.model_args = model_args
		# a layer of the same dimensions as the original, with no biases.
		# gradients are computed outside of the backward pass
		self.decorr_layer = nn.Parameter(
			torch.eye(original_layer.weight.shape[1]), requires_grad=False)

		self.fuse = fuse # used during the forward pass

		self.compute_grad = True # controlled by self.train()
		self.compute_loss = True 

		self.sample_frac = kwargs.get("sample_frac")
		self.kappa = kwargs.get("kappa")

		self.correlation_loss = 0
		self.whitening_loss = 0
		self.grad = None

		self.loss = DecorrLoss()


	def forward(self, x):

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
				b, l, d = x.shape
				n_samples = int(self.sample_frac*l)

				sample_idx = torch.multinomial(
					torch.ones(b, l), n_samples, replacement=False)
				batch_idx = torch.arange(b).unsqueeze(1).expand(b, n_samples)

				selected = x[batch_idx, sample_idx]

				selected_decorr = selected @ self.decorr_layer.T

				grad, correlation_loss, whitening_loss = self.loss(
					selected_decorr, self.kappa, self.model_args,
					compute_grad=self.compute_grad, compute_loss=self.compute_loss)

				self.correlation_loss += correlation_loss
				self.whitening_loss += whitening_loss
				# updates happen on entire network at once in training loop
				self.grad = grad 

		return y

	def reset(self):
		self.correlation_loss = 0
		self.whitening_loss = 0
		self.grad = None

	def reset_grad(self):
		self.grad = None

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
		model_args (MambaArgs): Arguments for the Mamba model architecture.
		mode (str): Mode of operation for decorrelation (`"token"` or `"patch"`).
		decorr_layer (nn.Parameter): Trainable decorrelation matrix, initialized at identity.

	"""
	def __init__(self, original_layer: nn.Module, model_args: MambaArgs, mode: str = "patch", 
		fuse: bool= True, **kwargs):
		"""
		Initializes the DecorrConv1d layer with a decorrelation matrix.

		Args:
			original_layer (nn.Module): The original convolutional layer to extend.
			model_args (MambaArgs): Arguments for the Mamba model architecture.
			mode (str, default="patch"): Decorrelation mode. 
				- `"token"`: Decorrelate features of each token independently.
				- `"patch"`: Decorrelate features across convolutional patches.
				- `"channel_shared"`: Decorrelate features across each embedding channel
						within every convolutional patch, using the same decorrelation
						matrix across all channels.	
			fuse (bool, default=True): Controls whether the decorrelation and main layer
				operations are fused into one transformation before being applied to the 
				input. More efficient. 								
			**kwargs: Additional arguments for decorrelation parameters, such as:
				- `kappa` (float): Decorrelation gradient hyperparameter.
				- `sample_frac` (float): Fraction of data sampled for decorrelation
					gradient calculation.

		Raises:
			AssertionError: If `mode` is not `"token"` or `"patch"`.
		"""
		super(DecorrConv1d, self).__init__(
			original_layer=original_layer, fuse=fuse, model_args=model_args, **kwargs)

		self.model_args = model_args
		self.mode = mode

		# determines how the decorrelation is applied.
		assert mode == "token" or mode == "patch" \
			or mode == "channel_shared" or mode == "channel_independent", \
			"conv_1d_mode must be \"token\", \"patch\", \"channel_shared\", or \"channel_independent\""

		if mode == "token":
			# decorrelate each token's features independently
			self.decorr_layer = nn.Parameter(torch.eye(model_args.D_inner), requires_grad=False)

		elif mode == "patch":
			# decorrelate all input features within each convolutional patch
			self.decorr_layer = nn.Parameter(
				torch.eye(model_args.D_inner*model_args.conv_1d_size), requires_grad=False)

		elif mode == "channel_shared":
			# decorrelate each patch channel's input features independently,
			# using the same decorrelation matrix for all channels
			self.decorr_layer = nn.Parameter(
				torch.eye(model_args.conv_1d_size), requires_grad=False)

		else:
			# decorrelate each patch channel's input features independently,
			# using a separate decorrelation matrix for all channels
			all_matrices = torch.eye(model_args.conv_1d_size).unsqueeze(0).repeat(
					model_args.D_inner, 1, 1)

			self.decorr_layer = nn.Parameter(all_matrices, requires_grad=False)

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
			x.unsqueeze(1), (d, self.model_args.conv_1d_size), 
			stride=1, padding=(0, self.model_args.conv_1d_size-1)).transpose(1,2)

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
					b, -1, d, self.model_args.conv_1d_size)

				# the unfused operation passes each token embedding through
				# the same decorrelation matrix, then performs convolution via
				# row-wise dot products of the kernels with the decorrelated 
				# patch matrix. We fuse these by defining a separate kernel-scaled version 
				# of the decorrelation matrix for each token in the patch, passing the 
				# tokens through their respective matrices, and then summing over the 
				# token dimension

				# (conv_1d_size, D, D)
				token_specific_decorr = self.decorr_layer.unsqueeze(0).repeat(
					self.model_args.conv_1d_size, 1, 1)

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
					decorr_inputs.transpose(1,2).unsqueeze(1), (d, self.model_args.conv_1d_size), 
					stride=1, padding=(0, self.model_args.conv_1d_size-1)).transpose(1,2)

				# (B, n_patches, D, conv_1d_size)
				decorr_patch_matrices = x_decorr_unfolded.reshape(
					b, -1, d, self.model_args.conv_1d_size)			

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

					grad, correlation_loss, whitening_loss = self.loss(
						selected_decorr, self.kappa, self.model_args,
						self.compute_grad, self.compute_loss)

					self.correlation_loss += correlation_loss
					self.whitening_loss += whitening_loss
					# updates happen on entire network at once in training loop
					self.grad = grad   				      

		# decorrelates all features in each input patch to the convolution
		elif self.mode == "patch":

			if self.fuse: # fused operation
				# split up decorrelation matrix to pre-multiply with each channel's kernel

				# (D, conv_1d_size, D*conv_1d_size (= n_patch_features))
				decorr_split = self.decorr_layer.reshape(
					d, self.model_args.conv_1d_size, d*self.model_args.conv_1d_size)

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
				decorr_patches = decorr.reshape(b,-1, d, self.model_args.conv_1d_size)

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

					selected_decorr = x_unfolded[batch_idx, sample_idx] @ self.decorr_layer.T

					grad, correlation_loss, whitening_loss = self.loss(
						selected_decorr, self.kappa, self.model_args,
						self.compute_grad, self.compute_loss)

					self.correlation_loss += correlation_loss
					self.whitening_loss += whitening_loss
					# updates happen on entire network at once in training loop
					self.grad = grad  

		# decorrelates each patch channel's input features independently,
		# using the same decorrelation matrix for all channels					
		elif self.mode == "channel_shared":   
			# represent each patch as a matrix, with embedding channels grouped
			# in the last dimension

			#(B, n_patches, D, conv_1d_size)
			patch_matrices = x_unfolded.reshape(
				b, -1, d, self.model_args.conv_1d_size)

			if self.fuse: # fused decorrelation and convolution

				# pre-multiply each kernel by a copy of the decorrelation matrix, 
				# then perform channel-independent 1d convolution using the resulting kernels

				# (D_inner, conv_1d_size, conv_1d_size)
				decorr_repeat = self.decorr_layer.unsqueeze(0).repeat(
						self.model_args.D_inner, 1, 1)

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

					grad, correlation_loss, whitening_loss = self.loss(
						selected_decorr, self.kappa, self.model_args,
						self.compute_grad, self.compute_loss)

					self.correlation_loss += correlation_loss
					self.whitening_loss += whitening_loss
					# updates happen on entire network at once in training loop
					self.grad = grad

		# decorrelates each patch channel's input features independently,
		# using a separate decorrelation matrix for all channels	
		else:
			# represent each patch as a matrix, with embedding channels grouped
			# in the last dimension

			#(B, n_patches, D, conv_1d_size)
			patch_matrices = x_unfolded.reshape(
				b, -1, d, self.model_args.conv_1d_size)

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
					# sample fraction of each batch from which to compute loss + update.
					b, n_patches, _, _ = patch_matrices.shape

					# we sample a fixed fraction of the number of patches within each batch,
					# and compute gradients/losses for all of the decorrelation matrices
					# using each sample
					n_samples = int(self.sample_frac*n_patches)

					sample_idx = torch.multinomial(
						torch.ones(b, n_patches), n_samples, replacement=False)
					batch_idx = torch.arange(b).unsqueeze(1).expand(b, n_samples)

					selected = patch_matrices[batch_idx, sample_idx]

					selected_decorr = einsum(self.decorr_layer, selected,
						'd conv_1d_size dummy, b n_samples d dummy -> b n_samples d conv_1d_size')

					grad, correlation_loss, whitening_loss = self.loss(
						selected_decorr, self.kappa, self.model_args,
						compute_grad=self.compute_grad, compute_loss=self.compute_loss,
						batched=True)

					self.correlation_loss += correlation_loss
					self.whitening_loss += whitening_loss
					# updates happen on entire network at once in training loop
					self.grad = grad 


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


class DecorrMamba(Mamba):
	"""
	Mamba architecture with built-in decorrelation layers and additional functionality
	for training and monitoring them.

	This class extends the `Mamba` architecture to add decorrelation layers, which 
	are initialized and managed dynamically. It supports the following functionalities:
	- Creation of decorrelation matrices in specific layers (e.g., `in_proj`, `out_proj`).
	- Loss monitoring for decorrelation layers (e.g., correlation loss, whitening loss).
	- Dynamic updates to decorrelation layers during training.

	Attributes:
		total_correlation_loss (float): The total correlation loss across all decorrelation layers.
		total_whitening_loss (float): The total whitening loss across all decorrelation layers.
		decorr_lr (float): Learning rate for updating decorrelation matrices.
		conv_1d_mode (str): The mode used for 1D convolution decorrelation layers. "patch"
			decorrelates all features seen by a convolutional kernel, "token" decorrelates
			all token features independently. 
		sample_frac (float): Fraction of the data sampled for decorrelation update calculations
		kappa (float): Hyperparameter controlling contribution of whitening and decorrelation
			loss terms in the gradient computation, for the decorrelation amtrices

	Methods:
		sum_decorr_losses():
			Calculates and stores the total correlation and whitening losses from all decorrelation layers.

		update_decorr_matrices():
			Updates decorrelation matrices for all decorrelation layers using their respective gradients.

		reset_decorr_grad():
			Resets the gradients of decorrelation matrices after a forward pass.

		reset_decorr_layers():
			Resets the gradients and losses of decorrelation layers and the total summed losses.

		compute_decorr_losses(mode: bool=True):
			Enables or disables the computation of decorrelation losses during the forward pass.
	"""

	def __init__(self,  conv_1d_mode: str, model_args: MambaArgs = None, 
		existing_model: Mamba = None, fuse: bool = True, **kwargs):

		"""
		Initializes the DecorrMamba model.

		Either creates a new Mamba model with decorrelation layers or extends an 
		existing Mamba model with decorrelation functionality. In the latter case,
		decorrelation matrices are initialized at identity.

		Args:
			conv_1d_mode (str): The mode used for 1D convolution decorrelation layers.
			model_args (MambaArgs, optional): Arguments to configure a new Mamba model.
			existing_model (Mamba, optional): Pre-existing Mamba model to modify.
			**kwargs: Additional keyword arguments for decorrelation parameters:
				- kappa (float): Hyperparameter for decorrelation gradient.
				- sample_frac (float): Fraction of the data sampled for decorrelation 
					matrix gradient calculations.
				- decorr_lr (float): Learning rate for decorrelation updates.
			fuse (bool, default=True): Controls whether the decorrelation and main layer
				operations are fused into one transformation before being applied to the 
				input. More efficient. 					

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


		def _create_decorr_matrices(module, kappa, sample_frac, conv_1d_mode, model_args, fuse):
			''' 
			Used to recursively traverse model and create decorrelation matrices 
			in pre-defined places
			'''
			for name, child in module.named_children():
				if name == "in_proj" or name == "out_proj" or name == "to_BCdelta":
					setattr(module, name, DecorrLinear(
						child, kappa=kappa, model_args = model_args,
						sample_frac=sample_frac, fuse=fuse))

				if name == "conv1d":
					setattr(module, name, DecorrConv1d(
						child, model_args, conv_1d_mode, kappa=kappa, 
						sample_frac=sample_frac, fuse=fuse))


		self.apply(partial(_create_decorr_matrices, 
			kappa=kwargs.get("kappa"), sample_frac=kwargs.get("sample_frac"), 
			conv_1d_mode=conv_1d_mode, model_args=self.model_args, fuse=fuse))

		# remove weight decay for decorrelation layers
		apply_to_decorr(self, 
			lambda decorr_module: setattr(
				getattr(decorr_module, "decorr_layer"), "_no_weight_decay", True))


		self.total_correlation_loss = 0
		self.total_whitening_loss = 0
		self.decorr_lr = kwargs.get("decorr_lr")

		# these are just here for reference, individual decorrelation layers
		# take care of using these values automatically
		self.conv_1d_mode = conv_1d_mode
		self.sample_frac = kwargs.get("sample_frac")
		self.kappa = kwargs.get("kappa")

	def sum_decorr_losses(self):
		''' 
		Calculates the summed total of correlation and whitening losses across all 
		layers implementing decorrelation, for a Mamba model
		'''

		def _sum_losses(module):

			for child in module.children():
				# DecorrConv1d extends DecorrLinear, should account for 
				# convolutional layers too
				if isinstance(child, DecorrLinear):
					self.total_correlation_loss += child.correlation_loss
					self.total_whitening_loss += child.whitening_loss

		self.apply(_sum_losses)

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
					assert child.grad is not None, "Gradient not computed"
					child.decorr_layer -= self.decorr_lr * child.grad @ child.decorr_layer

		self.apply(_update_decorr_matrices)

	def reset_decorr_grad(self):
		''' 
		Resets gradients of decorrelation matrices after forward pass. 
		Used between each minibatch update. 
		'''

		apply_to_decorr(self, lambda x: x.reset_grad())

	def reset_decorr_layers(self):
		''' 
		Resets gradients and losses of decorrelation layers after parameter
		updates. Also resets the summed total losses across all decorrelation
		layers. Used before beginning a new training epoch.
		'''
		apply_to_decorr(self, lambda x: x.reset())
		self.total_correlation_loss = 0
		self.total_whitening_loss = 0

	def compute_decorr_losses(self, mode: bool=True):
		"""
		Enables or disables the computation of decorrelation losses during the 
		forward pass. Useful for switching between training and inference modes.

		Args:
			mode (bool, default=True): If True, computes decorrelation losses; 
									   if False, skips loss computation.
		"""
		apply_to_decorr(self, lambda x: setattr(x, "compute_loss", mode))

			





	
