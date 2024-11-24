import torch
import torch.nn as nn
import torch.nn.functional as F 
from model.mamba import Mamba
from einops import einsum
from utils.helpers import MambaArgs
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


	def forward(self, x, kappa: float, compute_grad: bool = True, 
		compute_loss: bool = True):

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

		# collapse across batch dimension
		b, l, d = x.shape
		x = x.reshape(b*l, d)

		# compute the individual loss elements
		D = torch.diag_embed(x**2)
		V = D - torch.eye(d)

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
	def __init__(self, original_layer: nn.Module, fuse: bool=True, **kwargs):
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

		# for efficiency, multiply decorrelation and standard weight matrices together before
		# passing input through the resulting single matrix
		y = x @ (self.original_layer.weight @ self.decorr_layer).T

		if self.original_layer.bias is not None:
			y += self.original_layer.bias

		# caveat: optimization above requires another forward pass through just the
		# decorrelation matrix to compute the loss and gradients
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
					selected_decorr, self.kappa, self.compute_grad, self.compute_loss)

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

# in case you want to compute the unfused token operation, here's the code (used during testing)
	# decorr_out = x_unfolded @ self.decorr_layer.T

	# decorr_out_reshape = decorr_out.reshape(
	# 	b, -1, d, self.model_args.conv_1d_size).transpose(2,3)

	# y = einsum(decorr_out_reshape, torch.squeeze(self.original_layer.weight),
	# 	'b n_patches conv_1d_size d, d conv_1d_size -> b n_patches d')




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
		super(DecorrConv1d, self).__init__(original_layer, **kwargs)

		self.model_args = model_args
		self.mode = mode

		# determines how the decorrelation is applied.
		assert mode == "token" or mode == "patch" or mode == "channel_shared", \
			"conv_1d_mode must be \"token\", \"patch\", or \"channel_shared\""

		if mode == "token":
			# decorrelate each token's features independently
			self.decorr_layer = nn.Parameter(torch.eye(model_args.D_inner), requires_grad=False)
		elif mode == "patch":
			# decorrelate all input features within each convolutional patch
			self.decorr_layer = nn.Parameter(
				torch.eye(model_args.D_inner*model_args.conv_1d_size), requires_grad=False)
		else:
			# decorrelate each patch channel's input features independently,
			# using the same decorrelation matrix for all of them
			self.decorr_layer = nn.Parameter(
				torch.eye(model_args.conv_1d_size), requires_grad=False)

	def forward(self, x):

		if self.compute_grad or self.compute_loss:
			assert self.sample_frac is not None , \
				"Specify sample_frac for loss and gradient computation"

			assert self.sample_frac > 0 and self.sample_frac <= 1.0, \
				"sample_frac must be between 0 and 1"   

		b, d, l = x.shape
		# (B, n_patches, conv_1d_size*D). All data in each convolution patch
		# is represented in a single vector
		x_unfolded = F.unfold(
			x.unsqueeze(1), (d, self.model_args.conv_1d_size), 
			stride=1, padding=(0, self.model_args.conv_1d_size-1)).transpose(1,2)

		# decorrelates each token's features independently. As in DecorrLinear, decorrelation
		# and convolutional operations are fused together into one matrix. Convolutions
		# in Mamba only mix information across the same token dimension,
		# (groups = D_inner), making the convolution operation a series of dot 
		# products within the unfolded patch, not a regular matrix multiplication.

		if self.mode == "token":
			# currently each patch's features are represented in
			# a single vector. Reshape these vectors into matrices, such
			# that the 0th dimension of each matrix contains all features
			# for the 0th dimension across all tokens, etc. 

			#( B, n_patches, conv_1d_size, D)
			patch_matrices = x_unfolded.reshape(
				b, -1, d, self.model_args.conv_1d_size).transpose(2,3)

			# decorr + conv1d for one patch is a matrix multiplication
			# of the matrix representation of the patch with the decorrelation
			# matrix, followed by row-wise dot products of the output with the 
			# convolution weights. This can be fused into one operation by defining
			# a separate scaled version of the decorrelation matrix for each token
			# in a convolutional patch, passing the tokens through their respective
			# matrices, and then summing over the token dimension

			token_specific_decorr = self.decorr_layer.unsqueeze(0).repeat(
				self.model_args.conv_1d_size, 1, 1)

			token_specific_decorr = einsum(token_specific_decorr, 
				torch.squeeze(self.original_layer.weight).T,

				'conv_1d_size out_dim in_dim, conv_1d_size out_dim' +
				' -> conv_1d_size out_dim in_dim')

			decorr_conv_outputs = einsum(patch_matrices, token_specific_decorr,
				'b n_patches conv_1d_size in_dim, conv_1d_size out_dim in_dim' + 
				' -> b n_patches conv_1d_size out_dim')

			# (B, n_patches, D). Summing over the decorrelated + scaled tokens
			# in each patch completes the dot product operation. 
			y = torch.sum(decorr_conv_outputs, dim=2)
			
			if self.original_layer.bias is not None:
				y += self.original_layer.bias 

			# like in DecorrLinear, fusion of decorrelation and layer operation requires
			# separate forward pass for gradient computation   
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
						selected_decorr, self.kappa, self.compute_grad, self.compute_loss)

					self.correlation_loss += correlation_loss
					self.whitening_loss += whitening_loss
					# updates happen on entire network at once in training loop
					self.grad = grad   				      

		# decorrelates all features in each input patch to the convolution
		elif self.mode == "patch":
			# split up decorrelation matrix to pre-multiply with each channel's kernel
			# (D, D*conv_1d_size (= n_patch_features), conv_1d_size)
			decorr_reshape = self.decorr_layer.reshape(
				d, self.model_args.conv_1d_size, d*self.model_args.conv_1d_size).transpose(1,2)

			fused_matrix = einsum(torch.squeeze(self.original_layer.weight), decorr_reshape,
				'd conv_1d_size, d n_patch_features conv_1d_size -> d n_patch_features')

			y = x_unfolded @ fused_matrix.T

			if self.original_layer.bias is not None:
				y += self.original_layer.bias

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
						selected_decorr, self.kappa, self.compute_grad, self.compute_loss)

					self.correlation_loss += correlation_loss
					self.whitening_loss += whitening_loss
					# updates happen on entire network at once in training loop
					self.grad = grad  

		# decorrelates each patch channel's input features independently,
		# using the same decorrelation matrix for all of them					
		else:   
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


			if self.original_layer.bias is not None:
				y += self.original_layer.bias

			if self.compute_grad or self.compute_loss:
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
					selected_decorr, self.kappa, self.compute_grad, self.compute_loss)

				self.correlation_loss += correlation_loss
				self.whitening_loss += whitening_loss
				# updates happen on entire network at once in training loop
				self.grad = grad   				      						           

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
						child, kappa=kappa, sample_frac=sample_frac, fuse=fuse))

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

			





	
