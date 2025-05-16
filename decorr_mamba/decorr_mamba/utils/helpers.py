from dataclasses import dataclass
from torch.utils.data import Dataset
import math
from typing import Callable, Union
import numpy as np
import matplotlib.pyplot as plt 
import torch  
from transformers import AutoTokenizer
from collections import Counter



@dataclass
class TrainingArgs():
	''' Contains all the arguments necessary to train a model. In case of learning schedule
		being selected, this employs a cosine decay to a certain pre-specified limit, with a 
		linear warmup period. Learning schedule can also be visualized before training. 
		Default values for these parameters (where applicable) can be found in the .json
		files in the template_experiment folder.'''

	n_steps: int
	L: int  
	B: int 
	lr: float
	adam_beta: tuple 
	adam_epsilon: float

	gradient_clip: float
	weight_decay: float
	ddp: bool

	# parameters of the learning rate schedule
	use_lr_sched: bool
	min_lr: float

	# n_steps and warmup_steps can refer to both epochs and gradient descent
	# steps
	warmup_steps: int

	optimizer: str = "adam"
	
	def __post_init__(self):
		# NB specification in the paper uses 5x the learning rate of a comparable 
		# GPT3 architecture as the peak schedule LR, for language modelling. The learning rate 
		# should be selected using this method, in language modelling cases!

		assert self.n_steps is not None, "Must specify n_steps"
		if self.use_lr_sched:
			assert self.warmup_steps is not None, "Warmup epochs/steps specification missing"
			assert self.warmup_steps <= self.n_steps, "Warmup epochs/steps > total epochs/steps"

			self.schedule_fn: Callable[[int], float] = self._lm_learning_schedule

			assert self.lr > self.min_lr, \
				"Minimum learning rate greater than the peak learning rate"

	def _lm_learning_schedule(self, step):
		# a cosine decay with a minimum value, with a linear warm-up
		if step < self.warmup_steps:
			return float(step+1) / float(max(1, self.warmup_steps))
		else:
			# calculate amount of decay progress
			progress = float(step - self.warmup_steps + 1) / \
							float(max(1, self.n_steps - self.warmup_steps))
			# shift cosine function up, rescale, and compute the appropriate amount
			# of decay
			cosine_decay = 0.5 * (1+ math.cos(math.pi * progress))
			# rescale the function again so that it doesn't go below the minimum
			# value
			return cosine_decay * (1 - self.min_lr / self.lr) + self.min_lr / self.lr
	
	
	def show_lr_schedule(self):
		# visualization of the learning rate schedule given the specified
		# training protocol
		steps = np.arange(0, self.n_steps)
		lrs = np.zeros(len(steps))
		for step in steps:
			lrs[step] = self.lr*self._lm_learning_schedule(step)

		# min_lr = np.min(lrs[self.warmup_steps+1:])
		# max_lr = np.max(lrs)

		plt.figure()
		plt.plot(steps, lrs)
		plt.xlim([0,self.n_steps-1])
		plt.xlabel("Step")
		plt.ylabel("Learning rate")
		# plt.title(f"Max = {max_lr}, \nMin = {min_lr:.2e}")
		plt.show()


@dataclass
class MambaArgs:
	''' 
	Contains all the arguments necessary to define a Mamba model. Default values
	for all of these are found in the .json files in the template_experiment folder, 
	and are coped from the original Mamba inplementation. 


	OUTDATED: ALL DEFAULT MAMBA ARCHITECTURES ARE TRAINED USING THE DEFAULT
	MambaConfig CLASS FROM THE MAMBA REPO, THIS WAS FOR PERSONAL USE ONLY

	'''

	N: int # hidden dimensionality
	D: int # dimensionality of token embeddings
	n_layers: int
	vocab_size: int
	pad_vocab_size_multiple: int
	device: str
	expansion_factor: int # input embeddings are upscaled by this factor
	conv_1d_size: int
	conv_bias: bool
	general_bias: bool # applies to the input and output projections

	# parameters governing how to initialize delta's
	# relevant projection weights
	delta_init: str # either "random" or "constant"
	delta_scale: float
	delta_rank: Union[int, str] # either "auto" or a specific number

	# for the biases to the delta projection layer
	delta_min: float 
	delta_max: float
	delta_init_floor: float 

	def __post_init__(self):

		assert self.vocab_size <= 50257, "Vocab size exceeds maximum of GPT2 tokenizer"

		# see discussion in paper about dimensionality of delta
		self.D_inner = int(self.expansion_factor * self.D)
		if self.delta_rank == "auto":
			self.delta_rank = math.ceil(self.D/16)

		# padding vocab size to be a nice number for GPU use. If using
		# the full vocab size of the tokenizer, we pad with extra tokens that are
		# never used. If using smaller than the full vocab size, whatever number the
		# user inputs will be rounded up to a nice value
		if self.vocab_size % self.pad_vocab_size_multiple != 0:
			self.vocab_size += (self.pad_vocab_size_multiple
								- self.vocab_size % self.pad_vocab_size_multiple)

class SeqDataset(Dataset):
	''' A simple way of creating datasets for next token prediction training

		Args:
			seq_size (int): length of the training sequences
			seqs (list[str]): the entire dataset expressed as a 
				list of strings, where each list entry is a word

		Attributes:
			device (device)
			seq_size (int)
			seqs (list[str])

		'''

		
	def __init__(self, seq_size: int, seqs: list[str]):
		super(SeqDataset, self).__init__()
		self.seq_size = seq_size
		self.seqs = seqs

	def __len__(self):
		return len(self.seqs) - self.seq_size - 1

	def __getitem__(self, idx):
		in_seq = torch.tensor(self.seqs[idx:idx + self.seq_size], 
			dtype=torch.long)
		target_seq = torch.tensor(self.seqs[idx + 1:idx + self.seq_size + 1], 
			dtype=torch.long)

		return in_seq, target_seq

class LanguageDatasetMaker:
	''' Takes a standard-form dataset (list of individual word strings) and 
		uses the GPT2 tokenizer with a user-defined vocabulary length to create
		usable standard PyTorch Datasets out of it

		Args:
			raw_dataset (list[str]): input dataset, where each entry is a word.
			model_args (MambaArgs): model configuration.
			train_args (TrainingArgs): training configuration (batch size, sequence length, etc.).
			total_dataset_frac (float, optional): fraction of the total dataset to be used. 
				Default is 1, meaning the full dataset.
			train_split (float, optional): Proportion of the dataset to be used for training. 
				Defaults to 0.8.
			val_split (float, optional): Proportion of the dataset to be used for validation. 
				Defaults to 0.1.

		Attributes:
			train_set (SeqDataset or None): PyTorch-compatible training data. 
				Set to `None` if `train_split` is 0.
			val_set (SeqDataset or None): PyTorch-compatible validation data. 
				Set to `None` if `val_split` is 0.
			test_set (SeqDataset or None): PyTorch-compatible testing data. 
				Set to `None` if the test split is 0.
			train_len (int): number of training sequences.
			val_len (int): number of validation sequences.
			test_len (int): number of test sequences).
			total_len (int): total number of sequences.
		'''

	def __init__(self, 
		raw_dataset: list[str], model_args: MambaArgs, train_args: TrainingArgs,
		total_dataset_frac: float = 1, train_split: float = 0.8, val_split: float = 0.1):

		assert train_split + val_split <= 1.0, \
			"Sum of split fractions exceeds 1"

		assert total_dataset_frac <= 1.0, \
			"Total dataset fraction must be smaller or equal to 1"

		self.total_dataset_frac = total_dataset_frac
		self.model_args = model_args
		self.train_args = train_args


		self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b', 
			clean_up_tokenization_spaces=True, unk_token="<|unk|>")


		self.train_split = train_split
		self.val_split = val_split
		self.test_split = 1.0 - self.train_split - self.val_split

		# functionality to allow some splits to not exist (e.g. use all
		# data in the training split)
		train_set, val_set, test_set = self._create_dataset(raw_dataset)

		self.train_set = train_set if train_split > 0 else None
		self.val_set = val_set if val_split > 0 else None
		self.test_set = test_set if self.test_split > 0 else None

		# calculating total dataset length
		self.train_len = len(self.train_set) if self.train_set is not None else 0
		self.val_len = len(self.val_set) if self.val_set is not None else 0
		self.test_len= len(self.test_set) if self.test_set is not None else 0

		self.total_len = self.train_len + self.val_len + self.test_len

	def _create_dataset(self, raw_dataset):
		''' Creates a PyTorch-compatible SeqDataset from the raw data''' 

		# shorten if necessary
		if self.total_dataset_frac < 1.0:
			raw_dataset = raw_dataset[:int(self.total_dataset_frac*(len(raw_dataset)))]

		# get tokens and corresponding IDs
		tokens = [self.tokenizer.tokenize(word) for word in raw_dataset]
		token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]
		token_ids = [item for sublist in token_ids for item in sublist] 

		# filter tokens if we're working with fewer than the max vocab size
		if self.model_args.vocab_size < 50264:  # NB this is after padding vocab size!

			# get the id of the unk token so we can replace all too-infrequent tokens
			# with it
			unk_token_id = self.tokenizer.convert_tokens_to_ids('<|unk|>')

			# find the top n most common tokens. Minus one because we want to leave
			# room for the unk token in the final vocabulary
			counter = Counter(token_ids)
			most_common = counter.most_common(self.model_args.vocab_size-1)
			# just want the items, not the counts
			most_common = [item for item, count in most_common]

			# replace all tokens in the original list that are not in most_common
			# with the unk token (possibly including the unk token itself)
			token_ids = [
				token_id if token_id in most_common else unk_token_id for token_id in token_ids]

			# rescale the IDs to be within range of 0:vocab_size-1, and keep track of this 
			# dictionary to translate model output after training.
			token_dict = dict(
				zip(most_common, 
					list(np.arange(0,len(most_common)))
					))
			token_dict[unk_token_id] = self.model_args.vocab_size-1

			if len(most_common) < self.model_args.vocab_size-1:
				print("Minimum vocab size of the dataset (including <|unk|>):" +\
					f" {len(most_common)}, can shrink further!")

			token_ids = [token_dict[token_id] for token_id in token_ids]


		else:
			token_dict = None


		# put everything in Datasets and return
		train_set = SeqDataset(
			self.train_args.L, 
			token_ids[:int(self.train_split * len(token_ids))])

		val_set   = SeqDataset(
			self.train_args.L, 
			token_ids[int(self.train_split * len(token_ids)) + 1:
				int((self.train_split+self.val_split) * len(token_ids))])

		# whatever is left over from the training and validation splits
		# becomes the testing dataset
		test_set  = SeqDataset(
			self.train_args.L, 
			token_ids[int((self.train_split+self.val_split) * len(token_ids)) + 1:])

		return train_set, val_set, test_set
	



