from dataclasses import dataclass
from torch.utils.data import Dataset
import math
from typing import Callable, Union
import numpy as np
import matplotlib.pyplot as plt 
import torch  
from transformers import GPT2Tokenizer
import os


class DefaultArgs:
    ''' Contains the default training protocols described in the Mamba paper.
        For the synthetic datasets, values were often not given. In these cases,
        values were copied from the language modelling protocol.'''

    def __init__(self):
        # language modelling args
        self.lm_args = {
            "adam_beta": (0.9, 0.95),
            "adam_epsilon": 1e-8,
            "gradient_clip": 1.0,
            "weight_decay": 0.1,
            "use_lr_sched": True,
            "min_lr": 1e-5
        }

        # synthetic dataset modelling args
        self.s_args = {
            "adam_beta": (0.9, 0.95),
            "adam_epsilon": 1e-8,
            "gradient_clip": None,
            "weight_decay": None,
            "use_lr_sched": False,
            "min_lr": None        
        }



@dataclass
class TrainingArgs():
    ''' Contains all the arguments necessary to train a model. In case of learning schedule
        being selected, this employs a cosine decay to a certain pre-specified limit, with a 
        linear warmup period. Learning schedule can also be visualized before training. '''

    n_epochs: int
    L: int  
    B: int 
    lr: float
    adam_beta: tuple 
    adam_epsilon: float

    gradient_clip: float = None
    weight_decay: float = None

    # parameters of the learning rate schedule
    use_lr_sched: bool = None
    min_lr: float = None
    warmup_epochs: int = None
    
    def __post_init__(self):
        # NB specification in the paper uses 5x the learning rate of a comparable 
        # GPT3 architecture as the peak schedule LR, for language modelling. The learning rate 
        # should be selected using this method, in language modelling cases!

        if self.use_lr_sched:
            assert self.warmup_epochs is not None, "Warmup epochs specification missing"
            assert self.warmup_epochs < self.n_epochs, "Warmup epochs > total epochs"
            self.schedule_fn: Callable[[int], float] = self._lm_learning_schedule
            assert self.lr > self.min_lr, \
                "Minimum learning rate greater than the peak learning rate"


    def _lm_learning_schedule(self, epoch):
        # a cosine decay with a minimum value, with a linear warm-up
        if epoch < self.warmup_epochs:
            return float(epoch+1) / float(max(1, self.warmup_epochs))
        else:
            # calculate amount of decay progress
            progress = float(epoch - self.warmup_epochs + 1) / \
                            float(max(1, self.n_epochs - self.warmup_epochs))
            # shift cosine function up, rescale, and compute the appropriate amount
            # of decay
            cosine_decay = 0.5 * (1+ math.cos(math.pi * progress))
            # rescale the function again so that it doesn't go below the minimum
            # value
            return cosine_decay * (1 - self.min_lr / self.lr) + self.min_lr / self.lr
    
    def show_lr_schedule(self):
        # visualization of the learning rate schedule given the specified
        # training protocol
        epochs = np.arange(0, self.n_epochs)
        lrs = np.zeros(len(epochs))
        for e in epochs:
            lrs[e] = self.lr*self._lm_learning_schedule(e)

        min_lr = np.min(lrs[self.warmup_epochs+1:])
        max_lr = np.max(lrs)

        plt.figure()
        plt.plot(epochs, lrs)
        plt.xlim([0,self.n_epochs-1])
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.title(f"Max = {max_lr}, \nMin = {min_lr:.2e}")
        plt.show()


@dataclass
class MambaArgs:
    ''' Contains all the arguments necessary to define a Mamba model'''

    N: int # hidden dimensionality
    D: int # dimensionality of token embeddings
    n_layers: int
    vocab_size: int = 50257 # GPT2 tokenizer default
    assert vocab_size <= 50257, "Vocab size exceeds maximum of GPT2 tokenier"
    pad_vocab_size_multiple: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    expansion_factor: int = 2 # input embeddings are upscaled by this factor
    conv_1d_size: int = 4
    conv_bias: bool = True
    general_bias: bool = False # applies to the input and output projections

    # parameters governing how to initialize delta's
    # relevant projection weights
    delta_init: str = "random" # other option is "constant"
    delta_scale: float = 1.0
    delta_rank: Union[int, str] = 'auto'

    # for the biases to the delta projection layer
    delta_min = 0.001
    delta_max = 0.1
    delta_init_floor=1e-4

    def __post_init__(self):
        # see discussion in paper about dimensionality of delta
        self.D_inner = int(self.expansion_factor * self.D)
        if self.delta_rank == "auto":
            self.delta_rank = math.ceil(self.D/16)

        # padding vocab size to be a nice number for parallel processing
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)

class SeqDataset(Dataset):
    ''' A simple way of creating datasets for next token prediction training

        Args:
            device (str): the device, cpu or cuda
            seq_size (int): length of the training sequences
            seqs (list[str]): the entire dataset expressed as a 
                list of strings, where each list entry is a word

        Attributes:
            device (device)
            seq_size (int)
            seqs (list[str])

        '''

        
    def __init__(self, device: str, seq_size: int, seqs: list[str]):
        super(SeqDataset, self).__init__()
        self.device = device
        self.seq_size = seq_size
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs) - self.seq_size - 1

    def __getitem__(self, idx):
        in_seq = torch.tensor(self.seqs[idx:idx + self.seq_size], 
            dtype=torch.long, device=self.device)
        target_seq = torch.tensor(self.seqs[idx + 1:idx + self.seq_size + 1], 
            dtype=torch.long, device=self.device)

        return in_seq, target_seq

class LanguageDatasetMaker:
    ''' Takes a standard-form dataset (list of individual word strings) and 
        uses the GPT2 tokenizer with a user-defined vocabulary length to create
        usable standard PyTorch Datasets out of it

        Args:
            raw_dataset (list[str]): input dataset, where each entry is a word.
            mamba_args (MambaArgs): model configuration.
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
        raw_dataset: list[str], mamba_args: MambaArgs, train_args: TrainingArgs,
        total_dataset_frac: float = 1, train_split: float = 0.8, val_split: float = 0.1):

        assert train_split + val_split <= 1.0, \
            "Sum of split fractions exceeds 1"

        assert total_dataset_frac <= 1.0, \
            "Total dataset fraction must be smaller or equal to 1"

        self.total_dataset_frac = total_dataset_frac
        self.mamba_args = mamba_args
        self.train_args = train_args


        self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")


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

        # get tokens and corresponding IDs
        tokens = [self.tokenizer.tokenize(word) for word in raw_dataset]
        token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]

        # flattening them all in one list
        token_ids = [item for sublist in token_ids for item in sublist] 

        # shorten if necessary
        if self.total_dataset_frac < 1.0:
            token_ids = token_ids[:int(self.total_dataset_frac*(len(token_ids)))]

        # limit all tokens to the top n most common. Replace the less common occurrences
        # with unk token 
        filtered_ids = []
        unk_id = self.tokenizer.get_vocab()['unk']
        # if given vocab size is not large enough to include the unk token itself, 
        # the vocab size must be reduced by 1 to fit this in
        if unk_id > self.mamba_args.vocab_size-1:
            vocab_limit = self.mamba_args.vocab_size-2
        else:
            vocab_limit = self.mamba_args.vocab_size-1

        for token_id in token_ids:
            # filter the text to only include top n most common words
            if token_id > vocab_limit and token_id != unk_id:
                filtered_ids.append(unk_id)
            else:
                filtered_ids.append(token_id)

        del token_ids 

        # put everything in Datasets and return
        train_set = SeqDataset(
            self.mamba_args.device, 
            self.train_args.L, 
            filtered_ids[:int(self.train_split * len(filtered_ids))])

        val_set   = SeqDataset(
            self.mamba_args.device, 
            self.train_args.L, 
            filtered_ids[int(self.train_split * len(filtered_ids)) + 1:
                int((self.train_split+self.val_split) * len(filtered_ids))])

        # whatever is left over from the training and validation splits
        # becomes the testing dataset
        test_set  = SeqDataset(
            self.mamba_args.device, 
            self.train_args.L, 
            filtered_ids[int((self.train_split+self.val_split) * len(filtered_ids)) + 1:])

        return train_set, val_set, test_set



