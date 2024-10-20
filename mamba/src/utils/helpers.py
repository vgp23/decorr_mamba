from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Callable
import math
import numpy as np
import matplotlib.pyplot as plt 
import torch  
from transformers import GPT2Tokenizer
import os

@dataclass
class LMTrainingArgs:
    # NB this is not the actual learning rate, see below! 
    gpt_3_peak_lr: float # GPT3 spec, copy from table depending on size
    warmup_epochs: int 
    n_epochs: int
    L: int  
    B: int 

    # default arguments, specified according to Mamba paper. This is the 
    # default for language modeling, for artificial tasks use the other
    # args class! TODO: implement this other class

    min_lr: float = 1e-5
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    adam_beta: tuple = (0.9, 0.95)
    adam_epsilon: float = 1e-8
    optimizer: str = "AdamW" # TODO: implement ability to use Adam, just because
    
    def __post_init__(self):
        self.peak_lr: float = 5*self.gpt_3_peak_lr # the actual lr in the training recipe
        self.schedule_fn: Callable[[int], float] = self._lm_learning_schedule

        assert self.warmup_epochs < self.n_epochs, "Warmup epochs > total epochs"
        assert self.optimizer == "AdamW" or self.optimizer == "Adam", 'Invalid optimizer'
        
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
            return cosine_decay * (1 - self.min_lr / self.peak_lr) + self.min_lr / self.peak_lr
    
    def show_lr_schedule(self):
        # visualization of the learning rate schedule given the specified
        # training protocol
        epochs = np.arange(0, self.n_epochs)
        lr = np.zeros(len(epochs))
        for e in epochs:
            lr[e] = self.peak_lr*self._lm_learning_schedule(e)

        min_lr = np.min(lr[self.warmup_epochs+1:])
        max_lr = np.max(lr)

        plt.figure()
        plt.plot(epochs, lr)
        plt.xlim([0,self.n_epochs-1])
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.title(f"Max = {max_lr}, \nMin = {min_lr:.2e}")
        plt.show()


@dataclass
class MambaArgs:
    N: int 
    D: int
    n_layers: int
    vocab_size: int = 50257
    assert vocab_size <= 50257, "Vocab size exceeds maximum of GPT2 tokenier"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    expansion_factor: int = 2
    conv_1d_size: int = 4
    conv_bias: bool = True
    general_bias: bool = False # applies to the input and output projections

    # parameters governing how to initialize delta's
    # relevant projection weights
    delta_init: str = "random" # other option is "constant"
    delta_scale: float = 1.0
    delta_rank = "auto"

    # for the biases to the projection layer
    delta_min = 0.001
    delta_max = 0.1
    delta_init_floor=1e-4

    def __post_init__(self):
        self.D_inner = int(self.expansion_factor * self.D)
        if self.delta_rank == "auto":
            self.delta_rank = math.ceil(self.N/16)

class SeqDataset(Dataset):
    def __init__(self, device, seq_size, seqs):
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
        usable standard PyTorch Datasets out of it'''
    def __init__(self, 
        raw_dataset: list[str], mamba_args: MambaArgs, train_args: LMTrainingArgs,
        total_dataset_frac: float = 1, train_split: float = 0.8, val_split: float = 0.1):

        assert train_split + val_split <= 1.0, \
            "Sum of split fractions exceeds 1"

        assert total_dataset_frac <= 1.0, \
            "Total dataset fraction must be smaller or equal to 1"

        self.total_dataset_frac = total_dataset_frac
        self.mamba_args = mamba_args
        self.train_args = train_args

        tokenizer_path = os.path.join(".", "gpt2tokenizer")
        if os.path.exists(tokenizer_path) and os.path.isdir(tokenizer_path):
            self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2", 
                                                            force_download=True)
            self.tokenizer.save_pretrained(tokenizer_path)

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
        for token_id in token_ids:
            # if given vocab size is not large enough to include the unk token itself, 
            # the vocab size must be reduced by 1 to fit this in
            if unk_id > self.mamba_args.vocab_size-1:
                vocab_limit = self.mamba_args.vocab_size-2
            else:
                vocab_limit = self.mamba_args.vocab_size-1

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



