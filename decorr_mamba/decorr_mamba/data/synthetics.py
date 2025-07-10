import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import os

class CopyData:
    def __init__(self, L, M, A, variable=False, variable_length=False, batch_shape=(), one_hot=False, reverse=False):
        """
        Generate a dataset for a sequence copying task.
        This code is adopted from the copying.py script in the S4 repository. The original code can be found at:
        https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/src/dataloaders/datasets/copying.py

        Parameters:
        L (int): Number of padding tokens
        M (int): Number of tokens to memorize
        A (int): Alphabet size
        variable (bool): If True, selective copying task
        variable_length (bool): If True, randomize number of tokens to memorize
        batch_shape (tuple): Shape of the batch
        one_hot (bool): If True, convert the input sequence into a one-hot encoded tensor
        reverse (bool): If True, reverse the order of the target sequence
        """

        self.L = L
        self.M = M
        self.A = A
        self.variable = variable
        self.variable_length = variable_length
        self.batch_shape = batch_shape
        self.one_hot = one_hot
        self.reverse = reverse

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.variable_length:
            self.M = int(random.random() * self.M) + 1

        tokens = torch.randint(low=1, high=self.A-1, size=self.batch_shape+(self.M,), dtype=torch.long)
        if self.variable:
            total_batch = int(np.prod(self.batch_shape))
            inds = torch.stack([
                torch.randperm(self.L+self.M)[:self.M]
                for _ in range(total_batch)
                ], 0)
            inds = inds.reshape(self.batch_shape+(self.M,))
            inds, _ = inds.sort()
        else:
            inds = torch.arange(self.M).repeat(self.batch_shape+(1,))
        zeros_x = torch.zeros(self.batch_shape+(self.M+self.L,), dtype=torch.long)
        zeros_x.scatter_(-1, inds, tokens)
        markers = (self.A-1) * torch.ones(self.batch_shape+(self.M,), dtype=torch.long)

        x_ = torch.cat([zeros_x, markers], dim=-1)
        y_ = torch.cat([tokens], dim=-1)
        if self.reverse: y_ = y_.flip(-1)
        if self.one_hot: x = F.one_hot(x_, self.A).float()
        else: x = x_

        y = y_

        return x, y

class InductionData:
    def __init__(self, B: int, vocab_size: int, L: int, prefix_len:int):
        """
        Generates synthetic data of the form:
        ... S M .... S M
        where S is an 'induction token' and M is the token to memorize / recall

        vocab_size: token alphabet size
        L: total sequence length to generate
        prefix_len: region where first S should occur
        """
        assert prefix_len < L - 4
        self.B = B # batch size
        self.vocab_size = vocab_size
        self.L = L # total sequence length
        self.prefix_len = prefix_len # region where first induction token occurs
        self.vocab = list(range(self.vocab_size))
        # this token is uniquely presented as the memorization and retrieval
        # cue, appearning nowhere else
        self.ind_tok = self.vocab_size-1

    def __iter__(self):
        return self

    def __next__(self):
        # vocab_size-1 is specified because we only want the last token to
        # come up as the memorization and retrieval cue. Final sequence
        # ends up one token longer than specified because of the retrieval
        # cue.

        # the token to memorize, for each sequence
        mem = torch.randint(0, self.vocab_size-1, (self.B,1))
        batch = torch.randint(0, self.vocab_size-1, (self.B, self.L+1))
        # index where the "cue" token will be placed. This position is the
        # same across an entire batch, but varies across batches. 
        inds = torch.randint(0, self.prefix_len, (1,))*torch.ones((self.B, 1)).to(torch.int64)
        # indices where the "retrieval" token will be placed
        inds2 = torch.full((self.B,1), self.L)

        # modify the randomly generated batch to include the S and M tokens in the 
        # correct places
        batch.scatter_(1, inds, self.ind_tok)
        batch.scatter_(1, inds+1, mem)
        batch.scatter_(1, inds2, self.ind_tok)

        return batch, mem


def create_selective_copy_set(M, L, A, n_seq, path, split):
    ''' Creates a fixed dataset on which the model will be tested'''

    # return the entire dataset as one batch
    dataset = CopyData(L=L,M=M,A=A, variable=True, batch_shape=(n_seq,))
    inputs, outputs = next(dataset)
    # cast to uint8, don't need more space than this
    os.makedirs(path, exist_ok=True)
    torch.save(
        (inputs.to(torch.uint8), outputs.to(torch.uint8)), 
        os.path.join(path, f"{split}.pt"))

def create_induction_set(L, vocab_size, prefix_len, n_seq, path, split):
    ''' Creates a fixed dataset on which the model will be tested'''

    # return the entire dataset as one batch
    dataset = InductionData(B=n_seq, vocab_size=vocab_size, L=L, prefix_len=prefix_len)
    inputs, outputs = next(dataset)
    # cast to uint8, don't need more space than this
    os.makedirs(path, exist_ok=True)
    torch.save(
        (inputs.to(torch.uint8), outputs.to(torch.uint8)), 
        os.path.join(path, f"{split}.pt"))

if __name__ == "__main__":

    # Generating data for Induction Heads
    # 5 percent of the data in each "epoch". Sequence length varies, number of sequences
    # is kept constant (this isn't an autoregressive task, the number of predictions is the
    # same regardless of sequence length)
    n_seq = int((8192*8)*0.05)
    lengths = [256, 2048, 32768]
    prefix_len = 64
    vocab_size=16
    for L in lengths:
        create_induction_set(L, vocab_size, prefix_len, n_seq,"induction_heads",f"valid_{L}")
        create_induction_set(L, vocab_size, prefix_len, n_seq,"induction_heads",f"test_{L}")
        
    # Generating data for Selective Copying. L is the number of padding tokens, not the full
    # sequence length.
    L = 4096-16
    create_selective_copy_set(M=16, L=L, A=vocab_size, n_seq=2048, path="selective_copy", split=f"valid")
    create_selective_copy_set(M=16, L=L, A=vocab_size, n_seq=2048, path="selective_copy", split=f"test")