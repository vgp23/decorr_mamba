import torch
import numpy as np
import pickle as pkl
import os

# credit to https://github.com/hrbigelow/mamba-recall.git 
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
        # come up as the memorization and retrieval cue

        # the token to memorize, for each sequence
        mem = torch.randint(0, self.vocab_size-1, (self.B,1))
        batch = torch.randint(0, self.vocab_size-1, (self.B, self.L))
        # index where the first "S" token will be placed. This position is the
        # same across an entire batch, but varies across batches. 
        inds = torch.randint(0, self.prefix_len, (1,))*torch.ones((self.B, 1)).to(torch.int64)
        # indices where the second "S" token will be placed
        inds2 = torch.full((self.B,1), self.L-2)

        # modify the randomly generated batch to include the S and M tokens in the 
        # correct places
        batch.scatter_(1, inds, self.ind_tok)
        batch.scatter_(1, inds+1, mem)
        batch.scatter_(1, inds2, self.ind_tok)
        batch.scatter_(1, inds2+1, mem)

        return batch
    
def create_validation_set(vocab_size, L, prefix_len, n_seq, path):
    ''' Creates a fixed validation dataset on which the model will be tested'''

    # return the entire dataset as one batch
    dataset = InductionData(1, vocab_size, L, prefix_len)
    dataset_iter = iter(dataset)
    dataset_storage = torch.zeros(n_seq, L)
    for i in range(n_seq):
        dataset_storage[i, :] = next(dataset_iter)

    with open(path, 'wb') as file:
        # Serialize the array and save it to the file
        pkl.dump(dataset_storage.numpy(), file)

if __name__ == "__main__":

    dataset_name = "val"
    vocab_size = 16
    n_seq = int(2048*8/10) # 10 percent of size of training "dataset"
    prefix_len = 64
    L = 256

    # construct the full path for the pickle file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "..", "..", "..", "datasets", "induction_heads")
    os.makedirs(dataset_dir, exist_ok=True)
    
    dataset_name = f"{dataset_name}_vocabsize_{vocab_size}_nseq_{n_seq}_L_{L}_prefixlen_{prefix_len}"
    file_path = os.path.join(dataset_dir, f'{dataset_name}.pkl')

    # create dataset
    create_validation_set(vocab_size, L, prefix_len, n_seq, file_path)


