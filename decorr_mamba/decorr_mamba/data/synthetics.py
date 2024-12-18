# credit to https://github.com/hrbigelow/mamba-recall.git 
import torch

class InductionData:
    def __init__(self, B, vocab_size, L, prefix_len):
        """
        Generates synthetic data of the form:
        ... S M .... S M
        where S is an 'induction token' and M is the token to memorize / recall

        n_vocab: token alphabet size
        seq_len: total sequence length to generate
        prefix_len: region where first S should occur
        """
        assert prefix_len < L - 4
        self.B = B # batch size
        self.vocab_size = vocab_size
        self.L = L # total sequence length
        self.prefix_len = prefix_len # region where first induction token occurs
        self.vocab = list(range(self.vocab_size))
        self.ind_tok = self.vocab_size

    def __iter__(self):
        return self

    def __next__(self):
        mem = torch.randint(0, self.vocab_size, (self.B,1))
        batch = torch.randint(0, self.vocab_size, (self.B, self.L))
        # inds = t.randint(0, self.P, (self.B,1))
        inds = torch.full((self.B,1), 5)
        inds2 = torch.full((self.B,1), self.L-2)
        batch.scatter_(1, inds, self.ind_tok)
        batch.scatter_(1, inds+1, mem)
        batch.scatter_(1, inds2, self.ind_tok)
        batch.scatter_(1, inds2+1, mem)
        return dict(starts=inds, tokens=batch)