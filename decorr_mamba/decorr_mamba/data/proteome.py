import torch
import pandas as pd
import os
import gzip
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class ProteomeDataset(Dataset):
    def __init__(self, data_dir):
        self.data = torch.load(data_dir, mmap=True, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def make_sequences(fasta_file: str, cutoff_len: int=50, seq_len:int=384):
        # not the most efficient, but maybe I want to work with the keys later
        seq_dict = {record.id: str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")}
        keys = list(seq_dict.keys())
        # most proteins are 300-400 amino acids long. If shorter than 50,
        # discard completely. Otherwise, pad up to 384 (128 x 3), or truncate down
        # to this length

        all_AAs = []
        for key in keys:
            protein_set = set(list(seq_dict[key]))
            for aa in protein_set:
                if not aa in all_AAs:
                    all_AAs.append(aa)

        # save 0 as the padding index
        AA_TO_INT = dict(zip(all_AAs, np.arange(1, len(all_AAs) + 1)))

        # iterate through all proteins and create sequences
        seqs = []
        for key in tqdm(keys):
            seq = seq_dict[key]
            if len(seq) > cutoff_len:
                # if shorter than pre-set length, pad with zeros at the end
                if len(seq) <= seq_len:
                    padded_seq_tensor = torch.zeros(seq_len, dtype=torch.uint8)
                    seq_tensor = torch.tensor(
                                        [AA_TO_INT.get(aa, -1) for aa in seq], 
                                        dtype=torch.uint8)    
                    padded_seq_tensor[:len(seq_tensor)] = seq_tensor
                    seqs.append(padded_seq_tensor)

                # if longer than pre-set length, cut up using sliding window
                else:
                    for i in range(len(seq) - seq_len + 1):
                        seq_tensor = torch.tensor(
                                            [AA_TO_INT.get(aa, -1) for aa in seq[i:i+seq_len]], 
                                            dtype=torch.uint8)
                        seqs.append(seq_tensor)

        # split into splits
        seqs = torch.stack(seqs)
        train_ratio = 0.8
        val_ratio = 0.1
        indices = torch.randperm(seqs.shape[0])

        train_end = int(train_ratio * seqs.shape[0])
        val_end = train_end + int(val_ratio * seqs.shape[0])

        train_data = seqs[indices[:train_end]]
        val_data = seqs[indices[train_end:val_end]]
        test_data = seqs[indices[val_end:]]

        parent_dir = f"hp_length_{seq_len}"
        os.makedirs(parent_dir, 
                    exist_ok=True)

        torch.save(train_data, os.path.join(parent_dir, f"train.pt"))
        torch.save(val_data, os.path.join(parent_dir, f"valid.pt"))
        torch.save(test_data, os.path.join(parent_dir, f"test.pt"))

    
