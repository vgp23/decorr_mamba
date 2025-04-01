import torch
import pandas as pd
import os
import gzip
from Bio import SeqIO
from torch.utils.data import Dataset
from tqdm import tqdm

class DNADataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Collect all sequence file paths recursively
        self.data = torch.load(data_dir, mmap=True, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def make_sequences(fasta_file: str, bed_file: str, split: str, L: int=2**17, 
            include_lowercase: bool=False):
        """Generates and saves DNA sequences as tensors from a FASTA file and BED file.

        All tensors saved in torch.uint8 because more is unnecessary (only 5 or 9
        different tokens are possible, depending on include_lowercase). All batches
        are cast back into torch.float32 during training. 

        Args:
            fasta_file (str): Path to the gzipped FASTA file containing genomic sequences.
            bed_file (str): Path to the BED file defining genomic regions of interest.
            split (str): Dataset split to process ('train', 'valid', or 'test').
            L (int, optional): Length of sequence segments. Default is 2^17.
            include_lowercase (bool, optional): Whether to distinguish between lowercase and uppercase bases. 
                Default is False.

        Raises:
            AssertionError: If `split` is not one of 'train', 'valid', or 'test'.
        """
        # import genome
        with gzip.open(fasta_file, 'rt') as handle:
            genome = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))

        # import sequence data
        assert split == "train" or split == "valid" or split =="test", \
            "invalid split specified"
        
        df = pd.read_csv(bed_file, sep="\t", header=None, comment="#")
        df.columns = ["chrom", "start", "end", "split"]
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)

        # filter data to contain only relevant split
        df = df[df["split"]==split]
        chrom_keys = df["chrom"] # chromosome of sequence
        idxs = df.to_numpy()[:,1:3] # chromosome indices of sequence

        # make distinction between repeats and main coding regions
        if include_lowercase:
            NUC_TO_INT = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4,
                               'a': 5, 't': 6, 'g': 7, 'c': 8}
        else:
            NUC_TO_INT = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4,
                               'a': 0, 't': 1, 'g': 2, 'c': 3}
                        
        segments = []
        print("Making sequences:")
        # if L is shorter than or equal 2^17, take each of the pre-defined segments
        # and divide them into as many non-overlapping sub-segments as possible
        if L <= 2**17:
            for chrom, (start_idx, end_idx) in tqdm(
                zip(chrom_keys, idxs), total=len(idxs)):

                # pull out the full sequence
                full_seq = str(genome[chrom].seq[start_idx:end_idx])
                # split the sequence up
                for i in range(0, 2**17, L):
                    if i+L <= 2**17:
                        seq_tensor = torch.tensor(
                            [NUC_TO_INT.get(base, 4) for base in full_seq[i:i+L]], 
                            dtype=torch.uint8)        
                        segments.append(seq_tensor)
        # if L is larger than 2^17, create two samples for each pre-defined segment,
        # one which starts where the pre-defined segment begins, and another which 
        # ends where the pre-defined segment ends
        else:
            for chrom, (start_idx, end_idx) in zip(chrom_keys, idxs):
                seq_1_tensor = torch.tensor(
                    [NUC_TO_INT.get(base, 4) for base in 
                        str(genome[chrom].seq[start_idx:start_idx+L])], 
                    dtype=torch.uint8)
                segments.append(seq_1_tensor)
                # we can't make the second sample if this is close to the beginning
                # of the chromosome, there may not be enough base pairs available
                # before the start index
                if end_idx - L >= 0:
                    seq_2_tensor = torch.tensor(
                        [NUC_TO_INT.get(base, 4) for base in 
                            str(genome[chrom].seq[end_idx-L:end_idx])], 
                        dtype=torch.uint8)
                    segments.append(seq_2_tensor)
        
        # Convert this list to a giant tensor
        tensor_data = torch.stack(segments)

        # make necessary directories
        parent_dir = f"hg38_length_{L}_include_lowercase_{include_lowercase}"
        os.makedirs(parent_dir, 
                    exist_ok=True)
        
        # save!
        torch.save(tensor_data, os.path.join(parent_dir, f"{split}.pt"))

if __name__ == "__main__":
    split = "train"
    L = 1024
    include_lowercase = False

    bed_file = "data_human_sequences.bed"
    fasta_file = os.path.join("data", "hg38.fa.gz")

    DNADataset.make_sequences(fasta_file, bed_file, split=split, 
                        L=L, include_lowercase=include_lowercase)
    

