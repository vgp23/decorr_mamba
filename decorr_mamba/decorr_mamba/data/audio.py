import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import os

class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.data = torch.load(data_dir, mmap=True, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @staticmethod
    def make_sequences(L_idx: int, mu=255, n_bins=256):
        def _mu_law_encode(audio_data, mu, n_bins):
            """
            Mu-law encoding. Returns a pytorch tensor corresponding to the sequence,
            doesn't support more than 16-bit quantization (unnecessary probably)
            """
            # Apply mu-law encoding formula. Use mu=255, as this is the standard in
            # telephony and whatnot. 
            encoded_audio = np.sign(audio_data) * np.log1p(mu * np.abs(audio_data)) / np.log1p(mu)
            
            # Scale to integer range [0, n_bins-1]
            if n_bins <= 256:
                quantized = torch.tensor(
                    np.round(
                        (encoded_audio + 1) / 2 * (n_bins-1)), dtype=torch.uint8)
            else:
                quantized = torch.tensor(
                    np.round(
                        (encoded_audio + 1) / 2 * (n_bins-1)), dtype=torch.uint16)
            
            return quantized
        
        # these are the sets of lengths used in the original Mamba paper,
        # defined to keep pooling & hardware happy. Index into this instead
        # of having to remember it
        valid_L = [8192, 16384, 30720, 61440, 120832, 239616, 479232, 958464]
        assert L_idx >= 0 and L_idx<=7, "Specify L_idx between 0 and 7 (inclusive)"
        # add 1 to account for the target creation shift
        L = valid_L[L_idx] + 1

        dataset = load_dataset("krandiash/youtubemix", split="train")
        # splits pre-defined by first paper to use this dataset
        train_split = [1, 212]
        val_split = [213, 226]
        test_split = [227, 241]

        train_seqs = []
        val_seqs = []
        test_seqs = []

        tot_len = 0
        for i in tqdm(range(1, 242)):
            # each full_seq is 1 minute long 
            full_seq = dataset[i]["audio"]["array"]

            # encode the sequence using mu-law compression + quantization
            full_seq = _mu_law_encode(full_seq, mu, n_bins)

            # chunk the sequence, dropping the stuff that's left over if applicable
            truncate_len = (len(full_seq) // L) * L
            chunks = list(full_seq[:truncate_len].reshape(-1, L))

            tot_len += len(chunks)*chunks[0].shape[0]

            # figure out which of the lists to append it to
            if train_split[0] <= i and train_split[1] >= i:
                train_seqs += chunks
            elif val_split[0] <= i and  val_split[1] >= i:
                val_seqs += chunks
            elif test_split[0] <= i and  test_split[1] >= i:
                test_seqs += chunks
            else:
                raise IndexError("Index out of bounds of pre-defined splits")

        parent_dir = f"ym_length_{L-1}_mu_{mu}_n_bins_{n_bins}"

        os.makedirs(parent_dir, 
                    exist_ok=True)
        
        train_data = torch.stack(train_seqs)
        print(train_data.dtype)
        val_data = torch.stack(val_seqs)
        test_data = torch.stack(test_seqs)

        torch.save(train_data, os.path.join(parent_dir, f"train.pt"))
        torch.save(val_data, os.path.join(parent_dir, f"valid.pt"))
        torch.save(test_data, os.path.join(parent_dir, f"test.pt"))
                


