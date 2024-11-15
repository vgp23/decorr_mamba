
from model.mamba import Mamba
from model.decorrelation import DecorrMamba, DecorrConv1d, apply_to_decorr
from utils.trainer import MambaTrainer
from torch.utils.data import DataLoader
from utils.helpers import MambaArgs, TrainingArgs, DefaultArgs, LanguageDatasetMaker
from einops import einsum
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import pickle

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def generate(model,
             tokenizer,
             prompt: str,
             n_tokens_to_gen: int = 50,
             sample: bool = True,
             top_k: int = 40):
    
    model.eval()
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to("mps")
    
    for token_n in range(n_tokens_to_gen):
        with torch.no_grad():
            indices_to_input = input_ids
            next_token_logits = model(indices_to_input)[:, -1]
        
        probs = F.softmax(next_token_logits, dim=-1)
        (batch, vocab_size) = probs.shape
        
        if top_k is not None:
            (values, indices) = torch.topk(probs, k=top_k)
            probs[probs < values[:, -1, None]] = 0
            probs = probs / probs.sum(axis=1, keepdims=True)
        
        if sample:
            next_indices = torch.multinomial(probs, num_samples=1)
        else:
            next_indices = torch.argmax(probs, dim=-1)[:, None]
        
        input_ids = torch.cat([input_ids, next_indices], dim=1)

    output_completions = [tokenizer.decode(output.tolist()) for output in input_ids][0]
    
    return output_completions


if __name__ == "__main__":

	torch.manual_seed(5)
	torch.autograd.set_detect_anomaly(True)

	with open("../../kaggle_song_lyrics_dataset/kaggle_song_lyrics_dataset.pkl", "rb") as f:
	    seqs = pickle.load(f)

	L = 32
	B = 8
	D = 16
	N = 8

	device = 'cpu' # hehe 
	mamba_args = MambaArgs(N, D, n_layers=2)
	decorr_model = DecorrMamba("patch", mamba_args, 
		sample_frac=0.1, kappa=0.5, decorr_lr=0.001)


	# defining the training protocol
	default_train_args = DefaultArgs()
	train_args = TrainingArgs(
	    n_epochs=2, L=L, B=B, lr=5*1.5e-3, **default_train_args.lm_args, warmup_epochs=0)

	datasets = LanguageDatasetMaker(seqs, mamba_args, train_args, total_dataset_frac=0.001,
	                                train_split=0.5, val_split=0.5)

	# creating datasets + trainer
	train_loader = DataLoader(datasets.train_set, B, shuffle=True)
	val_loader   = DataLoader(datasets.val_set, B, shuffle=False)

	trainer = MambaTrainer(mamba_args, train_args, decorr_model)

	trainer.train(train_loader, val_loader)





	










	