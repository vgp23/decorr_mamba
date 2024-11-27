from transformers import AutoTokenizer
from decorr_mamba.utils.helpers import LanguageDatasetMaker, MambaArgs, TrainingArgs, DefaultArgs
import torch
import pickle
import numpy as np

if __name__ == "__main__":
	with open("datasets/kaggle_song_lyrics_dataset/kaggle_song_lyrics_dataset.pkl", "rb") as f:
	    seqs = pickle.load(f)

	print(np.arange(0,10))

	B = 64
	D = 128
	N = 32
	L = 8

	mamba_args = MambaArgs(N, D, n_layers=2, device="cpu")
	default_train_args = DefaultArgs()
	train_args = TrainingArgs(
	    n_epochs=20, L=L, B=B, lr=1*1.5e-3, **default_train_args.lm_args, warmup_epochs=2)


	# datasets sent to the device specified in mamba_args by default 
	datasets = LanguageDatasetMaker(seqs, mamba_args, train_args, total_dataset_frac=0.1,
	                                train_split=0.8, val_split=0.2)

	print(mamba_args.vocab_size)

	# vocab_size = 1000  # We want the top 999 tokens and the <unk> token

	# # get tokens and corresponding IDs
	# tokens = [self.tokenizer.tokenize(word) for word in raw_dataset]
	# token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]

	# # flattening them all in one list
	# token_ids = [item for sublist in token_ids for item in sublist] 



	# # limit all tokens to the top n most common. Replace the less common occurrences
	# # with unk token 
	# filtered_ids = []
	# unk_id = self.tokenizer.get_vocab()['unk']
	# # if given vocab size is not large enough to include the unk token itself, 
	# # the vocab size must be reduced by 1 to fit this in
	# if unk_id > self.model_args.vocab_size-1:
	# 	vocab_limit = self.model_args.vocab_size-2
	# else:
	# 	vocab_limit = self.model_args.vocab_size-1

	# for token_id in token_ids:
	# 	# filter the text to only include top n most common words
	# 	if token_id > vocab_limit and token_id != unk_id:
	# 		filtered_ids.append(unk_id)
	# 	else:
	# 		filtered_ids.append(token_id)

	# del token_ids 