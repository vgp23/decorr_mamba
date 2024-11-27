import torch
import pickle
from torch.utils.data import DataLoader
import os

from decorr_mamba.utils.helpers import MambaArgs, TrainingArgs, DefaultArgs, LanguageDatasetMaker
from decorr_mamba.model.decorrelation import DecorrMamba
from decorr_mamba.utils.trainer import MambaTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
GPU = 6


if __name__ == "__main__":

	torch.manual_seed(5)
	torch.autograd.set_detect_anomaly(True)

	# train on song lyrics dataset for now
	print("Loading dataset...")
	with open("../../datasets/kaggle_song_lyrics_dataset/kaggle_song_lyrics_dataset.pkl", "rb") as f:
		seqs = pickle.load(f)

	print("Dataset loaded.")

	# inner model dimensionalities and batch size
	L = 64
	B = 128

	D = 256 
	N = 64

	device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")
	mamba_args = MambaArgs(N, D, n_layers=2, device=device, vocab_size=1024)

	print("Creating model...")
	decorr_model = DecorrMamba("channel_independent", mamba_args, 
		sample_frac=0.1, kappa=0.5, decorr_lr=0.002).to(device)
	print("Model created.")

	# defining the training protocol
	default_train_args = DefaultArgs()
	train_args = TrainingArgs(
	    n_epochs=20, L=L, B=B, lr=1*1.5e-3, **default_train_args.lm_args, warmup_epochs=2)
	print(f"Training with following training arguments:\n{train_args}")

	# datasets sent to the device specified in mamba_args by default 
	datasets = LanguageDatasetMaker(seqs, mamba_args, train_args, total_dataset_frac=0.001,
	                                train_split=0.8, val_split=0.2)

	# creating datasets + trainer
	train_loader = DataLoader(datasets.train_set, B, shuffle=True)
	val_loader   = DataLoader(datasets.val_set, B, shuffle=False)

	trainer = MambaTrainer(mamba_args, train_args, decorr_model)

	trainer.train(train_loader, val_loader, 
		save_checkpoints=True, save_all_checkpoints=True)