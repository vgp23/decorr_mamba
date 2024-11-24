import torch
import pickle
from utils.helpers import MambaArgs, TrainingArgs, DefaultArgs, LanguageDatasetMaker
from torch.utils.data import DataLoader
from model.mamba import Mamba 
from utils.trainer import MambaTrainer
from model.decorrelation import DecorrMamba


if __name__ == "__main__":

	torch.manual_seed(5)
	torch.autograd.set_detect_anomaly(True)

	# train on song lyrics dataset for now
	print("Loading dataset")
	with open("../../kaggle_song_lyrics_dataset/kaggle_song_lyrics_dataset.pkl", "rb") as f:
	    seqs = pickle.load(f)
	print("Dataset loaded")

	# inner model dimensionalities and batch size
	L = 32
	B = 10
	D = 16
	N = 8

	device = 'cpu' # hehe 
	mamba_args = MambaArgs(N, D, n_layers=2)

	print("Creating model")
	decorr_model = DecorrMamba("patch", mamba_args, 
		sample_frac=1.0, kappa=0.5, decorr_lr=0.001)


	# defining the training protocol
	default_train_args = DefaultArgs()
	train_args = TrainingArgs(
	    n_epochs=20, L=L, B=B, lr=5*1.5e-3, **default_train_args.lm_args, warmup_epochs=0)
	print(f"Training with following training arguments\n{train_args}")

	datasets = LanguageDatasetMaker(seqs, mamba_args, train_args, total_dataset_frac=0.001,
	                                train_split=0.5, val_split=0.5)

	# creating datasets + trainer
	train_loader = DataLoader(datasets.train_set, B, shuffle=True)
	val_loader   = DataLoader(datasets.val_set, B, shuffle=False)

	trainer = MambaTrainer(mamba_args, train_args, decorr_model)

	trainer.train(train_loader, val_loader)