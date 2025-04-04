import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
import json
from functools import partial
import pickle as pkl

from decorr_mamba.utils.helpers import MambaArgs, TrainingArgs, LanguageDatasetMaker
from decorr_mamba.model.decorrelation import DecorrMamba
from decorr_mamba.utils.trainer import MambaTrainer
from decorr_mamba.model.mamba import Mamba
from decorr_mamba.data.synthetics import InductionData
import os
import multiprocessing

import argparse
import wandb
# wandb.login()

# tokenizer throws a warning if this isn't set here
os.environ["TOKENIZERS_PARALLELISM"] = 'false'


def print_args(args):
    print("\nTraining with the following parameters:")
    for arg, value in vars(args).items():
        print(f"{arg.upper()}: {value}")


def get_all_args():

	parser = argparse.ArgumentParser(description="Specify global training and model parameters.")

	# loading the dictionary for default values. By default, the training
	# scheme is simple (no weight decay, learning rate schedulers, or gradient
	# clipping)
	with open( "standard_defaults.json") as json_file:
		default_args = json.load(json_file)

	# -------------------------------------------
	# universal model specifications without default values
	parser.add_argument("--d", type=int, required=True, help="Token embedding dimensionality")
	parser.add_argument("--n", type=int, required=True, help="Hidden state dimensionality")
	parser.add_argument("--use_decorr", action="store_true", help="Using decorrelation or not")
	parser.add_argument("--n_layers", type=int, required=True, help="Number of Mamba blocks")
	parser.add_argument("--n_epoch_steps", type=int, help="Number of steps in each epoch for synthetic training")
	# --------------------------------------------------------------------
	# model specifications with default values specified in original Mamba code
	parser.add_argument("--vocab_size", type=int, default=default_args["vocab_size"], 
		help="Vocabulary size for tokenizer and embedding layer")
	parser.add_argument("--pad_vocab_size_multiple", type=int, default=default_args["pad_vocab_size_multiple"], 
		help="Makes vocabulary size a multiple of this value")
	parser.add_argument("--expansion_factor", type=int, default=default_args["expansion_factor"], 
		help="Factor by which input embeddings are upscaled")
	parser.add_argument("--conv_1d_size", type=int, default=default_args["conv_1d_size"], 
		help="Size of the 1D convolution")
	parser.add_argument("--use_conv_bias",
					type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], 
					default=default_args["conv_bias"], 
		help="Whether to use bias in the convolutional layer")
	parser.add_argument("--use_general_bias", 
					type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], 
					default=default_args["general_bias"], 
		help="Whether to use bias in input and output projections")
	# delta projection weight initialization	
	parser.add_argument("--delta_init", type=str, default=default_args["delta_init"], 
		choices=["random", "constant"], 
		help="Initialization method for delta projection weights")
	parser.add_argument("--delta_scale", type=float, default=default_args["delta_scale"], 
		help="Scale for delta projection weights")
	
	def str_or_int(value):
		# needed because delta_rank can be either a string or an integer
		try:
			return int(value)  # try converting to an integer
		except ValueError:
			return value  # return as a string if conversion fails	
		
	parser.add_argument("--delta_rank", type=str_or_int, default=default_args["delta_rank"], 
		help="Rank for delta projection weights")

	# initialization for biases to delta projection layer
	parser.add_argument("--delta_min", type=float, default=default_args["delta_min"], 
		help="Minimum value for delta projection biases")
	parser.add_argument("--delta_max", type=float, default=default_args["delta_max"], 
		help="Maximum value for delta projection biases")
	parser.add_argument("--delta_init_floor", type=float, default=default_args["delta_max"], 
		help="Initialization floor for delta projection biases")

	# -----------------------------------------
	# training constants without default values
	parser.add_argument("--n_epochs", type=int, required=True, help="Number of epochs to train for")	
	parser.add_argument("--l", type=int, required=True, help="Training sequence length")
	parser.add_argument("--b", type=int, required=True, help="Batch size")
	parser.add_argument("--warmup_epochs", type=int, help="Number of linear warmup epochs used with scheduler")
	parser.add_argument("--backprop_lr", type=float, required=True, help="Learning rate for backpropagation")
	parser.add_argument("--gpu", type=int, default=int(os.getenv("CUDA_VISIBLE_DEVICES", "0")), 
					 help="ID of the GPU to train with")
	parser.add_argument("--dataset", type=str, required=True, help="Dataset folder name")	
	parser.add_argument("--save_checkpoints", action="store_true", default=False, 
					 help="Stores epoch state_dict when validation performance improves")
	parser.add_argument("--save_all_checkpoints", action="store_true", default=False,
					 help="Stores state_dicts at each epoch, requires save_checkpoints to also be true")
	parser.add_argument("--use_amp", action="store_true", help="Training with automatic mixed precision or not")

	# --------------------------------------
	# training constants with default values
	parser.add_argument("--adam_beta", type=tuple, default=default_args["adam_beta"], 
		help="Adam beta values")
	parser.add_argument("--adam_epsilon", type=float, default=default_args["adam_epsilon"], 
		help="Adam epsilon value")
	parser.add_argument("--gradient_clip", type=float, default=default_args["gradient_clip"], 
		help="Gradient clipping value")
	parser.add_argument("--weight_decay", type=float, default=default_args["weight_decay"], 
		help="Weight decay value")
	parser.add_argument("--use_lr_sched",
					type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], 
					default=default_args["use_lr_sched"], 
		help="Whether to use a learning rate scheduler for backpropagation")
	parser.add_argument("--min_lr", type=float, default=default_args["min_lr"],
		help="Minimum backpropagation learning rate value for the scheduler to decay to")
	parser.add_argument("--total_dataset_frac", type=float, default=1.0, 
		help="Fraction of total dataset to train on")
	parser.add_argument("--train_split", type=float, default=0.8,
		help="Fraction of total_dataset_frac to use as the training split")
	parser.add_argument("--val_split", type=float, default=0.1,
		help="Fraction of total_dataset_frac to use as the validation split")

	# -----------------------------------------------------------------------

	args, _ = parser.parse_known_args()
	# arguments specific to decorrelation models
	if args.use_decorr:
		parser.add_argument("--decorr_conv_1d_mode", type=str, default="channel_independent", 
			choices=["patch", "token", "channel_independent", "channel_universal"], 
			help="Decorrelation mode to use for the conv1d layers, if applicable")
		parser.add_argument("--unfuse_decorr_operations", 
			action="store_true", default=False, 
			help="Controls fusing of decorrelation and standard layer operations for debugging")	
		parser.add_argument("--decorr_sample_frac", type=float, default=0.1,
			help="Fraction of each batch to sample for computing decorrelation layer gradients")
		parser.add_argument("--decorr_kappa", type=float, default=0.5, 
			help="Value of kappa for decorrelation layer gradient computation, if applicable")	
			
		parser.add_argument("--decorr_lr", type=float, 
			help="Learning rate for decorrelation layers")	
		parser.add_argument("--demeaning_lr", type=float,
			help="Learning rate for EMA estimate of data means, required for de-meaning in decorrelation layers.")	

		parser.add_argument("--use_gain_scaling", 
			action="store_true",
			help="Whether to use gain scaling when updating decorrelation matrices")
		parser.add_argument("--use_demeaning", 
			action="store_true",
			help="Whether to use de-meaning prior to decorrelation")	

	args = parser.parse_args()

	assert args.train_split + args.val_split < 1.0, \
		"Training and validation splits sum to 1, leaving no testing split left over."

	return args

def main(args):
	# NB whatever values were specified for the learning rates are overwritten
	# by a sweep!

	# RESUMING TRAINING AFTER BROKEN PIPE
	wandb.init(project="decorr_mamba", id="b441a1t8", resume="must") # can add notes and tags in here!
	wandb.config.update(vars(args))

	print_args(args)
	
	# validation dataset loading
	print("\nLoading validation dataset...")

	with open(
		os.path.join("..", "..", "..", "datasets",
			   "induction_heads", 
			   f"{wandb.config.dataset}.pkl"), "rb") as f:
		seqs = pickle.load(f)

	print("Dataset loaded. ")

	DEVICE = torch.device(f'cuda' if \
					   torch.cuda.is_available() else 'cpu')

	print(f"\nTraining with device: {DEVICE}")

	# model creation
	mamba_args = MambaArgs(N=wandb.config.n, 
						   D=wandb.config.d,
						   n_layers=wandb.config.n_layers, 
						   vocab_size=wandb.config.vocab_size, 
						   pad_vocab_size_multiple=wandb.config.pad_vocab_size_multiple,
						   device=DEVICE,
						   expansion_factor=wandb.config.expansion_factor,
						   conv_1d_size=wandb.config.conv_1d_size,
						   conv_bias=wandb.config.use_conv_bias,
						   general_bias=wandb.config.use_general_bias,
						   delta_init=wandb.config.delta_init,
						   delta_scale=wandb.config.delta_scale,
						   delta_rank=wandb.config.delta_rank,
						   delta_min=wandb.config.delta_min,
						   delta_max=wandb.config.delta_max,
						   delta_init_floor=wandb.config.delta_init_floor)


	if wandb.config.use_decorr and wandb.config.decorr_lr > 0:
		print(f"\nCreating model WITH DECORRELATION...")
		model = DecorrMamba(conv_1d_mode=wandb.config.decorr_conv_1d_mode,
							model_args=mamba_args,
						    fuse=bool(1-wandb.config.unfuse_decorr_operations),
						    kappa=wandb.config.decorr_kappa,
						    sample_frac=wandb.config.decorr_sample_frac,
						    decorr_lr=wandb.config.decorr_lr,
						    use_gain_scaling=wandb.config.use_gain_scaling,
						    use_demeaning=wandb.config.use_demeaning,
						    demeaning_lr=wandb.config.demeaning_lr
						    ).to(DEVICE)
		
	# if decorr_lr is set to 0, create a model without decorrelation 
	# (overriding the use_decorr argument).
	# we do this so that it's easy to run sweeps for both types of model
	# (can run a single sweep and cover them both)		
	else:
		print("\nCreating model WITHOUT DECORRELATION...")
		model = Mamba(model_args=mamba_args).to(DEVICE)

	print("Model created.\n")

	# load in the checkpoint state_dict... lovely
	state_dict_path = os.path.join(
		os.getcwd(), "checkpoints", "epoch_4.pt")
	state_dict = torch.load(state_dict_path, weights_only=True)
	model.load_state_dict(state_dict)

	wandb.watch(model, log="all", log_freq=10)

	# defining the training protocol
	train_args = TrainingArgs(n_epochs=wandb.config.n_epochs,
							  L=wandb.config.l,
							  B=wandb.config.b,
							  lr=wandb.config.backprop_lr,
							  adam_beta=wandb.config.adam_beta,
							  adam_epsilon=wandb.config.adam_epsilon,
							  gradient_clip=wandb.config.gradient_clip,
							  weight_decay=wandb.config.weight_decay,
							  use_lr_sched=wandb.config.use_lr_sched,
							  min_lr=wandb.config.min_lr,
							  warmup_epochs=wandb.config.warmup_epochs)

	# loading in the validation data and creating validation loader
	val_dataset = TensorDataset(torch.tensor(seqs, dtype=torch.long))
	
	val_loader = DataLoader(
		val_dataset, wandb.config.b, shuffle=False, num_workers=8, pin_memory=True)

	
	# for training, we generate new data each time
	train_loader = InductionData(
		B=wandb.config.b, 
		vocab_size=wandb.config.vocab_size,
		L=wandb.config.l,
		prefix_len=int(0.25*wandb.config.l)
		)

	trainer = MambaTrainer(mamba_args, train_args, model, DEVICE)

	trainer.train_induction(iter(train_loader), val_loader, 
			   save_checkpoints=wandb.config.save_checkpoints, 
			   save_all_checkpoints=wandb.config.save_all_checkpoints,
			   n_epoch_steps=wandb.config.n_epoch_steps,
			   use_amp=wandb.config.use_amp)

if __name__ == "__main__":

	torch.manual_seed(5)
	# torch.autograd.set_detect_anomaly(True)
	multiprocessing.set_start_method('spawn', force=True)

	all_args = get_all_args()
	main(all_args)
