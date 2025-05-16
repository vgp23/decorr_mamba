import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import json
import numpy as np
import torch.utils.data.distributed
from functools import partial
import time
import matplotlib.pyplot as plt
import random

from decorr_mamba.utils.helpers import TrainingArgs
from decorr_mamba.model.decorrelation import DecorrSaShiMiMamba
from decorr_mamba.utils.trainer import MambaTrainer
from decorr_mamba.data.audio import AudioDataset

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import multiprocessing

import argparse
import wandb

from decorr_mamba.model.sashimi_mamba import SaShiMiMamba
from mamba_ssm.models.config_mamba import MambaConfig
# wandb.login()

# tokenizer throws a warning if this isn't set here
os.environ["TOKENIZERS_PARALLELISM"] = 'false'


def print_args(args):
	print("\nTraining with the following parameters:")
	for arg, value in args.items():
		print(f"{arg.upper()}: {value}")

def parse_none(value):
    """ Custom type that interprets 'None' as a NoneType object """
    if value == "None":
        return None
    return float(value)

def get_all_args():

	parser = argparse.ArgumentParser(description="Specify global training and model parameters.")

	# loading the dictionary for default values. By default, the training
	# scheme is simple (no weight decay, learning rate schedulers, or gradient
	# clipping)
	with open( "language_defaults.json") as json_file:
		default_args = json.load(json_file)

	# -------------------------------------------
	# universal model specifications without default values
	parser.add_argument("--d", type=int, required=True, help="Token embedding dimensionality")
	# parser.add_argument("--model_n", type=int, required=True, help="Hidden state dimensionality")
	parser.add_argument("--use_decorr", action="store_true", help="Using decorrelation or not")
	# --------------------------------------------------------------------
	# model specifications with default values specified in original Mamba code
	parser.add_argument("--vocab_size", type=int, default=default_args["vocab_size"], 
		help="Vocabulary size for tokenizer and embedding layer")
	# -----------------------------------------
	# training constants without default values
	parser.add_argument("--optimizer", type=str, help="Optimizer to train with. Either \'adam\' or \'soap\'.",
					 default="adam")
	parser.add_argument("--n_steps", type=int, required=True, help="Number of epochs/gradient descent steps to train for")	
	parser.add_argument("--seq_len", type=int, required=True, help="Training sequence length")
	parser.add_argument("--b", type=int, required=True, help="Batch size")
	parser.add_argument("--warmup_steps", type=int, help="Number of linear warmup epochs/gradient descent steps used with scheduler")
	parser.add_argument("--backprop_lr", type=float, help="Learning rate for backpropagation")
	parser.add_argument("--n_val", type=int, default=10, help="Number of validation steps in the entire process.")	
	parser.add_argument("--dataset", type=str, required=True, help="Dataset folder name")	
	parser.add_argument("--save_checkpoints", action="store_true", default=False, 
					 help="Stores epoch state_dict at each epoch")
	parser.add_argument("--metric", type=str, default="ppl", help="Evaluation metric, either perplexity (ppl) or bits per byte (bpb)")
	
	# parser.add_argument("--save_all_checkpoints", action="store_true", default=False,
	# 				 help="Stores state_dicts at each epoch, requires save_checkpoints to also be true")
	
	parser.add_argument("--use_amp", action="store_true", help="Training with automatic mixed precision or not")
	parser.add_argument("--log_freq", type=int, default=50, help="Number of batch updates between wandb logging events")
	parser.add_argument("--val_sched_base", type=int, default=20, help="Base of the logarithm determining the spacing of validation events. Larger = more in initial stages.")	
	parser.add_argument("--trial_id", type=int, help="ID of trial")
	parser.add_argument("--skip_init_val", action="store_true", help="Skips validation of untrained models")
	parser.add_argument("--seed", type=int, 
					 help="Sets the same seed for model initialization and dataloaders. If not specified, random seed chosen each run.")
	parser.add_argument("--machine", type=str, default="em", help="Machine to train with. Either \"em\' or \"snellius\"")
	# --------------------------------------
	# training constants with default values
	parser.add_argument("--adam_beta", type=tuple, default=default_args["adam_beta"], 
		help="Adam beta values")
	parser.add_argument("--adam_epsilon", type=float, default=default_args["adam_epsilon"], 
		help="Adam epsilon value")
	parser.add_argument("--gradient_clip", type=parse_none, default=default_args["gradient_clip"], 
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
	# parser.add_argument("--train_split", type=float, default=0.8,
	# 	help="Fraction of total_dataset_frac to use as the training split")
	# parser.add_argument("--val_split", type=float, default=0.1,
	# 	help="Fraction of total_dataset_frac to use as the validation split")

	# -----------------------------------------------------------------------

	args, _ = parser.parse_known_args()
	# arguments specific to decorrelation models
	if args.use_decorr:	
		parser.add_argument("--decorr_sample_frac", type=float,
			help="Fraction of each batch to sample for computing decorrelation layer gradients")
		parser.add_argument("--decorr_kappa", type=float, default=0.0, 
			help="Value of kappa for decorrelation layer gradient computation, if applicable")	
			
		parser.add_argument("--decorr_lr", type=float,
			help="Learning rate for decorrelation layers")	
		# parser.add_argument("--demeaning_lr", type=float,
		# 	help="Learning rate for EMA estimate of data means, required for de-meaning in decorrelation layers.")	

		# parser.add_argument("--use_gain_scaling", 
		# 	action="store_true",
		# 	help="Whether to use gain scaling when updating decorrelation matrices")
		# parser.add_argument("--use_demeaning", 
		# 	action="store_true",
		# 	help="Whether to use de-meaning prior to decorrelation")	

	args = parser.parse_args()

	# assert args.train_split + args.val_split < 1.0, \
	# 	"Training and validation splits sum to 1, leaving no testing split left over."

	return args


# DDP-specific helper functions
def setup_ddp():
	dist.init_process_group(backend='nccl', init_method='env://')
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)
	return local_rank

def cleanup_ddp():
	dist.destroy_process_group()

def get_shared_trial_seed(local_seed: int, rank):
	"""Broadcast a seed from rank 0 to all other processes."""
	seed_tensor = torch.tensor([local_seed], dtype=torch.int32, device=f"cuda:{rank}")
	if dist.is_initialized():
		dist.broadcast(seed_tensor, src=0)
	return seed_tensor.item()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config):
	# NB whatever values were specified for the learning rates are overwritten
	# by a sweep!

	ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
	rank = int(os.environ.get("RANK", 0))
	world_size = int(os.environ.get("WORLD_SIZE", 1))
	local_rank = setup_ddp() if ddp else 0

	is_main = rank == 0

	# if no random seed provided, generate one and feed it to everything. If 
	# we're using ddp, this also has to go to the samplers used in the training
	# loop. 
	if not config["seed"]:
		if ddp:
			if is_main:
				seed = int(time.time()) % 2**31 
			else:
				seed = 0  # placeholder
			# broadcast from 0 to other processes
			seed = get_shared_trial_seed(seed, local_rank)
		else:
			seed = int(time.time()) % 2**31 
		
		config["seed"] = seed
	else:
		seed = config["seed"]

	seed_everything(seed)

	if is_main:
		wandb.init(project="decorr_mamba", config=config)
		if wandb.config.trial_id:
			wandb.run.name = f"run-audio-bplr-{wandb.config.backprop_lr}-dlr-{wandb.config.decorr_lr}-trial-{wandb.config.trial_id}"
		else:
			wandb.run.name = f"run-audio-bplr-{wandb.config.backprop_lr}-dlr-{wandb.config.decorr_lr}"
		wandb.run.save()
		print_args(config)

	else:
		# This is only true if we don't pass anything in, which we only do when
		# running a sweep (wandb will automatically overwrite this stuff when
		# not running in disabled mode, but it causes problems here)
		if config["backprop_lr"] == None:
			del config['backprop_lr']
		if config["decorr_lr"] == None:
			del config['decorr_lr']
		if config["decorr_sample_frac"] == None:
			del config["decorr_sample_frac"]

		wandb.init(mode="disabled", config=config)

	# dataset loading
	if wandb.config.machine == "snellius":
		dataset_dir = os.path.join(os.environ["TMPDIR"], wandb.config["dataset"])
	elif wandb.config.machine == "em":
		dataset_dir = os.path.join("/scratch", "fast", "vicpet", "audio", wandb.config["dataset"])

	train_dir = os.path.join(dataset_dir, "train.pt")
	val_dir = os.path.join(dataset_dir, "valid.pt")	
		
	train_dataset = AudioDataset(train_dir)
	val_dataset = AudioDataset(val_dir)

	if is_main:
		print("\nDataset loaded. ")

	# Shortening dataset, if needed
	if not wandb.config["total_dataset_frac"] == 1.0:
		train_len = int(wandb.config["total_dataset_frac"]*len(train_dataset))
		train_dataset = train_dataset[:train_len]
		val_len = int(wandb.config["total_dataset_frac"]*len(val_dataset))
		val_dataset = val_dataset[:val_len]	
	
	if is_main:
		print(f"Train dataset length: {len(train_dataset)}")
		print(f"Val dataset length: {len(val_dataset)}\n")	

	# creating dataloaders
	if ddp:
		train_sampler = torch.utils.data.distributed.DistributedSampler(
			train_dataset, num_replicas=world_size, rank=rank, shuffle=True
		)
		val_sampler = torch.utils.data.distributed.DistributedSampler(
			val_dataset, num_replicas=world_size, rank=rank, shuffle=False
		)
		loader_shuffle = False	
		
	else:
		train_sampler = val_sampler = None
		loader_shuffle = True

	train_loader = DataLoader(
		train_dataset, int(wandb.config.b//world_size), num_workers=3, pin_memory=True, 
		drop_last=False, shuffle=loader_shuffle, sampler=train_sampler)
		
	val_loader = DataLoader(
		val_dataset, int(wandb.config.b//world_size), num_workers=3, pin_memory=True, 
		drop_last=False, sampler=val_sampler)
	
	if is_main:
		print(f"\nTrain loader length: {len(train_loader)}")
		print(f"Val loader length: {len(val_loader)}")	

	# model creation
	mamba_args = MambaConfig(d_model=wandb.config.d,
						   vocab_size=wandb.config.vocab_size)


	DEVICE = torch.device(f'cuda:{local_rank}' if \
					torch.cuda.is_available() else 'cpu')

	if is_main:
		print(f"\nTraining with device: {DEVICE}")

	if wandb.config.use_decorr and wandb.config.decorr_lr > 0:
		if is_main:
			print(f"\nCreating model WITH DECORRELATION...")
		model = DecorrSaShiMiMamba(kappa=wandb.config.decorr_kappa,
							sample_frac=wandb.config.decorr_sample_frac,
							decorr_lr=wandb.config.decorr_lr,
							config=mamba_args, p=16, q=2, depth=2,
							blocks_per_stage=3, device=DEVICE)
		
		
	# if decorr_lr is set to 0, create a model without decorrelation 
	# (overriding the use_decorr argument).
	# we do this so that it's easy to run sweeps for both types of model
	# (can run a single sweep and cover them both)		
	else:
		if is_main:
			print("\nCreating model WITHOUT DECORRELATION...")
		model = SaShiMiMamba(config=mamba_args, p=16, q=2, depth=2, 
					   blocks_per_stage=3, device=DEVICE)	

	if is_main:
		print("Model created.\n")
		print(model)

		total_params = sum(p.numel() for p in model.parameters())
		print(f"Total number of parameters: {total_params}\n")

	if ddp:
		model = DDP(model, device_ids=[local_rank])
		# sync all parameters across processes
		for param in model.parameters():
			torch.distributed.broadcast(param.data, src=0)

	wandb.watch(model, log="all", log_freq=wandb.config.log_freq)

	# defining the training protocol
	train_args = TrainingArgs(n_steps=wandb.config.n_steps,
							  L=wandb.config.seq_len,
							  B=wandb.config.b,
							  lr=wandb.config.backprop_lr,
							  adam_beta=wandb.config.adam_beta,
							  adam_epsilon=wandb.config.adam_epsilon,
							  gradient_clip=wandb.config.gradient_clip,
							  weight_decay=wandb.config.weight_decay,
							  use_lr_sched=wandb.config.use_lr_sched,
							  min_lr=wandb.config.min_lr,
							  warmup_steps=wandb.config.warmup_steps,
							  ddp=ddp,
							  optimizer=wandb.config.optimizer)
	
	if wandb.config.use_lr_sched and is_main:
		train_args.show_lr_schedule()
		plt.savefig("schedule.png")

	trainer = MambaTrainer(mamba_args, train_args, model, rank, local_rank)

	trainer.train_sequence_steps(train_loader, val_loader, 
		use_amp=False, log_freq=wandb.config.log_freq, 
		train_backprop=True, train_decorr=True, 
		save_checkpoints=wandb.config.save_checkpoints, 
		n_val=wandb.config.n_val, skip_init_val=wandb.config.skip_init_val,
		datashuffle_seed=seed, metric=wandb.config.metric, val_sched_base=wandb.config.val_sched_base)
	
	if ddp:
		cleanup_ddp()
	
	wandb.finish()

if __name__ == "__main__":
	torch.autograd.set_detect_anomaly(True)
	multiprocessing.set_start_method('spawn', force=True)

	config = vars(get_all_args())
	main(config)


