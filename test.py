'''
Check for mamba_inner_fn
'''
import math
import torch
from torch import nn
from tqdm import tqdm
from einops import repeat
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, mamba_inner_ref
from decorr_mamba.model.sashimi_mamba import MambaBlocks, SaShiMiMamba
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import os 
from decorr_mamba.data.audio import AudioDataset
from torch.utils.data import DataLoader

# Define a random seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Set device to CUDA if available, otherwise CPU
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# config = MambaConfig(d_model=16, n_layer=3, vocab_size=256)

config = MambaConfig(d_model=64, vocab_size=256, n_layer=3)

# model = SaShiMiMamba(p=16, q=2, depth=2, 
# 				  config=config, blocks_per_stage=3).to("cuda:3")

model = MambaLMHeadModel(config=config, device=device)


# model1 = MambaBlocks(config.d_model,
#                     config.n_layer,
#                     config.d_intermediate,
#                     config.ssm_cfg,
#                     config.attn_layer_idx,
#                     config.attn_cfg,
#                     1e-5,
#                     config.rms_norm,
#                     None,
#                     config.fused_add_norm,
#                     config.residual_in_fp32,
#                     device,
#                     None)

# model2 = MambaBlocks(config.d_model,
#                     config.n_layer,
#                     config.d_intermediate,
#                     config.ssm_cfg,
#                     config.attn_layer_idx,
#                     config.attn_cfg, 
#                     1e-5,
#                     config.rms_norm,
#                     None,
#                     config.fused_add_norm,
#                     config.residual_in_fp32,
#                     device,
#                     None)

def print_memory(tag=""):
	allocated = torch.cuda.memory_allocated(4) / 1024**2
	reserved = torch.cuda.memory_reserved(4) / 1024**2
	max_allocated = torch.cuda.max_memory_allocated(4) / 1024**2
	max_reserved = torch.cuda.max_memory_reserved(4) / 1024**2

	print(f"\n[{tag}]")
	print(f"Allocated: {allocated:.2f} MB")
	print(f"Reserved: {reserved:.2f} MB")
	print(f"Max Allocated: {max_allocated:.2f} MB")
	print(f"Max Reserved: {max_reserved:.2f} MB")
	
def test_gradient_implementation(device=device):
	# Create random input tensors and parameters
	batch_size = 1
	seqlen = 1024

	# optimizer = torch.optim.AdamW(model.parameters())
	# optimizer.zero_grad()

	# dataset_dir = os.path.join("/scratch", "fast", "vicpet", "audio", "ym_length_16384_mu_255_n_bins_256")

	# train_dir = os.path.join(dataset_dir, "train.pt")	
		
	# train_dataset = AudioDataset(train_dir)
	# train_loader = DataLoader(
	# 	train_dataset, batch_size, num_workers=3, pin_memory=True, 
	# 	drop_last=False, shuffle=True)
	
	# ins = next(iter(train_loader))[:,:-1].long().to(device)
	# print(type(ins))
	ins = torch.randint(0, config.vocab_size, 
					(batch_size, seqlen), device=device).long()
	print(type(ins))

	outs = model(ins).logits

	criterion = nn.CrossEntropyLoss()
	torch.autograd.set_detect_anomaly(True)


	# Create dummy targets
	target = torch.randint(0, config.vocab_size, (batch_size, seqlen)).to(device)
	outs = outs.view(-1, config.vocab_size)
	target = target.view(-1)
	loss = criterion(outs, target)

	# Backward pass through mamba_inner_fn

	loss.backward()

	print("DONE")


# Call the test function
test_gradient_implementation(device)