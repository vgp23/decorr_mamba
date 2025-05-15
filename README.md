# Decorrelated Mamba

Improving the convergence speed of Mamba architectures using feature decorrelation methods. This repository serves primarily as internal version control, and is aligned with the experiments I am carrying out in my thesis on the original [Mamba](https://arxiv.org/abs/2312.00752), [Mamba2](https://arxiv.org/abs/2405.21060), and incorporating these within the [SaShiMi](https://arxiv.org/pdf/2202.09729) architecture for autoregressive modeling of raw audio. For now, the package is only installable after cloning the repository, but will be packaged properly once the research work is done. 

## Installation

Use [pip](https://pip.pypa.io/en/stable/) to install the original Mamba package, and (optionally) the efficient 1D convolution implementation it relies on.

```bash
pip install mamba-ssm
pip install causal-conv1d>=1.4.0 #optional
```

Use [pip](https://pip.pypa.io/en/stable/) to install the decorrelated Mamba extensions directly from the repository.

```bash
pip install git+https://github.com/s1097736/thesis_work
```

## Usage

Usage for inference follows the standard format of models from HuggingFace, expecting a Tensor of token indices as input and returning a CausalLMOutput object with the associated output logits. An example for training one such model has been provided under the example_use folder. A minimal process for creating a new decorrelated model from scratch has been shown below.

```python
from decorr_mamba.model.decorrelation import DecorrMamba
from mamba_ssm.models.config_mamba import MambaConfig
import torch

device = torch.device("cuda")
mamba_args = MambaConfig(d_model=64, vocab_size=256)

model = DecorrMamba(config=mamba_args, device=device)
```
