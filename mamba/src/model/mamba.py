import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange, repeat, einsum # might be useful later
from utils.helpers import MambaArgs
import math
from functools import partial
import json


class Mamba(nn.Module):
    ''' Full Mamba architecture '''
    def __init__(self, args: MambaArgs):
        super(Mamba, self).__init__()

        self.args = args

        self.embedding = nn.Embedding(args.vocab_size, args.D)

        self.layers = nn.ModuleList([ResidualMambaBlock(args) 
                                     for _ in range(args.n_layers)])
        self.rms = RMSNorm(args.D)

        self.logits = nn.Linear(args.D, args.vocab_size, bias=False)
        self.logits.weight = self.embedding.weight # weight tying! 

        # https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
        def _init_weights(
            module,
            n_layers: int = args.n_layers,
            initializer_range: float = 0.02,  # only used for embedding layer 
            rescale_prenorm_residual=True):

            # slightly modified from original code. Use of biases in the projection layers
            # is controlled by the MambaArgs class, no need to enforce that here. 

            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=initializer_range)

            if rescale_prenorm_residual:
                # scaling the weights of the output projection initializations
                # depending on the number of layers, as per the GPT2 paper.
                # this only affects the output of the SSM, not the residual
                # connections around the entire block
                for name, p in module.named_parameters():
                    if name == "out_proj.weight":
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                        with torch.no_grad():
                            p /= math.sqrt(n_layers)

        self.apply(partial(_init_weights))

    # https://github.com/johnma2006/mamba-minimal/blob/master/model.py 
    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        config_data = load_config_hf(pretrained_model_name)
        args = MambaArgs(
            D=config_data['d_model'],
            n_layers=config_data['n_layer'],
            vocab_size=config_data['vocab_size'],
            N=16
        )
        model = Mamba(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)

        # define a new dictionary to make the names match
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_key = new_key.replace('mixer', 'block')
            new_key = new_key.replace('D', 's6_block.D')
            new_key = new_key.replace('A_log', 's6_block.log_minus_A')
            new_key = new_key.replace('norm_f', 'rms')
            new_key = new_key.replace('norm', 'rms')
            new_key = new_key.replace('dt_proj', 's6_block.delta_upscale')
            new_key = new_key.replace('x_proj', 's6_block.to_BCdelta')    
            new_key = new_key.replace('lm_head', 'logits')    
            new_state_dict[new_key] = state_dict[key]
        
        model.load_state_dict(new_state_dict)

        return model


    def forward(self, x):

        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.rms(x)
        logits = self.logits(x)

        return logits

class ResidualMambaBlock(nn.Module):
    ''' Wraps the standard Mamba block with RMS normalization and residual
        connections (used everywhere)'''
    
    def __init__(self, args: MambaArgs):
        super(ResidualMambaBlock, self).__init__()

        self.args = args
        self.block = MambaBlock(args)
        self.rms = RMSNorm(args.D)
    
    def forward(self, x):
        return self.block(self.rms(x)) + x
    
class MambaBlock(nn.Module):
    ''' Standard Mamba block as illustrated in the paper '''
    def __init__(self, args: MambaArgs):
        super(MambaBlock, self).__init__()

        self.args = args

        # takes care of both of the upscale projections, factor of 2!
        self.in_proj = nn.Linear(args.D, 2*args.D_inner, bias=args.general_bias)
        self.conv1d = nn.Conv1d(
            in_channels=args.D_inner,
            out_channels=args.D_inner,
            bias=args.conv_bias,
            kernel_size=args.conv_1d_size,
            groups=args.D_inner,
            padding=args.conv_1d_size - 1,
        )    
        self.s6_block = S6Block(args)    

        self.out_proj = nn.Linear(args.D_inner, args.D, bias=args.general_bias)

    def forward(self, x):
        b, l, _ = x.shape # used to avoid specifying these in 

        x = self.in_proj(x)
        # split the input into the two paths
        (x, res) = x.split(
            split_size=[self.args.D_inner, self.args.D_inner], dim=-1)

        # input of shape (B,L,D), dimensions need switching for convolution
        x = torch.transpose(x, 1,2)
        x = self.conv1d(x)[:,:,:l] # the limit is needed because of the padding
        x = torch.transpose(x, 1,2)

        x = F.silu(x)
        x = self.s6_block(x)
        x = x * F.silu(res)

        y = self.out_proj(x)

        return y

class S6Block(nn.Module):
    ''' Inner SSM block '''
    def __init__(self, args: MambaArgs):

        super(S6Block, self).__init__()
        self.args = args 

        def s4d_real():
            # initialization for A used in the paper. Other complex-valued 
            # initializations also possible

            # compute one diagonal, then broadcast across D dimensions. NB
            # that this output is missing a minus sign; we update A in log space,
            # so this minus is only added in during the forward pass
            A = torch.arange(1, args.N + 1, dtype=torch.float32)

            return A.unsqueeze(0).repeat(args.D_inner,1)

        def get_delta_bias():
            # sample biases such that passing them through a softplus
            # leads to them being between a and b
            samples = torch.exp(
                (math.log(args.delta_max)-math.log(args.delta_min)) * \
                 torch.rand(args.D_inner) + math.log(args.delta_min)).clamp(
                                            min=args.delta_init_floor)

            # inverse softplus
            return samples + torch.log(-torch.expm1(-samples))

        self.log_minus_A = nn.Parameter(torch.log(s4d_real()))
        self.log_minus_A._no_weight_decay = True
        
        # these are strictly linear projections, no biases used ever.
        # delta uses one, which we manually add later. 
        self.to_BCdelta = nn.Linear(args.D_inner, 2*args.N+args.delta_rank, bias=False)

        # although the theory states that delta is 1-dimensional and then
        # broadcasted across D dimensions, the implementation actually projects it 
        # up to these D dimensions with a trainable linear layer. 
        # this is just a generalization which allows for more expressivity.
        # within this generalization, delta doesn't have to be 1-dimensional anymore!
        self.delta_upscale = nn.Linear(args.delta_rank, args.D_inner, bias=True)

        # certain initializations are equivalent to broadcasting! 
        delta_init_std = args.delta_rank**-0.5 * args.delta_scale
        if args.delta_init == "constant":
            nn.init.constant_(self.delta_upscale.weight, delta_init_std)
        elif args.delta_init == "random":
            nn.init.uniform_(self.delta_upscale.weight, -delta_init_std, delta_init_std)
        else:
            raise NotImplementedError

        # bias based on empirical work
        with torch.no_grad():
            self.delta_upscale.bias.copy_(get_delta_bias())

        # papers imply this is taken care of by residual connections
        # around the block, but it seems they also implement it here
        self.D = nn.Parameter(torch.ones(args.D_inner))
        self.D._no_weight_decay = True
        
        
    def discretize(self, delta, B, x):

        # ZOH discretization. NB that the log space A is being cast back into
        # A here, as the equation in the paper requires
        delta_A = torch.einsum('bld, dn -> bldn', delta, -torch.exp(self.log_minus_A.float()))
        A_bar = torch.exp(delta_A)

        # below is the full ZOH discretization of B according to the paper.
        # the official implementation doesn't actually do this, instead using 
        # an Euler discretization, as it's cheaper to compute with minimal
        # performance cost. Not sure what we should do for our implementation...

        # delta_B = torch.einsum('bld,bln->bldn', delta, B)
        # # diagonal matrices, so 1/A is the inverse, subtracting 1 instead 
        # # of the identity matrix, and directly multiplying elementwise for the 
        # # first multiplication (second is defined elementwise anyway)
        # B_bar = 1/(delta_A) * (A_bar - 1) * delta_B

        # Euler discretization. Computes the product with the input
        # at this step, removes unnecessary computation later
        B_bar_x = torch.einsum('bld, bln, bld -> bldn', delta, B, x)

        return A_bar, B_bar_x

    def forward(self, x):
        b, l, _ = x.shape 
        # generate all projected parameters and split them up
        BCdelta = self.to_BCdelta(x)

        # delta: (B, L, 1). B, C: (B, L, N)
        (delta, B, C) = BCdelta.split(
            split_size=[self.args.delta_rank, self.args.N, self.args.N], dim=-1)

        # "broadcasting" for delta and computing final parameters
        delta = self.delta_upscale(delta) # (B,L,D)
        delta = F.softplus(delta)

        # discretization. NB that the discretized version of B is 
        # already applied to the input sequence here!
        A_bar, B_bar_x = self.discretize(delta, B, x) # (B, L, D, N)
        
        # scan through each individual token to compute hidden states
        hidden_states = torch.zeros(
            b, l+1, self.args.D_inner, self.args.N).to(self.args.device)
        
        for i in range(0,l):
            # because A is represented only through diagonal, Ah_t-1 is 
            # equivalent to taking the elementwise product of the diagonal
            # and the hidden state
            hidden_states[:,i+1,:,:] = A_bar[:,i,:,:]*hidden_states[:,i,:,:].clone() + \
                B_bar_x[:,i,:,:] # (B,D,N)
        
        # compute outputs in parallel
        outputs = torch.einsum('bln, bldn -> bld', C, hidden_states[:,1:,:,:])

        # throw in D as residual connections with no bias
        outputs = outputs + x * self.D.float()

        return outputs


class RMSNorm(nn.Module):
    ''' Simple implementation of RMSNorm. Default implementation is bugged
        in this version of PyTorch, don't want to mess with version updating '''
    def __init__(self,
                 D: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(D))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
