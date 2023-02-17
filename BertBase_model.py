import torch
from torch import nn
from einops import rearrange
import copy
import numpy as np

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertLayerNorm(nn.Module):
    def __init__(self, num_hid, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_hid))
        self.bias = nn.Parameter(torch.zeros(num_hid))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertPreTrainedModel(nn.Module):
    '''
    An abstract class to handel weights initialization
    and train the model.
    '''
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        self.config = config

    def init_weights(self, module):
        '''
        Initialize the weights.
        '''
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

        
