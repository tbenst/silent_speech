##
from magneto.models.hyena import HyenaOperator

import pytorch_lightning as pl
from magneto.models.hyena import HyenaOperator
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# use_amp = False
use_amp = True
##
seq_len = 2**18
device = "cuda:1"
def hyena_layer(d_model=512, seq_len=seq_len, order=2, filter_order=64):
    layer = HyenaOperator(
        d_model=d_model, 
        l_max=seq_len,
        order=order,
        filter_order=filter_order
    )
    return layer
##
# d_model   filter_order   memory usage
# 128       64             5974 MiB
# 128       32             5622 MiB
# 256       32             10778 MiB
# 256       64             10970 MiB
# 64        64             3126 MiB
d_model = 64
filter_order = 32
layer1 = hyena_layer(d_model=d_model, filter_order=filter_order)
layer2 = hyena_layer(d_model=d_model, filter_order=filter_order)
model = nn.Sequential(layer1, layer2).to(device)
with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
    x = torch.randn(1, seq_len, d_model).to(device)
    y = model(x)
    
##
layer1 = hyena_layer()
layer2 = hyena_layer()
model = nn.Sequential(layer1, layer2).to(device)
# model = layer1.to(device)
with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
    # batch x seq_len x d_model
    x = torch.randn(1, seq_len, 512).to(device)
    y = model(x)
    
# memory usage: 15102 MiB with one layer
# memory usage: 13066 MiB with one layer + AMP
# memory usage: 21294 MiB with two layers + AMP

print(x.shape, y.shape)
##
