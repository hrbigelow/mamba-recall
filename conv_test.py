import torch
from causal_conv1d import causal_conv1d_fn

batch, dim, seq, width = 10, 5, 17, 4
x = torch.zeros((batch, dim, seq)).to('cuda')
weight = torch.zeros((dim, width)).to('cuda')
bias = torch.zeros((dim, )).to('cuda')

causal_conv1d_fn(x, weight, bias, None)

