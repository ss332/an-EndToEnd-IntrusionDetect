import torch
import torch.nn as nn


rnn = nn.GRU(10, 20, 1)
input = torch.randn(5, 3, 10)
h0 = torch.randn(1, 3, 20)
output, hn = rnn(input, h0)
print(output.shape,output)
print(hn.shape,hn)