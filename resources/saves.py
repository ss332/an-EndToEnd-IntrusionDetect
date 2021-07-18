import numpy as np
import pandas as pd
import time
import re
# packets = all.rdpcap(r'E:\Amazon\DDOS-LOIC-UDP.pcap')
# print(packets[131].time)
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(packets[131].time)))
# print(packets[131].show())
import torch
import torch.nn as nn
rnn = nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
print(output.size())
