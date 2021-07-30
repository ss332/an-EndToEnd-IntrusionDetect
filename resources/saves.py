import numpy as np
import pandas as pd
import time
import re
# packets = all.rdpcap(r'E:\Amazon\DDOS-LOIC-UDP.pcap')
# print(packets[131].time)
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(packets[131].time)))
# print(packets[131].show())
import torch

s = "you're asking me out..?that's so cute.what's your name again?"
print(s)
print(re.sub(r"([.!?])", r" \1", s))

s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
print(s)
a= torch.Tensor([1,2,3,4])

print(a.squeeze(0))
print(torch.tensor([[41]]).size())
c=np.zeros((1,3))
print(c.size)
for i in range(10):
    i=i+4
    print(i)
