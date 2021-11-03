from scapy import all
import time
import operator
import re
import pandas as pd
import math
import torch


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()


def getSpace():
    for i in range(100001):
        record_name = r'E:\bot\bot{}.pt'.format(i)
        # 归一化
        record_flow = torch.load(record_name) / 255.0
        space_tensor = torch.zeros(784)
        size = 0
        for k in range(record_flow.size(0)):
            x = torch.flatten(record_flow[k], 0)
            l = 255
            while l >= 0:
                if x[l] == 0:
                    l = l - 1
                else:
                    break
            for j in range(l):
                if size < 784:
                    space_tensor[size] = x[j]
                    size = size + 1


        space_name = r'E:\space\space{}.pt'.format(i)
        torch.save(space_tensor, space_name)

        if (i + 1) % 2000 == 0:
            print('%d %d%% (%s) ' % (i + 1, 100 * (i + 1) / 100001, timeSince(start),))


getSpace()
