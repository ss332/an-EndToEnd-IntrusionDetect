# This is a sample Python script.
import torch
from scapy import all
import re
"""
author:alice，
time:2020.12.28
"""
import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import recall_score
import pandas as pd
import torch.nn as nn
import torch
import time
from sklearn.utils import shuffle
import operator

def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    record_name = r'E:\bot\FlowId{}.pt'.format(177)
    # 归一化
    record_flow = torch.load(record_name) / 255.0
    space_tensor = torch.zeros(784)
    flows = pd.read_csv('flows1.csv')
    flows = shuffle(flows)
    # 标签集
    labelSet = flows["label"]
    s=labelSet[77]
    target_tensor = torch.tensor(s)
    print(target_tensor)
