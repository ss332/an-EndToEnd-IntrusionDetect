# This is a sample Python script.
import torch;
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

def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    flows = pd.read_csv('flows1.csv')
    flows=shuffle(flows)
    label_1 = flows[flows["label"]==1]
    set=flows["label"]
    label=[143867,148505,114363,46950,181006]
    print(label_1.iloc[233])
    print(set[75882])

