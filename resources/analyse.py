import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils import shuffle
import numpy as np
from numpy import float64
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
files = [
    # 1
    r"E:\data\CICIDS2018\Wednesday-14-02-2018_TrafficForML_CICFlowMeter.txt",
    # 2
    r"E:\data\CICIDS2018\Thursday-15-02-2018_TrafficForML_CICFlowMeter.txt",
    # 3
    r"E:\data\CICIDS2018\Friday-16-02-2018_TrafficForML_CICFlowMeter.txt",
    # 4
    r"E:\data\CICIDS2018\tues-20-02.txt",
    # 5
    r"E:\data\CICIDS2018\Wednesday-21-02-2018_TrafficForML_CICFlowMeter.txt",
    # 6
    r"E:\data\CICIDS2018\Thursday-22-02-2018_TrafficForML_CICFlowMeter.txt",
    # 7
    r"E:\data\CICIDS2018\infilatration.txt",
    # 8
    r"E:\data\CICIDS2018\Friday-02-03-2018_TrafficForML_CICFlowMeter.txt",
]

# 只加载想要的列
columns = ["Flow Duration", "Fwd Pkt Len Mean", "Bwd Pkt Len Mean", "Tot Fwd Pkts", "Tot Bwd Pkts"]


def statics():
    se = 0
    for file in files:
        se += 1
        df = pd.read_csv(file, usecols=columns)

        for i in range(5):
            col = df[columns[i]]
            print(col.describe())
            print('--------------')

        print('-------------', se, '-------------------------')


# statics()




def anlysistues():
    reader = pd.read_csv(r"E:\data\CICIDS2018\infilatration.txt",)
    i = 0
    j = 0
    k = 0
    all = 0
    print(reader["Label"].value_counts())
    # for df in reader:
    #     all += df.shape[0]
    #     print(df["Label"].value_counts())
    #     benign_df = df[df["Label"] == 'Benign']
    #     udp_df = df[df["Label"] == 'DDOS attack-LOIC-UDP']
    #     http_df = df[df["Label"] == 'DDoS attacks-LOIC-HTTP']
    #
    #     i += benign_df.shape[0]
    #     j += udp_df.shape[0]
    #     k += http_df.shape[0]
    #     print(i, j, k)
    #
    #     print('-------------------')
    print(i, j, k, i + j + k, all)

anlysistues()