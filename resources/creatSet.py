import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils import shuffle
import numpy as np
from numpy import float64
import sys

# 0.7训练，0.3测试

# 有却失值，无穷值，连续值填0
def ids2018():

    top_limit=0.1
    ratio=3
    df = pd.read_csv(r"E:\data\CICIDS2018\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv")
    print(df.shape)
    df = shuffle(df)
    bot_df = df[df["Label"] == 'Bot']
    benign_df = df[df["Label"] != 'Bot']
    print(benign_df.info())
    # print(bot_df)

    train_df = pd.concat([benign_df.iloc[:533668, :], bot_df.iloc[:200334, :]], axis=0)
    test_df = pd.concat([benign_df.iloc[533668:, :], bot_df.iloc[200334:, :]], axis=0)

    train_df.fillna(0.0, inplace=True)
    test_df.fillna(0.0, inplace=True)