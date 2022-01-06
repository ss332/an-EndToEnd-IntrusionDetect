import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils import shuffle
import numpy as np
from numpy import float64
import sys


# 0.6训练，0.4测试  533668 200334
# 1048575 252583 481419
# 有却失值，无穷值，连续值填0
def ids2018():
    top_limit = 0.1
    ratio = 3
    df2 = pd.read_csv(r"E:\data\CICIDS2018\Wednesday-21-02-2018_TrafficForML_CICFlowMeter.txt")
    df = pd.read_csv(r"E:\data\CICIDS2018\Friday-02-03-2018_TrafficForML_CICFlowMeter.txt",)
    bot_ddos = pd.concat([df, df2], axis=0)
    print(bot_ddos.shape)

    bot_ddos = shuffle(bot_ddos)
    bot_df = df[df["Label"] == 'Bot']
    benign_df1 = df[df["Label"] == 'Benign']

    print(benign_df1.shape)
    print(bot_df.shape)
    print("----")


    # train_df = pd.concat([benign_df.iloc[:252583, :], bot_df.iloc[:481419, :]], axis=0)
    # test_df = pd.concat([benign_df.iloc[252583:, :], bot_df.iloc[481419:, :]], axis=0)
    train_df = bot_ddos.iloc[:1468005, :]
    test_df = bot_ddos.iloc[1468005:, :]

    train_df.fillna(0.0, inplace=True)
    test_df.fillna(0.0, inplace=True)

    # 查明字符类型，这些类别特征是
    # 3 Timestamp  31419
    # 80 Label  2
    print("Train SET:")
    i = 0
    for feature in train_df.columns:

        if train_df[feature].dtypes == 'object':
            print(feature, "index:{index} {num} ".format(index=i, num=len(train_df[feature].unique())))
        i = i + 1

    train_df['Flow Byts/s'] = train_df['Flow Byts/s'].replace({float('inf'): sys.float_info.max})
    print(np.isfinite(train_df['Flow Byts/s']))

    x_train = train_df.drop(labels=['Flow Byts/s', 'Flow Pkts/s', 'Dst Port', 'Protocol', 'Timestamp', 'Label'], axis=1)
    y_train = train_df['Label']
    y_train = y_train.replace({'Benign': 0, 'DDOS attack-LOIC-UDP': 1, 'DDOS attack-HOIC': 2, 'Bot': 3})
    # y_train = y_train.replace({'Benign': 0, 'DoS attacks - SlowHTTPTest': 1, ' DoS attacks - Hulk': 2, 'Bot': 3})


    x_test = test_df.drop(labels=['Flow Byts/s', 'Flow Pkts/s', 'Dst Port', 'Protocol', 'Timestamp', 'Label'], axis=1)
    y_test = test_df['Label']
    y_test = y_test.replace({'Benign': 0, 'DDOS attack-LOIC-UDP': 1, 'DDOS attack-HOIC': 2, 'Bot': 3})
    # y_test = y_train.replace({'Benign': 0, 'DoS attacks - SlowHTTPTest': 1, ' DoS attacks - Hulk': 2, 'Bot': 3})

    return x_train, y_train, x_test, y_test

ids2018()
