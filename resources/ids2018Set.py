import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils import shuffle
import numpy as np
from numpy import float64
import sys

# 0.6训练，0.4测试  533668 200334
# 1048575 252583 481419
# 有却失值，无穷值，连续值填0
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
l = {'Benign': 0, 'FTP-BruteForce': 1, 'SSH-Bruteforce': 2, 'DoS attacks-GoldenEye': 3, 'DoS attacks-Slowloris': 4,
     'DoS attacks-Hulk': 5, 'DoS attacks-SlowHTTPTest': 6, 'DDoS attacks-LOIC-HTTP': 7, 'DDOS attack-HOIC': 9,
     'DDOS attack-LOIC-UDP': 8, 'Brute Force -Web': 10, 'Brute Force -XSS': 11, 'SQL Injection': 12,
     'Infilteration': 13,
     'Bot': 14, }


def ids2018():
    top_limit = 0.1
    ratio = 3
    df = pd.read_csv('pick_up.txt')
    df=shuffle(df)
    print(df.shape)
    df['Flow Byts/s'] = df['Flow Byts/s'].replace({float('inf'): sys.float_info.max})
    df = df.drop(labels=['Flow Byts/s', 'Flow Pkts/s', 'Dst Port', 'Protocol', 'Timestamp'], axis=1)
    df = df.replace(l)

    # print(np.isfinite(train_df['Flow Byts/s']))
    df.fillna(0.0, inplace=True)

    train_df = df.iloc[:560000, :]

    test_df = df.iloc[560000:, :]

    # 查明字符类型，这些类别特征是
    print("Train SET:")
    i = 0
    for feature in train_df.columns:

        if train_df[feature].dtypes == 'object':
            print(feature, "index:{index} {num} ".format(index=i, num=len(train_df[feature].unique())))
        i = i + 1

    # y_train = y_train.replace({'Benign': 0, 'DoS attacks - SlowHTTPTest': 1, ' DoS attacks - Hulk': 2, 'Bot': 3})
    y_train = train_df['Label']
    y_test = test_df['Label']
    x_train = train_df.drop(labels=['Label'], axis=1)
    x_test = test_df.drop(labels=['Label'], axis=1)
    a=np.array(y_test)
    print(a.shape)

    return x_train, y_train, x_test, y_test


ids2018()

# Benign            667626   5
# FTP-BruteForce    193360   2.5
# SSH-Bruteforce    187589   2.5

# Name: Label, dtype: int64
# Benign                   996077   5
# DoS attacks-GoldenEye     41508   4
# DoS attacks-Slowloris     10990   1

# Name: Label, dtype: int64
# DoS attacks-Hulk            461912    2.5
# Benign                      446772    5
# DoS attacks-SlowHTTPTest    139890   2.5

# Name: Label, dtype: int64
# DDoS attacks-LOIC-HTTP    576191     5
# Benign                    423809     5

# Name: Label, dtype: int64
# DDOS attack-HOIC        686012       48270
# Benign                  360833       5
# DDOS attack-LOIC-UDP      1730      1730

# Name: Label, dtype: int64
# Benign              1048213        99638
# Brute Force -Web        249       -
# Brute Force -XSS         79       -
# SQL Injection            34       -

# Name: Label, dtype: int64
# Benign           782237       5
# Infilteration    161934       5

# Name: Label, dtype: int64
# Benign    762384           5
# Bot       286191           5

pick1 = {'Benign': 50000, 'FTP-BruteForce': 25000, 'SSH-Bruteforce': 25000}
pick2 = {'Benign': 50000, 'DoS attacks-GoldenEye': 40000, 'DoS attacks-Slowloris': 10000}
pick3 = {'Benign': 50000, 'DoS attacks-Hulk': 25000, 'DoS attacks-SlowHTTPTest': 25000}
pick4 = {'Benign': 50000, 'DDoS attacks-LOIC-HTTP': 50000}
pick5 = {'Benign': 50000, 'DDOS attack-HOIC': 48270, 'DDOS attack-LOIC-UDP': 1730}
pick6 = {'Benign': 99638, 'Brute Force -Web': 249, 'Brute Force -XSS': 79, 'SQL Injection': 34}
pick7 = {'Benign': 50000, 'Infilteration': 50000}
pick8 = {'Benign': 50000, 'Bot': 50000}
pick=[pick1,pick2,pick3,pick4,pick5,pick6,pick7,pick8]
def picks():
    i = 0
    ups = ''
    for file in files:
        df = pd.read_csv(file)
        df=shuffle(df)

        picks = pick[i]
        j = 0
        for k, v in zip(picks.keys(), picks.values()):

            up_df = df[df['Label'] == '{}'.format(k)].iloc[:v, :]

            if i == 0 and j == 0:
                ups = up_df
            else:
                ups = pd.concat([ups, up_df], axis=0)

            j += 1

        i += 1
    print(ups.shape)
    ups.to_csv('pick_up.txt', index=0)

# picks()
