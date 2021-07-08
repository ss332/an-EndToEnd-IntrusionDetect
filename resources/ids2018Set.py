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

    # 查明字符类型，这些类别特征是
    # 3 Timestamp  31419
    # 80 Label  2
    print("Train SET:")
    i = 0
    for feature in train_df.columns:

        if train_df[feature].dtypes == 'object':
            print(feature, "index:{index} {num} ".format(index=i, num=len(train_df[feature].unique())))
        i = i + 1

    train_df['Flow Byts/s']=train_df['Flow Byts/s'].replace({float('inf'):sys.float_info.max})
    print(np.isfinite( train_df['Flow Byts/s']))

    x_train = train_df.drop(labels=['Flow Byts/s','Flow Pkts/s','Dst Port', 'Protocol', 'Timestamp', 'Label'], axis=1)
    y_train = train_df['Label']
    y_train = y_train.replace({'Benign': 0, 'Bot': 1})

    x_test = test_df.drop(labels=['Flow Byts/s','Flow Pkts/s','Dst Port', 'Protocol', 'Timestamp', 'Label'], axis=1)
    y_test = test_df['Label']
    y_test = y_test.replace({'Benign': 0, 'Bot': 1})

    return x_train, y_train, x_test, y_test
