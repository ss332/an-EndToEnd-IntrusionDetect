import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils import shuffle


# 0.7训练，0.3测试


def ids2018():
    # df = pd.read_csv(r"E:\data\CICIDS2018\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv")
    # df = shuffle(df)
    # bot_df = df[df["Label"] == 'Bot']
    # benign_df = df[df["Label"] != 'Bot']
    # print(benign_df.info())
    # print(bot_df.info())
    #
    # train_df = pd.concat([benign_df.iloc[:533668, :], bot_df.iloc[:200334, :]], axis=0)
    # train_df.to_csv("Friday-02-03-2018-train.csv")
    #
    # test_df = pd.concat([benign_df.iloc[533668:, :], bot_df.iloc[200334:, :]], axis=0)
    # test_df.to_csv("Friday-02-03-2018-test.csv")

    train_df = pd.read_csv("Friday-02-03-2018-train.csv")
    print(train_df.info())

    test_df = pd.read_csv("Friday-02-03-2018-test.csv")
    print(test_df.shape)

    # 查明字符类型，这些类别特征是
    # 3 Timestamp  31419
    # 80 Label  2
    print("Train SET:\n")
    i = 0
    for feature in train_df.columns:
        if train_df[feature].dtypes == 'object':
            print(feature, "index:{index} {num} ".format(index=i, num=len(train_df[feature].unique())))
        i = i + 1

    x_train = train_df.drop(labels=['Unnamed: 0', 'Dst Port', 'Protocol', 'Timestamp'], axis=1)
    print(x_train.head())
    y_train = train_df['Label']
    y_train = y_train.replace({'Benign': 0, 'Bot': 1})

    x_test = test_df.drop(labels=['Unnamed: 0', 'Dst Port', 'Protocol', 'Timestamp'], axis=1)
    y_test = test_df['Label']
    y_test = y_test.replace({'Benign': 0, 'Bot': 1})

    return x_train, y_train, x_test, y_test


ids2018()
