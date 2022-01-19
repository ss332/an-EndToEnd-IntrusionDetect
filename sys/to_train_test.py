import pandas as pd

from sklearn.utils import shuffle


def to_split(file, train=0.7):
    flows = pd.read_csv(file)
    flows = shuffle(flows)
    # 标签集
    set_size = 800000
    train_size = 560000
    test_size = 240000

    train = flows.iloc[:train_size, :]
    test = flows.iloc[train_size:, :]

    train.to_csv('all_train.csv',index=0)
    train.to_csv('all_test.csv',index=0)


to_split('all_flows.txt', 0.7)
