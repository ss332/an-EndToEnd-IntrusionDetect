import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def unsw_set():
    train_df = pd.read_csv(r"E:\data\UNSW-NB15\UNSW_NB15_training-set.csv")
    train_df.drop('id', axis=1, inplace=True)
    print(train_df)
    print("trainSet labelDistribution:\n", train_df['attack_cat'].value_counts())

    test_df = pd.read_csv(r"E:\data\UNSW-NB15\UNSW_NB15_testing-set.csv")
    test_df.drop('id', axis=1, inplace=True)
    print(test_df.info())
    print("testSet labelDistribution:\n", test_df['attack_cat'].value_counts())

    # 查明字符类型，这些类别特征是：1 proto  131
    # 2 service  13
    # 4 state  7
    # 42 attack_cat  10
    print("\nTrain SET:")
    i = 0
    for feature in train_df.columns:
        if train_df[feature].dtypes == 'object':
            print(feature, "index:{index} {num} ".format(index=i, num=len(train_df[feature].unique())))
        i = i + 1
    print("\nTest SET:")

    i = 0
    for feature in test_df.columns:
        if test_df[feature].dtypes == 'object':
            print(feature, "index:{index} {num} ".format(index=i, num=len(test_df[feature].unique())))
        i = i + 1

    # 测试集和训练集中这几种字符类型数目一致,单独将这几列取出来
    extract_cat_columns = ['proto', 'service', 'state']
    train_cat = train_df[extract_cat_columns]
    test_cat = test_df[extract_cat_columns]

    print(train_cat.head(5))

    # 将所有类别形成一个字符串列表
    def create_list(lis, name):
        return [name + x for x in lis]

    # for train
    proto = sorted(train_df.proto.unique())
    service = sorted(train_df.service.unique())
    print(service)
    state = sorted(train_df.state.unique())
    train_cat_col_names = create_list(proto, 'proto_') + create_list(service, 'service_') + create_list(state, 'state_')

    # for _test
    proto_test = sorted(test_df.proto.unique())
    service_test = sorted(test_df.service.unique())
    state_test = sorted(test_df.state.unique())
    test_cat_col_names = create_list(proto_test, 'proto_') + create_list(service_test, 'service_') + create_list(
        state_test, 'state_')

    print(test_cat_col_names)


    # 将类别变为序号，列入有三个种类，则对应3个种类记为1，2，3,然后转化为oneHot
    # for train
    train_cat_encoder = train_cat.apply(LabelEncoder().fit_transform)
    train_cat_one_hot = OneHotEncoder().fit_transform(train_cat_encoder)
    train_cat_df = pd.DataFrame(train_cat_one_hot.toarray(), columns=train_cat_col_names)

    # for _test
    test_cat_encoder = test_cat.apply(LabelEncoder().fit_transform)
    test_cat_one_hot = OneHotEncoder().fit_transform(test_cat_encoder)
    test_cat_df = pd.DataFrame(test_cat_one_hot.toarray(), columns=test_cat_col_names)

    # 训练集补齐缺少的state,proto类型，填0
    train_state = train_df['state'].tolist()
    test_state = test_df['state'].tolist()
    difference = list(set(test_state) - set(train_state))
    print('difference', difference)
    for col in difference:
        train_cat_df['state_' + col] = 0

    difference2 = list(set(train_state) - set(test_state))
    print('difference', difference2)
    for col in difference2:
        test_cat_df['state_' + col] = 0

    train_proto = train_df['proto'].tolist()
    test_proto = test_df['proto'].tolist()
    difference = list(set(test_proto) - set(train_proto))
    print('difference', difference)
    for col in difference:
        train_cat_df['proto_' + col] = 0

    print(len(train_cat_df.columns))
    print(len(test_cat_df.columns))

    # 拼接
    y_train = train_df.label
    train_df.drop(['proto', 'service', 'state', 'attack_cat', 'label'], axis=1, inplace=True)
    # scaler1 = StandardScaler().fit(train_df)
    # train_df = pd.DataFrame(scaler1.transform(train_df))

    new_train_df = train_df.join(train_cat_df)
    print(len(new_train_df.columns))

    # 拼接
    y_test = test_df.label
    test_df.drop(['proto', 'service', 'state', 'attack_cat', 'label'], axis=1, inplace=True)
    # scaler2 = StandardScaler().fit(test_df)
    # test_df = pd.DataFrame(scaler2.transform(test_df))

    new_test_df = test_df.join(test_cat_df)
    print(len(new_test_df.columns))

    # 两分类
    x_train = new_train_df
    x_test = new_test_df

    # 可选的多分类
    multi_cat = {'Normal': 0, 'Generic': 1, 'Exploits': 2, 'Fuzzers': 3, 'DoS': 4, 'Reconnaissance': 5, 'Analysis': 6,
                 'Backdoor': 7, 'Shellcode': 8, 'Worms': 9}

    return x_train, y_train, x_test, y_test

