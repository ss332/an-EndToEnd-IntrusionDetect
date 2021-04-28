import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from resources import values


def nslSet():
    # 获取所有列的名字
    col_names = values.getNslColNames()

    # 读取数据集
    nsl_train = pd.read_csv(r"E:\data\NSL_KDD\KDDTrain+2.csv", header=None, names=col_names)
    nsl_test = pd.read_csv(r"E:\data\NSL_KDD\KDDTest+2.csv", header=None, names=col_names)
    print('trainSet shape：', nsl_train.shape)
    print('testSet shape：', nsl_test.shape)
    # print(nsl_train.head(8))
    # print(nsl_train.describe())

    # 看一下样本类别分布
    print("trainSet labelDistribution:\n", nsl_train['label'].value_counts())
    print("testSet labelDistribution:\n", nsl_test['label'].value_counts())

    # 数据处理，准备将string字符类型特征转化为one_hot向量
    # 查明那些事字符类型，这些类别特征是：protocol_type2,service3,flag4.
    print("\nTrain SET:")
    for feature in nsl_train.columns:
        if nsl_train[feature].dtypes == 'object':
            print(feature, "has {num} 种".format(num=len(nsl_train[feature].unique())))
            # print(nsl_train[feature].value_counts().sort_values(ascending=False).head())

    print("\nTEST SET:")
    for feature in nsl_train.columns:
        if nsl_test[feature].dtypes == 'object':
            print(feature, "has {num} 种".format(num=len(nsl_test[feature].unique())))

    # 测试集比样本集中的service类型少了6个，样本类别多了15种,单独将这几列取出来
    categorical_columns = ['protocol_type', 'service', 'flag']
    train_categorical_columns = nsl_train[categorical_columns]
    test_categorical_columns = nsl_test[categorical_columns]
    print(test_categorical_columns.head(5))

    # 将所有类别形成一个字符串列表

    def create_list(lis, name):
        return [name + x for x in lis]
    # protocol type
    protocol = sorted(nsl_train.protocol_type.unique())
    service = sorted(nsl_train.service.unique())
    flag = sorted(nsl_train.flag.unique())

    # put together
    processed_col = create_list(protocol, 'Protocol_type_') + create_list(service, 'service_') + create_list(flag, 'flag_')
    print(processed_col)

    # for test
    protocol_test = sorted(nsl_test.protocol_type.unique())
    service_test = sorted(nsl_test.service.unique())
    flag_test = sorted(nsl_test.flag.unique())

    # put together
    processed_test_col = create_list(protocol_test, 'Protocol_type_') + create_list(service_test, 'service_') + create_list(flag_test, 'flag_')

    # 将类别变为序号，列入有三个种类，则对应3个种类记为1，2，3,然后转化为onehot
    train_categorical_columns_encoder = train_categorical_columns.apply(LabelEncoder().fit_transform)
    test_categorical_columns_encoder = test_categorical_columns.apply(LabelEncoder().fit_transform)
    print(train_categorical_columns_encoder.head())

    train_categorical_one_hot = OneHotEncoder().fit_transform(train_categorical_columns_encoder)
    train_cat_data = pd.DataFrame(train_categorical_one_hot.toarray(), columns=processed_col)

    test_categorical_one_hot = OneHotEncoder().fit_transform(test_categorical_columns_encoder)
    test_cat_data = pd.DataFrame(test_categorical_one_hot.toarray(), columns=processed_test_col)

    print(train_cat_data.head(8))

    # 测试集补齐缺少的服务类型，填0
    train_service = nsl_train['service'].tolist()
    test_service = nsl_test['service'].tolist()
    difference = list(set(train_service) - set(test_service))
    for col in difference:
        test_cat_data[col] = 0

    # 去掉被转化成one_hot向量的3个字符特征，然后拼接one_hot dataFrame
    new_train_df = nsl_train.join(train_cat_data)
    new_train_df.drop('flag', axis=1, inplace=True)
    new_train_df.drop('protocol_type', axis=1, inplace=True)
    new_train_df.drop('service', axis=1, inplace=True)
    # new_train_df.to_csv("train.csv")
    new_test_df = nsl_train.join(train_cat_data)
    new_test_df.drop('flag', axis=1, inplace=True)
    new_test_df.drop('protocol_type', axis=1, inplace=True)
    new_test_df.drop('service', axis=1, inplace=True)
    # new_test_df.to_csv("test.csv")
    # print(new_train_df.shape)

    # 对样本攻击类别标签分类，定义：0=normal，1=Dos，2=Probe,3=R2L,4=U2R
    train_class = new_train_df['label']
    test_class = new_test_df['label']

    new_train_class = train_class.replace(values.getNslLabelReplace())
    new_test_class = test_class.replace(values.getNslLabelReplace())

    new_test_df['label'] = new_test_class
    new_train_df['label'] = new_train_class
    # print(new_train_df['class'].head())

    # 分为训练样本和标签
    x_train = new_train_df.drop('label', 1)
    y_train = new_train_df['label']

    x_test = new_test_df.drop('label', 1)
    y_test = new_test_df['label']

    return x_train, y_train, x_test, y_test
