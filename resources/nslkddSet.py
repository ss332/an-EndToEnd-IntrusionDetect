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

    # 将所有类别形成一个字符串
    # protocol type
    protocol = sorted(nsl_train.protocol_type.unique())
    protocol2 = ['Protocol_type_' + x for x in protocol]
    # service
    service = sorted(nsl_train.service.unique())
    service2 = ['service_' + x for x in service]
    # flag
    flag = sorted(nsl_train.flag.unique())
    flag2 = ['flag_' + x for x in flag]
    # put together
    dumcols = protocol2 + service2 + flag2
    print(dumcols)

    # for test
    protocol_test = sorted(nsl_test.protocol_type.unique())
    protocol2_test = ['Protocol_type_' + x for x in protocol_test]
    # service
    service_test = sorted(nsl_test.service.unique())
    service2_test = ['service_' + x for x in service_test]
    # flag
    flag_test = sorted(nsl_test.flag.unique())
    flag2_test = ['flag_' + x for x in flag_test]
    # put together
    testDumcols = protocol2_test + service2_test + flag2_test

    # 将类别变为序号，列入有三个种类，则对应3个种类记为1，2，3,然后转化为onehot
    train_categorical_columns_encoder = train_categorical_columns.apply(LabelEncoder().fit_transform)
    test_categorical_columns_encoder = test_categorical_columns.apply(LabelEncoder().fit_transform)
    print(train_categorical_columns_encoder.head())

    train_categorical_onehot = OneHotEncoder().fit_transform(train_categorical_columns_encoder)
    train_cat_data = pd.DataFrame(train_categorical_onehot.toarray(), columns=dumcols)

    test_categorical_onehot = OneHotEncoder().fit_transform(test_categorical_columns_encoder)
    test_cat_data = pd.DataFrame(test_categorical_onehot.toarray(), columns=testDumcols)

    print(train_cat_data.head(8))

    # 测试集补齐缺少的服务类型，填0
    trainservice = nsl_train['service'].tolist()
    testservice = nsl_test['service'].tolist()
    difference = list(set(trainservice) - set(testservice))
    for col in difference:
        test_cat_data[col] = 0

    # 去掉被转化成onehot向量的3个字符特征，然后拼接onehot dataFrame
    newTrainDf = nsl_train.join(train_cat_data)
    newTrainDf.drop('flag', axis=1, inplace=True)
    newTrainDf.drop('protocol_type', axis=1, inplace=True)
    newTrainDf.drop('service', axis=1, inplace=True)
    # newTrainDf.to_csv("train.csv")
    newTestDf = nsl_train.join(train_cat_data)
    newTestDf.drop('flag', axis=1, inplace=True)
    newTestDf.drop('protocol_type', axis=1, inplace=True)
    newTestDf.drop('service', axis=1, inplace=True)
    # newTestDf.to_csv("test.csv")
    # print(newTrainDf.shape)

    # 对样本攻击类别标签分类，定义：0=normal，1=Dos，2=Probe,3=R2L,4=U2R
    trainclass = newTrainDf['label']
    testclass = newTestDf['label']

    newTrainClass = trainclass.replace(values.getNslLabelReplace())
    newTestClass = testclass.replace(values.getNslLabelReplace())

    newTestDf['label'] = newTestClass
    newTrainDf['label'] = newTrainClass
    # print(newTrainDf['class'].head())

    # 将其分为训练样本和标签
    x_train = newTrainDf.drop('label', 1)
    y_train = newTrainDf['label']

    x_test = newTestDf.drop('label', 1)
    y_test = newTestDf['label']

    return x_train,y_train,x_test,y_test
