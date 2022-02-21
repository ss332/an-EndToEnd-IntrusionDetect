from resources import nslkddSet
from resources import unswSet
from resources import ids2018Set
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from resources import values
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import warnings

warnings.filterwarnings("ignore")


def train_sets(data_set):

    x_train, y_train, x_test, y_test = data_set
    print("传送完毕----------------||")
    # iterate over classifiers,遍历分类器，打印指标
    for name, clf in zip(names, classifiers):
        print(name + "分类方法训练开始>")
        starts = time.time()
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        print(score)
        use = time.time() - starts
        y_predict = clf.predict(x_test)
        print(y_predict)
        values.print_metrics(name, y_test, y_predict, use)


if __name__ == '__main__':
    names = ["Decision Tree",
             "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes",
             # "QDA",
             # "RBF SVM",
             # "Linear SVM",
             "Nearest Neighbors"]
    classifiers = [
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100),
        MLPClassifier(hidden_layer_sizes=256,random_state=0,max_iter=500,early_stopping=True,batch_size=32),
        AdaBoostClassifier(n_estimators=200,random_state=0),
        GaussianNB(),
        # QuadraticDiscriminantAnalysis(),
        # make_pipeline(StandardScaler(), SVC(gamma='auto')),
        # make_pipeline(StandardScaler(), SVC(gamma='auto', kernel="linear")),
        KNeighborsClassifier(n_neighbors=5)]

    # train_sets(nslkddSet.nsl_set())
    # train_sets(unswSet.unsw_set())
    train_sets(ids2018Set.ids2018())




