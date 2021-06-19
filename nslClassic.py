from resources import nslkddSet
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

h = .02  # step size in the mesh

names = ["Decision Tree",
         "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA",
         "RBF SVM", "Linear SVM",
         "Nearest Neighbors",
         ]

classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(n_estimators=100),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    make_pipeline(StandardScaler(), SVC(gamma='auto')),
    make_pipeline(StandardScaler(), SVC(gamma='auto', kernel="linear", C=0.025)),
    KNeighborsClassifier(n_neighbors=5),
]

x_train, y_train, x_test, y_test = nslkddSet.nsl_set()
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

    values.print_metrics(name, y_test, y_predict, use)
