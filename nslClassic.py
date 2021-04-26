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

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(n_estimators=100),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

x_train, y_train, x_test, y_test = nslkddSet.nslSet()

# iterate over classifiers,遍历分类器，打印指标
for name, clf in zip(names, classifiers):
    starts = time.time()
    clf.fit(x_train, y_train)
    finish = time.time()
    use = starts - finish
    score = clf.score(x_test, y_test)
    print(score)
    y_predict = clf.predict(x_test)
    values.printMetrics(name, y_test, y_predict, use)

