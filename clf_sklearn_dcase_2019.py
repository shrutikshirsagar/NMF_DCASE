

### without NMF
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
train  = np.load('//nas/shared/workspace/i21_skshirsagar/foa_mel_norm_nmf/fold12_nmf.npy')
val = np.load('//nas/shared/workspace/i21_skshirsagar/foa_mel_norm_nmf/fold3_nmf.npy')
test = np.load('//nas/shared/workspace/i21_skshirsagar/foa_mel_norm_nmf/fold4_nmf.npy')
print(train.shape)
print(val.shape)
X_train = train[:, :-226]
X_valid = val[:, :-226]
X_test = test[:, :-226]
y_train = train[:, 448:459]
y_valid = val[:, 448:459]
y_test = test[:, 448:459]
print(X_train.shape, y_train.shape)
print(X_valid .shape)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train1 = scaler.transform(X_train)
X_valid1 = scaler.transform(X_valid)
x_test1 = scaler.transform(X_test)

y_train1 = np.argmax(y_train, axis=1)
y_valid1 = np.argmax(y_valid, axis=1)
y_test1 = np.argmax(y_test, axis=1)

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


for name, clf in zip(names, classifiers):
    print(name, clf)
    clf.fit(X_train, y_train)
    score = clf.score(x_test1, y_test1)
    print(score)
    print(X_train1.shape)
    clf.fit(X_train1, y_train1)
    print(y_train1.shape)
    y_pred = clf.predict(x_test1)
    print(y_pred)
    y_pred = (y_pred > 0.5)
    acc = balanced_accuracy_score(y_test1, y_pred)
    print(acc*100)

    class_names = ['alarm', 'baby', 'crash', 'dog',  'female_speech', 'footsteps', 'knock', 'male_scream', 'male_speech', 'phone', 'piano']


    y_pred = clf.predict(x_test1)

    y_pred = (y_pred > 0.5)
    accuracy_score(y_test1, y_pred, normalize=False)
    print(classification_report(y_test1, y_pred, target_names=class_names, digits=4))
