from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import numpy as np

label_names = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'P', 'R']


def readData(filename) -> (np.array, np.array):
    '''
    Read data from pattern-learn or pattern-test
    '''
    one_hots = []
    features = []
    with open(filename) as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            eles = line.strip().split()
            features.append(list(map(float, eles[:96])))
            one_hots.append(list(map(int, eles[96:])))
            #print(idx, features, one_hots)
    labels = [np.argmax(oh) for oh in one_hots]
    return np.array(features), np.array(labels)


split_radio = 0.7

x_train, y_train = readData("../data/pattern-test")
x_test, y_test = readData("../data/pattern-learn")

train_num = int(5000 * split_radio)
# x_test.append(x_train[train_num:])
# y_test.append(y_train[train_num:])

lr_clf = LogisticRegression(random_state=1, max_iter=1000)
rf_clf = RandomForestClassifier(oob_score=True, random_state=10)
sgd_clf = SGDClassifier(loss='log', penalty='l2', max_iter=100)
svm_clf = SVC(gamma='scale',
              C=1.0,
              cache_size=200,
              degree=3,
              decision_function_shape='ovr',
              kernel='rbf')
dt_clf = DecisionTreeClassifier()
eclf = VotingClassifier(estimators=[('lr', lr_clf), ('rf', rf_clf),
                                    ('sgd', sgd_clf), ('svm', svm_clf),
                                    ('dt', dt_clf)],
                        voting='hard')

for clf, clf_name in zip([lr_clf, rf_clf, sgd_clf, svm_clf, dt_clf, eclf], [
        'Logistic Regrsssion', 'Random Forest', 'SGD', 'SVM', 'Decision Tree',
        'Ensemble'
]):
    scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')
    print('Accuracy: {:.2f} (+/- {:.2f}) [{}]'.format(scores.mean(),
                                                      scores.std(), clf_name))
    clf.fit(x_train[:train_num], y_train[:train_num])
    y_pred = clf.predict(x_test)
    print("The classification report:\n",
          classification_report(y_test, y_pred, target_names=label_names))
