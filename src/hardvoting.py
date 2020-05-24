from sklearn import datasets
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import numpy as np
from sklearn.preprocessing import label_binarize

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


x_train_r, y_train_r = readData("../data/pattern-test")
x_test_r, y_test_r = readData("../data/pattern-learn")

x_total = np.concatenate((x_train_r, x_test_r))
y_total = np.concatenate((y_train_r, y_test_r))

y_bin = label_binarize(y_total, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

n_classes = x_total.shape[1]
n_samples, n_features = x_total.shape

x_train, x_test, y_train, y_test = train_test_split(x_total,
                                                    y_total,
                                                    test_size=.3,
                                                    random_state=0)

lr_clf = LogisticRegression(random_state=1,
                            max_iter=1000,
                            multi_class="multinomial",
                            solver='saga',
                            penalty='l1',
                            n_jobs=-1)
rf_clf = RandomForestClassifier(oob_score=True,
                                n_estimators=100,
                                max_features=10,
                                random_state=30)
ext_clf = ExtraTreesClassifier(n_estimators=10,
                               max_depth=None,
                               min_samples_split=2,
                               random_state=0,
                               n_jobs=-1)
sgd_clf = SGDClassifier(loss='log', penalty='l2', max_iter=1000, n_jobs=-1)
svm_clf = SVC(C=4.5,
              gamma=0.06,
              cache_size=200,
              degree=3,
              decision_function_shape='ovr',
              kernel='rbf',
              probability=True)
dt_clf = DecisionTreeClassifier()

mlp_clf = MLPClassifier(
    solver='sgd',
    activation='relu',
    alpha=1e-4,
    hidden_layer_sizes=(96, 192),
    random_state=1,
    max_iter=100,
    #verbose=10,
    learning_rate_init=.1)

heclf = VotingClassifier(estimators=[('lr', lr_clf), ('rf', rf_clf),
                                     ('ext', ext_clf), ('sgd', sgd_clf),
                                     ('svm', svm_clf), ('dt', dt_clf),
                                     ('mlp', mlp_clf)],
                         voting='hard',
                         n_jobs=-1)

seclf = VotingClassifier(estimators=[('lr', lr_clf), ('rf', rf_clf),
                                     ('ext', ext_clf), ('sgd', sgd_clf),
                                     ('svm', svm_clf), ('dt', dt_clf),
                                     ('mlp', mlp_clf)],
                         voting='soft',
                         weights=[0.92, 0.97, 0.97, 0.95, 0.98, 0.94, 0.97],
                         flatten_transform=True,
                         n_jobs=-1)

for clf, clf_name in zip(
    [lr_clf, rf_clf, ext_clf, sgd_clf, svm_clf, dt_clf, mlp_clf, heclf, seclf],
    [
        'Logistic Regrsssion', 'Random Forest', 'Extra Trees', 'SGD', 'SVM',
        'Decision Tree', 'MLP', 'Ensemble(hard voting)',
        'Ensemble(soft voting)'
    ]):
    scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')
    print('Accuracy: {:.2f} (+/- {:.2f}) [{}]'.format(scores.mean(),
                                                      scores.std(), clf_name))
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("The classification report:\n",
          classification_report(y_test, y_pred, target_names=label_names))
