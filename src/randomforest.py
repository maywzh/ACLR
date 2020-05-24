import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import calinski_harabasz_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score

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

model = RandomForestClassifier(oob_score=True,
                               n_estimators=100,
                               max_features=10,
                               random_state=30)

scores_clf_svc_cv = cross_val_score(model, x_train, y_train, cv=5)
print(scores_clf_svc_cv)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores_clf_svc_cv.mean(), scores_clf_svc_cv.std() * 2))
rftree = model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_score = model.predict_proba(x_test)

print("The classification report:\n",
      classification_report(y_test, y_pred, target_names=label_names))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

print(fpr)