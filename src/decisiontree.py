import numpy as np
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


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


x_train, y_train = readData("../data/pattern-test")
x_test, y_test = readData("../data/pattern-learn")

# for i in range(np.shape(x_train)[0]):
#     if y_train[i] == 1:
#         plt.scatter(x_train[i][0], x_train[i][1], c='b', s=20)
#     else:
#         plt.scatter(x_train[i][0], x_train[i][1], c='y', s=20)
# #plt.show()

train_num = 5000
test_num = 1000

model = DecisionTreeClassifier()
scores_clf_svc_cv = cross_val_score(model, x_train, y_train, cv=5)
print(scores_clf_svc_cv)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores_clf_svc_cv.mean(), scores_clf_svc_cv.std() * 2))

model.fit(x_train, y_train)

precition = model.score(x_test, y_test)
print('precition is : ', precition * 100, "%")
