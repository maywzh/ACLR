import numpy as np
import seaborn as sns
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


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
model = svm.SVC(gamma='scale',
                C=1.0,
                cache_size=200,
                degree=3,
                decision_function_shape='ovr',
                kernel='rbf')
model.fit(x_train[:train_num], y_train[:train_num])
scores_clf_svc_cv = cross_val_score(model, x_train, y_train, cv=5)
print(scores_clf_svc_cv)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores_clf_svc_cv.mean(), scores_clf_svc_cv.std() * 2))

precition = model.score(x_test, y_test)
print('precition is : ', precition * 100, "%")

#获取模型返回值
n_Support_vector = model.n_support_  #支持向量个数
print("The number of support vectors is： ", n_Support_vector)

y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
