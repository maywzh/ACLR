import numpy as np
import seaborn as sns
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
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

n_classes = x_total.shape[1]
n_samples, n_features = x_total.shape

x_train, x_test, y_train, y_test = train_test_split(x_total,
                                                    y_total,
                                                    test_size=.3,
                                                    random_state=0)
param_grid = {
    'C': [4.25, 4.5, 4.75, 5],
    'gamma': [0.055, 0.06, 0.05, 0.045, 0.04]
}

model = svm.SVC()

# #构建自动调参容器，n_jobs参数支持同时多个进程运行并行测试
# grid_search = GridSearchCV(model, param_grid, n_jobs=-1, verbose=10)
# grid_search.fit(x_train, y_train)
# #选出最优参数
# best_parameters = grid_search.best_estimator_.get_params()
# for para, val in list(best_parameters.items()):
#     print(para, val)

# model = svm.SVC(kernel='rbf',
#                 C=best_parameters['C'],
#                 gamma=best_parameters['gamma'],
#                 decision_function_shape='ovr',
#                 degree=3,
#                 probability=True)
model = svm.SVC(
    kernel='rbf',
    C=4.5,
    gamma=0.06,
    decision_function_shape='ovr',
    degree=3,
    #verbose=1,
    probability=True)

model.fit(x_train, y_train)
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
print(model.n_support_)