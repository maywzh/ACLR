import numpy as np
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.decomposition import PCA

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
# pca = PCA()
# pca.fit(x_train)
# x_new = pca.transform(x_train)

# print(x_new.shape)

# param_grid = {
#     'cross_val_score': ['gini', 'entropy'],
#     'splitter': ['best', 'random'],
#     'max_depth': ['None', 3, 4, 5]
# }
# 用GridSearchCV寻找最优参数（列表）
param = [{
    'criterion': ['gini'],
    'max_depth': [30, 50, 60, 100],
    'min_samples_leaf': [2, 3, 5, 10],
    'min_impurity_decrease': [0.1, 0.2, 0.5]
}, {
    'criterion': ['gini', 'entropy']
}, {
    'max_depth': [30, 60, 100],
    'min_impurity_decrease': [0.1, 0.2, 0.5]
}]
grid = GridSearchCV(DecisionTreeClassifier(), param_grid=param, cv=6)
grid.fit(x_train, y_train)
print('最优分类器:', grid.best_params_, '最优分数:', grid.best_score_)  # 得到最优的参数和分值

#构建自动调参容器，n_jobs参数支持同时多个进程运行并行测试
#选出最优参数
best_parameters = grid.best_estimator_.get_params()
for para, val in list(best_parameters.items()):
    print(para, val)

model = DecisionTreeClassifier(
    max_depth=best_parameters['max_depth'],
    min_samples_leaf=best_parameters['min_samples_leaf'],
    min_impurity_decrease=best_parameters['min_impurity_decrease'],
    splitter=best_parameters['splitter'],
    criterion=best_parameters['criterion'])
scores_clf_svc_cv = cross_val_score(model, x_train, y_train, cv=5)
print(scores_clf_svc_cv)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores_clf_svc_cv.mean(), scores_clf_svc_cv.std() * 2))

model.fit(x_train, y_train)

precition = model.score(x_test, y_test)
print('precition is : ', precition * 100, "%")
