from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

iris = datasets.load_iris()
x_train = iris.data[:, [0, 2]]  #取两列，方便绘图
y = iris.target

clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                    ('svc', clf3)],
                        voting='soft',
                        weights=[2, 1, 1])
#weights控制每个算法的权重, voting=’soft' 使用了软权重

clf1.fit(x_train, y)
clf2.fit(x_train, y)
clf3.fit(x_train, y)
eclf.fit(x_train, y)

x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))  #创建网格

fig, axes = plt.subplots(2, 2, sharex='col', sharey='row',
                         figsize=(10, 8))  #共享x_train轴和Y轴

for idx, clf, title in zip(
        product([0, 1], [0, 1]), [clf1, clf2, clf3, eclf],
    ['Decision Tree (depth=4)', 'KNN (k=7)', 'Kernel SVM', 'Soft Voting']):
    Z = clf.predict(
        np.c_[xx.ravel(),
              yy.ravel()])  #起初我以为是预测的x_train的值，实际上是预测了上面创建的网格的值，以这些值来进行描绘区域
    Z = Z.reshape(xx.shape)
    axes[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axes[idx[0], idx[1]].scatter(x_train[:, 0],
                                 x_train[:, 1],
                                 c=y,
                                 s=20,
                                 edgecolor='k')
    axes[idx[0], idx[1]].set_title(title)
plt.show()