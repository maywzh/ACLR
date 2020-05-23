import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import jaccard_similarity_score, cohen_kappa_score, hamming_loss
from sklearn.multiclass import OneVsRestClassifier
#海明距离也适用于多分类的问题，简单来说就是衡量预测标签与真实标签之间的距离，取值在0~1之间。距离为0说明预测结果与真实结果完全相同，距离为1就说明模型与我们想要的结果完全就是背道而驰。
#kappa系数是用在统计学中评估一致性的一种方法，取值范围是[-1,1]，实际应用中，一般是[0,1]，与ROC曲线中一般不会出现下凸形曲线的原理类似。这个系数的值越高，则代表模型实现的分类准确度越高。
#它与海明距离的不同之处在于分母。当预测结果与实际情况完全相符时，系数为1；当预测结果与实际情况完全不符时，系数为0；当预测结果是实际情况的真子集或真超集时，距离介于0到1之间。我们可以通过对所有样本的预测情况求平均得到算法在测试集上的总体表现情况。
#铰链损失（Hinge loss）一般用来使“边缘最大化”（maximal margin）。损失取值在0~1之间，当取值为0，表示多分类模型分类完全准确，取值为1表明完全不起作用。


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

model = KNeighborsClassifier(n_neighbors=10)
scores_clf_svc_cv = cross_val_score(model, x_train, y_train, cv=5)
print(scores_clf_svc_cv)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores_clf_svc_cv.mean(), scores_clf_svc_cv.std() * 2))

model.fit(x_train, y_train)

precition = model.score(x_test, y_test)

print('precition is : ', precition * 100, "%")