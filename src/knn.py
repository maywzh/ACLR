import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import jaccard_similarity_score, cohen_kappa_score, hamming_loss
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import calinski_harabasz_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
#海明距离也适用于多分类的问题，简单来说就是衡量预测标签与真实标签之间的距离，取值在0~1之间。距离为0说明预测结果与真实结果完全相同，距离为1就说明模型与我们想要的结果完全就是背道而驰。
#kappa系数是用在统计学中评估一致性的一种方法，取值范围是[-1,1]，实际应用中，一般是[0,1]，与ROC曲线中一般不会出现下凸形曲线的原理类似。这个系数的值越高，则代表模型实现的分类准确度越高。
#它与海明距离的不同之处在于分母。当预测结果与实际情况完全相符时，系数为1；当预测结果与实际情况完全不符时，系数为0；当预测结果是实际情况的真子集或真超集时，距离介于0到1之间。我们可以通过对所有样本的预测情况求平均得到算法在测试集上的总体表现情况。
#铰链损失（Hinge loss）一般用来使“边缘最大化”（maximal margin）。损失取值在0~1之间，当取值为0，表示多分类模型分类完全准确，取值为1表明完全不起作用。
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

model = KNeighborsClassifier(n_neighbors=10,
                             algorithm='auto',
                             weights='distance',
                             n_jobs=-1)
scores_clf_svc_cv = cross_val_score(model, x_train, y_train, cv=5)
print(scores_clf_svc_cv)
print("Accuracy: %0.2f (+/- %0.2f)" %
      (scores_clf_svc_cv.mean(), scores_clf_svc_cv.std() * 2))

model.fit(x_train, y_train)

precition = model.score(x_test, y_test)

print('precition is : ', precition * 100, "%")