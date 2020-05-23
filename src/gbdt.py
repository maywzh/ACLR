import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from sklearn import svm


def readData(filename) -> (list, list):
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
            one_hots.append(list(map(float, eles[96:])))
            #print(idx, features, one_hots)
    labels = [np.argmax(oh) for oh in one_hots]
    return np.array(features), np.array(labels)


x_train, y_train = readData("../data/pattern-learn")
x_test, y_test = readData("../data/pattern-test")

train_num = 5000
test_num = 1000
predictor = svm.SVC(gamma='scale',
                    C=1.0,
                    decision_function_shape='ovr',
                    kernel='rbf')
predictor.fit(x_test[:train_num], y_test[:train_num])
result = predictor.predict(x_train[:test_num])
accurancy = np.sum(np.equal(result, y_train[:test_num])) / test_num
print(accurancy)
