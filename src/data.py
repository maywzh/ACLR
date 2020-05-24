import pandas as pd
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
