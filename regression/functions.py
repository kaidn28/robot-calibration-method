import numpy as np
import pandas as pd

def foldSplit(data, k):
    #print(data.shape[0])
    fold_length = data.shape[0]//k
    folds = []
    for i in range(k):
        ith_fold = data[i*fold_length: (i+1)*fold_length]
        #print(ith_fold.shape)
        folds.append(ith_fold)
    #print(folds)
    return folds
##linear no bias
def ithFoldTrainTestSplit(folds, i):
    folds_ = folds.copy()
    #print(len(folds_))
    val = folds_.pop(i)

    x_val = val[:, :2]
    y_val = val[:, 2:]
    #print(len(folds_))
    train = np.concatenate(folds_)
    x_train = train[:, :2]
    y_train = train[:, 2:]
    #print(train.shape)
    return x_train, x_val, y_train, y_val