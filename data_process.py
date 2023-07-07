from sklearn import preprocessing
from sklearn import impute
import numpy as np
import pandas as pd


def impute_data(data: np.ndarray, imputer=impute.SimpleImputer(strategy='mean', keep_empty_features=True)):
    return imputer.fit_transform(data)


def standardize_data(data: np.ndarray, scaler=preprocessing.StandardScaler()):
    return scaler.fit_transform(data)


def split_data_into_windows(X: np.ndarray, y, window_size: int):
    result_X = []
    result_y = []
    for i in range(0, np.size(X, 0)-window_size-1):
        temp = []
        for j in range(0, np.size(X, 1)):
            temp.append(X[i:i+window_size:1, j])

        result_X.append(temp)
        result_y.append(y[i+window_size+1])

    return np.array(result_X), np.array(result_y)
