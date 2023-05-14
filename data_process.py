from sklearn import preprocessing
from sklearn import impute
import numpy as np
import pandas as pd


def impute_data(data: np.ndarray, imputer=impute.SimpleImputer(strategy='mean')):
    return imputer.fit_transform(data)


def standardize_data(data: np.ndarray, scaler=preprocessing.StandardScaler()):
    return scaler.fit_transform(data)
