from sklearn import preprocessing
from sklearn import impute
import numpy as np


def impute_data(data: np.ndarray, imputer=impute.SimpleImputer(strategy='mean')):
    return imputer.fit_transform(data)


def normalize_data(data: np.ndarray, scaler=preprocessing.StandardScaler()):
    return scaler.fit_transform(data)
