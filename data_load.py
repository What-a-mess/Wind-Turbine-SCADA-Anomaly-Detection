import numpy as np
import pandas as pd
from sklearn import preprocessing

import data_process


@DeprecationWarning
def load_data(data_path: str):
    data = np.loadtxt(data_path, dtype=str, delimiter=",")
    return data


def load_turbine_data_without_time(data_path: str):
    data = pd.read_csv(data_path, sep=",", header=None, comment='#')
    data = data.drop(0, axis=1)
    return data


def load_turbine_standardized_data_without_time(data_path: str):
    data = load_turbine_data_without_time(data_path)
    return data


if __name__ == '__main__':
    test_data = load_turbine_standardized_data_without_time("data/Turbine_Data_Penmanshiel_11_2021-01-01_-_2021-07-01_1051.csv")
    print(test_data)
