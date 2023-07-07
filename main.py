import numpy as np
from sklearn import preprocessing

import data_load
import data_process

if __name__ == '__main__':

    DATA_PATH = "data/Turbine_Data_Penmanshiel_11_2021-01-01_-_2021-07-01_1051.csv"
    LOGS_PATH = "data/Status_Penmanshiel_11_2021-01-01_-_2021-07-01_1051.csv"

    data = data_load.load_turbine_data_without_time(DATA_PATH)
    logs = data_load.load_turbine_logs_with_endtime(LOGS_PATH)
    labels = data_load.get_data_label(data_load.load_turbine_data(DATA_PATH), logs)
    scaler = preprocessing.StandardScaler()
    data = data_process.impute_data(data)
    data = data_process.standardize_data(data, scaler=scaler)
    print(data.shape)
    data = np.array(data, dtype='float32')

    X, y = data_process.split_data_into_windows(data, labels, 100)

    print(X, y)
    print(X.shape)