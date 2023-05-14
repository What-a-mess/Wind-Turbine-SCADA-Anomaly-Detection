import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing

import data_process


def load_turbine_data(data_path: str):
    data = pd.read_csv(data_path, sep=",", header=[9])
    return data


def load_turbine_logs(log_path: str):
    logs = pd.read_csv(log_path, sep=",", comment="#")
    return logs


def load_turbine_data_without_time(data_path: str):
    data = load_turbine_data(data_path)
    data = data.drop("# Date and time", axis=1)
    return data


def load_turbine_logs_with_endtime(logs_path: str):
    logs = load_turbine_logs(logs_path)
    for index, row in logs.iterrows():
        if row["Timestamp end"] == '-':
            logs.loc[index, "Timestamp end"] = row["Timestamp start"]

    return logs


def get_data_label(data, logs):
    def get_row_type(data):
        has_low_wind_speed = False
        had_stop = False
        for index, row in data.iterrows():
            if row["IEC category"] == "Full Performance":
                continue
            if row["Service contract category"] != "External stop (low wind speed)" and row["IEC category"] != "Technical Standby":
                if row["Message"] != "Brake program 50":
                    return 1
                else:
                    return -1 # 目前还不知道Brake program 50是什么
            if row["Status"] == "Stop":
                had_stop = True
            if row["Service contract category"] == "External stop (low wind speed)  (5)":
                has_low_wind_speed = True

        if had_stop:
            return 2
        elif has_low_wind_speed:
            return 3
        else:
            return 0

    data = pd.DataFrame(data)
    logs = pd.DataFrame(logs)

    logs["Timestamp start"] = pd.to_datetime(logs["Timestamp start"], format="%Y-%m-%d %H:%M:%S")
    logs["Timestamp end"] = pd.to_datetime(logs["Timestamp end"], format="%Y-%m-%d %H:%M:%S")
    res = []
    for index, row in data.iterrows():
        row_date = row["# Date and time"]
        row_date = datetime.datetime.strptime(row_date, "%Y-%m-%d %H:%M:%S")
        duration = datetime.timedelta(minutes=10)
        # 分为两种情况：
        # 1. 记录时间在log时间前
        # 2. 记录时间在log时间后
        # two cases here:
        # 1. the data recorded time is before the log you need
        # 2. the data time is after the log time (record is inside log event's duration)
        valid_rows = ((row_date < logs["Timestamp start"]) &
                      (row_date + duration > logs["Timestamp start"])) | \
                     ((row_date > logs["Timestamp start"]) &
                      (row_date < logs["Timestamp end"]))

        # valid_logs = logs[(logs["Timestamp start"] < row_date) & (logs["Timestamp end"] > row_date)]
        valid_logs = logs[valid_rows]
        if valid_logs.size > 0:
            row_type = get_row_type(valid_logs)
            if row_type != 0:
                res.append(1)
            else:
                res.append(0)
        else:
            res.append(0)

    return res


if __name__ == '__main__':
    data_path = "data/Turbine_Data_Penmanshiel_11_2021-01-01_-_2021-07-01_1051.csv"
    logs_path = "data/Status_Penmanshiel_11_2021-01-01_-_2021-07-01_1051.csv"
    test_data = load_turbine_data(data_path)
    test_logs = load_turbine_logs_with_endtime(logs_path)
    label = get_data_label(test_data, test_logs)
    print(label)
    print(np.sum(label), "/", len(label))
    # print(test_data)
