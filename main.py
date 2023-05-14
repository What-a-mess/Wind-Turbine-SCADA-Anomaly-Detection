import numpy as np
import torch
from torch import nn
from sklearn.model_selection import RepeatedKFold
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

import data_load
import data_process
import model
import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_vanilla_autoencoder():
    epochs = 3000
    data = data_load.load_turbine_standardized_data_without_time(
        "data/Turbine_Data_Penmanshiel_11_2021-01-01_-_2021-07-01_1051.csv")
    data_imputer = KNNImputer(n_neighbors=10)
    data = data_process.impute_data(data)
    data = data_process.normalize_data(data)
    data = np.array(data, dtype='float32')

    k = 10
    kf = RepeatedKFold(n_splits=k, n_repeats=1)
    i=0

    for train, test in kf.split(data):
        ae_model = model.AutoEncoder(input_size=np.size(data, axis=1)).to(device)
        loss = nn.MSELoss()
        # optimizer = torch.optim.SGD(ae_model.parameters(), lr=0.001)
        optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.0015)
        X_train, X_test = data[train], data[test]
        for epoch in range(epochs):
            print(f"\n=====epoch {epoch}/{epochs}=====")
            train_model.train(data_x=X_train, data_y=X_train, model=ae_model, loss_fn=loss, optimizer=optimizer,
                              batch_size=30000)

        res = train_model.autoencoder_test(X_test, X_test, ae_model, 1000)
        print(res)
        plt.hist(res)
        plt.show()
        i += 1
        if i == 1:
            break

    # ae_model = model.AutoEncoder(input_size=np.size(data, axis=1)).to(device)
    # loss = nn.MSELoss()
    # # optimizer = torch.optim.SGD(ae_model.parameters(), lr=0.001)
    # optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.001)
    # for epoch in range(epochs):
    #     print(f"\n=====epoch {epoch}/{epochs}=====")
    #     train_model.train(data_x=data, data_y=data, model=ae_model, loss_fn=loss, optimizer=optimizer,
    #                       batch_size=100)


if __name__ == '__main__':
    test_vanilla_autoencoder()
