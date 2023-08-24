import math

import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, feature_num: int, window_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pool_size = 3
        conv_kernel_sizes = [3, 1]
        conv_channels = [feature_num, feature_num, 40]
        maxpooling_kernel_sizes = [3, 3]
        fc_in_feature = self.cal_fc_features(window_size, conv_kernel_sizes, maxpooling_kernel_sizes)
        self.conv_layers = []
        self.maxPooling_layers = []
        for i in range(len(conv_kernel_sizes)):
            self.conv_layers.append(
                nn.Conv1d(in_channels=conv_channels[i], out_channels=conv_channels[i+1], kernel_size=conv_kernel_sizes[i]).to(device)
            )
            self.maxPooling_layers.append(
                nn.MaxPool1d(kernel_size=maxpooling_kernel_sizes[i]).to(device)
            )

        # self.conv1 = nn.Conv1d(in_channels=feature_num, out_channels=feature_num, kernel_size=20).to(device)
        # self.maxPooling = nn.MaxPool1d(kernel_size=pool_size).to(device)
        # self.conv2 = nn.Conv1d(in_channels=feature_num, out_channels=5, kernel_size=5).to(device)
        self.fc1 = nn.Linear(in_features=conv_channels[-1] * fc_in_feature, out_features=20).to(device)
        self.fc2 = nn.Linear(in_features=20, out_features=2).to(device)
        self.relu = nn.ReLU(True).to(device)

    @staticmethod
    def cal_fc_features(L_in, conv_kernel_sizes, maxpooling_kernel_sizes, conv_layers_num: int = None):
        if conv_layers_num is None:
            conv_layers_num = len(conv_kernel_sizes)

        L_res = L_in
        for i in range(conv_layers_num):
            L_res = math.floor(L_res - conv_kernel_sizes[i] + 1)
            L_res = math.floor((L_res - maxpooling_kernel_sizes[i]) / maxpooling_kernel_sizes[i] + 1)

        return L_res

    def forward(self, x):
        # x = self.relu(self.conv1(x))
        # x = self.maxPooling(x)
        # x = self.relu(self.conv2(x))
        # x = self.maxPooling(x)
        for i in range(len(self.conv_layers)):
            x = self.relu(self.conv_layers[i](x))
            x = self.maxPooling_layers[i](x)

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        return x
