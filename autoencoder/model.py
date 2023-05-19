import torch
from torch import nn, optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoEncoder(nn.Module):
    def __init__(self, input_size: int, inner_size=12):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 32),
                                     nn.ReLU(True),
                                     nn.Linear(32, inner_size)).to(device)
        self.decoder = nn.Sequential(nn.Linear(inner_size, 32),
                                     nn.ReLU(True),
                                     nn.Linear(32, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, input_size)).to(device)

    def forward(self, x):
        encode_res = self.encoder(x)
        decode_res = self.decoder(encode_res)

        return decode_res


class GetRnnOutput(nn.Module):
    def forward(self, x):
        out, _ = x
        return out


class RnnAutoEncoder(nn.Module):
    def __init__(self, input_size: int, inner_size=20):
        super(RnnAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.ReLU(),
                                     nn.RNN(32, inner_size),
                                     GetRnnOutput()).to(device)
        self.decoder = nn.Sequential(nn.RNN(inner_size, 32),
                                     GetRnnOutput(),
                                     nn.ReLU(),
                                     nn.Linear(32, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, input_size)).to(device)

    def forward(self, x):
        encode_res = self.encoder(x)
        decode_res = self.decoder(encode_res)

        return decode_res


class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size: int, inner_size=20):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.ReLU(),
                                     nn.LSTM(32, inner_size),
                                     GetRnnOutput()).to(device)
        self.decoder = nn.Sequential(nn.LSTM(inner_size, 32),
                                     GetRnnOutput(),
                                     nn.ReLU(),
                                     nn.Linear(32, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, input_size)).to(device)

    def forward(self, x):
        encode_res = self.encoder(x)
        decode_res = self.decoder(encode_res)

        return decode_res
