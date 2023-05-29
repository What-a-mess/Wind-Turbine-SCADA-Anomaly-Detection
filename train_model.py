import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(data_x, data_y, model, loss_fn, optimizer, batch_size):
    size = np.size(data_x, 0)
    model.train()
    batch_num = int(size / batch_size + (0 if size % batch_size == 0 else 1))
    output_batches = range(0, batch_num, int(max(batch_num / 5, batch_num)))
    for batch in range(batch_num):
        left = batch*batch_size
        x, y = data_x[left: left + batch_size], data_y[left: left + batch_size]
        x, y = torch.tensor(x), torch.tensor(y)
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch in output_batches:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def autoencoder_test(test_X, test_y, model, batch_size):
    model.eval()
    test_X, test_y = torch.tensor(test_X), torch.tensor(test_y)
    test_X, test_y = test_X.to(device), test_y.to(device)
    test_decode = model.forward(test_X)
    diff_vec = (test_decode - test_y).detach().cpu().numpy()
    res_error = np.sum(np.power(diff_vec, 2), axis=1)
    # print(test_decode)
    # print(test_y)
    # print(diff_vec)

    return res_error
