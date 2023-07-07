import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cnn_train(train_X, train_y, model, loss_fn, optimizer, batch_size):
    size = np.size(train_X, 0)
    model.train()
    batch_num = int(size / batch_size + (0 if size % batch_size == 0 else 1))
    output_batches = range(0, batch_num, int(max(batch_num / 5, 1)))
    print(output_batches)
    for batch in range(batch_num):
        left = batch * batch_size
        X, y = train_X[left: left + batch_size], train_y[left: left + batch_size]
        X, y = torch.tensor(X), torch.tensor(y).to(torch.int64)
        X, y = X.to(device), y.to(device)

        # print(np.shape(X), np.shape(y))

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch in output_batches:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def cnn_test(test_X, model):
    model.eval()
    test_X = torch.tensor(test_X)
    test_X = test_X.to(device)
    predict_y = model.forward(test_X)

    return predict_y
