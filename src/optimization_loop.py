import torch
from torch import nn


def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    total_loss = 0
    model.train()

    for batch, (X, y) in enumerate(dataloader):

        #Prediction e loss
        pred = model(X)
        loss = loss_fn(pred, y)

        #Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if batch % 50 == 0 or batch == len(dataloader) - 1:
            current = batch * batch_size + len(X)
            avg_loss = total_loss / (batch + 1)
            print(f"Avg loss: {avg_loss:>7f}, Progress: [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:

            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            pred_probs = torch.sigmoid(pred)

            pred_labels = (pred_probs > 0.5).float()

            correct += (pred_labels == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test error: \n Accuracy: {(100 * correct):>0.1f}%, Average loss: {test_loss:>8f}% \n")
