import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    total_loss = 0
    model.train()

    train_losses = []

    for batch, (X, y) in enumerate(dataloader):

        # Prediction e loss
        pred = model(X).squeeze()
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if batch % 50 == 0 or batch == len(dataloader) - 1:
            current = batch * batch_size + len(X)
            avg_loss = total_loss / (batch + 1)
            train_losses.append(avg_loss)
            print(f"Avg loss: {avg_loss:>7f}, Progress: [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()

    size = len(dataloader.dataset)
    print(f"Size: {size}")
    num_batches = len(dataloader)
    print(f"Num batches: {num_batches}")
    test_loss, correct = 0, 0

    # Liste in cui memorizzo etichette vere e previste
    all_pred_labels = []
    all_true_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            pred_probs = model(X).squeeze()

            test_loss += loss_fn(pred_probs, y).item()

            #pred_probs = torch.sigmoid(pred)

            pred_labels = (pred_probs > 0.5).float()

            all_pred_labels.extend(pred_labels.cpu().numpy().astype(int))
            all_true_labels.extend(y.cpu().numpy().astype(int))

            correct += (pred_labels == y.float()).sum().item()

    print(f"correct: {correct}")
    test_loss /= num_batches
    correct /= size

    # Calcolo precision, recall e f1
    precision = precision_score(all_true_labels, all_pred_labels)
    recall = recall_score(all_true_labels, all_pred_labels)
    f1 = f1_score(all_true_labels, all_pred_labels)

    print(f"Test error: \n Accuracy: {(100 * correct):>0.1f}%, Average loss: {test_loss:>8f}% \n")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f} \n")
