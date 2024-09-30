import torch
from sklearn.metrics import precision_score, recall_score, f1_score




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

    #Liste in cui memorizzo etichette vere e previste
    all_pred_labels = []
    all_true_labels = []

    with torch.no_grad():
        for X, y in dataloader:

            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            pred_probs = torch.sigmoid(pred)

            pred_labels = (pred_probs > 0.5).float()

            all_pred_labels.extend(pred_labels.cpu().numpy())
            all_true_labels.extend(y.cpu().numpy())

            correct += (pred_labels == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    #Calcolo precision, recall e f1
    precision = precision_score(all_true_labels, all_pred_labels)
    recall = recall_score(all_true_labels, all_pred_labels)
    f1 = f1_score(all_true_labels, all_pred_labels)

    print(f"Test error: \n Accuracy: {(100 * correct):>0.1f}%, Average loss: {test_loss:>8f}% \n")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f} \n")