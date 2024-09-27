from torch.utils.data import Dataset
import torch
import scipy.sparse as sp


class DocumentDataset(Dataset):
    def __init__(self, data_matrix, labels):
        if sp.issparse(data_matrix):
            # Converte la matrice sparsa in una densa
            data_matrix = data_matrix.todense()
        self.data_matrix = torch.tensor(data_matrix, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if idx >= len(self.data_matrix):
            raise IndexError("Index out of bounds")
        return self.data_matrix[idx], self.labels[idx]
