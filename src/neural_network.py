import torch
from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device}")


class NeuralNetwork(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=vocab_size, out_features=512),
            #nn.ReLU(),
            #nn.Dropout(0.5),
            #nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        logits = self.linear_relu_stack(x)
        return torch.sigmoid(logits)#.squeeze()

