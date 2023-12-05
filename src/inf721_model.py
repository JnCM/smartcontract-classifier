import torch

class SmartContractClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(100, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.out = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        y_hat = torch.sigmoid(self.out(x))
        return y_hat
