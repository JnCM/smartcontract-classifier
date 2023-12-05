from torch.utils.data import Dataset

class SmartContractDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_value = self.X[idx]
        y_value = self.y[idx]
        return x_value, y_value
