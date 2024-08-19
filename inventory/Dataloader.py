from os.path import *
from torch.utils.data import Dataset, DataLoader
class ICNNDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def ICNN_data_loader(X, y, batch_size, shuffle=False):
    icnn_dataset = ICNNDataset(X, y)
    icnn_dataloader = DataLoader(icnn_dataset, batch_size=batch_size, shuffle=shuffle)
    return icnn_dataloader