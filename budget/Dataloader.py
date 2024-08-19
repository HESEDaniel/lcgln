from os.path import *
from torch.utils.data import Dataset, DataLoader

class ICLNDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
    
def ICLN_data_loader(X, y, batch_size):
    icln_dataset = ICLNDataset(X, y)
    icln_dataloader = DataLoader(icln_dataset, batch_size=batch_size, shuffle=False)
    
    return icln_dataloader
