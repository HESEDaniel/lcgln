import numpy as np
import os
import pickle
import torch
import random
from os.path import *
from torch.utils.data import Dataset, DataLoader

class LCGLNDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
    
def LCGLN_data_loader(X, y, batch_size, shuffle=False):
    lcgln_dataset = LCGLNDataset(X, y)
    lcgln_dataloader = DataLoader(lcgln_dataset, batch_size=batch_size, shuffle=shuffle)
    return lcgln_dataloader
