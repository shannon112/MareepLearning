import torch
from torch.utils import data
import random

class TwitterDataset(data.Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)
    
    __len__ will return the number of data
    """
    def __init__(self, X, y):
        self.data = X
        self.label = y
    def __getitem__(self, idx):
        # randomly sample sub-sentencce
        '''
        first_idx = random.randint(0,30-30)
        last_idx = first_idx+30
        if self.label is None: return self.data[idx][first_idx:last_idx]
        return self.data[idx][first_idx:last_idx], self.label[idx]
        '''
        # directly output
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)