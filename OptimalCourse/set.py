import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class jotDataset(Dataset):
    def __init__(self):
        self.x_csv_file = np.array(pd.read_csv("./data/labdata_x.csv"), dtype=float)
        self.y_csv_file = np.array(pd.read_csv("./data/labdata_y.csv"), dtype=float) 
        
    def __len__(self):
        return len(self.x_csv_file)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_csv_file[idx])
        y = torch.FloatTensor(self.y_csv_file[idx])
        return x,y