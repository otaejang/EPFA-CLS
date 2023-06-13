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
        self.x_csv_file = np.array(pd.read_csv("./data/train/labdata_x.csv"), dtype=float)
        self.y_csv_file = np.array(pd.read_csv("./data/train/labdata_y.csv"), dtype=float) 
        
    def __len__(self):
        return len(self.x_csv_file)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_csv_file[idx])
        y = torch.FloatTensor(self.y_csv_file[idx])
        return x,y
    
def data_transform(data):
    x_data = (data[0]-3.61)/8.59,(data[1]-11.36)/23.29,(data[2]-11.13)/6.85,(data[3]-0.069)/0.2537,(data[4]-0.5546)/0.1157,(data[5]-6.28)/0.7019,(data[6]-68.57)/28.12,(data[7]-3.79)/2.103,(data[8]-9.5494)/8.698,(data[9]-408.23)/168.37,(data[10]-18.45)/2.16,(data[11]-356.67)/91.2,(data[12]-12.653)/7.13
    x_data = torch.Tensor(x_data)
    return x_data