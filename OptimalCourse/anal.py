from numpy import testing
from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import T
from Model import Net

import matplotlib.pyplot as plt 
from matplotlib import pyplot
from sklearn.metrics import r2_score
import math

def tensor_to_np(data):
    data_np = data.detach().cpu().numpy()
    return data_np

def np_to_tensor(data):
    cuda = torch.device("cuda")
    data_tn = torch.FloatTensor(data).cuda()

    return data_tn

def saveModel(model, path):
    torch.save(model, path)

def loadModel(path):
    model = torch.load(path).cuda()
    model.eval()
    return model

def loadFile(path):
    file = np.array(pd.read_csv(path))
    return file

def saveFile(path, file):
    np.savetxt(path, file, delimiter=",")

def showGraph(actual,pred,path):
    plt.scatter(pred, actual)
    plt.title('Actual VS Prediction')
    plt.xlabel('pred')
    plt.ylabel('actual')
    plt.show()
    plt.savefig('./picture_data/Train')

def showR2(actual, pred):
    TestR2Value = r2_score(actual, pred)
    return TestR2Value

def showMean(data):
    mean = np.mean(data)
    return mean

def predict_index(input):
    for i, v in enumerate(input):
        if v == max(input):
            index = i
            max_value = v
    return index, max_value

def predict_input(input, index):
    for i,v in enumerate(input):
        if i == index:
            pred = v
    return pred
