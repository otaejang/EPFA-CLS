import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from anal import np_to_tensor
import csv
from datetime import datetime

#The rows in "boston_dataset.csv" represent the initial values of the algorithm, 
# and the columns represent the desired control parameters.
def EPFA_CLS(row, col,boston_x):
    arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    Bdmin = np.min(boston_x[:,col])
    Bdmax = np.max(boston_x[:,col])
    mean = [3.61, 11.36, 11.13, 0.069, 0.5546, 6.28, 68.57, 3.79, 9.5494, 408.23, 18.45, 356.67, 12.653]
    std = [8.59, 23.29, 6.85, 0.2537, 0.1157, 0.7019, 28.12, 2.103, 8.698, 168.37, 2.16, 91.2, 7.13]
    
    data = data_transform(boston_x[row],mean, std)
    data = np.asarray(data)
    EPFA_data = torch.tensor(data, requires_grad=True, dtype=torch.float32)
    EPFA_data = torch.tensor(EPFA_data)
    data_new = EPFA_data
    eps = 0.4
    i=0
    now=datetime.now()
    while i<1000:
        a = [item.item() for item in data_new]
        data_old = torch.tensor(a, requires_grad=True)
        gd = chainr(data_old,model)
        data_new = data_old + eps*gd
        k = (Bdmax-mean[col])/std[col]
        l = (Bdmin-mean[col])/std[col]
        if (data_new[col]>=k):
            data_new[col] = k
        elif (data_new[col]<=l):
            data_new[col] = l
        result = arr[:col] + arr[col+1:]
        for j in range(12):
            data_new[result[j]] = data[result[j]]
        i=i+1
    print("Initial Price : ",model(EPFA_data).item()," CLS's Price : " ,model(data_new).item())
    cls = data_Denorm(data_new, mean, std)
    ori = data_Denorm(EPFA_data, mean, std)
    print("initial point : {}, cls : {}".format(ori[col], cls[col]))
    return ori, cls

def chainr(x,y):
    z = y(x)
    z.backward()
    return x.grad

def data_transform(data, mean, std):
    x_data = [(data[i] - mean[i]) / std[i] for i in range(13)]
    return x_data

def data_Denorm(data, mean, std):
    x_data = [data[i].item() * std[i] + mean[i] for i in range(13)]
    return x_data

def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # 문자열 리스트를 숫자 리스트로 변환
            row_numeric = [float(num) for num in row]
            data.append(row_numeric)
    return data

model = torch.load('./model/BostonNet.pth',map_location=torch.device('cpu'))
csv_file_path = './boston_housing.csv'
csv_data = read_csv_file(csv_file_path)
boston = np.array(csv_data)
boston_x = boston[:,:-1]

# epfa-cls
ori, cls = EPFA_CLS(1,0,boston_x)