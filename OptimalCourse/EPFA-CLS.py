import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn.functional import relu
import torch.optim as optim
import torch.nn.functional as F
from anal import np_to_tensor, tensor_to_np
import visdom
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import animation 
import csv
from matplotlib import cm
import random
from datetime import datetime

def chainrule(x,y):
    z = y(x)
    z.backward()
    return x.grad

#load a model
model = torch.load('./model/OptimalCourseNet.pth',map_location=torch.device('cpu'))

#first input value
[x1, x2, x3, x4] = [80.0, 10.0 ,315.0 ,5.0]
initData = [80.0, 10.0 ,315.0 ,5.0]
data = [x1,x2,x3,x4]
gradient_data = torch.tensor(data, requires_grad=True)
model_data = gradient_data
data_new = gradient_data
eps = 0.3
i=0
k=0
now=datetime.now()
while i<1000:
    a = [data_new[0].item(),data_new[1].item(),data_new[2].item(),data_new[3].item()]
    data_old = torch.tensor(a, requires_grad=True)
    gd = chainrule(data_old,model)

    # print(i, " ", data_new, " ", gd)
    data_new = data_old + eps*gd
    if (data_new[1]>15):
        data_new[1] = 15
    elif (data_new[1]<=0):
        data_new[1] = 0
    data_new[2] = x3
    data_new[3] = x4
    if (gd[0]>=-0.09) & (gd[0]<=0.09):
        k=k+1
    if (k>=50):
        break
    i=i+1
    
print("Initial Point : " + str(initData))
print("Local minimum occurs at: " + str(tensor_to_np(data_new)))
print("inital md : ", model(model_data).item()," CLS's MD : ",model(data_new).item())
past=datetime.now()
diff=past-now
print(diff)




