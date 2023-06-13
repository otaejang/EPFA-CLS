from set import jotDataset
from Model import Net
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from anal import saveModel
import datetime
import argparse
import os

def get_args():
    parser=argparse.ArgumentParser("otjang, shjo, shkim")
    parser.add_argument('--data_path', type=str, default='data/', help='the root folder of dataset')
    parser.add_argument('--saved_path', type=str, default='model/')
    args = parser.parse_args()
    return args

def train():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    dataset = jotDataset()
    dataloader = DataLoader(dataset,batch_size=1000,shuffle=True)

    model = Net()
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    np_epochs = 15000
    for epoch in range(np_epochs+1):
        for batch_idx, samples in enumerate(dataloader):
            x_train, y_train = samples
            
            if torch.cuda.is_available():
                x_train=x_train.cuda()
                y_train=y_train.cuda()

            prediction = model(x_train)
            cost = F.mse_loss(prediction, y_train)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            print('epoch {:4d}/{} batch {}/{} cost {:.6f}'.format(epoch,np_epochs,batch_idx+1,len(dataloader),cost.item()))
    return model

if __name__ == '__main__':
    args = get_args()
    dt_now = datetime.datetime.now()
    save_time=str(dt_now.year)+'-'+str(dt_now.month)+'-'+str(dt_now.day)+'-'+str(dt_now.hour)+'-'+str(dt_now.minute)
    args.saved_path=args.saved_path+f'{save_time}/'
    os.makedirs(args.saved_path, exist_ok=True)
    path_model = os.path.join(args.saved_path, f'BostonNet_DV_{save_time}.pth')
    model = train()
    saveModel(model, path_model)
    print(model)
