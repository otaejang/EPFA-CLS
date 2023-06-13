import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        h1=nn.Linear(13,1024)
        h2=nn.Linear(1024,1024)
        h3=nn.Linear(1024,1)
        self.hidden = nn.Sequential(
            h1,
            nn.ReLU(),
            h2,
            nn.ReLU(),
            h3
        )
    def forward(self,x):
        o=self.hidden(x)
        return o