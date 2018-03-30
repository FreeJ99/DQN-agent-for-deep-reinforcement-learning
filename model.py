import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

class DQN(nn.Module):
    def __init__(self,n_actions):
        super(DQN,self).__init__()
        self.conv1=nn.Conv2d(4,16,kernel_size=8,stride=4)
        self.conv2=nn.Conv2d(16,32,kernel_size=4,stride=2)
        self.fc1=nn.Linear(32*9*8,256)
        self.fc2=nn.Linear(256,n_actions)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        return self.fc2(x)
