import gym
import matplotlib.pyplot as plt
from matplotlib.pyplot import ion
import random
import numpy as np
from collections import namedtuple
from scipy import ndimage
import matplotlib.image as mpimg
import cv2
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

nActions=4
rm_capacity=100000
nEpisodes=100
BATCH_SIZE=32
GAMMA=0.99

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition=namedtuple('Transition',('state','action','next_state','reward'))

class ReplayMemory():
    def __init__(self,capacity):
        self.capacity=capacity
        self.position=0
        self.memory=[]
    def push(self,*args):
        if(len(self.memory)<self.capacity):
            self.memory.append(None)
        self.memory[self.position]=Transition(*args)
        self.position=(self.position+1)%self.capacity
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)

class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()
        self.conv1=nn.Conv2d(4,16,kernel_size=8,stride=4)
        self.conv2=nn.Conv2d(16,32,kernel_size=4,stride=2)
        self.fc1=nn.Linear(32*9*8,256)
        self.fc2=nn.Linear(256,nActions)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        return self.fc2(x)


def optimize_model():
    if len(memory.memory)<BATCH_SIZE:
        return
    transitions=memory.sample(BATCH_SIZE)
    optimizer.zero_grad()
    for transition in transitions:
        if transition.next_state is None:
            target=Variable(torch.FloatTensor([transition.reward]).unsqueeze(0).cuda())
        else:
            target=transition.reward
            q=GAMMA*model.forward(transition.next_state.cuda())
            target+=torch.max(q)
        target=target.detach()
        output=model.forward(transition.state.cuda())
        output=output[0,transition.action]
        loss=criterion(output,target)
        loss.backward()
    optimizer.step()

class Agent():
    def __init__(self,epsilon):
        self.epsilon=epsilon
        random.seed()
    def take_action(self,state):
        r=random.random()
        state=state.cuda()
        if(r<self.epsilon):
            return random.randint(0,nActions-1)
        else:
            q=model.forward(state)
            maxi=q.data[0][0]
            maxidx=0
            for i in range(1,nActions):
                if q.data[0][i]>maxi:
                    maxi=q.data[0][i]
                    maxidx=i
            return i

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def editScreen(observation):
    observation=cv2.pyrDown(observation)
    observation=rgb2gray(observation)
    observation=observation[15:99,:]
    return observation

def makeState(observations):
    observations[3]=editScreen(observations[3])
    return Variable((torch.from_numpy(np.stack(observations,axis=0))).unsqueeze(0).float())

def plot_durations():
    plt.clf()
    plt.figure(1)
    plt.subplot(121)
    durations_np=np.asarray(ep_durations)
    plt.title('Training...Durations')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_np)
    plt.subplot(122)
    returns_np=np.asarray(returns)
    plt.title('Training...Returns')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(returns)
    plt.show()

model=DQN()

if use_cuda:
    model=model.cuda()
agent=Agent(1)
memory=ReplayMemory(rm_capacity)
criterion=nn.MSELoss()
optimizer=optim.RMSprop(model.parameters())

ep_durations=[]
returns=[]

env=gym.make('Pong-v0')

for _ in range(10):
    observations=[None,None,None,None]
    observations[0]=env.reset()
    done=False
    for i in range(3):
        #env.render()
        observations[i+1],r,_,_=env.step(env.action_space.sample())
        observations[i]=editScreen(observations[i])
    cur_state=makeState(observations)
    while not done:
        #env.render()
        action=agent.take_action(cur_state)
        newO,reward,done,info=env.step(action)
        observations.pop(0)
        observations.append(newO)
        next_state=makeState(observations)
        if done:
            next_state=None
        memory.push(cur_state,action,next_state,reward)
        cur_state=next_state



for i_episode in range(nEpisodes):
    #if i_episode<=nEpisodes/10 and agent.epsilon>0.1:
    agent.epsilon=1-(i_episode/nEpisodes)
    observations=[None,None,None,None]
    observations[0]=env.reset()
    done=False
    t=0
    G=0
    for i in range(3):
        #env.render()
        observations[i+1],r,_,_=env.step(env.action_space.sample())
        G+=r
        t+=1
        observations[i]=editScreen(observations[i])
    cur_state=makeState(observations)
    while not done:
        t+=1
        #env.render()
        action=agent.take_action(cur_state)
        newO,reward,done,info=env.step(action)
        G+=reward
        observations.pop(0)
        observations.append(newO)
        next_state=makeState(observations)
        if done:
            next_state=None
        memory.push(cur_state,action,next_state,reward)
        cur_state=next_state
        optimize_model()
        if done:
            ep_durations.append(t+1)
            returns.append(G)
            print(datetime.datetime.now(), t+1, G)

plot_durations()
input("Press Enter to exit...")
