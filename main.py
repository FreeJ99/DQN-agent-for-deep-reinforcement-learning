import gym
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import namedtuple
import cv2,datetime,time,math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

from agent import Agent
from memory import ReplayMemory

#constants
ENV='Pong-v0'
N_ACTIONS=6
RM_CAPACITY=100000 #replay memory capacity
BATCH_SIZE=32
GAMMA=0.99

N_EPISODES=100
INIT_RM=1 #number of episodes used to fill replay memory

EPS_START = 1
EPS_END  = .1
EPS_STEPS = 75000

TRAIN=True

#check if gpu is is available
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


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

#init agent, memory and environment
agent=Agent(N_ACTIONS,EPS_START,EPS_END,EPS_STEPS,GAMMA,TRAIN,use_cuda,BATCH_SIZE)
memory=ReplayMemory(RM_CAPACITY)

env=gym.make(ENV)

ep_durations=[] #used for ploting
returns=[]

start_time=time.time()

#fill replay memory with random episodes
for _ in range(INIT_RM):
    observations=[None,None,None,None]
    observations[0]=env.reset()
    done=False
    for i in range(3):
        env.render()
        action=env.action_space.sample()
        observations[i+1],r,_,_=env.step(action)
        observations[i]=editScreen(observations[i])
    cur_state=makeState(observations)
    while not done:
        env.render()
        newO,reward,done,info=env.step(env.action_space.sample())
        observations.pop(0)
        observations.append(newO)
        next_state=makeState(observations)
        if done:
            next_state=None
        memory.push(cur_state,action,next_state,reward)
        cur_state=next_state


print(datetime.datetime.now(),'Pocetak treninga')
frames=0

for i_episode in range(N_EPISODES): #start of training
    observations=[None,None,None,None]
    observations[0]=env.reset()
    done=False
    t=0
    G=0
    for i in range(3):
        env.render()
        observations[i+1],r,_,_=env.step(env.action_space.sample())
        G+=r
        t+=1
        observations[i]=editScreen(observations[i])
    cur_state=makeState(observations)
    while not done:
        env.render()
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
        agent.optimize_model(memory)
        if done:
            ep_durations.append(t+1)
            returns.append(G)
            print(datetime.datetime.now(),math.floor(time.time()-start_time),'ASD', t+1, G)
        t+=1

agent.save()

plot_durations()
input("Press Enter to exit...")
