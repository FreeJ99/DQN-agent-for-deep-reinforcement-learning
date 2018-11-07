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
ENV='CartPole-v0'
N_ACTIONS=2
RM_CAPACITY=10000 #replay memory capacity
BATCH_SIZE=64
GAMMA=0.8

N_EPISODES=200
INIT_RM=10 #number of episodes used to fill replay memory

EPS_START = 1
EPS_END  = .1
EPS_STEPS = 75000

TRAIN=True

#check if gpu is is available
use_cuda = torch.cuda.is_available()
use_cuda=False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
use_cuda=True

def plot_durations():
    plt.clf()
    plt.figure(1)
    returns_np=np.asarray(returns)
    plt.title('Training...Returns')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(returns)
    plt.pause(0.001)

#init agent, memory and environment
agent=Agent(N_ACTIONS,EPS_START,EPS_END,EPS_STEPS,GAMMA,TRAIN,use_cuda,BATCH_SIZE)
memory=ReplayMemory(RM_CAPACITY)

env=gym.make(ENV)

ep_durations=[] #used for ploting
returns=[]

start_time=time.time()
frames=0
for i_episode in range(N_EPISODES): #start of training
    steps=0
    G=0
    cur_state=env.reset()
    while True:
        #env.render()
        action=agent.take_action(FloatTensor([cur_state]))
        next_state,reward,done,_=env.step(action)

        G+=reward
        if done:       #ZANEMARIO SAM NULL STATE NA KRAJU ZA SAD DA BIH LAKSE ISTESTIRAO,ALI MOZE BITI BITNO ZA TRENING
            reward=-1   #BEZ OVOGA NECE DA IZKONVERGIRA WTF
        #    next_state=None

        #tensors of shape 1Xstateshape,1,1x4,1
        memory.push(FloatTensor([cur_state]),LongTensor([action]),FloatTensor([next_state]),FloatTensor([reward]))

        agent.optimize_model(memory)

        cur_state=next_state
        steps+=1

        if done:
            ep_durations.append(steps)
            returns.append(G)
            frames+=steps
            print("{3} Frames {0} Episode {1} finished after {2} steps"
                  .format(frames,i_episode, steps, '\033[92m' if steps >= 195 else '\033[99m'))
            plot_durations()
            break

agent.save()

plot_durations()
plt.show()
input("Press Enter to exit...")
