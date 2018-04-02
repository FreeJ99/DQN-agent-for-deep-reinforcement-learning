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
ENV='Breakout-v0'
N_ACTIONS=4
RM_CAPACITY=10000 #replay memory capacity
BATCH_SIZE=32
GAMMA=0.99

N_EPISODES=1000
INIT_RM=1 #number of episodes used to fill replay memory

EPS_START = 1
EPS_END  = .1
EPS_STEPS = 10000

TRAIN=True


#check if gpu is is available
use_cuda = torch.cuda.is_available()
use_cuda=False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
use_cuda=True

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def editScreen(observation):
    observation=cv2.pyrDown(observation)
    observation=rgb2gray(observation)
    observation=observation[15:99,:]
    return observation

def makeState(observation_buffer):
    observation_buffer[3]=editScreen(observation_buffer[3])
    return FloatTensor(np.stack(observation_buffer,axis=0))

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
    plt.pause(0.001)

#init agent, memory and environment
agent=Agent(N_ACTIONS,EPS_START,EPS_END,EPS_STEPS,GAMMA,TRAIN,use_cuda,BATCH_SIZE)
memory=ReplayMemory(RM_CAPACITY)

env=gym.make(ENV)

ep_durations=[] #used for ploting
returns=[]



#fill replay memory with random episodes
for _ in range(INIT_RM):
    observation_buffer=[None,None,None,None]
    observation_buffer[0]=env.reset()
    for i in range(3):
        #env.render()
        action=env.action_space.sample()
        observation_buffer[i+1],r,_,_=env.step(action)
        observation_buffer[i]=editScreen(observation_buffer[i])
    cur_state=makeState(observation_buffer)
    while True:
        #env.render()
        action=env.action_space.sample()
        newO,reward,done,info=env.step(action)

        observation_buffer.pop(0)
        observation_buffer.append(newO)
        next_state=makeState(observation_buffer)

        memory.push(cur_state.unsqueeze(0),LongTensor([action]),next_state.unsqueeze(0),FloatTensor([reward]))

        cur_state=next_state

        if done:
            break

#TODO Implement None for terminal states
start_time=datetime.datetime.now()

frames=0

for i_episode in range(N_EPISODES): #start of training
    if(i_episode%10==0):
        print("Time is {0}".format(datetime.datetime.now()))
    observation_buffer=[None,None,None,None]
    observation_buffer[0]=env.reset()
    steps=0
    G=0
    for i in range(3):
        env.render()
        observation_buffer[i+1],r,_,_=env.step(env.action_space.sample())
        G+=r
        steps+=1
        observation_buffer[i]=editScreen(observation_buffer[i])
    cur_state=makeState(observation_buffer)#cur_state is already a tensor
    while True:
        env.render()
        action=agent.take_action(cur_state.unsqueeze(0))
        newo,reward,done,info=env.step(action)

        G+=reward

        observation_buffer.pop(0)
        observation_buffer.append(newo)
        next_state=makeState(observation_buffer)

        memory.push(cur_state.unsqueeze(0),LongTensor([action]),next_state.unsqueeze(0),FloatTensor([reward]))

        agent.optimize_model(memory)

        cur_state=next_state
        steps+=1

        if done:
            ep_durations.append(steps+1)
            returns.append(G)
            frames+=steps
            print("{4} Frames {0} Episode {1} finished after {2} steps with reward {3}"
                  .format(frames,i_episode, steps,G, '\033[92m' if G >= 0 else '\033[99m'))
            #plot_durations()
            break

agent.save()

plot_durations()
plt.show()
input("Press Enter to exit...")
