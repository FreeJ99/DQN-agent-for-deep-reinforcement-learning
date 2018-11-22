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
ENV='CartPole-v1'
N_ACTIONS=2
RM_CAPACITY=10000 #replay memory capacity
BATCH_SIZE=64
GAMMA=0.8

N_EPISODES=1000
N_FRAMES=500000
INIT_RM=10 #number of episodes used to fill replay memory

EPS_START = 1
EPS_END  = .01
EPS_STEPS = 5000

TRAIN=True
if not TRAIN:
    EPS_START=.02

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
    fsv=np.asarray(first_state_values)
    lsv=np.asarray(last_state_values)
    plt.title('Training...Returns')
    plt.xlabel('Frames')
    plt.ylabel('Return')
    plt.plot(ep_durations,returns)
    plt.plot(ep_durations,fsv,'C1')
    plt.plot(ep_durations,lsv,'C2')
    plt.pause(0.002)

#init agent, memory and environment
agent=Agent(N_ACTIONS,EPS_START,EPS_END,EPS_STEPS,GAMMA,TRAIN,use_cuda,BATCH_SIZE,'CP')
memory=ReplayMemory(RM_CAPACITY)

env=gym.make(ENV)

ep_durations=[0] #used for ploting
returns=[0]
last_state_values=[0]
first_state_values=[0]

for i_episode in range(INIT_RM):
    if not TRAIN:
        break
    cur_state=env.reset()
    while True:
        action=agent.take_action(FloatTensor([cur_state]))
        next_state,reward,done,_=env.step(env.action_space.sample())

        if done:
            reward=-1
            memory.push(FloatTensor([cur_state]),LongTensor([action]),None,FloatTensor([reward]))
        else:
            #tensors of shape 1Xstateshape,1,1x4,1
            memory.push(FloatTensor([cur_state]),LongTensor([action]),FloatTensor([next_state]),FloatTensor([reward]))

        cur_state=next_state

        if done:
            break


start_time=time.time()
frames=0
i_episode=0

while frames<N_FRAMES: #start of training
    steps=0
    G=0
    cur_state=env.reset()
    first_state_values.append(agent.policy_net(Variable(FloatTensor([cur_state]).cuda())).max(1)[0])
    while True:
        if not TRAIN:
            env.render()
        action=agent.take_action(FloatTensor([cur_state]))
        next_state,reward,done,_=env.step(action)

        G+=reward
        if done:
            reward=-1
            memory.push(FloatTensor([cur_state]),LongTensor([action]),None,FloatTensor([reward]))
        else:
            #tensors of shape 1Xstateshape,1,1x4,1
            memory.push(FloatTensor([cur_state]),LongTensor([action]),FloatTensor([next_state]),FloatTensor([reward]))

        if TRAIN:
            agent.optimize_model(memory)

        steps+=1

        if done:
            frames+=steps
            ep_durations.append(frames)
            returns.append(G)
            last_state_values.append(agent.policy_net(Variable(FloatTensor([cur_state]).cuda())).max(1)[0])

            print("{3} Frames {0} Episode {1} finished after {2} steps"
                  .format(frames,i_episode, steps, '\033[92m' if steps >= 195 else '\033[99m'))

            if i_episode%100==0:
                plot_durations()
            i_episode+=1
            break

        cur_state=next_state


agent.save()

plot_durations()
plt.show()
input("Press Enter to exit...")
