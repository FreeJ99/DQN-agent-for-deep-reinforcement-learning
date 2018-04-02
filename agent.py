import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import math

import random
from model import DQN,DQNCP
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
ByteTensor = torch.cuda.ByteTensor

class Agent():
    def __init__(self,n_actions,eps_start,eps_end,eps_steps,gamma,train,cuda,batch_size):
        self.eps_start=eps_start
        self.eps_end=eps_end
        self.eps_steps=eps_steps
        self.gamma=gamma
        self.batch_size=batch_size
        self.n_actions=n_actions
        self.steps_done=0

        self.policy_net=DQN(n_actions) # CHANGE THESE TWO LINES FOR TESTING ON CART POLE
        self.target_net=DQN(n_actions) # CHANGE THESE TWO LINES FOR TESTING ON CART POLE
        if not train:
            self.policy_net.load_state_dict(torch.load('NetParameters.txt'))
        self.update_target_net()
        if cuda:
            self.policy_net=self.policy_net.cuda()
            self.target_net=self.target_net.cuda()

        self.criterion=nn.MSELoss()
        self.optimizer=optim.RMSprop(self.policy_net.parameters())
        #self.optimizer=optim.Adam(self.policy_net.parameters(),0.001)

    def take_action(self,state):
        r=random.random()

        epsilon=self.eps_start-((self.eps_start-self.eps_end)/self.eps_steps)*self.steps_done
        #epsilon=EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)

        self.steps_done+=1
        if epsilon<self.eps_end:
            epsilon=self.eps_end
        if r<epsilon:
            return random.randint(0,self.n_actions-1)
        else:
            return self.policy_net(Variable(state.cuda(),volatile=True)).data.max(1)[1][0] #without [0] it was a long tensor of size 1,but env.step() takes a number,which is size 0

    def optimize_model(self,memory):
        if len(memory.memory)<self.batch_size:
            return
        transitions = memory.sample(self.batch_size)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)
        batch_state = Variable(torch.cat(batch_state)).cuda()
        batch_action = Variable(torch.cat(batch_action)).cuda()
        batch_reward = Variable(torch.cat(batch_reward)).cuda()
        batch_next_state = Variable(torch.cat(batch_next_state)).cuda()

        current_q_values = self.policy_net(batch_state).gather(1, batch_action.unsqueeze(1)) #action was 1 dimensional,dimensions need to match batch_state

        max_next_q_values = self.target_net(batch_next_state).detach().max(1)[0]
        expected_q_values = batch_reward + (self.gamma * max_next_q_values)

        loss=self.criterion(current_q_values,expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done%400==0: #update target net
            self.update_target_net()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self):
        print("Cuva model")
        torch.save(self.policy_net.state_dict(), 'NetParameters.txt')
        print("Sacuvao ga je")
