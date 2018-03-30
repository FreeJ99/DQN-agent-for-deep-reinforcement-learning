import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

import random
from model import DQN

class Agent():
    def __init__(self,n_actions,eps_start,eps_end,eps_steps,gamma,train,cuda,batch_size):
        self.eps_start=eps_start
        self.eps_end=eps_end
        self.eps_steps=eps_steps
        self.gamma=gamma
        self.batch_size=batch_size
        self.n_actions=n_actions
        self.frames=0

        self.policy_net=DQN(n_actions)
        self.target_net=DQN(n_actions)
        if not train:
            self.policy_net.load_state_dict(torch.load('NetParameters.txt'))
        self.update_target_net()
        if cuda:
            self.policy_net=self.policy_net.cuda()
            self.target_net=self.target_net.cuda()

        self.criterion=nn.MSELoss()
        self.optimizer=optim.RMSprop(self.policy_net.parameters())
        random.seed(10)

    def take_action(self,state):
        r=random.random()
        self.frames+=1
        epsilon=self.eps_start-((self.eps_start-self.eps_end)/self.eps_steps)*self.frames
        if epsilon<self.eps_end:
            epsilon=self.eps.end
        if r<epsilon:
            return random.randint(0,self.n_actions-1)
        else:
            state=state.cuda()
            q=self.policy_net.forward(state)
            maxi=q.data[0][0]
            maxidx=0
            for i in range(1,self.n_actions):
                if q.data[0][i]>maxi:
                    maxi=q.data[0][i]
                    maxidx=i
            return i

    def optimize_model(self,mem):
        if len(mem.memory)<self.batch_size:
            return
        transitions=mem.sample(self.batch_size)
        #q=target_model.forward()
        self.optimizer.zero_grad()
        for transition in transitions:
            if transition.next_state is None:
                target=Variable(torch.FloatTensor([transition.reward]).unsqueeze(0).cuda())
            else:
                target=transition.reward
                q=self.gamma*self.target_net.forward(transition.next_state.cuda())
                target+=torch.max(q)
            target=target.detach() #for some reason it won't work without this
            output=self.policy_net.forward(transition.state.cuda())
            output=output[0,transition.action]
            loss=self.criterion(output,target)
            loss.backward()
        self.optimizer.step()

        if self.frames%400==0: #update target net
            self.update_target_net()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self):
        torch.save(policy_net.state_dict(), 'NetParameters.txt')
