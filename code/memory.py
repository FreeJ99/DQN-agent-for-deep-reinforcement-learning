from collections import namedtuple
import random

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
