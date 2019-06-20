#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 08:15:14 2019

@author: omrisee
"""

import torch
import torchvision
import function_gen


class ModelNN(torch.nn.Module):
    
    def __init__(self,n1,n2):
        super(ModelNN, self).__init__()
        self.lin = torch.nn.Linear()
        
    def forward(self):
        raise Exception('Need to implement this function.')

class Learner:
    
    def __init__(self, func, batch_size = 50):
        self.func = func
        self.model = ModelNN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.learning_results = {'loss': []}
        self.batch_size = batch_size
    
    def learn(self, n_batches=100):
        for i in range(n_batches):
            self.optimizer.zero_grad()
            batched_input, target = self.func.generateBatch(self.batch_size)
            output = self.model(batched_input)
            loss = self.func.lossBatch(output,target)
            loss.backward()
            self.optimizer.step()


if __name__ == '__main__':
    print(1)