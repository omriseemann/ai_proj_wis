#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 08:15:14 2019

@author: omrisee
"""

import torch
import function_gen
import numpy as np


class ModelNN(torch.nn.Module):
    
    def __init__(self, input_shape, output_shape):
        super(ModelNN, self).__init__()
        self.activation = torch.nn.LeakyReLU()
        self.module_list = torch.nn.ModuleList()
        flag_end = False
        cnum = input_shape[0]
        cnum2 = cnum*4
        damping = 1.5
        h = int(input_shape[1]//damping)
        w = int(input_shape[2]//damping)
        while(not flag_end):
                if h==1 or w<output_shape[0]:
                    cnum2 = output_shape[1]
                    h = 1
                    w = output_shape[0]
                    flag_end = True
                self.module_list.append(torch.nn.Conv2d(cnum,cnum2,kernel_size=3, stride=1,padding=1))
                self.module_list.append(torch.nn.BatchNorm2d(cnum2))
                self.module_list.append(self.activation)
                self.module_list.append(torch.nn.AdaptiveAvgPool2d((h, w)))
                h = int(h//damping)
                w = int(w//damping)
                cnum = cnum2
                if cnum<50:
                    cnum2 = cnum2*4
    def forward(self, x):
        for i in range(len(self.module_list)):
            x = self.module_list[i](x)
        x = x.squeeze(2)
        x = x.transpose(1,2)
        return x

class Learner:
    
    def __init__(self, func_gen, batch_size = 50):
        self.functional_generator = func_gen()
        _input, _output = self.functional_generator.generateImage()
        self.input_shape = _input.shape
        self.output_shape = _output.shape
        self.learning_results = {'loss': [], 'error': []}
        self.batch_size = batch_size
        self.lr_ratio = 1e-2
        self.lr_period = 200
        self.lr_start = 1e-3
        self.reset()
        
    def reset(self, flag_model = True):
        if flag_model:
            self.model = ModelNN(self.input_shape, self.output_shape)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_start)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.cyclyc_lr())      
    
    def learn(self, n_batches=100):
        for i in range(n_batches):
            self.optimizer.zero_grad()
            batched_input, target = self.functional_generator.generateBatch(self.batch_size)
            output = self.model(batched_input)
            loss = self.functional_generator.lossBatch(output,target)
            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()
            self.learning_results['loss'].append(loss.detach().numpy())
            error = self.functional_generator.errorBatch(output, target)
            self.learning_results['error'].append(error.numpy())
            lr = self.optimizer.param_groups[0]['lr']
            print(f'epoch: {i}, loss: {loss:.3f}, error: {error:.3f}, LR:{lr:.3E}')
            
    def cyclyc_lr(self):
        down = np.log(self.lr_ratio)        
        f = lambda epoch: np.exp(down -down*(np.cos(epoch*np.pi / self.lr_period)**2))
        return f


if __name__ == '__main__':
    learner = Learner(function_gen.CaptchaGen_OS_Fixed,batch_size=10)
    learner.learn(10000)