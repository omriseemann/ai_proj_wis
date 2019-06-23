#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 08:15:14 2019

@author: omrisee
"""

import torch
import function_gen
import numpy as np
import matplotlib.pyplot as plt
import os

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
        self.data_params = {}
        self.data_params['input_shape'] = _input.shape
        self.data_params['output_shape'] = _output.shape
        self.data_params['batch_size'] = batch_size
        self.learning_results = {'loss': [], 'error': []}
        self.lr_params = {'lr_ratio' : 1e-2, 'lr_period' : 200000, 'lr_start': 1e-3}
        self.save_params = {'save_dir' : './saved_data', 'name': 'test'}
        self.reset()
        
    def reset(self, flag_model = True):
        if flag_model:
            self.model = ModelNN(self.data_params['input_shape'], self.data_params['output_shape'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_params['lr_start'])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.cyclyc_lr())      
    
    def learn(self, n_batches=100):
        self.model.train()
        for i in range(n_batches):
            self.optimizer.zero_grad()
            batched_input, target = self.functional_generator.generateBatch(self.data_params['batch_size'])
            output = self.model(batched_input)
            loss = self.functional_generator.lossBatch(output,target)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.learning_results['loss'].append(loss.detach().numpy())
            error = self.functional_generator.errorBatch(output, target)
            self.learning_results['error'].append(error.numpy())
            lr = self.optimizer.param_groups[0]['lr']
            print(f'epoch: {i}, loss: {loss:.3f}, error: {error:.3f}, LR:{lr:.3E}')
        self.model.eval()
            
    def cyclyc_lr(self):
        down = np.log(self.lr_params['lr_ratio'])        
        f = lambda epoch: np.exp(down -down*(np.cos(epoch*np.pi / self.lr_params['lr_period'])**2))
        return f
    
    def plot(self, fnum=1):
        f = plt.figure(fnum)
        plt.plot(self.learning_results['loss'])
        plt.plot(self.learning_results['error'])
    
    def save(self):
        name = self.save_params['name']+'.pt'
        save_dir = self.save_params['save_dir']
        optimizer_state = self.optimizer.state_dict()
        scheduler_state = self.scheduler.state_dict()
        model_state = self.model.state_dict()
        data = {'learning_results' : self.learning_results,
                'model_state' : model_state,
                'optimizer_state' : optimizer_state,
                'scheduler_state' : scheduler_state,
                'lr_params' : self.lr_params,
                'data_params' : self.data_params,
                'save_params' : self.save_params
                }
        spath = os.path.join(save_dir,name)
        spath = os.path.realpath(spath)
        torch.save(data,spath)
    
    def load(self,lpath = None):
        if lpath is None:
            name = self.save_params['name']+'.pt'
            save_dir = self.save_params['save_dir']
            lpath = os.path.join(save_dir,name)
            lpath = os.path.realpath(lpath)
        data = torch.load(lpath)
        self.learning_results = data['learning_results']
        self.save_params = data['save_params']
        self.data_params = data['data_params']
        self.lr_params = data['lr_params']
        self.model.load_state_dict(data['model_state'])
        self.optimizer.load_state_dict(data['optimizer_state'])
        self.scheduler.load_state_dict(data['scheduler_state'])


if __name__ == '__main__':
    learner = Learner(function_gen.CaptchaGen_OS_Fixed,batch_size=50)
    for i in range(5):
        learner.learn(1000)
        learner.save()
        learner.plot()