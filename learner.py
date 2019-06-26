#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 08:15:14 2019

@author: omrisee
"""

import torch
import capcha_gen
import numpy as np
import matplotlib.pyplot as plt
import os


class ModelCNN(torch.nn.Module):
    '''Cnn generic model, morphs input tensor C*H*W into output tensor
    C2*H2*W2'''

    def __init__(self, input_shape, output_shape):
        '''Gets input_shape of tensor and output_shape of tensor (not batches).
        Builds the model by stacking conv2d, batchnorm2d, leakyrelu and
        adaptive average pooling 2d.'''
        super(ModelCNN, self).__init__()
        self.activation = torch.nn.LeakyReLU()
        self.module_list = torch.nn.ModuleList()
        self.generate_architecture(input_shape, output_shape)
        cl = self.architecture['C']
        hl = self.architecture['H']
        wl = self.architecture['W']
        for i in range(len(cl)-1):
            c1, c2 = cl[i], cl[i+1]
            w2 = wl[i+1]
            h2 = hl[i+1]
            self.module_list.append(torch.nn.Conv2d(c1, c2, kernel_size=3,
                                                    stride=1, padding=1))
            self.module_list.append(torch.nn.BatchNorm2d(c2))
            self.module_list.append(self.activation)
            self.module_list.append(torch.nn.AdaptiveAvgPool2d((h2, w2)))

    def generate_architecture(self, input_shape, output_shape):
        ''' get the input and output shape tensors and sets achitecture dict
        of lists to help build the model.'''
        flag_end = False
        shape_multiplyer = 2
        c, h, w = input_shape[0], input_shape[1], input_shape[2]
        c2, h2, w2 = output_shape[0], output_shape[1], output_shape[2]
        cl, hl, wl = [], [], []
        cl.append(c)
        hl.append(h)
        wl.append(w)
        flag_compress = False
        c_before_compress = 200
        while not flag_end:
            if not flag_compress:
                cm = max(c_before_compress, c2 * shape_multiplyer**2)
                if cl[-1]*shape_multiplyer**2 < cm:
                    cl.append(round(cl[-1]*shape_multiplyer**2))
                else:
                    cl.append(round(cm))
                    flag_compress = True
            else:
                if not cl[-1]//shape_multiplyer**2 < c2:
                    cl.append(cl[-1]//shape_multiplyer**2)
                else:
                    cl.append(c2)
            if hl[-1]//shape_multiplyer > h2:
                hl.append(hl[-1]//shape_multiplyer)
            else:
                hl.append(h2)
            if wl[-1]//shape_multiplyer > w2:
                wl.append(wl[-1]//shape_multiplyer)
            else:
                wl.append(w2)
            if cl[-2] == c2 and wl[-2] == w2 and hl[-2] == h2:
                flag_end = True
        self.architecture = {'C': cl, 'H': hl, 'W': wl}

    def forward(self, x):
        for m in self.module_list:
            x = m(x)
        return x


class Learner:
    '''learner class to help with learning procedure, save data and plots
    results'''
    def __init__(self, func_gen, batch_size=50):
        self.functional_generator = func_gen
        _input, _output = self.functional_generator.generateImage()
        self.data_params = {}
        self.data_params['input_shape'] = _input.shape
        self.data_params['output_shape'] = _output.shape
        self.data_params['batch_size'] = batch_size
        self.learning_results = {'loss': [], 'error': []}
        self.lr_params = {'lr_ratio': 1e-2, 'lr_period': 200000,
                          'lr_start': 1e-3}
        self.save_params = {'save_dir': './saved_data', 'name': 'test'}
        self.reset()

    def reset(self, flag_model=True):
        '''helper function if you want to tweek learning parameters'''
        if flag_model:
            self.model = ModelCNN(self.data_params['input_shape'],
                                  self.data_params['output_shape'])
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr_params['lr_start'])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           self.cyclyc_lr())

    def plot_examples(self, n_examples=9):
        '''Plots examples of model results'''
        data, target = self.functional_generator.generateBatch(n_examples)
        output = self.model(data)
        self.functional_generator.plot_examples(data, target, output)

    def learn(self, n_batches=100, save_period=None):
        '''optimize the model for n_batches number of batches, save it every
        save_period number of batches.'''
        self.model.train()
        for i in range(n_batches):
            self.optimizer.zero_grad()
            batched_input, target = self.functional_generator.generateBatch(
                    self.data_params['batch_size'])
            output = self.model(batched_input)
            loss = self.functional_generator.lossBatch(output, target)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.learning_results['loss'].append(loss.detach().numpy())
            error = self.functional_generator.errorBatch(output, target)
            self.learning_results['error'].append(error.numpy())
            lr = self.optimizer.param_groups[0]['lr']
            print(f'epoch: {i}, loss: {loss:.3f}, error: {error:.3f}, ' +
                  f'LR:{lr:.3E}')
            if save_period is not None:
                if i % save_period == 0:
                    self.save()
        self.model.eval()

    def cyclyc_lr(self):
        down = np.log(self.lr_params['lr_ratio'])

        def f(epoch):
            return np.exp(down - down*(np.cos(epoch*np.pi /
                                              self.lr_params['lr_period'])**2))
        return f

    def plot(self, fnum=1):
        plt.figure(fnum)
        plt.plot(self.learning_results['loss'])
        plt.plot(self.learning_results['error'])
        plt.yscale('log')
        plt.grid(True)

    def save(self):
        name = self.save_params['name']+'.pt'
        save_dir = self.save_params['save_dir']
        optimizer_state = self.optimizer.state_dict()
        scheduler_state = self.scheduler.state_dict()
        model_state = self.model.state_dict()
        data = {'learning_results': self.learning_results,
                'model_state': model_state,
                'optimizer_state': optimizer_state,
                'scheduler_state': scheduler_state,
                'lr_params': self.lr_params,
                'data_params': self.data_params,
                'save_params': self.save_params
                }
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        spath = os.path.join(save_dir, name)
        spath = os.path.realpath(spath)
        torch.save(data, spath)

    def load(self, lpath=None):
        if lpath is None:
            name = self.save_params['name']+'.pt'
            save_dir = self.save_params['save_dir']
            lpath = os.path.join(save_dir, name)
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
    learner = Learner(capcha_gen.CaptchaGen_OS_Fixed(), batch_size=50)
    learner.save_params['name'] = 'try_after_arch'
    learner.load()
    learner.data_params['batch_size'] = 100
    learner.learn(1000)
    learner.save()
    learner.plot()
