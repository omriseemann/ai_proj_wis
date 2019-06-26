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
from models import ModelCNN


class Learner:
    '''learner interface '''
    def __init__(self, func_gen, model_class, lr_start=1e-3, lr_ratio=1e-2,
                 lr_period=1e6):
        self.model_class = model_class
        self.functional_generator = func_gen
        self.learning_results = {}
        self.log = []
        self.save_params = {'save_dir': './saved_data', 'name': 'test'}
        self.reset(lr_start, lr_ratio, lr_period)

    def reset(self, lr_start, lr_ratio, lr_period, flag_model=True):
        '''helper function if you want to tweek learning parameters'''
        if flag_model:
            self.model = self.model_class(
                    self.functional_generator.input_shape,
                    self.functional_generator.output_shape)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr_start)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, self.cyclyc_lr(lr_ratio, lr_period))
        self.log.append(f'reset: init {flag_model}, lr_start:{lr_start:.3E}, '
                        + f'lr_ratio:{lr_ratio:.3E}, lr_period:{lr_period:.3E}'
                        + '.')

    def cyclyc_lr(self, lr_ratio, lr_period):
        '''helper function for cyclic learning rate'''
        down = np.log(lr_ratio)

        def f(epoch):
            return np.exp(down - down*(np.cos(epoch*np.pi / lr_period)**2))
        return f

    def to_data_struct(self):
        '''data structure generator for saveing'''
        optimizer_state = self.optimizer.state_dict()
        scheduler_state = self.scheduler.state_dict()
        model_state = self.model.state_dict()
        data = {'learning_results': self.learning_results,
                'model_state': model_state,
                'optimizer_state': optimizer_state,
                'scheduler_state': scheduler_state,
                'save_params': self.save_params,
                'log': self.log
                }
        return data

    def to_self(self, data):
        '''data structure generator for loading'''
        self.learning_results = data['learning_results']
        self.save_params = data['save_params']
        self.model.load_state_dict(data['model_state'])
        self.optimizer.load_state_dict(data['optimizer_state'])
        self.scheduler.load_state_dict(data['scheduler_state'])
        self.log = data['log']

    def save(self):
        data = self.to_data_struct()
        name = self.save_params['name']+'.pt'
        save_dir = self.save_params['save_dir']
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
        self.to_self(data)
        return data

    def plot_examples(self, n_examples=9):
        '''Plots examples of model results'''
        data, target = self.functional_generator.generateBatch(n_examples)
        output = self.model(data)
        self.functional_generator.plot_examples(data, target, output)


class LearnerDatasetGenerative(Learner):
    '''learner class for dataset generated'''
    def __init__(self, func_gen, model_class, lr_start=1e-3, lr_ratio=1e-2,
                 lr_period=1e6):
        super(LearnerDatasetGenerative, self).__init__(func_gen, model_class,
                                                       lr_start, lr_ratio,
                                                       lr_period)
        self.learning_results = {'train_loss': [], 'valid_loss': [],
                                 'error': []}
        self.datasets = {'train': None, 'valid': None}

    def generateNewDataset(self, train_size=1, valid_size=1,
                           flag_verbose=False):
        train_ds = self.functional_generator.generateNewDataset(
                            train_size, flag_verbose=flag_verbose)
        valid_ds = self.functional_generator.generateNewDataset(
                            valid_size, flag_verbose=flag_verbose)
        self.datasets = {'train': train_ds, 'valid': valid_ds}
        self.log.append(f'Generated datasets of size ({train_size}, ' +
                        f'{valid_size}).')

    def learn(self, batch_size=10, save_period=None):
        pass

    def plot(self, fnum=1):
        plt.figure(fnum)
        plt.plot(self.learning_results['train_loss'])
        plt.plot(self.learning_results['valid_loss'])
        plt.plot(self.learning_results['error'])
        plt.legend(['train', 'valid', 'error'])
        plt.yscale('log')
        plt.grid(True)

    def to_data_struct(self):
        '''data structure generator for saveing'''
        data = super(LearnerDatasetGenerative, self).to_data_struct()
        data['datasets'] = self.datasets
        return data

    def to_self(self, data):
        '''data structure generator for loading'''
        super(LearnerDatasetGenerative, self).to_self(data)
        self.datasets = data['datasets']


class LearnerGenerative(Learner):
    '''learner class to help with learning procedure, save data and plots
    results'''
    def __init__(self, func_gen, model_class, lr_start=1e-3, lr_ratio=1e-2,
                 lr_period=1e6):
        super(LearnerGenerative, self).__init__(func_gen, model_class,
                                                lr_start, lr_ratio, lr_period)
        self.learning_results = {'loss': [], 'error': []}

    def learn(self, n_batches=100, batch_size=10, save_period=None):
        '''optimize the model for n_batches number of batches, save it every
        save_period number of batches.'''
        self.model.train()
        self.log.append(f'training {n_batches} batches of size {batch_size}')
        for i in range(n_batches):
            self.optimizer.zero_grad()
            batched_input, target = self.functional_generator.generateBatch(
                    batch_size)
            output = self.model(batched_input)
            loss = self.functional_generator.lossBatch(output, target)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.learning_results['loss'].append(loss.detach().numpy())
            error = self.functional_generator.errorBatch(output, target)
            self.learning_results['error'].append(error.detach().numpy())
            lr = self.optimizer.param_groups[0]['lr']
            print(f'epoch: {i}, loss: {loss:.3f}, error: {error:.3f}, ' +
                  f'LR:{lr:.3E}')
            if save_period is not None:
                if (i+1) % save_period == 0:
                    self.log.append(f'completed {i+1} / {n_batches} batches.')
                    self.model.eval()
                    self.save()
                    self.model.train()
        self.model.eval()
        if save_period is not None:
            if (i+1) % save_period != 0:
                self.log.append(f'completed {i+1} / {n_batches} batches.')
                self.save()

    def plot(self, fnum=1):
        plt.figure(fnum)
        plt.plot(self.learning_results['loss'])
        plt.plot(self.learning_results['error'])
        plt.legend(['train', 'error'])
        plt.yscale('log')
        plt.grid(True)


if __name__ == '__main__':
    learner = LearnerGenerative(capcha_gen.CaptchaGenOSFixed(), ModelCNN)
    learner.save_params['name'] = 'try_after_arch'
    learner.load()
    learner.learn(500, 200)
    learner.plot()
