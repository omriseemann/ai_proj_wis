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
from models import GenModel, ResModule


class Learner:
    '''learner interface '''
    def __init__(self, func_gen, model_module, model_class=GenModel):
        self.model_class = model_class
        self.model_module = model_module
        self.functional_generator = func_gen
        self.learning_results = {}
        self.log = []
        self.save_params = {'save_dir': './saved_data', 'name': 'test'}
        self.reset(flag_model=True)

    def reset(self,  lr_start=1e-3, lr_ratio=1e-2, lr_period=1e6,
              flag_model=False):
        '''helper function if you want to tweek learning parameters'''
        if flag_model:
            self.model = self.model_class(
                    self.functional_generator.input_shape,
                    self.functional_generator.output_shape,
                    self.model_module)
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
        try:
            data = torch.load(lpath)
            self.to_self(data)
        except FileNotFoundError:
            self.save()
            data = torch.load(lpath)
        return data

    def plot_examples(self, n_examples=9):
        '''Plots examples of model results'''
        data, target = self.functional_generator.generateBatch(n_examples)
        output = self.model(data)
        self.functional_generator.plot_examples(data, target, output)


class LearnerDatasetGenerative(Learner):
    '''learner class for dataset generated'''
    def __init__(self, func_gen, model_class, model_module):
        super(LearnerDatasetGenerative, self).__init__(func_gen, model_class,
                                                       model_module)
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

    def learn(self, n_epochs=1, batch_size=10, save_period=None):
        '''optimize the model for number of epochs with spacified batch size'''
        self.log.append(f'training {n_epochs} epochs with batch size' +
                        f'{batch_size}')
        train_dl = torch.utils.data.DataLoader(self.datasets['train'],
                                               batch_size=batch_size,
                                               shuffle=True)
        valid_dl = torch.utils.data.DataLoader(self.datasets['valid'])
        for i in range(n_epochs):
            self.model.train()
            j = 0
            tloss = []
            for batched_input, target in train_dl:
                self.optimizer.zero_grad()
                output = self.model(batched_input)
                loss = self.functional_generator.lossBatch(output, target)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                tloss.append(loss.detach().numpy())
                error = self.functional_generator.errorBatch(output, target)
                j += 1
            self.model.eval()
            vloss = []
            verror = []
            for batched_input, target in valid_dl:
                loss = self.functional_generator.lossBatch(output, target)
                error = self.functional_generator.errorBatch(output, target)
                vloss.append(loss.detach().numpy())
                verror.append(error.detach().numpy())
            vloss = np.array(vloss).mean()
            tloss = np.array(tloss).mean()
            verror = np.array(verror).mean()
            self.learning_results['error'].append(verror)
            self.learning_results['train_loss'].append(tloss)
            self.learning_results['valid_loss'].append(vloss)
            print(f'epoch: {i}, tloss: {tloss:.3f}, vloss: {vloss:.3f}, ' +
                  f'error: {verror:.3f}.')

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
    def __init__(self, func_gen, model_module, model_class=GenModel):
        super(LearnerGenerative, self).__init__(func_gen, model_module,
                                                model_class)
        self.learning_results = {'loss': [], 'error': []}

    def learn(self, n_batches=100, batch_size=10):
        '''optimize the model for n_batches number of batches'''
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
            print(f'batch: {i}, loss: {loss:.3f}, error: {error:.3f}, ' +
                  f'LR:{lr:.3E}')
        self.model.eval()

    def plot(self, fnum=1):
        plt.figure(fnum)
        plt.plot(self.learning_results['loss'])
        plt.plot(self.learning_results['error'])
        plt.legend(['train', 'error'])
        plt.yscale('log')
        plt.grid(True)


if __name__ == '__main__':
    learner = LearnerGenerative(capcha_gen.CaptchaGenOSFixed(), ResModule)
    learner.save_params['name'] = 'test'
    learner.learn(100, 10)
    learner.save()
    learner.plot()
