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
from models import GenModel, ResModule, SamplerModel


class Learner:
    '''learner interface '''
    def __init__(self, func_gen, model_dict, loss_dict, error_dict=None):
        self.model_dict = model_dict
        self.loss_dict = loss_dict
        self.error_dict = error_dict
        self.functional_generator = func_gen
        self.learning_results = {'loss': [], 'error': []}
        self.log = []
        self.save_params = {'save_dir': './saved_data', 'name': 'test'}
        self.reset(flag_model=True)

    def reset(self,  lr_start=1e-3, lr_ratio=1e-2, lr_period=1e6,
              flag_model=False):
        '''helper function if you want to tweek learning parameters'''
        self.optimizer_dict = {}
        # self.scheduler_list = []
        for k in self.model_dict.keys():
            if flag_model:
                self.model_dict[k].reset_parameters()
            self.optimizer_dict[k] = torch.optim.Adam(
                    self.model_dict[k].parameters(), lr=lr_start)
            '''
            self.scheduler_list.append(torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_list[-1], self.cyclyc_lr(lr_ratio, lr_period))
            '''
        self.log.append(f'reset: init {flag_model}, lr_start:{lr_start:.3E}')

    def learn(self, key, n_batches=100, batch_size=10):
        '''optimize the model for n_batches number of batches'''
        self.model_dict[key].train()
        self.log.append(f'training {key} for {n_batches} batches of ' +
                        f'size {batch_size}')
        for i in range(n_batches):
            self.optimizer_dict[key].zero_grad()
            batched_input, target = self.functional_generator.generateBatch(
                    batch_size)
            output = self.model_dict[key](batched_input)
            loss = self.loss_dict[key](output, target)
            loss.backward()
            self.optimizer_dict[key].step()
            self.learning_results['loss'].append(loss.detach().numpy())
            if self.error_dict is not None:
                error = self.error_dict[key](output, target)
                self.learning_results['error'].append(error.detach().numpy())
            else:
                error = 'None'
                self.learning_results['error'].append(1)
            lr = self.optimizer_dict[key].param_groups[0]['lr']
            print(f'batch: {i}, loss: {loss:.3f}, error: {error:.3f}, ' +
                  f'LR:{lr:.3E}')
        self.model_dict[key].eval()

    def cyclyc_lr(self, lr_ratio, lr_period):
        '''helper function for cyclic learning rate'''
        down = np.log(lr_ratio)

        def f(epoch):
            return np.exp(down - down*(np.cos(epoch*np.pi / lr_period)**2))
        return f

    def to_data_struct(self):
        '''data structure generator for saveing'''
        osl = {}
        msl = {}
        for k in self.optimizer_dict.keys():
            optimizer_state = self.optimizer_dict[k].state_dict()
            osl[k] = optimizer_state
            model_state = self.model_dict[k].state_dict()
            msl[k] = model_state
        data = {'learning_results': self.learning_results,
                'model_state': msl,
                'optimizer_state': osl,
                'save_params': self.save_params,
                'log': self.log
                }
        return data

    def to_self(self, data):
        '''data structure generator for loading'''
        self.learning_results = data['learning_results']
        self.save_params = data['save_params']
        for k in data['model_state'].keys():
            self.model_dict[k].load_state_dict(data['model_state'][k])
            self.optimizer_dict[k].load_state_dict(data['optimizer_state'][k])
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
        output = self.model_list[0](data)
        self.functional_generator.plot_examples(data, target, output)

    def plot(self, fnum=1):
        plt.figure(fnum)
        leg = []
        for v, k in enumerate(self.learning_results):
            leg.append(k)
            plt.plot(v)
        plt.legend(leg)
        plt.yscale('log')
        plt.grid(True)
        plt.box(True)


class LearnerDatasetGenerative(Learner):
    '''learner class for dataset generated'''
    def __init__(self, func_gen, model):
        super(LearnerDatasetGenerative, self).__init__(func_gen, [model])
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
    def __init__(self, func_gen, model):
        super(LearnerGenerative, self).__init__(func_gen, [model])
        self.learning_results = {'loss': [], 'error': []}

    def learn(self, n_batches=100, batch_size=10):
        '''optimize the model for n_batches number of batches'''
        self.model_list[0].train()
        self.log.append(f'training {n_batches} batches of size {batch_size}')
        for i in range(n_batches):
            self.optimizer_list[0].zero_grad()
            batched_input, target = self.functional_generator.generateBatch(
                    batch_size)
            output = self.model_list[0](batched_input)
            loss = self.functional_generator.lossBatch(output, target)
            loss.backward()
            self.optimizer_list[0].step()
            self.learning_results['loss'].append(loss.detach().numpy())
            error = self.functional_generator.errorBatch(output, target)
            self.learning_results['error'].append(error.detach().numpy())
            lr = self.optimizer_list[0].param_groups[0]['lr']
            print(f'batch: {i}, loss: {loss:.3f}, error: {error:.3f}, ' +
                  f'LR:{lr:.3E}')
        self.model_list[0].eval()

    def plot(self, fnum=1):
        plt.figure(fnum)
        plt.plot(self.learning_results['loss'])
        plt.plot(self.learning_results['error'])
        plt.legend(['train', 'error'])
        plt.yscale('log')
        plt.grid(True)


class LearnerGeneratorDescreminator(Learner):
    def __init__(self, func_gen, model_list):
        super(LearnerGeneratorDescreminator, self).__init__(
                func_gen, model_list)
        self.learning_results = {'Gen loss': [], 'Desc Loss': []}


class LearnerGenerativeSampler(Learner):
    '''sampler for efficeint learning'''
    def __init__(self, func_gen, model):
        model_list = [model, SamplerModel(func_gen.output_shape)]
        super(LearnerGenerativeSampler, self).__init__(func_gen, model_list)
        self.learning_results = {'loss': [], 'error': [], 'Samploss': []}

    def learn(self, n_batches=100, batch_size=10, flag_model=True,
              flag_sampler=True):
        '''optimize the model for n_batches number of batches'''
        model = self.model_list[0]
        sampler = self.model_list[1]
        optim_model = self.optimizer_list[0]
        optim_sampler = self.optimizer_list[1]
        self.log.append(f'training {n_batches} batches of size {batch_size}' +
                        f'model: {flag_model}, sampler: {flag_sampler}')
        sampler.train()
        model.train()
        for i in range(n_batches):
            optim_model.zero_grad()
            optim_sampler.zero_grad()
            targets = sampler.generate(batch_size//2)
            targets2 = sampler(targets)
            batched_input, target = self.functional_generator.generateBatch(
                    batch_size//2, targets2)
            batched_input2, target2 = self.functional_generator.generateBatch(
                    batch_size)
            batched_input = torch.cat((batched_input, batched_input2), 0)
            target = torch.cat((target, target2), 0)
            output = model(batched_input2)
            loss = self.functional_generator.lossBatch(output, target2)
            loss.backward()
            if flag_model:
                optim_model.step()
            if flag_sampler:
                sampler.reverse_grad()
                optim_sampler.step()
            self.learning_results['loss'].append(loss.detach().numpy())
            error = self.functional_generator.errorBatch(output, target)
            self.learning_results['error'].append(error.detach().numpy())
            lr = optim_model.param_groups[0]['lr']
            print(f'batch: {i}, loss: {loss:.3f}, error: {error:.3f}, ' +
                  f'LR:{lr:.3E}')
        model.eval()
        sampler.eval()


if __name__ == '__main__':
    gen = capcha_gen.CaptchaGenOSFixed()
    model = GenModel(gen.input_shape, gen.output_shape, ResModule)
    learner = Learner(gen, {'main': model}, {'main': gen.lossBatch},
                      {'main': gen.errorBatch})
    learner.save_params['name'] = 'test'
    learner.load()
    learner.learn('main', 1000, 5)
    learner.save()
    learner.plot()
    learner.learn('main', 500, 5)
    learner.save()
    learner.plot()
