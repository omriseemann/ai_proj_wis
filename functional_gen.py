#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:46:12 2019

@author: omrisee
"""


import torch


class FunctionalGenerator:
    '''general interface to work with the learner classes'''
    def __init__(self):
        self.input_shape = None
        self.output_shape = None

    def generateImage(self):
        ''' generates an Image as a tensor, returns target as well, must be
        implemented
        Image is tensor of type C1*H*W
        target is tensor of type C2*H2*W2
        '''
        Image = None
        Target = None
        raise NotImplementedError
        return Image, Target

    def loss(self, output, target):
        '''loss function, must be implemented, returns valid pytorch loss class
        This can be achived using torch.nn.functional'''
        raise NotImplementedError

    def error(self, output, target):
        '''error function for self analysis, must be implemented, returns
        tensor'''
        raise NotImplementedError

    def errorBatch(self, output, target):
        '''creates a tensor of errors based on two baches of targets'''
        L = []
        for o, t in zip(output, target):
            error = self.error(o, t)
            error = error.unsqueeze(0)
            L.append(error)
        return torch.cat(L).mean()

    def lossBatch(self, output, target):
        '''creates a tensor of losses based on two batches of targets'''
        L = []
        for o, t in zip(output, target):
            loss = self.loss(o, t)
            loss = loss.unsqueeze(0)
            L.append(loss)
        return torch.cat(L).mean()

    def generateBatch(self, n):
        '''generate batch of images and a list of targets
        batch index is first'''
        T = []
        data = []
        for i in range(n):
            d, t = self.generateImage()
            data.append(d.unsqueeze(0))
            T.append(t.unsqueeze(0))
        data = torch.cat(data)
        T = torch.cat(T)
        return data, T

    def plot_examples(self, input_, target, output):
        '''plot function for the generator, input_ is the data input,
        target is real target, output is model output. Gets batch of examples
        '''
        raise NotImplementedError

    def generateNewDataset(self, size=100, flag_verbose=False):
        '''generates new datasets for the wrapper'''
        data, T = [], []
        for i in range(size):
            d, t = self.generateImage()
            data.append(d.unsqueeze(0))
            T.append(t.unsqueeze(0))
            if flag_verbose:
                print(f'{i+1} / {size}')
        data = torch.cat(data)
        T = torch.cat(T)
        dataset = torch.utils.data.dataset.TensorDataset(
                data, T)
        return dataset


if __name__ == "__main__":
    # testing function
    from capcha_gen import CaptchaGenOSFixed
    main = CaptchaGenOSFixed(6, flag_verbose=True)
    db = main.generateNewDataset()
