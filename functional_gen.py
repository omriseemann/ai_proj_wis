#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:46:12 2019

@author: omrisee
"""


import torch


class FunctionalGenerator:
    '''general interface to work with the learner classes'''
    def generateImage(self):
        ''' generates an Image as a tensor, returns target as well, must be
        implemented
        Image is tensor of type C1*H*W
        target is tensor of type C2*H2*W2
        '''
        Image = None
        Target = None
        raise Exception('Need to implement this function.')
        return Image, Target

    def loss(self, a, b):
        '''loss function, must be implemented, returns valid pytorch loss class
        This can be achived using torch.nn.functional'''
        raise Exception('Need to implement this function.')

    def error(self, a, b):
        '''error function for self analysis, must be implemented, returns
        tensor'''
        raise Exception('Need to implement this function.')

    def errorBatch(self, A, B):
        '''creates a tensor of errors based on two baches of targets'''
        L = []
        for a, b in zip(A, B):
            error = self.error(a, b)
            error = error.unsqueeze(0)
            L.append(error)
        return torch.cat(L).mean()

    def lossBatch(self, A, B):
        '''creates a tensor of losses based on two batches of targets'''
        L = []
        for a, b in zip(A, B):
            loss = self.loss(a, b)
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
        raise Exception('Need to implement this function.')


if __name__ == "__main__":
    # testing function
    pass
