#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:46:12 2019

@author: omrisee
"""


import torch


class FunctionalGenerator:
    # general interface to work with the learner class
    size_outputs = [1] # dim size of the output tensor
    def generateImage(self):
        # generates an Image as a tensor, returns target as well, must be
        # implemented
        raise Exception('Need to implement this function.')
    
    def loss(self,a,b):
        # loss function, must be implemented, returns tensor
        raise Exception('Need to implement this function.')
        
    def error(self,a,b):
        # loss function, must be implemented, returns tensor
        raise Exception('Need to implement this function.')
        
    def errorBatch(self,A,B):
        # created a tensor of errors based on two baches of targets
        L = []
        for a,b in zip(A,B):
            l = self.error(a,b)
            l = l.unsqueeze(0)
            L.append(l)
        return torch.cat(L).mean()
    
    def lossBatch(self,A,B):
        # created a tensor of losses based on two baches of targets
        L = []
        for a,b in zip(A,B):
            l = self.loss(a,b)
            l = l.unsqueeze(0)
            L.append(l)
        return torch.cat(L).mean()
    
    def generateBatch(self,n):
        # generate batch of images and a list of targets
        # batch index is first
        T = []
        data = []
        for i in range(n):
            d,t = self.generateImage()
            data.append(d.unsqueeze(0))
            T.append(t.unsqueeze(0))
        data = torch.cat(data)
        T = torch.cat(T)
        return data, T
    
    def plot_examples(self, input_, target, output):
        # plot function for the generator, input_ is the data input, 
        # target is real target, output is model output. Gets batched examples
        raise Exception('Need to implement this function.')


if __name__ == "__main__":
    pass # for testing