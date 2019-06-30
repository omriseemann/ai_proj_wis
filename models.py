#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 08:15:14 2019

@author: omrisee
"""

import torch
import numpy as np


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
        self.module_list.append(torch.nn.Conv2d(c2, c2, kernel_size=1,
                                                stride=1, padding=0))

    def generate_architecture(self, input_shape, output_shape):
        ''' get the input and output shape tensors and sets achitecture dict
        of lists to help build the model.'''
        flag_end = False
        shape_multiplyer = max(np.round(np.log10(input_shape)))
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
                cm = max(c_before_compress, c2 * shape_multiplyer)
                if cl[-1]*shape_multiplyer < cm:
                    cl.append(int(cl[-1]*shape_multiplyer))
                else:
                    cl.append(int(cm))
                    flag_compress = True
            else:
                if not cl[-1]//shape_multiplyer < c2:
                    cl.append(int(cl[-1]//shape_multiplyer))
                else:
                    cl.append(int(c2))
            if hl[-1]//shape_multiplyer > h2:
                hl.append(int(hl[-1]//shape_multiplyer))
            else:
                hl.append(int(h2))
            if wl[-1]//shape_multiplyer > w2:
                wl.append(int(wl[-1]//shape_multiplyer))
            else:
                wl.append(int(w2))
            if cl[-2] == c2 and wl[-2] == w2 and hl[-2] == h2:
                flag_end = True
        self.architecture = {'C': cl, 'H': hl, 'W': wl}

    def forward(self, x):
        for m in self.module_list:
            x = m(x)
        return x


if __name__ == '__main__':
    pass
