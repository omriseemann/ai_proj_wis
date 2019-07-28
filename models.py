#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 08:15:14 2019

@author: omrisee
"""

import torch


class ResModule(torch.nn.Module):
    ''' Module of inception '''
    def __init__(self,  ci, co, hi, ho, wi, wo):
        super(ResModule, self).__init__()
        self.module_list = torch.nn.ModuleList()
        self.module_list.append(CNNModule(ci, co, hi, ho, wi, wo))
        self.module_list.append(torch.nn.Conv2d(co, co, kernel_size=3,
                                                stride=1, padding=1))
        self.module_list.append(torch.nn.LeakyReLU())  # activation
        self.module_list.append(torch.nn.Conv2d(co, co, kernel_size=3,
                                                stride=1, padding=1))
        self.module_list.append(torch.nn.LeakyReLU())  # activation
        self.module_res = torch.nn.ModuleList()
        self.module_res.append(torch.nn.Conv2d(ci, co, kernel_size=3,
                                               stride=1, padding=1))
        self.module_res.append(torch.nn.AdaptiveAvgPool2d((ho, wo)))

    def forward(self, x):
        x1 = x
        for m in self.module_res:
            x1 = m(x1)
        for m in self.module_list:
            x = m(x)
        x += x1
        return x

    def reset_parameters(self):
        for m in self.module_list:
            try:
                m.reset_parameters()
            except AttributeError:
                pass

        for m in self.module_res:
            try:
                m.reset_parameters()
            except AttributeError:
                pass


class CNNModule(torch.nn.Module):
    ''' Module of Fully connected CNN '''
    def __init__(self, ci, co, hi, ho, wi, wo):
        super(CNNModule, self).__init__()
        self.module_list = torch.nn.ModuleList()
        self.module_list.append(torch.nn.Conv2d(ci, co, kernel_size=3,
                                                stride=1, padding=1))
        self.module_list.append(torch.nn.AdaptiveAvgPool2d((ho, wo)))
        self.module_list.append(torch.nn.BatchNorm2d(co))
        self.module_list.append(torch.nn.LeakyReLU())  # activation

    def forward(self, x):
        for m in self.module_list:
            x = m(x)
        return x

    def reset_parameters(self):
        for m in self.module_list:
            try:
                m.reset_parameters()
            except AttributeError:
                pass


class SamplerModel(torch.nn.Module):
    '''sampler model'''
    def __init__(self, output_shape):
        super(SamplerModel, self).__init__()
        os = output_shape
        self.output_shape = output_shape
        self.model = ResModule(os[0], os[0], os[1], os[1], os[2], os[2])

    def generate(self, n):
        os = self.output_shape
        x = torch.randn((n, os[0], os[1], os[2]))
        return x

    def reverse_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad = -p.grad

    def forward(self, x):
        x = self.model(x)
        return x

    def reset_parameters(self):
        self.model.reset_parameters()


class GenModel(torch.nn.Module):
    '''Cnn generic model, morphs input tensor C*H*W into output tensor
    C2*H2*W2'''

    def __init__(self, input_shape, output_shape, model_module,
                 flag_reversed=False):
        '''Gets input_shape of tensor and output_shape of tensor (not batches).
        Builds the model by stacking conv2d, batchnorm2d, leakyrelu and
        adaptive average pooling 2d.'''
        super(GenModel, self).__init__()
        self.module_list = torch.nn.ModuleList()
        self.generate_architecture(input_shape, output_shape)
        cl = self.architecture['C']
        hl = self.architecture['H']
        wl = self.architecture['W']
        for k in range(len(cl)-1):
            if flag_reversed:
                i = len(cl)-1 - k
            else:
                i = k
            self.module_list.append(model_module(cl[i], cl[i+1], hl[i],
                                                 hl[i+1], wl[i], wl[i+1]))
        if flag_reversed:
            i = 0
        else:
            i = -1
        self.module_list.append(torch.nn.AdaptiveAvgPool2d((hl[i], wl[i])))
        self.module_list.append(torch.nn.Conv2d(cl[i], cl[i], kernel_size=1,
                                                stride=1, padding=0))

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
        c_before_compress = 100
        while not flag_end:
            if not flag_compress:
                if cl[-1]*10 < c_before_compress:
                    cl.append(int(cl[-1]*10))
                else:
                    cl.append(int(c_before_compress))
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
            if cl[-2] == c2 and wl[-1] == w2 and hl[-1] == h2 and len(cl) > 2:
                flag_end = True
        self.architecture = {'C': cl, 'H': hl, 'W': wl}

    def forward(self, x):
        for m in self.module_list:
            x = m(x)
        return x

    def reset_parameters(self):
        for m in self.module_list:
            try:
                m.reset_parameters()
            except AttributeError:
                pass


if __name__ == '__main__':
    m = ResModule(1, 10, 10, 10, 10, 10)
    s = SamplerModel((10, 1, 10))
    x = s.generate(5)
    o = s(x)
