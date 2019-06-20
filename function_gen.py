#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:46:12 2019

@author: omrisee
"""

import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
from captcha.image import ImageCaptcha
# pip install captcha
import random
import string
#from difflib import SequenceMatcher


class FunctionalGen:
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


class ScintImageGen(FunctionalGen):
    
    def __init__(self, nx=2048,ny=2048,bits=16,rbright=50, noise_level=200):
        self.nx = nx
        self.ny = ny
        self.noise_level = noise_level
        self.bits = bits
        self.rbright = rbright
        return
    
    def generateImage(self):
        xm,ym = np.meshgrid(range(self.nx), range(self.ny))
        I = np.round(np.random.rand(self.nx,self.ny)*self.noise_level)
        return I
            


class CaptchaGen_OS_Fixed(FunctionalGen):
    # generator of captcha with fixed length
    char_to_num = {}
    num_to_char = {}
    letters = string.ascii_letters
    for i,c in enumerate(letters):
        char_to_num[c] = i 
        num_to_char[i] = c
    
    def __init__(self, string_length=6):
        self.string_length = string_length
        self.generator = ImageCaptcha()
        self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()])
        self.size_outputs = [string_length, len(self.letters)]
        return
    
    def generateRandomString(self, pspace=0):
        s = ''
        n = len(self.letters)
        for i in range(self.string_length):
            k = random.randint(0,n-1)
            r = random.random()
            if r<pspace:
                s += ' '
            else:
                s += self.num_to_char[k]
        return s
    
    def generateImage(self):
        s = self.generateRandomString()
        I = self.generator.generate_image(s)
        I = self.transforms(I)
        t = self.string_to_tensor(s)
        return I, t
    
    def string_to_index_tensor(self,s):
        res = np.zeros(self.string_length)
        for i,c in enumerate(s):
            res[i] = self.char_to_num[c]
        return torch.tensor(res)
    
    def string_to_tensor(self,s):
        t = self.string_to_index_tensor(s)
        T = np.zeros([self.string_length, len(self.char_to_num)])
        for i,x in enumerate(t):
            x = int(x)
            T[i,x] = 1
        return torch.tensor(T, requires_grad=True)
    
    def tensor_to_string(self,t):
        s = ''
        for x in t.argmax(1):
            c = self.num_to_char[int(x)]
            s += c
        return s
    
    def loss(self,output,target):
        lossf = torch.nn.CrossEntropyLoss()
        l = lossf(output,target.argmax(-1))
        return l
    

if __name__ == "__main__":
    main = CaptchaGen_OS_Fixed(6)
    data,t = main.generateImage()
    s = main.tensor_to_string(t)
    tp = torchvision.transforms.ToPILImage()
    im = tp(data)
    plt.imshow(im)
    print(s)
    data,T = main.generateBatch(10)
    print([main.tensor_to_string(T[i]) for i in range(10)])
    print(data.shape)
    print(T.shape)