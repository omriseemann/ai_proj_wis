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
from difflib import SequenceMatcher


class FunctionalGen:
    # general interface to work with the learner class
    def generateImage(self):
        # generates an Image as a tensor, returns target as well, must be
        # implemented
        raise Exception('Need to implement this function.')
    
    def loss(self,a,b):
        # loss function, must be implemented
        raise Exception('Need to implement this function.')
    
    def lossBatch(self,A,B):
        # created a tensor of losses based on two lists of targets
        L = np.array(np.zeros(len(A)))
        i = 0
        for a,b in zip(A,B):
            l = self.loss(a,b)
            L[i] = l
            i += 1
        return L
    
    def generateBatch(self,n):
        # generate batch of images and a list of targets
        # batch index is first
        S = []
        data = []
        for i in range(n):
            d,s = self.generateImage()
            data.append(d.unsqueeze(0))
            S.append(s)
        data = torch.cat(data)
        return data, S


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
    def __init__(self, string_length=6):
        self.string_length = string_length
        self.generator = ImageCaptcha()
        self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()])
        return
    
    def generateRandomString(self):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(self.string_length))
    
    def generateImage(self):
        s = self.generateRandomString()
        I = self.generator.generate_image(s)
        I = self.transforms(I)
        return I, s
    
    def loss(self,s1,s2):
        l = 1 - SequenceMatcher(None,s1,s2).ratio()
        l  = l + abs(len(s1) - len(s2))
        return l
    

if __name__ == "__main__":
    main = CaptchaGen_OS_Fixed(6)
    data,s = main.generateImage()
    tp = torchvision.transforms.ToPILImage()
    im = tp(data)
    plt.imshow(im)
    print(s)
    data,s = main.generateBatch(10)
    print(s)