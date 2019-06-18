#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:46:12 2019

@author: omrisee
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from captcha.image import ImageCaptcha
# pip install captcha
import random
import string


class FunctionalGen:
    def generateImage(self):
        raise Exception('Need to implement this function.')
    
    def loss(self):
        raise Exception('Need to implement this function.')
    
    def generateBatch(self,n):
        raise Exception('Need to implement this function.')


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

    def generateBatch(self, n):
        pass


class CaptchaGen_OS(FunctionalGen):
    
    def __init__(self, string_length=6):
        self.string_length = string_length
        self.generator = ImageCaptcha()
        return
    
    def generateRandomString(self):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(self.string_length))
    
    def generateImage(self):
        s = self.generateRandomString()
        I = self.generator.generate_image(s)
        return I, s

    def generateBatch(self, n):
        pass


if __name__ == "__main__":
    main = CaptchaGen_OS()
    data,s = main.generateImage()
    plt.imshow(data)
    print(s)