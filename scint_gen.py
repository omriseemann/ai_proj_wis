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
import random
from functional_gen import FunctionalGenerator


class ScintImageGen(FunctionalGenerator):
    
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
    

if __name__ == "__main__":
    pass