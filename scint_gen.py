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
from functional_gen import FunctionalGenerator
import scipy.signal as sig


class ScintImageGen(FunctionalGenerator):
    
    def __init__(self, nx=2048,ny=2048,rbright=50, noise_level=0.2):
        self.nx = nx
        self.ny = ny
        xm,ym = np.meshgrid(range(self.nx), range(self.ny))
        self.xm = xm
        self.ym = ym
        self.noise_level = noise_level
        self.rbright = rbright        
        return
    
    def generateImage(self,Nx=80,Ny=80,R=(3,10), distance=(1.5,2.5), tan_a=0.3):
        a1 = (np.random.rand()-0.5)*tan_a
        a2 = (np.random.rand()-0.5)*tan_a
        cx = (1/4+np.random.rand()/2) * self.nx
        cy = (1/4+np.random.rand()/2) * self.ny
        r = np.random.rand()*(R[1]-R[0]) + R[0]
        r = round(r)
        s = (np.random.rand()*(distance[1] - distance[0])+distance[0])*r*2
        I = np.zeros((self.nx,self.ny))
        ux,uy = np.meshgrid(range(round(r*2)),range(round(r*2)))
        Ic = (((ux-r)**2+(uy-r)**2)<round(r)**2)
        noise = np.random.rand(self.nx,self.ny)*self.noise_level
        target = np.zeros((Nx,Ny))
        for i in range(Nx):
            for j in range(Ny):
                x = cx + (i-Nx/2)*s + (j-Ny/2)*s*a1
                y = cy + (i-Nx/2)*s*a2 + (j-Ny/2)*s
                x = round(x)
                y = round(y)
                if x<(self.nx-r) and y<(self.ny-r) and x>=r and y>=r:
                    I0 = np.random.rand()
                    I[x,y]= I0
                    target[i,j] = I0 + np.mean(noise[x-r:x+r,y-r:y+r])
        target = torch.tensor(target)
        target = target.unsqueeze(0)
        I = sig.fftconvolve(I,Ic,'same')
        I += noise
        I = torch.FloatTensor(I)
        I = I.unsqueeze(0)
        return I, target

if __name__ == "__main__":
    main = ScintImageGen()
    I,t = main.generateBatch(5)
    