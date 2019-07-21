#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:53:38 2019

@author: omrisee
"""

from scint_gen import ScintImageGen
from models import ResModule
from learner import LearnerGenerative
import torch
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    N = 2048
    gen = ScintImageGen(pixel_Nx=N, pixel_Ny=N, spot_Nx=80, spot_Ny=80)
    I, t = gen.generateBatch(2)
    # gen.plot_examples(I, t, t)
    a = gen.lossBatch(t, t)
    learner = LearnerGenerative(gen, ResModule)
    learner.model = torch.nn.Sequential(ResModule(1, 10, N, N//4, N, N//4),
                                        ResModule(10, 2, N//4, N, N//4, N))
    learner.reset(lr_start=1e-3)
    learner.save_params['name'] = 'scint_test'
    learner.load()
    o = learner.model(I)
    learner.learn(100, 5)
    learner.save()
    learner.plot()
    #learner.plot_examples(2)
    
    fpath = '/home/omri/Dropbox/lab_server - Omri Seemann/exp_data/20190604/hamamatsu/shot012.TIF'
    fpath = '/home/omrisee/Dropbox/lab_server - Omri Seemann/exp_data/20190604/hamamatsu/shot012.TIF'


    a = plt.imread(fpath,-1)
    b = np.log(a+1)
    c = torch.tensor(b*1.).unsqueeze(0)
    d = gen.normt(c)
    o = learner.model(d.unsqueeze(0))
    gen.plot_examples(d.unsqueeze(0),o,o)
    
    #plt.figure(1)
    #plt.imshow((a), cmap='nipy_spectral')
