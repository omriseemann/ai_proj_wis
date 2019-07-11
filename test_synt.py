#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:53:38 2019

@author: omrisee
"""

from scint_gen import ScintImageGen
from models import ModelCNN, ResModule
from learner import LearnerGenerative

if __name__ == '__main__':
    gen = ScintImageGen(pixel_Nx=200, pixel_Ny=200, spot_Nx=10, spot_Ny=10)
    I, t = gen.generateBatch(2)
    # gen.plot_examples(I, t, t)
    a = gen.lossBatch(t, t)
    learner = LearnerGenerative(gen, ModelCNN, ResModule)
    learner.reset(lr_start=1e-2)
    learner.save_params['name'] = 'scint_test'
    learner.load()
    o = learner.model(I)
    learner.learn(500, 50)
    learner.save()
    learner.plot()
