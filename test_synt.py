#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:53:38 2019

@author: omrisee
"""

from scint_gen import ScintImageGen
from models import ModelCNN
from learner import LearnerGenerative

if __name__ == '__main__':
    gen = ScintImageGen()
    I, t = gen.generateBatch(2)
    a = gen.lossBatch(t, t)
    learner = LearnerGenerative(gen, ModelCNN, lr_start=1e-2)
    learner.save_params['name'] = 'scint_test'
    learner.load()
    o = learner.model(I)
    learner.learn(1000, 10)
    learner.save()
    learner.plot_examples(2)
