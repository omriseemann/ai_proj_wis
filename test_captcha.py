#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 08:31:14 2019

@author: omrisee
"""

from capcha_gen import CaptchaGenOSFixed
from models import ModelCNN, ResModule, CNNModule
from learner import LearnerGenerative

if __name__ == '__main__':
    gen = CaptchaGenOSFixed()
    I, t = gen.generateBatch(2)
    a = gen.lossBatch(t, t)
    learner = LearnerGenerative(gen, ModelCNN, ResModule)
    learner.save_params['name'] = 'try3'
    learner.reset(lr_start=1e-2)
    learner.load()
    o = learner.model(I)
    learner.learn(100, 100)
    learner.save()
    learner.plot()
