#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 08:31:14 2019

@author: omrisee
"""

from capcha_gen import CaptchaGenOSFixed
from models import ResModule, CNNModule
from learner import LearnerGenerative

if __name__ == '__main__':
    gen = CaptchaGenOSFixed()
    I, t = gen.generateBatch(2)
    a = gen.lossBatch(t, t)
    learner = LearnerGenerative(gen, ResModule)
    learner.save_params['name'] = 'try4'
    learner.reset(lr_start=1e-3)
    learner.load()
    o = learner.model(I)
    learner.learn(2000, 10)
    learner.save()
    learner.plot()
    learner.learn(1000, 20)
    learner.save()
    learner.plot()
    learner.learn(500, 50)
    learner.save()
    learner.plot()
    learner.learn(200, 100)
    learner.save()
    learner.plot()
    learner.learn(100, 200)
    learner.save()
    learner.plot()
