#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 08:31:14 2019

@author: omrisee
"""

from capcha_gen import CaptchaGenOSFixed
from models import ResModule, GenModel
from learner import Learner

if __name__ == '__main__':
    gen = CaptchaGenOSFixed()
    model = GenModel(gen.input_shape, gen.output_shape, ResModule)
    I, t = gen.generateBatch(2)
    a = gen.lossBatch(t, t)
    learner = Learner(gen, {'main': model}, {'main': gen.lossBatch},
                      {'main': gen.errorBatch})
    learner.save_params['name'] = 'try6'
    learner.reset(lr_start=1e-3)
    learner.load()
    o = learner.model_dict['main'](I)
    learner.learn('main', 2000, 10)
    learner.save()
    learner.plot()
    learner.learn('main', 1000, 20)
    learner.save()
    learner.plot()
    learner.learn('main', 500, 50)
    learner.save()
    learner.plot()
    learner.learn('main', 200, 100)
    learner.save()
    learner.plot()
    learner.learn('main', 100, 200)
    learner.save()
    learner.plot(2)
