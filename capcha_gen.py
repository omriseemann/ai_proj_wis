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
#from difflib import SequenceMatcher
from functional_gen import FunctionalGenerator


class CaptchaGen_OS_Fixed(FunctionalGenerator):
    # generator of captcha with fixed length
    char_to_num = {}
    num_to_char = {}
    letters = string.ascii_letters
    for i,c in enumerate(letters):
        char_to_num[c] = i 
        num_to_char[i] = c
    
    def __init__(self, string_length=6):
        self.string_length = string_length
        self.generator = ImageCaptcha()
        self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()])
        self.size_outputs = [string_length, len(self.letters)]
        return
    
    def generateRandomString(self, pspace=0):
        s = ''
        n = len(self.letters)
        for i in range(self.string_length):
            k = random.randint(0,n-1)
            r = random.random()
            if r<pspace:
                s += ' '
            else:
                s += self.num_to_char[k]
        return s
    
    def generateImage(self):
        s = self.generateRandomString()
        I = self.generator.generate_image(s)
        I = self.transforms(I)
        t = self.string_to_tensor(s)
        return I, t
    
    def string_to_index_tensor(self,s):
        res = np.zeros(self.string_length)
        for i,c in enumerate(s):
            res[i] = self.char_to_num[c]
        return torch.FloatTensor(res)
    
    def string_to_tensor(self,s):
        t = self.string_to_index_tensor(s)
        T = np.zeros([self.string_length, len(self.char_to_num)])
        for i,x in enumerate(t):
            x = int(x)
            T[i,x] = 1.
        T = torch.FloatTensor(T)
        return T
    
    def tensor_to_string(self,t):
        s = ''
        for x in t.argmax(1):
            c = self.num_to_char[int(x)]
            s += c
        return s
    
    def loss(self,output,target):
        lossf = torch.nn.CrossEntropyLoss()
        l = lossf(output,target.argmax(-1))
        return l
    
    def error(self, output,target):
        r = output.argmax(-1) == target.argmax(-1)
        r = 1 - int(r.sum())/ len(r)
        return torch.FloatTensor([r])
    
    def plot_examples(self, input_, target, output):
        # todo add varieing number of examples
        n = target.shape[0]
        plt.figure()
        tp = torchvision.transforms.ToPILImage()
        for i in range(n):
            plt.subplot(3,3,i+1)
            I = input_[i]
            t = target[i]
            o= output[i]
            plt.imshow(tp(I))
            st = self.tensor_to_string(t)
            so = self.tensor_to_string(o)
            plt.title(f'{st} / {so}')
            
        
    

if __name__ == "__main__":
    main = CaptchaGen_OS_Fixed(6)
    data,t = main.generateImage()
    s = main.tensor_to_string(t)
    tp = torchvision.transforms.ToPILImage()
    im = tp(data)
    plt.imshow(im)
    print(s)
    data,T = main.generateBatch(5)
    print([main.tensor_to_string(T[i]) for i in range(5)])
    print(data.shape)
    print(T.shape)
    data,T2 = main.generateBatch(5)
    print(main.lossBatch(T,T2))
    print(main.errorBatch(T,T2))