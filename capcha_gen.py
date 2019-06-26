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
from captcha.image import ImageCaptcha  # pip install captcha
import random
import string
from functional_gen import FunctionalGenerator


class CaptchaGen_OS_Fixed(FunctionalGenerator):
    '''generator of captcha with fixed string length'''
    char_to_num = {}
    num_to_char = {}
    letters = string.ascii_letters
    for i, c in enumerate(letters):
        char_to_num[c] = i
        num_to_char[i] = c

    def __init__(self, string_length=6):
        '''string_length is the length of the capcha'''
        self.string_length = string_length
        self.generator = ImageCaptcha()
        self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()])
        self.size_outputs = [string_length, len(self.letters)]
        return

    def generateRandomString(self, pspace=0):
        '''generate random string with probability pspace for spaces'''
        s = ''
        n = len(self.letters)
        for i in range(self.string_length):
            k = random.randint(0, n-1)
            r = random.random()
            if r < pspace:
                s += ' '
            else:
                s += self.num_to_char[k]
        return s

    def generateImage(self):
        '''implimantation of FunctionalGenerator, check it for detail'''
        s = self.generateRandomString()
        Im = self.generator.generate_image(s)
        Im = self.transforms(Im)
        t = self.string_to_tensor(s)
        return Im, t

    def string_to_index_list(self, s):
        ''' give tensor of intexes of letters for the string'''
        res = np.zeros(len(s))
        for i, c in enumerate(s):
            res[i] = self.char_to_num[c]
        return res

    def string_to_tensor(self, s):
        ''' give tensor representation of string size = num of characters in
        dictionary * length of string'''
        t = self.string_to_index_list(s)
        T = torch.zeros([len(self.char_to_num), 1, len(s)])
        for i, x in enumerate(t):
            x = int(x)
            T[x, 0, i] = 1.
        return T

    def tensor_to_string(self, t):
        '''inverse of string_to_tensor'''
        s = ''
        t = t.argmax(0)
        t = t.squeeze(0)
        for x in t:
            c = self.num_to_char[int(x)]
            s += c
        return s

    def loss(self, output, target):
        '''implimantation of FunctionalGenerator, check it for detail'''
        lossf = torch.nn.CrossEntropyLoss()
        target = target.argmax(0).transpose(0, 1)
        output = output.transpose(0, 1).transpose(0, -1)
        loss = lossf(output, target)
        return loss

    def error(self, output, target):
        '''implimantation of FunctionalGenerator, check it for detail'''
        r = output.argmax(0) == target.argmax(0)
        r = 1 - float(r.sum()) / r.shape[1]
        return torch.FloatTensor([r])

    def plot_examples(self, input_, target, output):
        '''implimantation of FunctionalGenerator, check it for detail'''
        # TODO: add varying number of examples
        n = target.shape[0]
        plt.figure()
        tp = torchvision.transforms.ToPILImage()
        for i in range(n):
            plt.subplot(3, 3, i+1)
            Im = input_[i]
            t = target[i]
            o = output[i]
            plt.imshow(tp(Im))
            st = self.tensor_to_string(t)
            so = self.tensor_to_string(o)
            plt.title(f'{st} / {so}')


if __name__ == "__main__":
    # testing function
    main = CaptchaGen_OS_Fixed(6)
    data, t = main.generateImage()
    s = main.tensor_to_string(t)
    tp = torchvision.transforms.ToPILImage()
    im = tp(data)
    plt.imshow(im)
    print(s)
    data, T = main.generateBatch(5)
    print([main.tensor_to_string(T[i]) for i in range(5)])
    print(data.shape)
    print(T.shape)
    data, T2 = main.generateBatch(5)
    print(main.lossBatch(T, T2))
    print(main.errorBatch(T, T2))
