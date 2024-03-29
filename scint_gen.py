#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:46:12 2019

@author: omrisee
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from functional_gen import FunctionalGenerator
import cv2


class ScintImageGen(FunctionalGenerator):

    def __init__(self, pixel_Nx=2048, pixel_Ny=2048, rbright=50,
                 noise_level=(0.1, 0.5), corr_size=2,
                 spot_size_range=(10, 20), distance_factor=(1.5, 3),
                 spot_Nx=40, spot_Ny=40, tan_angle=0.1):
        self.pixel_Nx = pixel_Nx
        self.pixel_Ny = pixel_Ny
        self.corr_size = corr_size
        self.spot_size_range = spot_size_range
        self.spot_Nx = spot_Nx
        self.spot_Ny = spot_Ny
        self.distance_factor = distance_factor
        data_x, data_y = np.meshgrid(range(pixel_Nx), range(pixel_Ny))
        self.data_x = torch.FloatTensor(data_x)
        self.data_y = torch.FloatTensor(data_y)
        self.noise_level = noise_level
        noise_map = np.abs(np.random.randn(
                2*pixel_Nx // corr_size,
                2*pixel_Ny // corr_size))
        noise_map = cv2.resize(noise_map, (pixel_Nx*2, pixel_Ny*2))
        self.noise_map = torch.tensor(noise_map)
        self.rbright = rbright
        self.tan_angle = 0.3
        image, t = self.generateImage()
        self.input_shape = image.shape
        self.output_shape = t.shape

    def generateImage(self, params=None):
        tan_angle1 = (2*np.random.rand()-1)*self.tan_angle
        tan_angle2 = (2*np.random.rand()-1)*self.tan_angle
        center_x = (0.5 + 0.25 * (np.random.rand()-0.5)) * self.pixel_Nx
        center_y = (0.5 + 0.25 * (np.random.rand()-0.5)) * self.pixel_Ny
        spot_size = self.spot_size_range[0] + np.random.rand() * (
                self.spot_size_range[1] - self.spot_size_range[0])
        distance1 = (self.distance_factor[0] + np.random.rand() * (
                self.distance_factor[1] - self.distance_factor[0])) * spot_size
        distance2 = (self.distance_factor[0] + np.random.rand() * (
                self.distance_factor[1] - self.distance_factor[0])) * spot_size
        data = np.zeros((self.pixel_Nx, self.pixel_Ny))
        spot_map_x, spot_map_y = np.meshgrid(range(round(spot_size) + 2),
                                             range(round(spot_size) + 2))
        spot_map = (((spot_map_x - spot_size / 2)**2 + (
                spot_map_y - spot_size / 2)**2) < (spot_size / 2)**2) * 1.
        spot_map = cv2.resize(
                spot_map, tuple(np.array(spot_map.shape) // self.corr_size))
        spot_map = cv2.resize(
                spot_map, tuple(np.array(spot_map.shape) * self.corr_size))
        noise_ix = np.random.randint(0, self.pixel_Nx-1)
        noise_iy = np.random.randint(0, self.pixel_Ny-1)
        noise_map = self.noise_map[noise_ix:noise_ix+self.pixel_Nx,
                                   noise_iy:noise_iy+self.pixel_Ny]
        noise_level = self.noise_level[0] + (
                self.noise_level[1] - self.noise_level[0]) * np.random.rand()
        noise_map = noise_map * noise_level
        for i in range(self.spot_Nx):
            for j in range(self.spot_Ny):
                x = center_x + ((i - self.spot_Nx // 2) +
                                (j - self.spot_Ny // 2) *
                                tan_angle1) * distance1
                y = center_y + ((i - self.spot_Nx // 2) * tan_angle2 +
                                (j - self.spot_Ny // 2)) * distance2
                x = round(x - spot_map.shape[0] / 2) + 1
                y = round(y - spot_map.shape[1] / 2) + 1
                if (x < (self.pixel_Nx - spot_map.shape[0])) and (
                        y < (self.pixel_Ny - spot_map.shape[1])) and (
                                x >= 0) and (
                                        y >= 0):
                    data[x:x+spot_map.shape[0],
                         y:y+spot_map.shape[1]] = spot_map * np.random.rand()
        target = torch.tensor([center_x, center_y, distance1, distance2,
                               tan_angle1*distance1, tan_angle2*distance2,
                               spot_size])
        target = target.unsqueeze(-1).unsqueeze(-1)
        data = torch.FloatTensor(data)
        data += noise_map.float()
        data = data.unsqueeze(0)
        return data.float(), target.float()

    def loss(self, output, target):
        '''loss function, must be implemented, returns valid pytorch loss class
        This can be achived using torch.nn.functional'''
        lossf = torch.nn.L1Loss()
        loss = lossf(output, target)
        return loss

    def error(self, output, target):
        loss = (target-output)/(target+output)
        return loss.abs().mean()

    def plot_examples(self, _input, target, output):
        k = 0
        for im, t, o in zip(_input, target, output):
            k += 1
            plt.figure(k)
            plt.imshow(im[0])
            cx = o[0]
            cy = o[1]
            d1 = o[2]
            d2 = o[3]
            a1 = o[4]/d1
            a2 = o[5]/d2
            xv = []
            yv = []
            for i in range(self.spot_Nx):
                for j in range(self.spot_Ny):
                    x = cx + ((i - self.spot_Nx // 2) + (j - self.spot_Ny // 2)
                              * a1) * d1
                    y = cy + ((i - self.spot_Ny // 2) * a2 +
                              (j - self.spot_Ny // 2)) * d2
                    if (x < self.pixel_Nx) and (y < self.pixel_Ny) and (
                                x >= 0) and (y >= 0):
                        xv.append(x)
                        yv.append(y)
            plt.plot(yv, xv, '+y')


if __name__ == "__main__":
    main = ScintImageGen()
    I, t = main.generateBatch(2)
    main.plot_examples(I, t, t)
