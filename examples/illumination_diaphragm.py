#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 00:31:27 2018

@author: andrej
"""
import matplotlib.pyplot as plt
from dtmm.field import illumination_diaphragm

subplots = (221,222,223,224)
diameters = (5.,5.4,6.,7)
plt.gray()
for subplot, diameter in zip(subplots,diameters):
    plt.subplot(subplot)
    plt.imshow(illumination_diaphragm(diameter,1))

