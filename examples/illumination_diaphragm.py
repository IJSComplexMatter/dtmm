#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 00:31:27 2018

@author: andrej
"""
import matplotlib.pyplot as plt
from dtmm.field import _illumination_diaphragm

subplots = (221,222,223,224)
diameters = (5.,5.4,6.,7)

for subplot, diameter in zip(subplots,diameters):
    plt.subplot(subplot)
    plt.imshow(_illumination_diaphragm(diameter,1))

plt.gray()