#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 02:01:15 2018

@author: andrej
"""

import matplotlib.pyplot as plt
import dtmm
import numpy as np

focals = (-40,-30,-20,10,0,10)
wavelengths = np.linspace(380,780,11)
field = np.load("example_380_780_11_200.npy")

viewer = dtmm.field_viewer(field)
viewer.set_parameters(polarizer = 0, analyzer = 90, sample = 0, intensity = 2)

fig,axes = plt.subplots(1,5)

for i,focus in enumerate(focals):
    print(i)
    im = viewer.calculate_image(focus = focus)
    axim = axes[i].imshow(im)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    

fig.show()
