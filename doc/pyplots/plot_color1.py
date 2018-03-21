#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot material ID example"""

import dtmm.color as dc
import matplotlib.pyplot as plt
import numpy as np

s0 = dc.load_specter(np.linspace(380,780,256))

cmf = dc.load_tcmf(np.linspace(380,780,256), single_wavelength = True)
#s0 = dc.normalize_specter(s0,cmf)
im = np.zeros(shape = (81,256,3))

for i,w in enumerate(np.linspace(380,780,256)):
    for j in range(81):
        s = np.zeros(shape = (256,))
        s[i] = j
        im[j,i] = dc.specter2color(s,cmf)
    
plt.imshow(im)
plt.show()