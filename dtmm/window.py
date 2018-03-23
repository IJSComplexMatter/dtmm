#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 14:07:16 2018

@author: andrej

Window functions.
"""

from __future__ import absolute_import, print_function, division

import numpy as np
from dtmm.conf import FDTYPE

def _r(shape, scale = 1.):
    """Returns radius array of given shape."""
    ny,nx = [l//2 for l in shape]
    ay, ax = [np.arange(-l / 2. + .5, l / 2. + .5) for l in shape]
    xx, yy = np.meshgrid(ax, ay, indexing = "xy")
    r = ((xx/(nx*scale))**2 + (yy/(ny*scale))**2) ** 0.5    
    return r

def blackman(shape):
    """Returns a blacman window of a given shape"""
    r = _r(shape)   
    out = np.ones(shape, FDTYPE)
    out[...] = 0.42 + 0.5*np.cos(1*np.pi*r)+0.08*np.cos(2*np.pi*r)
    mask = (r>= 1.)
    out[mask] = 0.
    return out

def tukey(shape,alpha=0.1):
    """Returns a tukey window of a given shape and alpha parameter"""
    return aperture(shape, diameter = 1., alpha = alpha)

def aperture(shape, diameter = 1., alpha = 0.1):
    """Returns aperture window function. It is basically a tukey filter with a given diameter"""
    r = _r(shape, scale = diameter)
    out = np.ones(shape, FDTYPE)
    mask = r > (1-alpha)
    if alpha > 0.:
        tmp = 1/2*(1-np.cos(np.pi*(r-1)/alpha))
        out[mask] = tmp[mask]
    mask = (r>= 1.)
    out[mask] = 0.
    
    return out  

__all__ = ["aperture", "blackman"]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(aperture((33,33),diameter = 0.8))
    plt.subplot(122)
    plt.imshow(blackman((32,39)))