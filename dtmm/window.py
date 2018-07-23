#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
    """Returns a blacman window of a given shape
    
    Parameters
    ----------
    shape : (int,int)
        A shape of the 2D window.
    """
    r = _r(shape)   
    out = np.ones(shape, FDTYPE)
    out[...] = 0.42 + 0.5*np.cos(1*np.pi*r)+0.08*np.cos(2*np.pi*r)
    mask = (r>= 1.)
    out[mask] = 0.
    return out

def aperture(shape, diameter = 1., alpha = 0.1):
    """Returns aperture window function. 
    
    Aperture is basically a tukey filter with a given diameter
    
    Parameters
    ----------
    shape : (int,int)
        A shape of the 2D window
    diameter : float
        Width of the aperture (1. for max height/width)
    alpha : float
        Smoothnes parameter (should be between 0. and 1.)
    """
    r = _r(shape, scale = diameter)
    return tukey(r,alpha)

def tukey(r,alpha = 0.1, rmax = 1.):
    out = np.ones(r.shape, FDTYPE)
    alpha = alpha * rmax
    mask = r > rmax -alpha
    if alpha > 0.:
        tmp = 1/2*(1-np.cos(np.pi*(r-rmax)/alpha))
        out[mask] = tmp[mask]
    mask = (r>= rmax)
    out[mask] = 0.
    return out  

__all__ = ["aperture"]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(aperture((128,128),diameter = 0.8))
    plt.subplot(122)
    plt.imshow(blackman((32,39)))