#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Window functions.
"""

from __future__ import absolute_import, print_function, division

import numpy as np
from dtmm.conf import FDTYPE, cached_function

def _r(shape, scale = 1.):
    """Returns radius array of a given shape."""
    ny,nx = [l/2 for l in shape]
    ay, ax = [np.arange(-l / 2. + .5, l / 2. + .5) for l in shape]
    xx, yy = np.meshgrid(ax, ay, indexing = "xy")
    r = ((xx/(nx*scale))**2 + (yy/(ny*scale))**2) ** 0.5    
    return r

def _r2(shape):
    """Returns radius array of a given shape."""
    ay, ax = [np.arange(-l / 2. + .5, l / 2. + .5) for l in shape]
    xx, yy = np.meshgrid(ax, ay, indexing = "xy")
    r = ((xx)**2 + (yy)**2) ** 0.5    
    return r

@cached_function
def blackman(shape, out = None):
    """Returns a blacman window of a given shape
    
    Parameters
    ----------
    shape : (int,int)
        A shape of the 2D window.
    """
    r = _r(shape)  
    if out is None:
        out = np.ones(shape, FDTYPE)
    out[...] = 0.42 + 0.5*np.cos(1*np.pi*r)+0.08*np.cos(2*np.pi*r)
    mask = (r>= 1.)
    out[mask] = 0.
    return out

def gaussian(shape, waist):
    r = _r(shape, waist)
    out = np.empty(shape, FDTYPE)
    return np.exp(-r**2, out = out)

def gaussian_beam(shape, w0, k0, z = 0., n = 1):
    k = k0*n
    z0 = w0**2 *k/2.
    Rm2 = z/(z0**2+z**2)/2.
    w = w0 * (1+ (z/z0)**2)**0.5
    psi = np.arctan(z/z0)
    r = _r2(shape)
    return w0/w*np.exp(-(r/w)**2)*np.exp(1j*(k*z- psi))*np.exp(1j*(r*Rm2)**2)
    
def aperture(shape, diameter = 1., alpha = 0.05):
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
    r = np.asarray(r, FDTYPE)
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
    plt.imshow(aperture((32,32),diameter = 0.9))
    plt.subplot(122)
    plt.imshow(blackman((32,39)))