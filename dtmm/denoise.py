#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Field denoising
"""

import numpy as np
from dtmm.conf import FDTYPE
from dtmm.wave import betaphi
from dtmm.fft import fft2,ifft2

def tukey_notch_filter(x,x0,alpha):
    x = np.asarray(x, FDTYPE)
    out = np.ones(x.shape, FDTYPE)
    alpha = alpha * x0
    mask = (x < x0 + alpha)
    mask = np.logical_and(mask, (x > x0 - alpha))
    if alpha > 0.:
        tmp = 1/2*(1-np.cos(np.pi*(x-x0)/alpha))
        out[mask] = tmp[mask]
    return out  

def exp_notch_filter(x,x0,sigma):
    return np.asarray((1 - 1*np.exp(-np.abs(x-x0).clip(0,x0)/sigma))/(1-1*np.exp(-x0/sigma)),FDTYPE)

def denoise_field(field, wavenumbers, beta , smooth = 1, filter_func = exp_notch_filter, out = None):
    """Denoises field by attenuating modes around the selected beta parameter.
    """
    ffield = fft2(field, out = out)
    ffield = denoise_fftfield(ffield, wavenumbers, beta, smooth = smooth, filter_func = filter_func, out = ffield)
    return ifft2(ffield, out = ffield)

def denoise_fftfield(ffield, wavenumbers, beta, smooth = 1, filter_func = exp_notch_filter, out = None):
    """Denoises fourier transformed field by attenuating modes around the selected beta parameter.
    """
    shape = ffield.shape[-2:]
    b, p = betaphi(shape,wavenumbers)
    f = filter_func(b, beta, smooth)
    f = f[...,None,:,:]
    ffield = np.multiply(ffield, f, out = out)
    return ffield
