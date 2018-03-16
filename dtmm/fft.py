#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom 2D FFT functions.

numpy, scipy and mkl_fft do not have fft implemented such that output argument 
can be provided. This implementation adds the output argument for fft2 and 
ifft2 functions.
"""

from dtmm.conf import DTMMConfig, CDTYPE
from dtmm.conf import MKL_FFT_INSTALLED
import numpy as np
import scipy.fftpack as spfft
import numpy.fft as npfft

from functools import reduce


if MKL_FFT_INSTALLED == True:
    import mkl_fft

def _set_out(a,out):
    if out is not a:
        if out is None:
            out = a.copy()
        else:
            out[...] = a 
    return out

def _reshape(a):
    shape = a.shape
    newshape = reduce((lambda x,y: x*y), shape[:-2] or [1])
    newshape = (newshape,) + shape[-2:]
    a = a.reshape(newshape)
    return shape, a    

def __mkl_fft(fft,a,out):
    out = _set_out(a,out)
    shape, out = _reshape(out)
    #I am reshaping, because doing fft sequentially is much faster
    [fft(d,overwrite_x = True) for d in out] 
    return out.reshape(shape)    

def _mkl_fft2(a,out = None):
    return __mkl_fft(mkl_fft.fft2,a,out)

def _mkl_ifft2(a,out = None):
    return __mkl_fft(mkl_fft.ifft2,a,out)

def __sp_fft(fft,a,out):
    if out is None:
        return fft(a)
    elif out is a:
        out = fft(a, overwrite_x = True)
        if out is a:
            return out
        else:
            a[...] = out
        return a
    else:
        out[...] = fft(a)
        return out
        
def _sp_fft2(a, out = None):
    return __sp_fft(spfft.fft2, a, out)
    
def _sp_ifft2(a, out = None):
    return __sp_fft(spfft.ifft2, a, out)        

def __np_fft(fft,a,out):
    if out is None:
        return fft(a)
    else:
        out[...] = fft(a)
        return out
        
def _np_fft2(a, out = None):
    return __np_fft(npfft.fft2, a, out)
    
def _np_ifft2(a, out = None):
    return __np_fft(npfft.ifft2, a, out)       

                
def fft2(a, out = None):
    """Computes fft2 of the input complex array.
    
    Parameters
    ----------
    a : array_like
        Input array (must be complex).
    out : array or None, optional
       Output array. Can be same as input for fast inplace transform.
       
    Returns
    -------
    out : complex ndarray
        Result os the transformation along the last two axes.
    """
    a = np.asarray(a, dtype = CDTYPE)
    libname = DTMMConfig["fftlib"]
    if libname == "mkl_fft":
        return _mkl_fft2(a, out)
    elif libname == "scipy":
        return _sp_fft2(a, out)
    else:
        return _np_fft2(a, out)    
    
def ifft2(a, out = None): 
    """Computes ifft2 of the input complex array.
    
    Parameters
    ----------
    a : array_like
        Input array (must be complex).
    out : array or None, optional
       Output array. Can be same as input for fast inplace transform.
       
    Returns
    -------
    out : complex ndarray
        Result os the transformation along the last two axes.
    """
    a = np.asarray(a, dtype = CDTYPE)      
    libname = DTMMConfig["fftlib"]
    if libname == "mkl_fft":
        return _mkl_ifft2(a, out)
    
    elif libname == "scipy":
        return _sp_ifft2(a, out)
    else:
        return _np_ifft2(a, out)        
 