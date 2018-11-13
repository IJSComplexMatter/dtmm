# -*- coding: utf-8 -*-
"""
Wave creation and wave characterization functions.
"""

from __future__ import absolute_import, print_function, division

import numpy as np
from dtmm.conf import FDTYPE,CDTYPE, cached_function
import dtmm.fft as fft

def betaphi(shape, k0, out = None):
    """Returns beta and phi arrays of all possible plane eigenwaves.
    
    Parameters
    ----------
    shape : (int,int)
        Shape of the plane eigenwave.
    k0 : float
        Wavenumber in pixel units.
    out : (ndarray, ndarray), optional
        Output arrays tuple
    
    Returns
    -------
    array, array
        beta, phi arrays    
    """
    if out is None:
        out = None, None
    k0 = np.asarray(k0, FDTYPE)[...,np.newaxis,np.newaxis] #make it broadcastable
    ay, ax = map(lambda x : np.asarray(np.fft.fftfreq(x), dtype = FDTYPE), shape[-2:])
    xx, yy = np.meshgrid(ax, ay,copy = False, indexing = "xy") 
    beta = np.multiply((2 * np.pi / k0) , np.sqrt(xx**2 + yy**2), out = out[0])
    phi  = np.arctan2(yy,xx, out = out[1])
    return beta, phi

@cached_function
def betaxy(shape, k0, out = None):
    """Returns betax, betay arrays of plane eigenwaves.
    
    Parameters
    ----------
    shape : (int,int)
        Shape of the plane eigenwave.
    k0 : float
        Wavenumber in pixel units.
    out : (ndarray, ndarray), optional
        Output arrays tuple
    
    Returns
    -------
    array, array
        beta, phi arrays      
    """
    #ax, ay = map(np.fft.fftfreq, shape,(d,)*len(shape))
    k0 = np.asarray(k0,dtype = FDTYPE)[...,np.newaxis,np.newaxis] #make it broadcastable
    ay, ax = map(lambda x : np.asarray(np.fft.fftfreq(x), dtype = FDTYPE), shape[-2:])
    xx, yy = np.meshgrid(ax, ay,copy = False, indexing = "xy") 
    if out is None:
        out = None, None
    l = (2 * np.pi / k0)
    return np.multiply(l,xx, out = out[0]), np.multiply(l,yy, out = out[1])

def k0(wavelength, pixelsize = 1.):
    """Calculate wave number in vacuum from a given wavelength and pixelsize
    
    Parameters
    ----------
    wavelength : float or array of floats
        Wavelength in nm
    pixelsize: float
        Pixelsize in nm
    
    Returns
    -------
    array
        Wavenumber array     
    """
    out = 2*np.pi/np.asarray(wavelength) * pixelsize
    return np.asarray(out, dtype = FDTYPE)

def eigenwave(shape, i, j, amplitude = None, out = None):
    """Returns a planewave with a given fourier coefficient indices i and j. 
    
    Parameters
    ----------
    shape : (int,int)
        Shape of the plane eigenwave.
    i : int
        i-th index of the fourier coefficient 
    j : float
        j-th index of the fourier coefficient 
    amplitude : complex
        Amplitude of the fourier mode.
    out : ndarray, optional
        Output array
    
    Returns
    -------
    array
        Plane wave array.       
    """    
    if out is None:
        f = np.zeros(shape, dtype = CDTYPE)
    else:
        f = np.asarray(out)
        f[...] = 0.
    if amplitude is None:
        amplitude = np.multiply.reduce(shape[-2:])
    f[...,i,j] = amplitude
    return fft.ifft2(f, out = out)

def planewave(shape, k0, beta , phi, out = None):
    """Returns a planewave array with a given beta, phi, wave number k0.
    
    Parameters
    ----------
    shape : (int,int)
        Shape of the plane eigenwave.
    k0 : float or array of floats
        Wavenumbers in pixel units.
    beta : float
        Beta parameter of the plane wave
    phi: float
        Phi parameter of the plane wave
    out : (ndarray, ndarray), optional
        Output arrays tuple
    
    Returns
    -------
    array
        Plane wave array.       
    """
    k0 = np.asarray(k0)[...,np.newaxis,np.newaxis] #make it broadcastable
    beta = np.asarray(beta)[...,np.newaxis,np.newaxis]
    phi = np.asarray(phi)[...,np.newaxis,np.newaxis]
    ay, ax = [np.arange(-l // 2 + 1., l // 2 + 1.) for l in shape[-2:]]
    xx, yy = np.meshgrid(ax, ay, indexing = "xy", copy = False)
    xx = np.asarray(xx, dtype = FDTYPE)
    yy = np.asarray(yy, dtype = FDTYPE)
    kx = np.asarray(k0*beta*np.cos(phi), dtype = FDTYPE)
    ky = np.asarray(k0*beta*np.sin(phi), dtype = FDTYPE)
    out = np.exp((1j*(kx*xx+ky*yy)), out = out)
    return np.divide(out,out[...,0,0][...,None,None],out)

__all__ = [ "betaphi","betaxy","eigenwave","planewave","k0"]