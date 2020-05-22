# -*- coding: utf-8 -*-
"""
Wave creation and wave characterization functions.
"""

from __future__ import absolute_import, print_function, division

import numpy as np
from dtmm.conf import FDTYPE,CDTYPE, cached_function, BETAMAX
import dtmm.fft as fft

def betax1(n, k0, out = None):
    """Returns beta values of all possible plane eigenwaves in 2D.
    
    Parameters
    ----------
    n : int
        Size of the planewave
    k0 : float
        Wavenumber in pixel units.
    out : ndarray, optional
        Output arrays tuple
    
    Returns
    -------
    array
        beta array  
    """
    k0 = np.abs(np.asarray(k0, FDTYPE)[...,np.newaxis]) #make it broadcastable
    xx = np.asarray(np.fft.fftfreq(n), dtype = FDTYPE)
    beta = np.multiply((2 * np.pi / k0) , xx, out = out)
    return beta

def betaxy2betaphi(betax, betay):
    phi  = np.arctan2(betay, betax)
    beta = np.sqrt(betax**2 + betay**2)
    return beta, phi

def betaxy2beta(betax, betay):
    return np.sqrt(betax**2 + betay**2)

def betaxy2phi(betax, betay):
    return np.arctan2(betay, betax)

def betaphi2betaxy(beta, phi):
    betax = beta*np.cos(phi)
    betay = beta*np.sin(phi)
    return betax, betay

def betaphi2betax(beta, phi):
    return beta*np.cos(phi)

def betaphi2betay(beta, phi):
    return beta*np.sin(phi)


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
    k0 = np.abs(np.asarray(k0, FDTYPE)[...,np.newaxis,np.newaxis]) #make it broadcastable
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

def wavelengths(start = 380,stop = 780, count = 9):
    """Raturns wavelengths (in nanometers) equaly spaced in wavenumbers between 
    start and stop.
    
    Parameters
    ----------
    start : float
        First wavelength
    stop : float
        Last wavelength    
    count : int
        How many wavelengths
        
    Returns
    -------
    out : ndarray
        A wavelength array
    """
    out = 1./np.linspace(1./start, 1./stop, count)
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

def eigenwave1(n, i,  amplitude = None):
    """Returns a planewave with a given fourier coefficient indices i 
    
    Parameters
    ----------
    n : int
        Shape of the plane eigenwave.
    i : int
        i-th index of the fourier coefficient 
    amplitude : complex
        Amplitude of the fourier mode.
    out : ndarray, optional
        Output array
    
    Returns
    -------
    array
        Plane wave array.       
    """    
    f = np.zeros((n,), dtype = CDTYPE)
    if amplitude is None:
        amplitude = n
    f[...,i] = amplitude
    return fft.ifft(f,overwrite_x = True)


@cached_function
def eigenmask(shape, k0, betamax = BETAMAX):
    b,p = betaphi(shape, k0)
    mask = b <= betamax
    return mask

@cached_function
def eigenmask1(n, k0, betamax = BETAMAX):
    b = betax1(n, k0)
    mask = np.abs(b) <= betamax
    return mask

def mask2beta(mask,k0):
    shape = mask.shape[-2:]
    b,p = betaphi(shape, k0)
    if mask.ndim == 3:
        return tuple([b[i][mask[i]] for i in range(mask.shape[0])])
    else:
        return b[mask]
    
def mask2beta1(mask,k0):
    n = mask.shape[-1]
    b = betax1(n, k0)
    if mask.ndim == 2:
        return tuple([b[i][mask[i]] for i in range(mask.shape[0])])
    else:
        return b[mask]
    
def mask2phi(mask,k0):
    shape = mask.shape[-2:]
    b,p = betaphi(shape, k0)
    if mask.ndim == 3:
        return tuple([p[mask[i]] for i in range(mask.shape[0])])
    else:
        return p[mask]

def mask2indices(mask, k0):
    shape = mask.shape[-2:]
    ii, jj = np.meshgrid(range(shape[-2]), range(shape[-1]),copy = False, indexing = "ij") 
    if mask.ndim == 3: #multiwavelength
        out = (_get_indices_array(ii,jj, mask[i]) for i in range(mask.shape[0]))
        return tuple(out)
    else:
        return _get_indices_array(ii,jj, mask)  

def mask2indices1(mask, k0):
    n = mask.shape[-1]
    ii = np.arange(n)
    if mask.ndim == 2: #multiwavelength
        out = (ii[mask[i]] for i in range(mask.shape[0]))
        return tuple(out)
    else:
        return ii[mask]

@cached_function
def eigenbeta(shape, k0, betamax = BETAMAX):
    b,p = betaphi(shape, k0)
    mask = b <= betamax
    if mask.ndim == 3:
        bp = tuple([b[i][mask[i]] for i in range(mask.shape[0])])
        return bp
    else:
        b = b[mask]
        return b
    
@cached_function
def eigenbeta1(n, k0, betamax = BETAMAX):
    b = betax1(n, k0)
    mask = np.abs(b) <= betamax
    if mask.ndim == 2:
        return tuple([b[i][mask[i]] for i in range(mask.shape[0])])
    else:
        return b[mask]
    
@cached_function
def eigenphi(shape, k0, betamax = BETAMAX):
    b,p = betaphi(shape, k0)
    mask = b <= betamax
    if mask.ndim == 3:
        pp = tuple([p[mask[i]] for i in range(mask.shape[0])])
        return pp
    else:
        p = p[mask]
        return p

def _get_indices_array(ii,jj, mask):
    itmp = ii[mask]
    jtmp = jj[mask]
    out = np.empty(shape = itmp.shape + (2,), dtype = itmp.dtype)
    out[:,0] = itmp
    out[:,1] = jtmp
    return out

@cached_function
def eigenindices(shape, k0, betamax = BETAMAX):
    ii, jj = np.meshgrid(range(shape[-2]), range(shape[-1]),copy = False, indexing = "ij") 
    mask = eigenmask(shape, k0, betamax)
    if mask.ndim == 3: #multiwavelength
        out = (_get_indices_array(ii,jj, mask[i]) for i in range(mask.shape[0]))
        return tuple(out)
    else:
        return _get_indices_array(ii,jj, mask)
 
@cached_function
def eigenindices1(n, k0, betamax = BETAMAX):
    ii = np.arange(n) 
    mask = eigenmask1(n, k0, betamax)
    if mask.ndim == 2: #multiwavelength
        out = (ii[mask[i]] for i in range(mask.shape[0]))
        return tuple(out)
    else:
        return ii[mask]
    
#@cached_function
#def eigenwaves(shape, k0, betamax = BETAMAX):
#    ii, jj = eigenindices(shape, k0, betamax)
#    out = np.empty(shape = (len(ii),) + shape, dtype = CDTYPE)
#    for n in range(len(ii)):
#        i = ii[n]
#        j = jj[n]
#        eigenwave(shape, i, j,  amplitude = 1, out =  out[n])
#    return out

def wave2eigenwave(wave, out = None):
    """Converts any wave to nearest eigenwave"""
    wave = np.asarray(wave)
    if out is None:
        out = np.empty(shape = wave.shape, dtype = CDTYPE)
    
    shape = wave.shape
    assert wave.ndim >= 2
    
    if len(shape) > 2:
        h = shape[-2]
        w = shape[-1]
        wave = wave.reshape(-1) #flatten
        n = len(wave)//h//w
        wave = wave.reshape((n,h,w))
        o = out.reshape((n,h,w))
    else:
        o = out
    
    s = np.abs(fft.fft2(wave))**2
    
    if len(shape) == 2:
        s = s[None,...]
        o = out[None,...]

    for i,a in enumerate(s):
        avg = a.sum()
        indices = np.argmax(a)
        ii,jj = np.unravel_index(indices,a.shape)
        eigenwave(a.shape,ii,jj, amplitude = avg**0.5, out = o[i])
    
    return out
    

def planewave(shape, k0, beta , phi, out = None):
    """Returns a 2D planewave array with a given beta, phi, wave number k0.
    
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

def planewave1(n, k0, beta, out = None):
    """Returns a 1D planewave array with a given beta, wave number k0.
    
    Parameters
    ----------
    n : int
        Shape of the plane eigenwave.
    k0 : float or array of floats
        Wavenumbers in pixel units.
    beta : float
        Beta parameter of the plane wave
    
    Returns
    -------
    array
        Plane wave array.       
    """
    k0 = np.asarray(k0)[...,np.newaxis] #make it broadcastable
    beta = np.asarray(beta)[...,np.newaxis]
    xx = np.arange(-n // 2 + 1., n // 2 + 1.)
    xx = np.asarray(xx, dtype = FDTYPE)
    kx = np.asarray(k0*beta, dtype = FDTYPE)
    out = np.exp((1j*(kx*xx)), out = out)
    return np.divide(out,out[...,0][...,None],out)

__all__ = [ "betaphi","betaxy","eigenwave","planewave","k0","wavelengths"]