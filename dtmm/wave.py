# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:44:09 2017

@author: andrej

Wave utilities...

"""

from __future__ import absolute_import, print_function, division

import numpy as np

from dtmm.conf import FDTYPE, CDTYPE, NCDTYPE,NFDTYPE, CDTYPE,  NUMBA_CACHE
from dtmm.window import blackman

#from dtmm.diffract import transmitted_field

import numba as nb

import dtmm.fft as fft

def betaphi(shape, k0):
    """Returns beta, phi of all possible plane eigenwaves.
    
    Parameters
    ----------
    shape : (int,int)
        Shape of the plane eigenwave.
    k0 : float
        Wavenumber in pixel units.
    
    Returns
    -------
    array, array
        beta, phi arrays    
    """
    k0 = np.asarray(k0)[...,np.newaxis,np.newaxis] #make it broadcastable
    #ax, ay = map(np.fft.fftfreq, shape,(d,)*len(shape))
    ay, ax = map(np.fft.fftfreq, shape)
    xx, yy = np.meshgrid(ax, ay,copy = False, indexing = "xy") 
    beta = (2 * np.pi / k0) * np.sqrt(xx**2 + yy**2)
    beta = np.asarray(beta, dtype = FDTYPE)
    phi = np.empty_like(beta)
    phi[...,:,:] = np.arctan2(yy,xx)
    return beta, phi

def betaxy(shape, k0):
    """Returns betax, betay arrays of plane eigenwaves with 
    a given wave number k0 and step size d"""
    #ax, ay = map(np.fft.fftfreq, shape,(d,)*len(shape))
    k0 = np.asarray(k0)[...,np.newaxis,np.newaxis] #make it broadcastable
    ay, ax = map(np.fft.fftfreq, shape)
    xx, yy = np.meshgrid(ax, ay,copy = False, indexing = "xy") 
    betax = 2 * np.pi * xx/k0
    betay = 2 * np.pi * yy/k0
    return np.asarray(betax, dtype = FDTYPE), np.asarray(betay, dtype = FDTYPE)

def k0(wavelength,d = 1.):
    """Calculate wave number in vacuum from a given wavelength"""
    out = 2*np.pi/np.asarray(wavelength) * d
    return np.asarray(out, dtype = FDTYPE)

def eigenwave(shape, i, j, amplitude = None, out = None):
    """Returns a planewave with a given fourier coefficients i and j"""
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
    """Returns a planewave array with a given beta, phi, wave number k0."""
    k0 = np.asarray(k0)[...,np.newaxis,np.newaxis] #make it broadcastable
    beta = np.asarray(beta)[...,np.newaxis,np.newaxis]
    phi = np.asarray(phi)[...,np.newaxis,np.newaxis]
    ay, ax = [np.arange(-l // 2 + 1., l // 2 + 1.) for l in shape]
    xx, yy = np.meshgrid(ax, ay, indexing = "xy", copy = False)
    xx = np.asarray(xx, dtype = FDTYPE)
    yy = np.asarray(yy, dtype = FDTYPE)
    kx = np.asarray(k0*beta*np.cos(phi), dtype = FDTYPE)
    ky = np.asarray(k0*beta*np.sin(phi), dtype = FDTYPE)
    #return _exp_ikr(kx,ky,xx,yy)
    out = np.exp((1j*(kx*xx+ky*yy)), out = out)
    return np.divide(out,out[...,0,0][...,None,None],out)


#def _wave2field(wave,k0,beta,phi, refind = 1, out = None):
#    k0 = np.asarray(k0)[...,np.newaxis,np.newaxis] #make it broadcastable
#    beta = np.asarray(beta)[...,np.newaxis,np.newaxis]
#    phi = np.asarray(phi)[...,np.newaxis,np.newaxis]
#    if out is None:
#        out = np.empty(wave.shape[0:-2] + (4,) + wave.shape[-2:], dtype = CDTYPE)
#    c = np.cos(phi)
#    s = np.sin(phi)
#    alpha = (n**2-(beta)**2)**0.5
#
#    out[...,0,:,:] = wave*alpha * c 
#    out[...,1,:,:] = wave * c 
#    out[...,2,:,:] = wave * s 
#    out[...,3,:,:] = -wave*alpha * s   
#    return out
#
#def wave2field(wave,k0, n = 1, out = None):
#    wave = fft.fft2(wave)
#    k0 = np.asarray(k0)
#    beta, phi = betaphi(wave.shape, k0)
#    if out is None:
#        out = np.empty(wave.shape[0:-2] + (4,) + wave.shape[-2:], dtype = CDTYPE)
#    c = np.cos(phi)
#    s = np.sin(phi)
#    alpha = (n**2-(beta)**2)**0.5
#    alpha[beta>1] = 0.
#    wave[beta>1] = 0.
#    out[...,0,:,:] = wave*alpha * c 
#    out[...,1,:,:] = wave * c 
#    out[...,2,:,:] = wave * s 
#    out[...,3,:,:] = -wave*alpha * s 
#    
#    fft.ifft2(out,out = out)
#    
#    return out

#def planewave_field(shape, k0, beta , phi, pol = (1,0),n = 1, out = None):
#    """Returns a planewave array with a given beta, phi, wave number k0."""
#    wave = planewave(shape, k0, beta , phi)
#    return _wave2field(wave,k0,beta,phi, n = n, out = out)
#
#def eigenwave_field(shape, k0, i,j, pol = (1,0),n = 1, out = None):
#    """Returns a planewave array with a given beta, phi, wave number k0."""
#    wave = eigenwave(shape, i,j)
#    beta, phi = mean_betaphi(wave, k0)
#    return _wave2field(wave,k0,beta,phi, n = n, out = out)
    
@nb.guvectorize([(NCDTYPE[:,:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:])], "(n,m),()->(),()", cache = NUMBA_CACHE)
def mean_betaphi(wave, k0, beta, phi):
    """Calculates mean beta and phi of a given wave array. """
    b = blackman(wave.shape)
    f = fft.fft2(wave*b) #filter it with blackman..
    s = np.abs(f)**2
    p = s/s.sum()#normalize probability coefficients
    betax, betay = betaxy(wave.shape,k0)
    betax = (betax*p).sum()
    betay = (betay*p).sum()
    #save results to output
    beta[0] = (betax**2+betay**2)**0.5
    phi[0] = np.arctan2(betay,betax)

__all__ = [ "betaphi","planewave","k0"]