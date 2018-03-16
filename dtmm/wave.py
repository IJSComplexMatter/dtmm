# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:44:09 2017

@author: andrej

Wave utilities...

"""

from __future__ import absolute_import, print_function, division

import numpy as np

from dtmm.conf import NCDTYPE,NFDTYPE, CDTYPE
from dtmm.window import blackman, aperture
from dtmm.color import load_tcmf

import numba as nb

import dtmm.fft as fft


#@nb.vectorize([NCDTYPE(NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE)], target = "parallel")
#def _exp_ikr(kx,ky,xx,yy):
#    kr = xx*kx + yy*ky
#    return np.exp(1j*kr)

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
    ax, ay = map(np.fft.fftfreq, shape)
    xx, yy = np.meshgrid(ax, ay,copy = False, indexing = "ij") 
    phi = np.arctan2(yy,xx)
    beta = (2 * np.pi / k0) * np.sqrt(xx**2 + yy**2)
    return beta, phi

def illumination_betaphi(NA, radius = 2.):
    """Returns beta, phi values for illumination.
    
    Parameters
    ----------
    NA : float
        Numerical aperture of the illumination.
    radius : float, optional
        Radius of the illumination in pixel values. Output length will be
        approx Pi*radius**2.
        
    Returns
    -------
    array, array
        beta, phi arrays 
        
    """
    shape = (1+2*int(radius),1+2*int(radius))
    ax, ay = [np.arange(-l // 2 + 1., l // 2 + 1.) for l in shape]
    xx, yy = np.meshgrid(ax, ay,copy = False, indexing = "ij") 
    phi = np.arctan2(yy,xx)
    beta = np.sqrt(xx**2 + yy**2)/radius*NA
    mask = (beta <= NA)
    return beta[mask], phi[mask]

 

def betaxy(shape, k0):
    """Returns betax, betay arrays of plane eigenwaves with 
    a given wave number k0 and step size d"""
    #ax, ay = map(np.fft.fftfreq, shape,(d,)*len(shape))
    k0 = np.asarray(k0)[...,np.newaxis,np.newaxis] #make it broadcastable
    ax, ay = map(np.fft.fftfreq, shape)
    xx, yy = np.meshgrid(ax, ay,copy = False, indexing = "ij") 
    betax = 2 * np.pi * xx/k0
    betay = 2 * np.pi * yy/k0
    return betax, betay

def k0(wavelength,d = 1.):
    """Calculate wave number in vacuum from a given wavelength"""
    return 2*np.pi/wavelength * d

def eigenwave(shape, i, j, amplitude = None, out = None):
    """Returns a planewave with a given fourier coefficients i and j"""
    if out is None:
        f = np.zeros(shape, dtype = CDTYPE)
    else:
        f = np.asarray(out)
        f[...] = 0.
    if amplitude is None:
        amplitude = np.multiply.reduce(shape)
    f[i,j] = amplitude
    return fft.ifft2(f, out = out)

def planewave(shape, k0, beta , phi, out = None):
    """Returns a planewave array with a given beta, phi, wave number k0."""
    k0 = np.asarray(k0)[...,np.newaxis,np.newaxis] #make it broadcastable
    beta = np.asarray(beta)[...,np.newaxis,np.newaxis]
    phi = np.asarray(phi)[...,np.newaxis,np.newaxis]
    ax, ay = [np.arange(-l // 2 + 1., l // 2 + 1.) for l in shape]
    xx, yy = np.meshgrid(ax, ay, indexing = "ij", copy = False)
    kx = k0*beta*np.cos(phi)
    ky = k0*beta*np.sin(phi)
    #return _exp_ikr(kx,ky,xx,yy)
    return np.exp((1j*(kx*xx+ky*yy)),out)

def illumination_waves(shape, k0, beta = 0., phi = 0., window = None, out = None):
    k0 = np.asarray(k0)
    beta = np.asarray(beta)[...,np.newaxis]
    phi = np.asarray(phi)[...,np.newaxis]
    if not k0.ndim in (0,1):
        raise TypeError("k0, must be an array with dimesion 1")
    out = planewave(shape, k0, beta, phi, out)
    if window is None:
        return out
    else:
        return out*window
    
def illumination2field(waves, k0, beta = 0., phi = 0., refind = 1., pol = None):
    beta = np.asarray(beta)
    phi = np.asarray(phi)
    k0 = np.asarray(k0)
    fieldv = np.zeros(beta.shape + (2,) + k0.shape + (4,) + waves.shape[-2:], dtype = CDTYPE)
    
    if beta.ndim > 0: 
        for i,data in enumerate(fieldv):
        
            data[0,...,0,:,:] = waves[i]
            data[0,...,1,:,:] = waves[i]
            
            data[1,...,2,:,:] = waves[i]
            data[1,...,3,:,:] = -waves[i]
    else:
            fieldv[0,...,0,:,:] = waves
            fieldv[0,...,1,:,:] = waves
            
            fieldv[1,...,2,:,:] = waves
            fieldv[1,...,3,:,:] = -waves
    return fieldv


def illumination_data(shape, wavelengths, beta = 0., phi = 0., 
                      refind = 1., pixelsize = 1., diameter = 0.9, alpha = 0.1, pol = None):
    wavenumbers = 2*np.pi/np.asarray(wavelengths) * pixelsize
    window = aperture(shape, diameter, alpha)
    waves = illumination_waves(shape, wavenumbers, beta = beta, phi = phi, window = window)
    field = illumination2field(waves, wavenumbers, beta = beta, phi = phi, refind = 1., pol = pol)
    cmf = load_tcmf(wavelengths)
    return field, wavenumbers, cmf
    

def _wave2field(wave,k0,beta,phi, refind = 1, out = None):
    k0 = np.asarray(k0)[...,np.newaxis,np.newaxis] #make it broadcastable
    beta = np.asarray(beta)[...,np.newaxis,np.newaxis]
    phi = np.asarray(phi)[...,np.newaxis,np.newaxis]
    if out is None:
        out = np.empty(wave.shape[0:-2] + (4,) + wave.shape[-2:], dtype = CDTYPE)
    c = np.cos(phi)
    s = np.sin(phi)
    alpha = (n**2-(beta)**2)**0.5

    out[...,0,:,:] = wave*alpha * c 
    out[...,1,:,:] = wave * c 
    out[...,2,:,:] = wave * s 
    out[...,3,:,:] = -wave*alpha * s   
    return out

def wave2field(wave,k0, n = 1, out = None):
    wave = fft.fft2(wave)
    k0 = np.asarray(k0)
    beta, phi = betaphi(wave.shape, k0)
    if out is None:
        out = np.empty(wave.shape[0:-2] + (4,) + wave.shape[-2:], dtype = CDTYPE)
    c = np.cos(phi)
    s = np.sin(phi)
    alpha = (n**2-(beta)**2)**0.5
    alpha[beta>1] = 0.
    wave[beta>1] = 0.
    out[...,0,:,:] = wave*alpha * c 
    out[...,1,:,:] = wave * c 
    out[...,2,:,:] = wave * s 
    out[...,3,:,:] = -wave*alpha * s 
    
    fft.ifft2(out,out = out)
    
    return out

def planewave_field(shape, k0, beta , phi, pol = (1,0),n = 1, out = None):
    """Returns a planewave array with a given beta, phi, wave number k0."""
    wave = planewave(shape, k0, beta , phi)
    return _wave2field(wave,k0,beta,phi, n = n, out = out)

def eigenwave_field(shape, k0, i,j, pol = (1,0),n = 1, out = None):
    """Returns a planewave array with a given beta, phi, wave number k0."""
    wave = eigenwave(shape, i,j)
    beta, phi = mean_betaphi(wave, k0)
    return _wave2field(wave,k0,beta,phi, n = n, out = out)
    
@nb.guvectorize([(NCDTYPE[:,:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:])], "(n,m),()->(),()")
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

__all__ = ["illumination_betaphi", "illumination_waves", "illumination2field", "betaphi","planewave","illumination_data"]