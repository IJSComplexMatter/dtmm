# -*- coding: utf-8 -*-
"""


"""

from __future__ import absolute_import, print_function, division

import numpy as np

from dtmm.conf import NCDTYPE,NFDTYPE, CDTYPE, NUDTYPE,  NUMBA_TARGET, NUMBA_PARALLEL, NUMBA_CACHE
from dtmm.wave import planewave
from dtmm.window import aperture
from dtmm.diffract import transmitted_field
#from dtmm.rotation import  _calc_rotations_uniaxial
#from dtmm.linalg import _inv4x4, _dotmr2
#from dtmm.dirdata import _uniaxial_order
#from dtmm.rotation import rotation_vector2

import numba as nb
from numba import prange

if NUMBA_PARALLEL == False:
    prange = range

sqrt = np.sqrt

def illumination_betaphi(NA, nrays = 13):
    """Returns beta, phi values for illumination.
    
    This function ca be used to define beta and phi arrays that can be used to
    construct illumination data with the :func:`illumination_data` function.
    The resulting beta,phi parameters define directions of rays for the input 
    light with a homogeneous angular distribution of rays - input
    light with a given numerical aperture.
    
    Parameters
    ----------
    NA : float
        Approximate numerical aperture of the illumination.
    nrays : int, optional
        Approximate number of rays. 
        
    Returns
    -------
    array, array
        beta, phi arrays 
        
    """
    radius = (nrays/np.pi)**0.5
    shape = (1+2*int(radius),1+2*int(radius))
    ay, ax = [np.arange(-l // 2 + 1., l // 2 + 1.) for l in shape]
    xx, yy = np.meshgrid(ax, ay,copy = False, indexing = "xy") 
    phi = np.arctan2(yy,xx)
    beta = np.sqrt(xx**2 + yy**2)/radius*NA
    mask = (beta <= NA)
    return beta[mask], phi[mask]

def illumination_waves(shape, k0, beta = 0., phi = 0., window = None, out = None):
    k0 = np.asarray(k0)
    beta = np.asarray(beta)[...,np.newaxis]
    phi = np.asarray(phi)[...,np.newaxis]
    if not k0.ndim in (0,1):
        raise ValueError("k0, must be an array with dimesion 1")
    out = planewave(shape, k0, beta, phi, out)
    if window is None:
        return out
    else:
        return np.multiply(out, window, out = out)
    
def waves2field(waves, k0, beta = 0., phi = 0., n = 1., jones = None, betamax = 0.9):
    beta = np.asarray(beta)
    phi = np.asarray(phi)
    k0 = np.asarray(k0)
    nrays = 1
    if jones is None:
        nrays = 2
    if beta.ndim == 1:
        nrays = len(beta)*nrays
        
    waves = waves/(nrays**0.5)   
        
    if jones is None:
        fieldv = np.zeros(beta.shape + (2,) + k0.shape + (4,) + waves.shape[-2:], dtype = CDTYPE)
    else:
        c,s = jones
        fieldv = np.zeros(beta.shape + k0.shape + (4,) + waves.shape[-2:], dtype = CDTYPE)
    
    if beta.ndim == 1: 
        for i,data in enumerate(fieldv):
            if jones is None:
                data[0,...,0,:,:] = waves[i]
                data[0,...,1,:,:] = waves[i]
                
                data[1,...,2,:,:] = waves[i]
                data[1,...,3,:,:] = -waves[i]
            else:
                data[...,0,:,:] = waves[i]*c
                data[...,1,:,:] = waves[i]*c
                data[...,2,:,:] = waves[i]*s
                data[...,3,:,:] = -waves[i]*s
                
    else:
        if jones is None:
            fieldv[0,...,0,:,:] = waves
            fieldv[0,...,1,:,:] = waves
            
            fieldv[1,...,2,:,:] = waves
            fieldv[1,...,3,:,:] = -waves
        else:
            fieldv[...,0,:,:] = waves*c
            fieldv[...,1,:,:] = waves*c
            fieldv[...,2,:,:] = waves*s
            fieldv[...,3,:,:] = -waves*s    
            
    return transmitted_field(fieldv,k0, n = n)


def illumination_data(shape, wavelengths, pixelsize = 1., beta = 0., phi = 0., 
                      n = 1.,  window = None, jones = None, betamax = 0.9):
    """Constructs forward propagating input illumination field data.
    
    Parameters
    ----------
    shape : (int,int)
        Shape of the illumination
    wavelengths : array_like
        A list of wavelengths.
    pixelsize : float, optional
        Size of the pixel in nm.
    beta : float or array_like of floats, optional
        Beta parameter(s) of the illumination. (Default 0. - normal incidence) 
    phi : float or array_like of floats, optional
        Azimuthal angle(s) of the illumination. 
    n : float, optional
        Refractive index of the media that this illumination field is assumed to
        be propagating in (default 1.)
    window : array or None, optional
        If None, no window function is applied. This window function
        is multiplied with the constructed plane waves to define field diafragm
        of the input light. See :func:`.window.aperture`.
    jones : jones vector or None, optional
        If specified it has to be a valid jones vector that defines polarization
        of the light. If not given (default), the resulting field will have two
        polarization components. See documentation for details and examples.
    
    """
    wavelengths = np.asarray(wavelengths)
    wavenumbers = 2*np.pi/wavelengths * pixelsize
    if wavenumbers.ndim not in (1,):
        raise ValueError("Wavelengths should be 1D array")
    waves = illumination_waves(shape, wavenumbers, beta = beta, phi = phi, window = window)
    field = waves2field(waves, wavenumbers, beta = beta, phi = phi, n = n, jones = jones, betamax = betamax)
    
    return (field, wavelengths, pixelsize)


def jonesvec(pol):
    """Returns normalized jones vector from an input length 2 vector. 
    Numpy broadcasting rules apply.
    
    Example
    -------
    
    >>> jonesvec((1,1j)) #left-handed circuar polarization
    array([ 0.70710678+0.j        ,  0.00000000+0.70710678j])
    
    """
    pol = np.asarray(pol)
    assert pol.shape[-1] == 2
    norm = (pol[...,0] * pol[...,0].conj() + pol[...,1] * pol[...,1].conj())**0.5
    return pol/norm[...,np.newaxis]

@nb.njit([(NCDTYPE[:,:,:,:],NFDTYPE[:,:,:])], parallel = NUMBA_PARALLEL, cache = NUMBA_CACHE)
def _field2specter(field, out):
    
    for j in prange(field.shape[2]):
        for i in range(field.shape[0]):
            for k in range(field.shape[3]):
                tmp1 = (field[i,0,j,k].real * field[i,1,j,k].real + field[i,0,j,k].imag * field[i,1,j,k].imag)
                tmp2 = (field[i,2,j,k].real * field[i,3,j,k].real + field[i,2,j,k].imag * field[i,3,j,k].imag)
                out[j,k,i] = tmp1-tmp2 

@nb.njit([(NCDTYPE[:,:,:,:,:],NFDTYPE[:,:,:])], parallel = NUMBA_PARALLEL, cache = NUMBA_CACHE)
def _field2spectersum(field, out):
    for n in prange(field.shape[0]):
        for j in range(field.shape[3]):
            for i in range(field.shape[1]):
                for k in range(field.shape[4]):
                    tmp1 = (field[n,i,0,j,k].real * field[n,i,1,j,k].real + field[n,i,0,j,k].imag * field[n,i,1,j,k].imag)
                    tmp2 = (field[n,i,2,j,k].real * field[n,i,3,j,k].real + field[n,i,2,j,k].imag * field[n,i,3,j,k].imag)
                    if n == 0:
                        out[j,k,i] = tmp1 - tmp2
                    else:
                        out[j,k,i] += (tmp1 -tmp2)

@nb.guvectorize([(NCDTYPE[:,:,:,:],NFDTYPE[:,:,:])],"(w,k,n,m)->(n,m,w)", target = "cpu", cache = NUMBA_CACHE)
def field2specter(field, out):
    _field2specter(field, out)

@nb.guvectorize([(NCDTYPE[:,:,:,:,:],NFDTYPE[:,:,:])],"(l,w,k,n,m)->(n,m,w)", target = "cpu", cache = NUMBA_CACHE)
def field2spectersum(field, out):
    _field2spectersum(field, out)    

    
def validate_field_data(field_waves):
    pass

__all__ = ["validate_field_data", "jonesvec","field2specter", "illumination_data","illumination_betaphi"]