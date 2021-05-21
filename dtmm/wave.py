# -*- coding: utf-8 -*-
"""
Wave creation and wave characterization functions.

Core functions
--------------

* :func:`.betaphi` computes beta and phi arrays of the eigenmodes.
* :func:`.betaxy` computes betax and betay arrays of the eigenmodes.
* :func:`.eigenbeta` returns beta eigenvalue array(s).
* :func:`.eigenindices` returns indices array(s) of all valid eigenwaves.
* :func:`.eigenphi` returns phi eigenvalue array(s).
* :func:`.eigenmask` returns a mask array(s) of all valid egenwaves.
* :func:`.eigenwave` returns eigenwave array.
* :func:`.k0` computes wave number from wavelength.
* :func:`.planewave` returns planewave array.
* :func:`.wavelengths` returns wavelengths equaly spaced in wave numbers.

Conversion functions
--------------------

* :func:`.betaxy2beta` converts betax, betay arrays to beta array.
* :func:`.betaxy2phi` converts betax, betay arrays to phi array.
* :func:`.betaxy2betaphi` converts betax, betay arrays to beta, phi arrays.
* :func:`.betaphi2betax` converts beta, phi arrays to betax array.
* :func:`.betaphi2betay` converts beta, phi arrays to betay array.
* :func:`.betaphi2betaxy` converts beta, phi arrays to betax, betay arrays.
* :func:`.mask2beta` converts mask array to beta array(s).
* :func:`.mask2phi` converts mask array to phi array(s).
* :func:`.mask2indices` converts mask array to indices array(s).
* :func:`.wave2eigenwave` converts any plane wave to nearest eigenwave.

1D functions
------------

1D waves and conversion functions (for 2D simulations)

* :func:`.betax1` computes betax and betay arrays of the eigenmodes.
* :func:`.eigenbeta1` returns beta eigenvalue array(s).
* :func:`.eigenindices1` returns indices array(s) of all valid eigenwaves.
* :func:`.eigenmask1` returns a mask array(s) of all valid egenwaves.
* :func:`.eigenwave1` returns eigenwave array.
* :func:`.mask2beta1` converts mask array to beta array(s).
* :func:`.mask2indices1` converts mask array to indices array(s).
* :func:`.planewave1` returns planewave array.
"""

from __future__ import absolute_import, print_function, division

import numpy as np
from dtmm.conf import FDTYPE,CDTYPE, cached_function, get_default_config_option
import dtmm.fft as fft

def deprecation(message):
    import warnings
    warnings.warn(message, DeprecationWarning, stacklevel=2)

def betaphi(shape, k0, out = None):
    """Returns beta and phi arrays of all possible plane eigenwaves.
    
    Parameters
    ----------
    shape : (int,int)
        Shape (height, width) of the plane eigenwave.
    k0 : float or array of floats
        Wavenumber (or wavenumbers) in pixel units.
    out : (ndarray, ndarray), optional
        Output arrays tuple.
    
    Returns
    -------
    out : array, array
        beta, phi arrays. The shape of the outputs is beta, phi: (height,width) or 
        beta: (len(k0),height,width) if k0 is an array.
    """
    if out is None:
        out = None, None
    if len(shape) not in (2,):
        1/0
        deprecation("In the future exception will be raised if shape is not of length 2 or 3")
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
        Shape (height, width) of the plane eigenwave.
    k0 : float or array of floats
        Wavenumber (or wavenumbers) in pixel units.
    out : (ndarray, ndarray), optional
        Output arrays tuple.
    
    Returns
    -------
    out : array, array
        beta, phi arrays      
    """
    if len(shape) not in (2,):
        1/0
        deprecation("In the future exception will be raised if shape is not of length 2 or 3")
    #ax, ay = map(np.fft.fftfreq, shape,(d,)*len(shape))
    k0 = np.asarray(k0,dtype = FDTYPE)[...,np.newaxis,np.newaxis] #make it broadcastable
    ay, ax = map(lambda x : np.asarray(np.fft.fftfreq(x), dtype = FDTYPE), shape[-2:])
    xx, yy = np.meshgrid(ax, ay,copy = False, indexing = "xy") 
    if out is None:
        out = None, None
    l = (2 * np.pi / k0)
    return np.multiply(l,xx, out = out[0]), np.multiply(l,yy, out = out[1])

def betaxy2betaphi(betax, betay):
    """Converts betax, betay arrays to beta, phi arrays."""
    phi  = betaxy2phi(betax, betay)
    beta = betaxy2beta(betax, betay)
    return beta, phi

def betaxy2beta(betax, betay):
    """Converts betax, betay arrays to beta array"""
    return np.sqrt(betax**2 + betay**2)

def betaxy2phi(betax, betay):
    """Converts betax, betay arrays to phi array"""
    return np.arctan2(betay, betax)

def betaphi2betaxy(beta, phi):
    """Converts beta, phi arrays to betax, betay arrays."""
    betax = betaphi2betax(beta, phi)
    betay = betaphi2betay(beta, phi)
    return betax, betay

def betaphi2betax(beta, phi):
    """Converts beta, phi arrays to betax array."""
    return beta*np.cos(phi)

def betaphi2betay(beta, phi):
    """Converts beta, phi arrays to betay array."""
    return beta*np.sin(phi)

def eigenwave(shape, i, j, amplitude = None, out = None):
    """Returns a planewave with a given fourier coefficient indices i and j. 
    
    Parameters
    ----------
    shape : (...,int,int)
        Shape (height, width) of the plane eigenwave.
    i : int
        i-th index of the fourier coefficient 
    j : float
        j-th index of the fourier coefficient 
    amplitude : complex, optional
        Amplitude of the fourier mode. If not specified it is set so that the 
        amplitude in real space equals one.
    out : ndarray, optional
        Output array.
    
    Returns
    -------
    out : array
        Plane wave array of shape (...,height,width).       
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

@cached_function
def eigenbeta(shape, k0, betamax = None):
    """Returns masked beta array(s) of all valid eigenwaves.
    
    Parameters
    ----------
    shape : (int,int)
        Shape (height, width) of the plane eigenwave.
    k0 : float or array of floats
        Wavenumber (or wavenumbers) in pixel units.
    betamax : float, optional
        The cutoff beta value. If not specified, it is set to default.
        
    Returns
    -------
    out : ndarray or tuple of ndarrays.
        Masked beta array(s).  
    """    
    betamax = get_default_config_option("betamax", betamax)
    b,p = betaphi(shape, k0)
    mask = b <= betamax
    if mask.ndim == 3:
        bp = tuple([b[i][mask[i]] for i in range(mask.shape[0])])
        return bp
    else:
        b = b[mask]
        return b
    
@cached_function
def eigenindices(shape, k0, betamax = None):
    """Returns masked indices array(s) of all valid eigenwaves.
    
    Parameters
    ----------
    shape : (int,int)
        Shape (height, width) of the plane eigenwave.
    k0 : float or array of floats
        Wavenumber (or wavenumbers) in pixel units.
    betamax : float, optional
        The cutoff beta value. If not specified, it is set to default.
        
    Returns
    -------
    out : ndarray or tuple of ndarrays.
        Masked indices array(s).  
    """
    betamax = get_default_config_option("betamax", betamax)
    ii, jj = np.meshgrid(range(shape[-2]), range(shape[-1]),copy = False, indexing = "ij") 
    mask = eigenmask(shape, k0, betamax)
    if mask.ndim == 3: #multiwavelength
        out = (_get_indices_array(ii,jj, mask[i]) for i in range(mask.shape[0]))
        return tuple(out)
    else:
        return _get_indices_array(ii,jj, mask)
    
@cached_function
def eigenphi(shape, k0, betamax = None):
    """Returns masked phi array(s) of all valid eigenwaves.
    
    Parameters
    ----------
    shape : (int,int)
        Shape (height, width) of the plane eigenwave.
    k0 : float or array of floats
        Wavenumber (or wavenumbers) in pixel units.
    betamax : float, optional
        The cutoff beta value. If not specified, it is set to default.
        
    Returns
    -------
    out : ndarray or tuple of ndarrays.
        Masked phi array(s).  
    """
    betamax = get_default_config_option("betamax", betamax)
    b,p = betaphi(shape, k0)
    mask = b <= betamax
    if mask.ndim == 3:
        pp = tuple([p[mask[i]] for i in range(mask.shape[0])])
        return pp
    else:
        p = p[mask]
        return p

@cached_function
def eigenmask(shape, k0, betamax = None):
    """Returns a boolean array of valid modal coefficents. 
    
    Valid coefficients are those that have beta <= betamax.
    
    Parameters
    ----------
    shape : (int,int)
        Shape (height, width) of the plane eigenwave.
    k0 : float or array of floats
        Wavenumber (or wavenumbers) in pixel units.
    betamax : float, optional
        The cutoff beta value. If not specified, it is set to default.
        
    Returns
    -------
    mask : ndarray
        A boolean array of shape (height, width) or (len(k0), height, width).
    """
    betamax = get_default_config_option("betamax", betamax)
    b,p = betaphi(shape, k0)
    mask = b <= betamax
    return mask

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
    out : array
        Wavenumber array     
    """
    out = 2*np.pi/np.asarray(wavelength) * pixelsize
    return np.asarray(out, dtype = FDTYPE)

def mask2beta(mask,k0):
    """Converts the input mask array to a masked beta array(s).
    
    Parameters
    ----------
    mask : ndarray
        A mask array of shape (n,height,width) or (height,width). If it is a
        3D array the first axis must match the length of k0.
    k0 : float or array of floats
        Wavenumber (or wavenumbers if mask.ndim = 3) in pixel units. 
        
    Returns
    -------
    beta : ndarray or tuple of ndarrays
        Array(s) of maksed beta values.
    """
    mask = np.asarray(mask)
    shape = mask.shape[-2:]
    b,p = betaphi(shape, k0)
    if mask.ndim == 3:
        return tuple([b[i][mask[i]] for i in range(mask.shape[0])])
    elif mask.ndim == 2:
        return b[mask]
    else:
        raise ValueError("Invalid mask shape")
    
def mask2phi(mask,k0):
    """Converts the input mask array to a masked phi array(s).
    
    Parameters
    ----------
    mask : ndarray
        A mask array of shape (n,height,width) or (height,width). If it is a
        3D array the first axis must match the length of k0.
    k0 : float or array of floats
        Wavenumber (or wavenumbers if mask.ndim = 3) in pixel units. 
        
    Returns
    -------
    beta : ndarray or tuple of ndarrays
        Array(s) of maksed phi values.
    """
    mask = np.asarray(mask)
    shape = mask.shape[-2:]
    b,p = betaphi(shape, k0)
    if mask.ndim == 3:
        return tuple([p[mask[i]] for i in range(mask.shape[0])])
    elif mask.ndim == 2:
        return p[mask]
    else:
        raise ValueError("Invalid mask shape")

def mask2indices(mask, k0 = None):
    """Converts the input mask array to a masked indices array(s).
    
    Parameters
    ----------
    mask : ndarray
        A mask array of shape (n,height,width) or (height,width).
        
    Returns
    -------
    beta : ndarray or tuple of ndarrays
        Array(s) of maksed indices values.
    """
    mask = np.asarray(mask)
    if k0 is not None:
        deprecation("This function will only take one argument in the future")
    shape = mask.shape[-2:]
    ii, jj = np.meshgrid(range(shape[-2]), range(shape[-1]),copy = False, indexing = "ij") 
    if mask.ndim == 3: #multiwavelength
        out = (_get_indices_array(ii,jj, mask[i]) for i in range(mask.shape[0]))
        return tuple(out)
    elif mask.ndim == 2:
        return _get_indices_array(ii,jj, mask) 
    else:
        raise ValueError("Invalid mask shape")
    
def planewave(shape, k0, beta , phi, out = None):
    """Returns a 2D planewave array with a given beta, phi, wave number k0.
    
    Broadcasting rules apply.
    
    Parameters
    ----------
    shape : (int,int)
        Shape (height, width) of the plane eigenwave.
    k0 : float or floats array
        Wavenumber in pixel units. 
    beta : float or floats array
        Beta parameter of the plane wave
    phi: float or floats array
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
        
def wave2eigenwave(wave, out = None):
    """Converts any 2D wave to nearest eigenwave.
    
    Parameters
    ----------
    wave : ndarray
        Input 2D plane wave.
    out : ndarray, optional
        Ouptut array.
    
    Returns
    -------
    out : ndarray
        Nearest eigenwave array.
    """
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

    s = np.abs(np.fft.fft2(wave))**2
    
    if len(shape) == 2:
        s = s[None,...]
        o = out[None,...]

    for i,a in enumerate(s):
        avg = a.sum()
        indices = np.argmax(a)
        ii,jj = np.unravel_index(indices,a.shape)
        eigenwave(a.shape,ii,jj, amplitude = avg**0.5, out = o[i])
    
    return out

#------------------
#1D implementations
#------------------

def eigenwave1(n, i,  amplitude = None):
    """Returns a 1D eigenwave with a given fourier coefficient indices i. 
    
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
    out : array
        Plane wave array.       
    """    
    f = np.zeros((n,), dtype = CDTYPE)
    if amplitude is None:
        amplitude = n
    f[...,i] = amplitude
    return fft.ifft(f,overwrite_x = True)

def betax1(n, k0, out = None):
    """Returns 1D beta values of all possible 1D eigenwaves.
    
    Parameters
    ----------
    n : int
        Shape of the planewave.
    k0 : float or array of floats
        Wavenumber in pixel units.
    out : ndarray, optional
        Output arrays tuple
    
    Returns
    -------
    out : array
        beta array.  
    """
    k0 = np.abs(np.asarray(k0, FDTYPE)[...,np.newaxis]) #make it broadcastable
    xx = np.asarray(np.fft.fftfreq(n), dtype = FDTYPE)
    beta = np.multiply((2 * np.pi / k0) , xx, out = out)
    return beta

@cached_function
def eigenmask1(n, k0, betamax = None):
    """Returns a boolean array of valid modal coefficents of 1D waves. 
    
    Valid coefficients are those that have beta <= betamax.
    
    Parameters
    ----------
    n : int
        Shape of the 1D eigenwave.
    k0 : float or array of floats
        Wavenumber (or wavenumbers) in pixel units.
    betamax : float, optional
        The cutoff beta value. If not specified, it is set to default.
        
    Returns
    -------
    mask : ndarray
        A boolean array of shape (n,) or (len(k0), n).
    """
    betamax = get_default_config_option("betamax", betamax)
    b = betax1(n, k0)
    mask = np.abs(b) <= betamax
    return mask
    
def mask2betax1(mask,k0):
    """Converts the input mask array to a masked betax array(s).
    
    Parameters
    ----------
    mask : ndarray
        A mask array of shape (n,height) or (height,). If it is a
        2D array the first axis must match the length of k0.
    k0 : float or array of floats
        Wavenumber (or wavenumbers) in pixel units. 
        
    Returns
    -------
    beta : ndarray or tuple of ndarrays
        Array(s) of maksed betax values.
    """
    n = mask.shape[-1]
    b = betax1(n, k0)
    if mask.ndim == 2:
        return tuple([b[i][mask[i]] for i in range(mask.shape[0])])
    elif mask.ndim == 1:
        return b[mask]
    else:
        raise ValueError("Invalid mask shape.")
        
def mask2beta1(*args,**kwargs):
    deprecation("This function is deprecated, use mask2betax1")
    return mask2betax1(*args,**kwargs)
    
def mask2indices1(mask, k0 = None):
    """Converts the input mask array to a masked indices array(s).
    
    Parameters
    ----------
    mask : ndarray
        A mask array of shape (n,height) or (height,).
        
    Returns
    -------
    beta : ndarray or tuple of ndarrays
        Array(s) of maksed indices values.
    """
    if k0 is not None:
        deprecation("This function will only take one argument in the future")
    n = mask.shape[-1]
    ii = np.arange(n)
    if mask.ndim == 2: #multiwavelength
        out = (ii[mask[i]] for i in range(mask.shape[0]))
        return tuple(out)
    elif mask.ndim == 1:
        return ii[mask]
    else:
        raise ValueError("Invalid mask shape.")

@cached_function
def eigenbetax1(n, k0, betamax = None):
    """Returns masked betax1 array(s) of all valid eigenwaves.
    
    Parameters
    ----------
    n : int
        Shape of the plane eigenwave.
    k0 : float or array of floats
        Wavenumber (or wavenumbers) in pixel units.
    betamax : float, optional
        The cutoff beta value. If not specified, it is set to default.
        
    Returns
    -------
    out : ndarray or tuple of ndarrays.
        Masked beta array(s).  
    """  
    betamax = get_default_config_option("betamax", betamax)
    b = betax1(n, k0)
    mask = np.abs(b) <= betamax
    if mask.ndim == 2:
        return tuple([b[i][mask[i]] for i in range(mask.shape[0])])
    else:
        return b[mask]
    
def eigenbeta1(*args,**kwargs):
    deprecation("This function is deprecated, use eigenbetax1")
    return eigenbetax1(*args,**kwargs)

def _get_indices_array(ii,jj, mask):
    itmp = ii[mask]
    jtmp = jj[mask]
    out = np.empty(shape = itmp.shape + (2,), dtype = itmp.dtype)
    out[:,0] = itmp
    out[:,1] = jtmp
    return out

@cached_function
def eigenindices1(n, k0, betamax = None):
    """Returns masked indices array(s) of all valid 1D eigenwaves.
    
    Parameters
    ----------
    n : int
        Shape of the 1D eigenwave.
    k0 : float or array of floats
        Wavenumber (or wavenumbers) in pixel units.
    betamax : float, optional
        The cutoff beta value. If not specified, it is set to default.
        
    Returns
    -------
    out : ndarray or tuple of ndarrays.
        Masked indices array(s).  
    """ 
    betamax = get_default_config_option("betamax", betamax)
    ii = np.arange(n) 
    mask = eigenmask1(n, k0, betamax)
    if mask.ndim == 2: #multiwavelength
        out = (ii[mask[i]] for i in range(mask.shape[0]))
        return tuple(out)
    else:
        return ii[mask]
    
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
