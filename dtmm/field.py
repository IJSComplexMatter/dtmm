"""
Field creation, transformation in IO functions
"""

from __future__ import absolute_import, print_function, division

import numpy as np

from dtmm.conf import NCDTYPE,NFDTYPE, FDTYPE, CDTYPE, NUMBA_PARALLEL, NUMBA_TARGET, NUMBA_CACHE, BETAMAX , DTMMConfig
from dtmm.wave import planewave, betaphi, wave2eigenwave
from dtmm.diffract import diffracted_field, diffraction_alphaf
from dtmm.window import aperture
from dtmm.fft import fft2, ifft2, fft, ifft
from dtmm.wave import betaxy, eigenmask, eigenmask1
from dtmm.window import blackman
from dtmm.tmm import alphaf ,fvec2E, E2fvec
from dtmm.tmm import fvec as field4
from dtmm.data import refind2eps
from dtmm.jones import jonesvec

import numba as nb
from numba import prange

if NUMBA_PARALLEL == False:
    prange = range
    
sqrt = np.sqrt

def field2fvec(field):
    """transposes field array from shape (..., k,n,m) to (...,n,m,k)."""
    return np.moveaxis(field,-3,-1)

def fvec2field(vec):
    """transposes vector from shape (..., n,m,k) to (...,k,n,m)"""
    return np.moveaxis(vec,-1,-3)

def field2jones(field, ks, beta = None, phi = None, epsv = (1.,1.,1.), epsa = (0.,0.,0.), mode = +1, input_fft = False, output_fft = False, betamax = BETAMAX):
    """Converts (..., 4,n,m) field array to (..., 2,n,m) jones array
    
    The conversion is done either in reciprocal space (default) or in real space,
    if you provide the beta or phi parameter.
    
    Parameters
    ----------
    field : ndarray
        Input field array data of shape (..., 4,n,m).
    ks : array
        A list of k-values (wave numbers).
    beta : float, optional
        If set, it defines the field beta parameter, conversion is then done
        in real space.
    phi : float, optional
        If set, it defines the field phi parameter, conversion is then done
        in real space.
    epsv : array
        The epsilon eigenvalues of the medium
    epsv : array
        The Euler rotation angles.
    mode : [+1 | -1]
        Propagation mode, +1 (forward) by default 
    input_fft : bool
        Specifies whether input array is fft data or not.
    output_fft : bool
        Specifies whether output array is fft data or not.
    betamax : float
        Defines the betamax parameter if transformation is done in Fourier space.
        
    Returns
    -------
    jones : ndarray
        Output jones array of shape (..., 2,n,m).
    """
    modal = beta is None and phi is None
    
    field = np.asarray(field)
    out = np.empty(field.shape[:-3] + (2,) + field.shape[-2:], field.dtype)
    if input_fft == False and modal:
        field = fft2(field)  
    elif input_fft == True and not modal:
        field = ifft2(field) 
    shape = field.shape[-3:]
    if modal:
        a,fmat = diffraction_alphaf(shape, ks, epsv, epsa, betamax = betamax)
    else:
        a, fmat, = alphaf(beta,phi,epsv,epsa) 
    fvec = field2fvec(field)
    evec = field2fvec(out)
    evec[...] = fvec2E(fvec, fmat = fmat, mode = mode)
    if output_fft == False and modal:
        ifft2(out, out = out)
    elif output_fft == True and not modal:
        fft2(out, out = out)
    return out

def jones2field(jones, ks, beta = None, phi = None, epsv = (1.,1.,1.), epsa = (0.,0.,0.),  mode = +1, input_fft = False, output_fft = False, betamax = BETAMAX):
    """Converts (..., 2,n,m) jones array to (..., 4,n,m) field array
    
    The conversion is done either in reciprocal space (default) or in real space,
    if you provide the beta or phi parameter.
    
    Parameters
    ----------
    jones : ndarray
        Input jones array data of shape (..., 2,n,m).
    ks : array
        A list of k-values (wave numbers).
    beta : float, optional
        If set, it defines the field beta parameter, conversion is then done
        in real space.
    phi : float, optional
        If set, it defines the field phi parameter, conversion is then done
        in real space.
    epsv : array
        The epsilon eigenvalues of the medium
    epsv : array
        The Euler rotation angles.
    mode : [+1 | -1]
        Propagation mode, +1 (forward) by default 
    input_fft : bool
        Specifies whether input array is fft data or not.
    output_fft : bool
        Specifies whether output array is fft data or not.
    betamax : float
        Defines the betamax parameter if transformation is done in Fourier space.
        
    Returns
    -------
    field : ndarray
        Output field array of shape (..., 4,n,m).
    """

    jones= np.asarray(jones)
    modal = beta is None and phi is None
    
    out = np.empty(jones.shape[:-3] + (4,) + jones.shape[-2:], jones.dtype)
    if input_fft == False and modal:
        jones = fft2(jones)   
    elif input_fft == True and not modal:
        jones = ifft2(jones)  
        
    shape = jones.shape[-3:]
    if modal:
        a,fmat = diffraction_alphaf(shape, ks, epsv, epsa, betamax = betamax)
    else:
        a, fmat, = alphaf(beta,phi,epsv,epsa) 
    fvec = field2fvec(out)
    jvec = field2fvec(jones)
    E2fvec(jvec, fmat = fmat, mode = mode, out = fvec)
    if output_fft == False and modal:
        ifft2(out, out = out)
    elif output_fft == True and not modal:
        fft2(out, out = out)
    return out

def _field2modes(field, k0, betamax = BETAMAX):
    f = fft2(field)
    mask = eigenmask(f.shape[-2:], k0, betamax)
    f = f[mask]
    return np.moveaxis(f,-2,-1)

def field2modes(field, k0, betamax = BETAMAX):
    """Converts 2D field array to modes array.
    
    Parameters
    ----------
    field : ndarray or tuple of ndarrays
        Input field array (or tuple of input fields for each wavenumber). The 
        shape of the input arrays must be (...,4,:,:) representing.
    k0 : float or a sequence of floats
        Defines the wavenumber. Fol multi-wavelength data, this must be a
        sequence of wawenumbers
    betamax : float, optional
        The beta cutoff parameter.
        
    Returns
    -------
    mask, modes : ndarray, ndarray or ndarray, tuple of ndarrays
        For a single-wavelength data it returns the mask array specifying
        the mode indices and modes coefficients array. For multi-wavelength data
        the modes is a tuple of ndarrays for each of the wavelengths.
            
    """
    if isinstance(field, tuple):
        out = tuple((field2modes(field[i], k0[i], betamax ) for i in range(len(field))))
        mask = tuple(o[0] for o in out)
        modes = tuple(o[1] for o in out)
        return mask, modes
    
    f = fft2(field)
    k0 = np.asarray(k0)
    mask = eigenmask(f.shape[-2:], k0, betamax)
    if k0.ndim == 0:
        return mask, np.moveaxis(f[...,mask],-2,-1)
    else:
        return mask, tuple((np.moveaxis(f[...,i,:,:,:][...,mask[i]],-2,-1) for i in range(len(k0))))

def ffield2modes(ffield, k0, betamax = BETAMAX):

    k0 = np.asarray(k0)
    mask = eigenmask(ffield.shape[-2:], k0, betamax)
    if k0.ndim == 0:
        return mask, np.moveaxis(ffield[...,mask],-2,-1)
    else:
        return mask, tuple((np.moveaxis(ffield[...,i,:,:,:][...,mask[i]],-2,-1) for i in range(len(k0))))
    

def field2modes1(field, k0, betamax = BETAMAX):
    """Converts field array to modes array.
    
    Parameters
    ----------
    field : ndarray or tuple of ndarrays
        Input field array (or tuple of input fields for each wavenumber). The 
        shape of the input arrays must be (...,4,:) representing.
    k0 : float or a sequence of floats
        Defines the wavenumber. Fol multi-wavelength data, this must be a
        sequence of wawenumbers
    betamax : float, optional
        The beta cutoff parameter.
        
    Returns
    -------
    mask, modes : ndarray, ndarray or ndarray, tuple of ndarrays
        For a single-wavelength data it returns the mask array specifying
        the mode indices and modes coefficients array. For multi-wavelength data
        the modes is a tuple of ndarrays for each of the wavelengths.
            
    """
    if isinstance(field, tuple):
        out = tuple((field2modes1(field[i], k0[i], betamax ) for i in range(len(field))))
        mask = tuple(o[0] for o in out)
        modes = tuple(o[1] for o in out)
        return mask, modes
    
    f = fft(field)
    k0 = np.asarray(k0)
    mask = eigenmask1(f.shape[-1], k0, betamax)
    if k0.ndim == 0:
        return mask, np.moveaxis(f[...,mask],-2,-1)
    else:
        return mask, tuple((np.moveaxis(f[...,i,:,:][...,mask[i]],-2,-1) for i in range(len(k0))))
    
    
def modes2ffield(mask, modes, out = None):
    """Inverse of ffield2modes. Takes the output of ffield2modes and recunstructs
    the field array.
    
    Parameters
    ----------
    mask : ndarray
        Mask array, as returned by field2modes
    modes : ndarray or tuple of ndarrays
        Modes array or a tuple of modes array (in case of multi-wavelength data).
    """
    if isinstance(mask, tuple):
        if out is None:
            out = (None,)* len(mask)
        return tuple((modes2ffield(mask[i], modes[i], out = out[i]) for i in range(len(mask))))

    shape = mask.shape[-2:]
    if mask.ndim == 2:
        shape = modes.shape[:-2] + (4,) + shape
        if out is None:
            out = np.zeros(shape =shape, dtype = CDTYPE )
        else:
            out[...] = 0.
        modes = np.moveaxis(modes,-1,-2)
        out[...,mask] = modes
    else:
        shape = modes[0].shape[:-2] + (len(mask),4,) + shape
        if out is None:
            out = np.zeros(shape =shape, dtype = CDTYPE )
        else:
            out[...] = 0.
        for i,(mode, m) in enumerate(zip(modes,mask)):
            mode = np.moveaxis(mode,-1,-2)
            o = out[...,i,:,:,:]
            o[...,m] = mode
            #print(o[...,m].shape)
            #print(mode.shape)
       
    return out

def modes2field(mask, modes, out = None):
    """Inverse of field2modes. Takes the output of field2modes and recunstructs
    the field array.
    
    Parameters
    ----------
    mask : ndarray
        Mask array, as returned by field2modes
    modes : ndarray or tuple of ndarrays
        Modes array or a tuple of modes array (in case of multi-wavelength data).
    """
    out = modes2ffield(mask, modes, out = out)
    return ifft2(out, out)

def modes2field1(mask, modes):
    """Inverse of field2modes. Takes the output of field2modes and recunstructs
    the field array.
    
    Parameters
    ----------
    mask : ndarray
        Mask array, as returned by field2modes
    modes : ndarray or tuple of ndarrays
        Modes array or a tuple of modes array (in case of multi-wavelength data).
    """
    if isinstance(mask, tuple):
        return tuple((modes2field1(mask[i], modes[i]) for i in range(len(mask))))

    shape = mask.shape[-1:]
    if mask.ndim == 1:
        shape = modes.shape[:-2] + (4,) + shape
        out = np.zeros(shape =shape, dtype = CDTYPE )

        modes = np.moveaxis(modes,-1,-2)
        out[...,mask] = modes
        return ifft(out, overwrite_x = True)
    else:
        shape = modes[0].shape[:-2] + (len(mask),4,) + shape
        out = np.zeros(shape =shape, dtype = CDTYPE )

        for i,(mode, m) in enumerate(zip(modes,mask)):
            mode = np.moveaxis(mode,-1,-2)
            o = out[...,i,:,:]
            o[...,m] = mode
            #print(o[...,m].shape)
            #print(mode.shape)
       
        return ifft(out, overwrite_x = True)

def aperture2rays(diaphragm, betastep = 0.1, norm = True):
    """Takes a 2D image of a diaphragm and converts it to beta, phi, intensity"""
    shape = diaphragm.shape

    ay, ax = [np.arange(-l / 2. + .5, l / 2. + .5) for l in shape]
    xx, yy = np.meshgrid(ax, ay, indexing = "xy")
    phi = np.arctan2(yy,xx)
    beta = np.sqrt(xx**2 + yy**2)*betastep
    mask = diaphragm > 0.
    intensity = diaphragm[mask]
    if norm == True:
        intensity = intensity/ intensity.sum() * len(intensity)
    return np.asarray(beta[mask],FDTYPE), np.asarray(phi[mask],FDTYPE), np.asarray(intensity,FDTYPE)

def illumination_aperture(diameter = 5., smooth = 0.1):
    n = int(round(diameter))
    alpha = min(max(smooth,0),1)
    a = aperture((n,n), diameter/n, alpha)
    return a
  
def illumination_rays(NA, diameter = 5., smooth = 0.1):
    """Returns beta, phi, intensity values for illumination.
    
    This function can be used to define beta,phi,intensiy arrays that can be used to
    construct illumination data with the :func:`illumination_data` function.
    The resulting beta,phi parameters define directions of rays for the input 
    light with a homogeneous angular distribution of rays - input
    light with a given numerical aperture.
    
    Parameters
    ----------
    NA : float
        Approximate numerical aperture of the illumination.
    diameter : int
        Field aperture diaphragm diameter in pixels. Approximate number of rays
        is np.pi*(diameter/2)**2
    smooth : float, optional
        Smoothness of diaphragm edge between 0. and 1.
        
    Returns
    -------
    beta,phi, intensity : ndarrays
        Ray parameters  
    """
    betastep = 2.*NA/(diameter-1)
    a = illumination_aperture(diameter, smooth)
    return aperture2rays(a, betastep = betastep, norm = True)


def illumination_eigenrays(shape, k0, NA = 0.2, scale = (1,1)):
    scale = np.asarray(scale, dtype = int)
    if scale.ndim == 0:
        scale = (scale, scale)
    elif scale.ndim != 1 or len(scale) != 2:
        raise ValueError("Unknown scale parameter!")
    b,p = betaphi(shape,k0)
    bs = b[...,::scale[0],::scale[1]]
    ps = p[...,::scale[0],::scale[1]]
    
    mask = bs <= NA
    
    if bs.ndim == 3:
        bs = tuple((b[m] for b,m in zip(bs,mask)))
        ps = tuple((ps[m] for m in mask))
        intensity = tuple((1./m.sum() for m in mask))
        return bs, ps, intensity
    else:
        return bs[mask], ps[mask], 1./mask.sum()


#def illumination_betaphi(NA, nrays = 13):
#    """Returns beta, phi values for illumination.
#    
#    This function can be used to define beta and phi arrays that can be used to
#    construct illumination data with the :func:`illumination_data` function.
#    The resulting beta,phi parameters define directions of rays for the input 
#    light with a homogeneous angular distribution of rays - input
#    light with a given numerical aperture.
#    
#    Parameters
#    ----------
#    NA : float
#        Approximate numerical aperture of the illumination.
#    nrays : int, optional
#        Approximate number of rays. 
#        
#    Returns
#    -------
#    array, array
#        beta, phi arrays 
#        
#    """
#    radius = (nrays/np.pi)**0.5
#    shape = (1+2*int(radius),1+2*int(radius))
#    ay, ax = [np.arange(-l // 2 + 1., l // 2 + 1.) for l in shape]
#    xx, yy = np.meshgrid(ax, ay,copy = False, indexing = "xy") 
#    phi = np.arctan2(yy,xx)
#    beta = np.sqrt(xx**2 + yy**2)/radius*NA
#    mask = (beta <= NA)
#    return np.asarray(beta[mask],FDTYPE), np.asarray(phi[mask],FDTYPE)

def illumination_waves(shape, k0, beta = 0., phi = 0., window = None, out = None):
    """Builds scalar illumination wave. 
    """
    k0 = np.asarray(k0)
    beta = np.asarray(beta)
    phi = np.asarray(phi)
    if beta.ndim < 1 :#and k0.ndim == 1:
        beta = beta[...,np.newaxis]
    if phi.ndim < 1 :#and k0.ndim == 1:
        phi = phi[...,np.newaxis]        
    if not k0.ndim in (0,1):
        raise ValueError("k0, must be an array with dimesion 1")
    out = planewave(shape, k0, beta, phi, out)
    if window is None:
        return out
    else:
        return np.multiply(out, window, out = out)

def illumination_eigenwaves(shape, k0, beta = 0., phi = 0., out = None):
    """Builds scalar illumination wave. 
    """
    k0 = np.asarray(k0)
    beta = np.asarray(beta)
    phi = np.asarray(phi)
    if beta.ndim < 1 :#and k0.ndim == 1:
        beta = beta[...,np.newaxis]
    if phi.ndim < 1 :#and k0.ndim == 1:
        phi = phi[...,np.newaxis]        
    if not k0.ndim in (0,1):
        raise ValueError("k0, must be an array with dimesion 1")
    out = planewave(shape, k0, beta, phi, out)


def field2betaphi(field_in,ks, multiray = False):
    beta, phi = mean_betaphi(field_in, ks)
    if field_in.ndim > 4 and multiray == True:  #must have at least two polarization states or multi-ray input
        beta = beta.mean(axis = tuple(range(1,field_in.ndim-3))) #average all, but first (multu-ray) axis
        phi = phi.mean(axis = tuple(range(1,field_in.ndim-3)))
    else:
        beta = beta.mean() #mean over all axes - single ray case
        phi = phi.mean()
    return beta, phi

    
def waves2field(waves, k0, beta = 0., phi = 0., n = 1., focus = 0., 
                jones = None, intensity = None, mode = "t", diffraction = True, betamax = BETAMAX):
    """Converts scalar waves to vector field data."""
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
    if diffraction == True:       
        diffracted_field(fieldv,k0, d = -focus, n = n, mode = mode, betamax = betamax, out = fieldv)
    
    #normalize field to these intensities
    if intensity is not None:
        intensity = intensity/nrays
        if jones is None:
            
            #-2 element must be polarization. make it broadcastable to field intensity
            intensity = intensity[...,None,:]
        norm = (intensity/(field2intensity(fieldv).sum((-2,-1))))**0.5
        np.multiply(fieldv, norm[...,None,None,None], fieldv)
        
    return fieldv


def waves2field2(waves, fmat, jones = None, phi = 0,mode = +1):
    """Converts scalar waves to vector field data."""
    if jones is None:
        fvec1 = field4(fmat, jones = jonesvec((1,0),phi), amplitude = waves, mode = mode)
        fvec2 = field4(fmat, jones = jonesvec((0,1),phi), amplitude = waves, mode = mode)
        field1 = fvec2field(fvec1)
        field2 = fvec2field(fvec2)
        
        shape = list(field1.shape)
        shape.insert(-4, 2)
        out = np.empty(shape = shape, dtype = field1.dtype)
        out[...,0,:,:,:,:] = field1
        out[...,1,:,:,:,:] = field2
    else:
        fvec = field4(fmat, jones = jonesvec(jones,phi), amplitude = waves, mode = mode)
        out = fvec2field(fvec).copy()
    return out



def illumination_data(shape, wavelengths, pixelsize = 1., beta = 0., phi = 0., intensity = 1.,
                      n = 1., focus = 0., window = None, backdir = False, 
                      jones = None, diffraction = True, eigenmodes = None, betamax = BETAMAX):
    """Constructs forward (or backward) propagating input illumination field data.
    
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
    focus : float, optional
        Focal plane of the field. By default it is set at z=0. 
    window : array or None, optional
        If None, no window function is applied. This window function
        is multiplied with the constructed plane waves to define field diafragm
        of the input light. See :func:`.window.aperture`.
    backdir : bool, optional
        Whether field is bacward propagating, instead of being forward
        propagating (default)
    jones : jones vector or None, optional
        If specified it has to be a valid jones vector that defines polarization
        of the light. If not given (default), the resulting field will have two
        polarization components. See documentation for details and examples.
    diffraction : bool, optional
        Specifies whether field is diffraction limited or not. By default, the 
        field is filtered so that it has only propagating waves. You can disable
        this by specifying diffraction = False.    
    eigenmodes : bool or None
        If set to True (sefault value when `window` = None.), it defines whether 
        to build eigenmodes from beta and phi values. In this case, beta and phi 
        are only approximate values. If set to False (default when `window` != None), 
        true beta and phi values are set. 
    betamax : float, optional
        The betamax parameter of the propagating field.

        
    Returns
    -------
    illumination : field_data
        A field data tuple.       
    """
    
    verbose_level = DTMMConfig.verbose
    if verbose_level > 0:
        print("Building illumination data.") 
    
    eigenmodes = (window is None) if eigenmodes is None else bool(eigenmodes)
    
    wavelengths = np.asarray(wavelengths)
    wavenumbers = 2*np.pi/wavelengths * pixelsize
    if wavenumbers.ndim not in (1,):
        raise ValueError("Wavelengths should be 1D array")
        
    if jones is None:
        intensity = intensity/2.
     
    #intensity = ((np.abs(waves)**2).sum((-2,-1)))* np.asarray(intensity)[...,None]#sum over pixels
    #intensity = intensity * intensity
    mode = -1 if backdir else +1
    _beta = np.asarray(beta, FDTYPE)
    _phi = np.asarray(phi, FDTYPE)
    _intensity = np.asarray(intensity, FDTYPE)

    nrays = len(_beta) if _beta.ndim > 0 else 1
    
    if _beta.ndim == 1:
        beta = _beta[:,None]
    else:
        beta = _beta
    if _phi.ndim == 1:
        phi = _phi[:,None]
    else:
        phi = _phi
        
    if verbose_level > 1:
        print("------------------------------------")
        print(" $ shape: {}".format(shape)) 
        print(" $ num wavelengths: {}".format(len(wavelengths)))  
        print(" $ num rays: {}".format(nrays))  
        print(" $ refractive index: {}".format(n)) 
        print(" $ diffraction limited: {}".format(diffraction))  
        print(" $ using eigenmodes: {}".format(eigenmodes))     
        print(" $ max beta: {}".format(betamax))   
        print(" $ jones: {}".format( jones if jones is not None else "unpolarized")) 
        print("------------------------------------")
        
    waves = illumination_waves(shape, wavenumbers, beta = beta, phi = phi, window = window)
 
    if window is None:
        waves = wave2eigenwave(waves)
     
    if _beta.ndim == 1:
        beta = _beta[:,None,None,None]
    else:
        beta = _beta[...,None,None]
    if _phi.ndim == 1:
        phi = _phi[:,None,None,None] 
    else:
        phi = _phi[...,None,None] 
    intensity = _intensity[...,None,None,None,None]   
    
    epsa = np.asarray((0.,0.,0.),FDTYPE)
    alpha, fmat = alphaf(beta, phi, refind2eps([n]*3), epsa)
    field = waves2field2(waves, fmat, jones = jones, phi = phi, mode = mode)
    intensity1 = field2intensity(field)
    norm = np.ones_like(intensity1)
    norm[:,...] = (intensity/nrays)**0.5    
    if diffraction == True:  
        diffracted_field(field,wavenumbers, d = -focus, n = n, mode = mode, betamax = betamax, out = field)
        intensity2 = field2intensity(field)
        ratio = (intensity1.sum((-2,-1))/intensity2.sum((-2,-1)))**0.5
        norm[...] = norm * ratio[...,None,None]
            
    np.multiply(norm[...,None,:,:], field, field)
    
    return (field, wavelengths, pixelsize)

#def illumination_dataold(shape, wavelengths, pixelsize = 1., beta = 0., phi = 0., intensity = 1.,
#                      n = 1., focus = 0., window = None, backdir = False, 
#                      jones = None, diffraction = True, betamax = BETAMAX):
#    """Constructs forward (or backward) propagating input illumination field data.
#    
#    Parameters
#    ----------
#    shape : (int,int)
#        Shape of the illumination
#    wavelengths : array_like
#        A list of wavelengths.
#    pixelsize : float, optional
#        Size of the pixel in nm.
#    beta : float or array_like of floats, optional
#        Beta parameter(s) of the illumination. (Default 0. - normal incidence) 
#    phi : float or array_like of floats, optional
#        Azimuthal angle(s) of the illumination. 
#    n : float, optional
#        Refractive index of the media that this illumination field is assumed to
#        be propagating in (default 1.)
#    focus : float, optional
#        Focal plane of the field. By default it is set at z=0. 
#    window : array or None, optional
#        If None, no window function is applied. This window function
#        is multiplied with the constructed plane waves to define field diafragm
#        of the input light. See :func:`.window.aperture`.
#    backdir : bool, optional
#        Whether field is bacward propagating, instead of being forward
#        propagating (default)
#    jones : jones vector or None, optional
#        If specified it has to be a valid jones vector that defines polarization
#        of the light. If not given (default), the resulting field will have two
#        polarization components. See documentation for details and examples.
#    diffraction : bool, optional
#        Specifies whether field is diffraction limited or not. By default, the 
#        field is filtered so that it has only propagating waves. You can disable
#        this by specifying diffraction = False.    
#    betamax : float, optional
#        The betamax parameter of the propagating field.
#    """
#    
#    verbose_level = DTMMConfig.verbose
#    if verbose_level > 0:
#        print("Building illumination data.") 
#    wavelengths = np.asarray(wavelengths)
#    wavenumbers = 2*np.pi/wavelengths * pixelsize
#    if wavenumbers.ndim not in (1,):
#        raise ValueError("Wavelengths should be 1D array")
#    waves = illumination_waves(shape, wavenumbers, beta = beta, phi = phi, window = window)
#    intensity = ((np.abs(waves)**2).sum((-2,-1)))* np.asarray(intensity)[...,None]#sum over pixels
#    mode = "r" if backdir else "t"
#    field = waves2field(waves, wavenumbers, intensity = intensity, beta = beta, phi = phi, n = n,
#                        focus = focus, jones = jones, mode = mode, betamax = betamax)
#    return (field, wavelengths, pixelsize)


@nb.njit([(NCDTYPE[:,:,:],NFDTYPE[:,:])], cache = NUMBA_CACHE)
def _field2intensity(field, out):
    for j in range(field.shape[1]):
        for k in range(field.shape[2]):
            tmp1 = (field[0,j,k].real * field[1,j,k].real + field[0,j,k].imag * field[1,j,k].imag)
            tmp2 = (field[2,j,k].real * field[3,j,k].real + field[2,j,k].imag * field[3,j,k].imag)
            out[j,k] = tmp1-tmp2 

@nb.guvectorize([(NCDTYPE[:,:,:],NFDTYPE[:,:])],"(k,n,m)->(n,m)", target = NUMBA_TARGET, cache = NUMBA_CACHE)
def field2intensity(field, out):
    """field2intensity(field)
    
Converts field array of shape [...,4,height,width] to intensity array
of shape [...,height,width]. For each pixel element, a normal
component of the Poynting vector is computed.
    
Parameters
----------
field : array_like
    Input field array
cmf : array_like
    Color matching function
    
Returns
-------
spec : ndarray
    Computed intensity array"""  
    assert len(field) == 4
    _field2intensity(field, out)

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
    """field2specter(field)
    
Converts field array of shape [...,nwavelengths,4,height,width] to specter array
of shape [...,height,width,nwavelengths]. For each pixel element, a normal
componentof Poynting vector is computed
    
Parameters
----------
field : array_like
    Input field array
cmf : array_like
    Color matching function
    
Returns
-------
spec : ndarray
    Computed specter array""" 
    _field2specter(field, out)

@nb.guvectorize([(NCDTYPE[:,:,:,:,:],NFDTYPE[:,:,:])],"(l,w,k,n,m)->(n,m,w)", target = "cpu", cache = NUMBA_CACHE)
def field2spectersum(field, out):
    _field2spectersum(field, out)  
    
    
@nb.guvectorize([(NCDTYPE[:,:,:],NFDTYPE[:,:],NFDTYPE[:,:],NFDTYPE[:],NFDTYPE[:])], "(k,n,m),(n,m),(n,m)->(),()", target = NUMBA_TARGET, cache = NUMBA_CACHE)
def _fft_betaphi(f, betax, betay, beta, phi):

    _betax = 0.
    _betay = 0.
    _ssum = 0.
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            for k in range(f.shape[2]):
                s = f[i,j,k].real**2 + f[i,j,k].imag**2
                _betax += s*betax[j,k]
                _betay += s*betay[j,k]
                _ssum += s
    #save results to output
    beta[0] = (_betax**2+_betay**2)**0.5 / _ssum
    phi[0] = np.arctan2(_betay,_betax)

def mean_betaphi(field,k0):
    """Calculates mean beta and phi of a given field."""
    b = blackman(field.shape[-2:])
    f = fft2(field*b) #filter it with blackman..
    betax, betay = betaxy(field.shape[-2:], k0)
    beta, phi = _fft_betaphi(f,betax,betay)
    return beta, phi

def validate_field_data(data):
    """Validates field data.
    
    This function inspects validity of the field data, and makes proper data
    conversions to match the field data format. In case data is not valid and 
    it cannot be converted to a valid data it raises an exception (ValueError). 
    
    Parameters
    ----------
    data : tuple of field data
        A valid field data tuple.
    
    Returns
    -------
    data : tuple
        Validated field data tuple. 
    """
    field, wavelengths, pixelsize = data
    field = np.asarray(field, dtype = CDTYPE)
    wavelengths = np.asarray(wavelengths, dtype = FDTYPE)
    pixelsize = float(pixelsize)
    if field.ndim < 4:
        raise ValueError("Invald field dimensions")
    if field.shape[-4] != len(wavelengths) or wavelengths.ndim != 1:
        raise ValueError("Incompatible wavelengths shape.")

    return field, wavelengths, pixelsize


MAGIC = b"dtmf" #legth 4 magic number for file ID
VERSION = b"\x00"

def save_field(file, field_data):
    """Saves field data to a binary file in ``.dtmf`` format.
    
    Parameters
    ----------
    file : file, str, or pathlib.Path
        File or filename to which the data is saved.  If file is a file-object,
        then the filename is unchanged.  If file is a string, a ``.dtmf``
        extension will be appended to the file name if it does not already
        have one.
    field_data: (field,wavelengths,pixelsize)
        A valid field data tuple
    
    """    
    own_fid = False
    field, wavelengths,pixelsize = validate_field_data(field_data)
    try:
        if isinstance(file, str):
            if not file.endswith('.dtmf'):
                file = file + '.dtmf'
            f = open(file, "wb")
            own_fid = True
        else:
            f = file
        f.write(MAGIC)
        f.write(VERSION)
        np.save(f,field)
        np.save(f,wavelengths)
        np.save(f,pixelsize)
    finally:
        if own_fid == True:
            f.close()


def load_field(file):
    """Load field data from file
    
    Parameters
    ----------
    file : file, str
        The file or filenam to read.
    """
    own_fid = False
    try:
        if isinstance(file, str):
            f = open(file, "rb")
            own_fid = True
        else:
            f = file
        magic = f.read(len(MAGIC))
        if magic == MAGIC:
            if f.read(1) != VERSION:
                raise OSError("This file was created with a more recent version of dtmm. Please upgrade your dtmm package!".format(file))
            field = np.load(f)
            wavelengths = np.load(f)
            pixelsize = float(np.load(f))
            return field, wavelengths, pixelsize
        else:
            raise OSError("Failed to interpret file {}".format(file))
    finally:
        if own_fid == True:
            f.close()
field2poynting = field2intensity
    
__all__ = ["illumination_rays","load_field", "save_field", "validate_field_data","field2specter","field2intensity", "illumination_data"]