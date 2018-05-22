"""
Main top level calculation functions for light propagation through optical data.
"""
from __future__ import absolute_import, print_function, division

from dtmm.conf import DTMMConfig, cached_function, BETAMAX
from dtmm.wave import k0

from dtmm.data import uniaxial_order, refind2eps, validate_optical_data
from dtmm.tmm import alphaffi_xy
from dtmm.linalg import dotmdm, dotmm, transmit, dotmf
from dtmm.print_tools import print_progress
from dtmm.diffract import projection_matrix, diffract, phase_matrix, diffraction_matrix
from dtmm.field import field2intensity
from dtmm.fft import fft2, ifft2
import numpy as np


def _isotropic_effective_data(data):
    d, material, angles = data
    n = len(d)
    epseff = uniaxial_order(0.,material).mean(axis = (0,1,2))
    epseff = np.broadcast_to(epseff,(n,3)).copy()#better to make copy.. to make it c contiguous
    aeff = np.array((0.,0.,0.))
    aeff = np.broadcast_to(aeff,(n,3)).copy()#better to make copy.. to make it c contiguous
    return validate_optical_data((d,epseff,aeff), homogeneous = True)

def _validate_betaphi(beta,phi, extendeddim = 0):
    beta = np.asarray(beta)
    phi = np.asarray(phi)  
    
    if beta.ndim != phi.ndim:
        raise ValueError("Beta nad phi should have same dimensions!")
    
    if beta.ndim == 1:
        if len(beta) != len(phi):
            raise ValueError("Beta nad phi should have same length!")
        #make arrays broadcastable to field by adding extra dimensions
        for i in range(extendeddim):
            beta = beta[...,None]
            phi = phi[...,None]
    elif beta.ndim != 0:
        raise ValueError("Only length 1 or scalar values are supported for beta and phi")
    return beta, phi

@cached_function
def correction_matrix(beta,phi,ks, d=1., epsv = (1,1,1), epsa = (0,0,0.), out = None):
    alpha, f, fi = alphaffi_xy(beta,phi + 0*np.pi,epsa,epsv)  
    kd = -np.asarray(ks)*d
    pmat = phase_matrix(alpha, kd)  
    return dotmdm(f,pmat,fi, out = out)

@cached_function
def corrected_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
    dmat = diffraction_matrix(shape, ks, d, epsv, epsa, betamax = betamax)
    cmat = correction_matrix(beta, phi, ks, d, epsv, epsa)
    if d > 0:
        return dotmm(dmat,cmat, out = None)
    else:
        return dotmm(cmat, dmat, out = None)
    
def _normalize_fft_field(field, intensity_in, intensity_out, out = None):
    m = intensity_out == 0.
    intensity_out[m] = 1.
    intensity_in[m] = 0.
    fact = (intensity_in/intensity_out)[...,None,:,:]
    fact[fact<=-1] = -1
    fact[fact>=1] = 1
    fact = np.abs(fact)
    return np.multiply(field,fact, out = out) 

def diffract_normalized(field, dmat, window = None, out = None):
    f1 = fft2(field, out = out)
    intensity1 = field2intensity(f1)
    f2 = dotmf(dmat, f1 ,out = f1)
    intensity2 = field2intensity(f2)
    f3 = _normalize_fft_field(f2, intensity1, intensity2, out = f2)
    out = ifft2(f3, out = out)
    if window is not None:
        out = np.multiply(out,window,out = out)
    return out   
    
def _projected_field(field, wavenumbers, mode, n = 1, betamax = BETAMAX, norm = False, out = None):
    eps = refind2eps([n]*3)
    pmat = projection_matrix(field.shape[-2:], wavenumbers, epsv = eps, epsa = (0.,0.,0.), mode = mode, betamax = betamax)

    if norm == False:
        return diffract(field, pmat, out = out) 
    else:
        return diffract_normalized(field, pmat, out = out)

def transmitted_field(field, wavenumbers, n = 1, betamax = BETAMAX, norm = False, out = None):
    """Computes transmitted (forward propagating) part of the field.
    
    Parameters
    ----------
    field : ndarray
        Input field array
    wavenumbers : array_like
        Wavenumbers of the field
    n : float, optional
        Refractive index of the media (1 by default)
    betamax : float
        Betamax perameter used.
    norm : bool, optional
        Whether to normalize field so that power spectrum of the output field
        remains the same as that of the input field.
    out : ndarray, optinal
        Output array
        
    Returns
    -------
    out : ndarray
       Transmitted field.
    """
    return _projected_field(np.asarray(field), wavenumbers, "t", n = n, betamax = betamax, norm = norm, out = out) 
    
def reflected_field(field, wavenumbers, n = 1, betamax = BETAMAX, norm = False, out = None):
    """Computes reflected (backward propagating) part of the field.
    
    Parameters
    ----------
    field : ndarray
        Input field array
    wavenumbers : array_like
        Wavenumbers of the field
    n : float, optional
        Refractive index of the media (1 by default)
    betamax : float
        Betamax perameter used.
    norm : bool, optional
        Whether to normalize field so that power spectrum of the output field
        remains the same as that of the input field.
    out : ndarray, optinal
        Output array
        
    Returns
    -------
    out : ndarray
       Reflected field.
    """
    return _projected_field(np.asarray(field), wavenumbers, "r", n = n, betamax = betamax, norm = norm, out = out) 
    

def transfer_field(field_data, optical_data, beta = 0., 
                   phi = 0., eff_data = None, nin = 1., nout = 1., npass = 1,nstep=1,
              diffraction = True,  window = None, betamax = BETAMAX):
    """Tranfers input field data through optical data.
    
    This function calculates transmitted field and possibly (when npass > 1) 
    updates input field to include reflected waves. 
    
    
    Parameters
    ----------
    field_data : Field data tuple
        Input field data tuple
    optical_data : Optical data tuple
        Optical data tuple through which input field is transmitted.
    beta : float or 1D array_like of floats
        Beta parameter of the input field. If it is a 1D array, beta[i] is the
        beta parameter of the field_data[0][i] field array.
    phi : float or 1D array_like of floats
        Phi angle of the input light field. If it is a 1D array, phi[i] is the
        phi parameter of the field_data[0][i] field array.
    eff_data : Optical data tuple or None
        Optical data tuple of homogeneous layers through which light is diffracted
        in the diffraction calculation. If not provided, an effective data is
        build from optical_data by taking an average isotropic refractive index
        of the material.
    nin : float, optional
        Refractive index of the input (bottom) surface (1. by default). Used
        in combination with npass > 1 to determine reflections from input layer.
    nout : float, optional
        Refractive index of the output (top) surface (1. by default). Used
        in combination with npass > 1 to determine reflections from output layer.
    npass: int, optional
        How many passes (iterations) to perform. For strongly reflecting elements
        this should be set to a higher value. If npass > 1, then input field data is
        overwritten and adds reflected light from the sample. (defaults to 1)
    nstep: int or 1D array_like of ints
        Specifies layer propagation computation steps (defaults to 1). For thick layers
        you may want to increase this number.
    diffraction : bool, optional
        Whether to perform difraction caclulation or not. Setting this to False 
        will dissable diffraction calculation (standard 4x4 method).
    window: array or None
        Additional window function that is multiplied after each layer propagation step.
        Computed field data is multiplied with this window after each layer.

    """
    #define optical data
    d, epsv, epsa = validate_optical_data(optical_data)
        
    #define effective optical data
    if eff_data is None:
        d_eff, epsv_eff, epsa_eff = _isotropic_effective_data((d, epsv, epsa))
    else:
        d_eff, epsv_eff, epsa_eff = validate_optical_data(eff_data, homogeneous = True)
        
    #define input field data
    field_in, wavelengths, pixelsize = field_data
    
    #define constants 
    ks = k0(wavelengths, pixelsize)
    n = len(d)
    substeps = np.broadcast_to(np.asarray(nstep),(n,))
    
    #define input ray directions. Either a scalar or 1D array
    beta, phi = _validate_betaphi(beta,phi,extendeddim = field_in.ndim-2)
    
    #define output field
    field_out = np.zeros_like(field_in)
    
    if npass > 1:
        field0 = field_in.copy()
        
    field = field_in
    out = field_out
    
    verbose_level = DTMMConfig.verbose
    indices = list(range(n))
    for i in range(npass):
        msg = "{}/{}".format(i+1,npass)
        for pindex, j in enumerate(indices):
            print_progress(pindex,n,level = verbose_level, suffix = msg) 
            thickness = d[j]*(-1)**i
            thickness_eff = d_eff[j]*(-1)**i
            p = phi + np.pi #not sure why I need to do this... but this makes it work for off axis propagation
            field = propagate_field(field, ks, (thickness, epsv[j], epsa[j]),(thickness_eff, epsv_eff[j], epsa_eff[j]), 
                            beta = beta, phi = p, nsteps = substeps[j], diffraction = diffraction,
                            betamax = betamax, out = out)
      
        print_progress(n,n,level = verbose_level, suffix = msg) 
        
        indices.reverse()
        if npass > 1:
            if i%2 == 0:
                if i != npass -1:
                    field = transmitted_field(field, ks, n = nout, betamax = betamax, norm = True, out = field_out)
                    out = field_in                    
            else:
                if i != npass -1:
                    field = np.subtract(field,field0,out = field)
                    field = reflected_field(field, ks, n = nin, betamax = betamax, norm = True, out = field_in)
                    field = np.add(field0, field, out = field_in)
                    field0 = field_in.copy()
                    out = field_out

    return field_out, wavelengths, pixelsize


def propagate_field(field, wavenumbers, layer, effective_layer, beta = 0, phi=0,
                    nsteps = 1, diffraction = True, 
                    betamax = BETAMAX, out = None):
    
    shape = field.shape[-2:]
    d, epsv, epsa = layer
    d_eff, epsv_eff, epsa_eff = effective_layer
    kd = wavenumbers*d/nsteps
    d_eff = d_eff/nsteps
    
    alpha, f, fi = alphaffi_xy(beta,phi,epsa,epsv)
    
    if diffraction == True:
        dmat = corrected_diffraction_matrix(shape, wavenumbers, beta,phi, d=d_eff,
                         epsv = epsv_eff, epsa = epsa_eff, betamax = betamax)

    for j in range(nsteps):
        if diffraction == True:
            if d > 0:
                field = diffract(field, dmat, out = out)
                field = transmit(f,alpha,fi, field, kd, out = field) 
            else:
                field = transmit(f,alpha,fi, field, kd, out = out)  
                field = diffract(field, dmat, out = field)
        else:
            field = transmit(f,alpha,fi, field, kd, out = out) 
        out = field
    return out

__all__ = ["transfer_field", "transmitted_field", "reflected_field"]
