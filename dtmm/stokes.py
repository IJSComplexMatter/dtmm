"""Stokes vector creation and conversion functions""" 

from dtmm.conf import FDTYPE, CDTYPE
import numpy as np
from dtmm.jones import jones_intensity

def validate_stokes(stokes):
    """Returns a validated stokes vector."""
    #stokesmust be a numpy array. If it is not, we try to convert it to numpy array
    stokes = np.asarray(stokes,FDTYPE)
    #stokes must at least 1D and the last axis must have length 4.
    if stokes.ndim < 1 or stokes.shape[-1] != 4:
        raise ValueError("Not a valid stokes array!")
    return stokes

def stokesvec(psi = 0., chi = 0., p = 1., intensity = 1., out = None):
    """Creates a stokes vector from the polarization ellipse parameters"""
    
    #determine output shape from input shapes
    psi = np.asarray(psi)
    chi = np.asarray(chi)
    p = np.asarray(p)
    intensity = np.asarray(intensity)
    array_shapes = (x.shape for x in (psi,chi,p,intensity))
    broadcast_shape = np.broadcast_shapes(*array_shapes)
    out_shape = broadcast_shape + (4,)
    
    if out is None:
        out = np.empty(shape = out_shape, dtype = FDTYPE)
    
    #tmp data
    a = intensity * p
    psi2 = 2 * psi
    chi2 = 2 * chi
    acoschi2 = a * np.cos(chi2)
    
    #fill out the components
    out[..., 0] = intensity
    out[..., 1] = acoschi2 * np.cos(psi2) 
    out[..., 2] = acoschi2 * np.sin(psi2)
    out[..., 3] = a * np.sin(chi2)
    
    return out

def stokes2p(stokes):
    """Returns the degree of polarization (the p parameter) from the stokes vector"""
    stokes = validate_stokes(stokes)  
    p = (stokes[...,1]**2 + stokes[...,2]**2 + stokes[...,3]**2)**0.5
    p/= stokes[...,0]
    return p
    
def stokes2psi(stokes):
    """Returns polarization elipse psi angle from the stokes vector of shape (...,4).
    """
    stokes = validate_stokes(stokes)
    return np.arctan2(stokes[...,2], stokes[...,1]) * 0.5

def stokes2chi(stokes):
    """Returns polarization elipse chi angle from the stokes vector of shape (...,4).
    See wikipedia for definition.
    """
    stokes = validate_stokes(stokes)
    return np.arctan2(stokes[...,3], np.sqrt(stokes[...,1]**2 + stokes[...,2]**2) )* 0.5

def jones2stokes(jvec, out = None):
    """Converts a jones vector to a valid stokes vector assuming perfect degree of polarization (p = 1)"""

    jvec = np.asarray(jvec)
    out_shape = jvec.shape[:-1] + (4,)
    
    if out is None:
        out = np.empty(shape = out_shape, dtype = FDTYPE)
        
    #tmp data
    jvecc = np.conj(jvec)
    jvec2 = (jvec * jvecc).real 
    ExEyc = 2 * jvec[...,0] * jvecc[...,1] 
    
    #fill out the components
    out[...,0] = jvec2[...,0] + jvec2[...,1]
    out[...,1] = jvec2[...,0] - jvec2[...,1]
    out[...,2] = ExEyc.real
    out[...,3] = -ExEyc.imag
    
    return out
    
def stokes2jones(stokes, out = None):
    """Converts a stokes vector to a valid jones vector.
    
    The polarization intensity of the stokes vector is set as the intensity of 
    the jones vector. The x component of the jones vector is assumed to be real 
    and positive."""
    
    stokes = validate_stokes(stokes)
    
    out_shape = stokes.shape[:-1] + (2,)
    
    Ip = (stokes[...,1]**2 +  stokes[...,2]**2 +  stokes[...,3]**2)**0.5
    
    a = ((Ip + stokes[...,1])/2.)**0.5
    b = ((Ip - stokes[...,1])/2.)**0.5
    
    phi = np.arctan2(stokes[...,3],stokes[...,2])
    
    if out is None:
        out = np.empty(shape = out_shape, dtype = CDTYPE)
        
    out[...,0] = a
    out[...,1] = b * np.exp(1j*phi)
    
    return out
    
    
    
    
    
    
        
