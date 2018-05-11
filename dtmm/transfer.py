"""
Main calculation functions for light propagation through optical data.

"""
from __future__ import absolute_import, print_function, division

from dtmm.conf import DTMMConfig, cached_function, BETAMAX
from dtmm.wave import k0

from dtmm.data import uniaxial_order, refind2eps, validate_optical_data
from dtmm.tmm import alphaffi_xy
from dtmm.linalg import dotmdm, dotmm, transmit
from dtmm.print_tools import print_progress
from dtmm.diffract import projection_matrix, diffract, phase_matrix, diffraction_matrix
from dtmm.field import field2intensity
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
    alpha, f, fi = alphaffi_xy(beta,phi + np.pi,epsa,epsv)  
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
        Beta parameter of the input field. If it is a 1D array, then input field
        data first axis is taken to be field data at beta value.
    phi : float or 1D array_like of floats
        Phi angle of the input light field. If it is a 1D array, then input field
        data first axis is taken to be field data at beta value.
    eff_data : Optical data tuple or None
        Optical data tuple of homogeneous layers through which light is diffracted
        in the diffraction calculation. If not provided an effective data is
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
        Specifies layer propagation computation steps (defaults to 1).
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
    
    shape = field_in.shape[-2:]
    
    dmat_in = projection_matrix(shape, ks,refind2eps([nin]*3), mode = "r", betamax = betamax)
    dmat_out = projection_matrix(shape, ks,refind2eps([nout]*3), mode = "t", betamax = betamax)
        
    if npass > 1:
        field0 = field_in.copy()
        
    field = field_in
    out = field_out
    
    verbose_level = DTMMConfig.verbose
    
    for i in range(npass):
        intensity_in = field2intensity(field)

        msg = "{}/{}".format(i+1,npass)
        for j in range(n):
            print_progress(j,n,level = verbose_level, suffix = msg) 
            thickness = d[j]*(-1)**i
            thickness_eff = d_eff[j]*(-1)**i
            if thickness < 0:
                p = phi 
            else:
                p = phi + np.pi #not sure why I need to do this... but this makes it work correclty for interference calculations
            field = propagate_field(field, ks, (thickness, epsv[j], epsa[j]),(thickness_eff, epsv_eff[j], epsa_eff[j]), 
                            beta = beta, phi = p, nsteps = substeps[j], diffraction = diffraction,
                            betamax = betamax, out = out)
        #intensity_out = field2specter(field).sum(axis = tuple(range(field.ndim-2)))
        intensity_out = field2intensity(field)
      
        print_progress(n,n,level = verbose_level, suffix = msg) 
        if npass > 1:
            if i%2 == 0:
                if i != npass -1:
                    field = diffract(field,dmat_out, window = window, out = field_out)
                    #intensity_refl = field2specter(field).sum(axis = tuple(range(field.ndim-2)))
                    intensity_refl = field2intensity(field)
                    m = intensity_refl == 0.
                    intensity_refl[m] = 1.
                    intensity_out[m] = 1.


                    fact = (intensity_out/intensity_refl)[...,None,:,:]
                    fact[fact<0.] = 0.
                    fact[fact>1] = 1.
                    fact = np.abs(fact)
                    #fact = fact + 0.1*np.random.randn(*fact.shape) + 0.1*np.random.randn(*fact.shape) *1j

                    #fact[...] = 0.4
    
     
                    field = np.multiply(field,fact, out = field_in)
                    
                    out = field_in
                    
            else:
                if i != npass -1:
                    field = np.subtract(field,field0,out = field)
                    intensity_in = field2intensity(field)
                    
                    field = diffract(field,dmat_in, window = window, out = field_in)
                    intensity_refl = field2intensity(field)

                    m = intensity_refl == 0.
                    intensity_refl[m] = 1.
                    intensity_in[m] = 1.
                    
                    out = field_in
                    fact = (intensity_in/intensity_refl)[...,None,:,:]
                    
                    fact[fact<0] = 0
                    fact[fact>1] = 1.
                    fact = np.abs(fact)
                    #fact = fact + np.random.randn(*fact.shape) + np.random.randn(*fact.shape) *1j

                    #fact[...] = 0.4

                    field = np.multiply(field,fact, out = field)
                    
                    field = np.add(field0, field, out = field_in)
                    field0 = field_in.copy()  
                    out = field_out  

    return field_out, wavelengths, pixelsize


def propagate_field(field, wavenumbers, layer_data, effective_data, beta = 0, phi=0,
                    nsteps = 1, diffraction = True, 
                    betamax = BETAMAX, out = None):
    
    shape = field.shape[-2:]
    d, epsv, epsa = layer_data
    d_eff, epsv_eff, epsa_eff = effective_data
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

    
__all__ = ["transfer_field"]
