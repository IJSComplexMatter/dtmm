#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:28:25 2018

@author: andrej

"""
from __future__ import absolute_import, print_function, division

from dtmm.conf import FDTYPE,DTMMConfig, cached_function
from dtmm.wave import betaphi, k0
from dtmm.window import tukey
from dtmm.data import uniaxial_order, refind2eps, validate_optical_data
from dtmm.tmm import alphaffi_xy, phasem,phasem_r, phasem_t
from dtmm.linalg import ftransmit, dotmdm, dotmf, dotmm, transmit
from dtmm.print_tools import print_progress
from dtmm.fft import fft2, ifft2
import numpy as np


DIFRACTION_PARAMETERS = ("distance", "mode")#, "refind")

#def propagate_full(fieldv,ks,stack,eps, out = None):
#
#    fieldv = np.asarray(fieldv)
#    if out is None:
#        out = np.empty_like(fieldv)
#    else:
#        if not isinstance(out, np.ndarray) or out.shape != fieldv.shape or out.dtype != fieldv.dtype:
#            raise TypeError("Output array invalid!")
#           
#    ks = np.asarray(ks)
#    stack = np.asarray(stack)
#    shape = fieldv.shape[-2:]
#        
#    beta, phi = betaphi(shape, ks)
#    mask = (beta <= 1.)
#    
#    xindices, yindices = np.indices(shape)
#    xindices, yindices = xindices[mask], yindices[mask]
#
#    n = len(stack)
#    
#    nk = len(xindices)
#    
#    verbose_level = DTMMConfig.verbose
#    for i,layer in enumerate(stack):
#        print_progress(i,n,level = verbose_level)
#        ffield = fft2(fieldv)
#        fieldv[...]=0.
#        for j in range(nk):
#            ii,jj = xindices[j], yindices[j]
#            fieldv = eigenwave(shape, amplitude = ffield[ii,jj])
#            alpha, f, fi = alphaffi_xy(beta[ii,jj],phi[ii,jj],layer,eps)
#            fieldv += ftransmit(f,alpha,fi, fieldv, ks) 
#    print_progress(n,n,level = verbose_level)
#    return fieldv


#def propagate(fieldv,ks,stack,eps, beta0 = 0., phi0 = 0., eps0 = None, layer0 = None,
#              diffraction = True, mode = "t", out = None):
#
#    fieldv = np.asarray(fieldv)
#    if out is None:
#        out = np.empty_like(fieldv)
#    else:
#        if not isinstance(out, np.ndarray) or out.shape != fieldv.shape or out.dtype != fieldv.dtype:
#            raise TypeError("Output array invalid!")
#           
#    ks = np.asarray(ks)
#    stack = np.asarray(stack)
#    if eps0 is None:
#        eps0 = uniaxial_order(0.,eps)
#        eps0 = eps0.mean(axis = tuple(range(eps0.ndim-1)))
#    else:
#        eps0 = np.asarray(eps0)
#        assert eps0.ndim == 1
#    if layer0 is None:
#        layer0 = np.array((0.,0.,0.))
#    else:
#        layer0 = np.asarray(layer0)
#        assert layer0.ndim == 1
#    shape = fieldv.shape[-2:]
#        
#    dmat = diffraction_matrix(shape, ks, eps = eps0, layer = layer0, d = 1., mode = mode)
#    
#    #modify diffriction matrix to remove accumulated average phase shift
#    alpha, f, fi = alphaffi_xy(beta0,phi0,layer0,eps0)
#    if mode == "t":
#        pmat = phasem_t(alpha, -ks)   
#    else:
#        pmat = phasem(alpha, -ks) 
#    #phase back shift matrix
#    cmat = dotmdm(f,pmat,fi)
#    #compute corrected diffraction matrix
#    dmat = dotmm(dmat,cmat[:,None,None,:,:],dmat)
#    
#    n = len(stack)
#    
#    verbose_level = DTMMConfig.verbose
#    for i,layer in enumerate(stack):
#        print_progress(i,n,level = verbose_level)
#        alpha, f, fi = alphaffi_xy(beta0,phi0,layer,eps)
#        if diffraction:
#            fieldv = diffract(fieldv, dmat, out = out)
#        fieldv = ftransmit(f,alpha,fi, fieldv, ks, out = out) 
#    print_progress(n,n,level = verbose_level)
#    return fieldv
#


#def transmit_field(field_waves, data, beta0 = 0., 
#                   phi0 = 0., eps0 = None, layer0 = None, n_in = 1., n_out = 1., npass = 1,nsteps=1,
#              diffraction = True, mode = "b", window = None, out = None):
#    d, mask, material, stack = data
#    fieldv, wavenumbers = field_waves
#    stack = np.asarray(stack)
#    ks = np.asarray(wavenumbers)
#    n = len(stack)
#    
#    if mask is None:
#        mask = np.zeros(shape = (n,)+fieldv.shape[-2:], dtype = "uint8")
#    else:
#        mask = np.asarray(mask)
#    fieldv = np.asarray(fieldv)
#    if out is None:
#        out = np.empty_like(fieldv)
#    else:
#        if not isinstance(out, np.ndarray) or out.shape != fieldv.shape or out.dtype != fieldv.dtype:
#            raise TypeError("Output array invalid!")
#           
#    if eps0 is None:
#        eps0 = uniaxial_order(0.,material[0])
#        eps0 = eps0.mean(axis = tuple(range(eps0.ndim-1)))
#    else:
#        eps0 = np.asarray(eps0)
#        assert eps0.ndim == 1
#    if layer0 is None:
#        layer0 = np.array((0.,0.,0.))
#    else:
#        layer0 = np.asarray(layer0)
#        assert layer0.ndim == 1
#    shape = fieldv.shape[-2:]
#    
#    dmat_in = projection_matrix(shape, ks,refind2eps([n_in]*3), mode = "r")
#    dmat_out = projection_matrix(shape, -ks,refind2eps([n_out]*3), mode = "r")
#
#        
#    dmatf = diffraction_matrix(shape, ks, eps = eps0, layer = layer0, d = d[0]/nsteps, mode = mode)
#    dmatb = diffraction_matrix(shape, -ks, eps = eps0, layer = layer0, d = d[0]/nsteps, mode = mode)
#
#    #modify diffriction matrix to remove accumulated average phase shift
#    alpha, f, fi = alphaffi_xy(beta0,phi0,layer0,eps0)
#    kd = ks*d[0]/nsteps
#    if mode == "t":
#        pmatf = phasem_t(alpha, -kd)
#        pmatb = phasem_t(alpha, kd)
#        _t = ftransmit
#    else:
#        pmatf = phasem(alpha, -kd)
#        pmatb = phasem(alpha, kd)
#        _t  = transmit
#
#    #phase back shift matrix
#    cmatf = dotmdm(f,pmatf,fi)
#    cmatb = dotmdm(f,pmatb,fi)
#    #compute corrected diffraction matrix
#    #dmatf = dotmm(dmatf,cmatf[:,None,None,:,:],dmatf)
#    dmatf = dotmm(cmatf[:,None,None,:,:],dmatf,dmatf)
#    #dmatb = dotmm(dmatb,cmatb[:,None,None,:,:],dmatb)
#    dmatb = dotmm(cmatb[:,None,None,:,:],dmatb,dmatb)
#    
#    if npass > 1:
#        fieldv0 = fieldv.copy()
#    
#    verbose_level = DTMMConfig.verbose
#    
#    def propagate(infield, is_forward, out = None):
#        field = infield.copy()
#        indices = list(range(n))
#        if not is_forward:
#            indices.reverse()
#        for i in indices:
#            layer = stack[i]
#            print_progress(i,n,level = verbose_level)
#            alpha, f, fi = alphaffi_xy2(beta0,phi0,layer,material,mask[i])
#            for j in range(nsteps):
#                if is_forward:
#                    if diffraction:
#                        field = diffract(field, dmatf, out = out)
#                    field = _t(f,alpha,fi, field, kd, out = out) 
#                else:
#                    field = _t(f,alpha,fi, field, -kd, out = out) 
#                    if diffraction:
#                        field = diffract(field, dmatb, out = out)
#        print_progress(n,n,level = verbose_level) 
#        return out
#    
#    forward = True
#    for i in range(npass):
#        
#        if forward:
#            out = propagate(fieldv,forward, out = out)
#        else:
#            fieldv = propagate(fieldv,forward, out = fieldv)
#   
#        forward = not forward
#        if npass > 1:
#            if i%2 == 0:
#                assert out is not fieldv
#                if i != npass -1:
#                    diffract(out,dmat_out, window = window, out = fieldv)
#            else:
#                diffract(fieldv,dmat_in, window = window, out = fieldv)
#                np.subtract(fieldv0, fieldv, out = fieldv)
#                fieldv0 = fieldv.copy()
#    
#    return out, wavenumbers




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


def transmit_field_old(field_data, data, beta = 0., 
                   phi = 0., eff_data = None, eps0 = None, layer0 = None, nin = 1., nout = 1., npass = 1,substeps=1,
              diffraction = True, mode = "b", window = None):
    #define optical data
    d, epsv, epsa = validate_optical_data(data)
    
    #define effective optical data
    if eff_data is None:
        d_eff, epsv_eff, epsa_eff = _isotropic_effective_data((d, epsv, epsa))
    else:
        d_eff, epsv_eff, epsa_eff = validate_optical_data(eff_data, homogeneous = True)
        
    #define input field data
    fieldv, wavelengths, pixelsize = field_data
    
    #define constants 
    ks = k0(wavelengths, pixelsize)
    n = len(d)
    substeps = np.broadcast_to(np.asarray(substeps),(n,))
    
    #define input ray directions. Either a scalar or 1D array
    beta0, phi0 = _validate_betaphi(beta,phi,extendeddim = fieldv.ndim-2)
    
    #define output field
    out = np.zeros_like(fieldv)
    
    if eps0 is None:
        eps0 = uniaxial_order(0.,epsv).mean(axis = (0,1,2))
    else:
        eps0 = np.asarray(eps0)
        assert eps0.ndim == 1
    if layer0 is None:
        layer0 = np.array((0.,0.,0.))
    else:
        layer0 = np.asarray(layer0)
        assert layer0.ndim == 1
    shape = fieldv.shape[-2:]
    
    dmat_in = projection_matrix(shape, ks,refind2eps([nin]*3), mode = "r")
    dmat_out = projection_matrix(shape, ks,refind2eps([nout]*3), mode = "r")
    
    #set initial diffraction matrices
    d_eff0, epsv_eff0, epsa_eff0 = d_eff[0], epsv_eff[0], epsa_eff[0]
    d0, nsteps0 = d[0], substeps[0]
    
    nsteps = nsteps0
    
    kd_eff0 = ks*d_eff0/nsteps0
    kd = ks*d0/nsteps0
    kd0 = kd
    
    alphad, fd, fid = diffraction_alphaffi_xy(shape, ks, epsv = epsv_eff0, epsa = epsa_eff0)
    pmatd = phase_matrix(alphad, kd_eff0)
    dmatd = dotmdm(fd,pmatd,fid) 
    
    alpha0, f0, fi0 = alphaffi_xy(beta0,phi0,epsa_eff0,epsv_eff0)  
    pmat0 = phase_matrix(alpha0, -kd0)  
    dmat0 = dotmdm(f0,pmat0,fi0)
    
    dmatf = diffraction_matrix(shape, ks, epsv = eps0, epsa = layer0, d = d[0]/nsteps, mode = mode)
    
    dmatb = diffraction_matrix(shape, ks, epsv = eps0, epsa = layer0, d = -d[0]/nsteps, mode = mode)

    dmatf = dotmm(dmat0,dmatd)
    
    update = False
    
    
#    for i in range(19):
#        d_effi, epsv_effi, epsa_effi = d_eff[i], epsv_eff[i], epsa_eff[i]
#        di, nstepsi = d[i], substeps[i] 
#        kd_effi = ks*d_effi/nstepsi
#        kdi = ks*di/nstepsi   
#        if not np.allclose(epsv_effi,epsv_eff0) or not np.allclose(epsa_effi,epsa_eff0):
#            pass
        
    
    
#    for i in range(10):
#        if not np.allclose(epsffd,epseff[i]) or not np.allclose(aeffd,aeff[i]):
#            aeffd = aeff[i]
#            epseffd = epseff[i]
#            alphad, fd, fid = diffraction_alphaffi_xy(shape, ks, epsv = epseffd, epsa = aeffd)
#            alpha0, f0, fi0 = alphaffi_xy(beta0,phi0,aeffd,epseffd) 
#            update = True
#        if d0 != d[i] or update:
#            d0 = d[i]
#            kd0 = ks*d0/nsteps
#            pmat0 = phase_matrix(alpha0, -kd0) 
#            
#            
#        if deffd != deff[i]
            
        
    
    
    #modify diffriction matrix to remove accumulated average phase shift
    alpha, f, fi = alphaffi_xy(beta0,phi0,layer0,eps0)
    kd = ks*d[0]/nsteps
    
    if mode == "t":
        pmatf = phasem_t(alpha, -kd[:,None,None])
        pmatb = phasem_t(alpha, kd[:,None,None])
        _t = ftransmit
    else:
        pmatf = phasem(alpha, -kd[:,None,None])
        pmatb = phasem(alpha, kd[:,None,None])
        _t  = transmit

    #phase back shift matrix
    cmatf = dotmdm(f,pmatf,fi)
    
    
    cmatb = dotmdm(f,pmatb,fi)
    #compute corrected diffraction matrix
    #dmatf = dotmm(dmatf,cmatf[:,None,None,:,:],dmatf)
    #return cmatf,dmatf

    #dmatf = dotmm(cmatf,dmatf)

    
    #dmatb = dotmm(dmatb,cmatb[:,None,None,:,:],dmatb)
    dmatb = dotmm(dmatb,cmatb)
    
    if npass > 1:
        fieldv0 = fieldv.copy()
    
    verbose_level = DTMMConfig.verbose
           
    def propagate(infield, is_forward, out = None, msg = ""):
        field = infield.copy()
        indices = list(range(n))
        if not is_forward:
            indices.reverse()
        for i in indices:
            layer = epsa[i]
            print_progress(i,n,level = verbose_level, suffix = msg)
            alpha, f, fi = alphaffi_xy(beta0,phi0,layer,epsv[i])
            for j in range(nsteps):
                if is_forward:
                    if diffraction:
                        field = diffract(field, dmatf, out = out, window = window)
                    field = _t(f,alpha,fi, field, kd, out = out) 
                else:
                    field = _t(f,alpha,fi, field, -kd, out = out) 
                    if diffraction:
                        field = diffract(field, dmatb, out = out, window = window)
        print_progress(n,n,level = verbose_level, suffix = msg) 
        return out
    
    forward = True
    for i in range(npass):
        msg = "{}/{}".format(i+1,npass)
        
        if forward:
            out = propagate(fieldv,forward, out = out,msg = msg)
        else:
            fieldv = propagate(fieldv,forward, out = fieldv, msg = msg)
   
        forward = not forward
        
        if npass > 1:
            if i%2 == 0:
                assert out is not fieldv
                if i != npass -1:
                    diffract(out,dmat_out, window = window, out = fieldv)
            else:
                
                diffract(fieldv,dmat_in, window = window, out = fieldv)
                np.subtract(fieldv0, fieldv, out = fieldv)
                fieldv0 = fieldv.copy()            
    
    return out, wavelengths, pixelsize

def transmit_field(field_data, optical_data, beta = 0., 
                   phi = 0., eff_data = None, nin = 1., nout = 1., npass = 1,nstep=1,
              diffraction = True, mode = "b", window = None, betamax = 0.9):
    """Transmits input field data through optical data.
    
    This function calculates transmitted field and possibly updates input field
    to include reflected waves. 
    
    
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
        in combination with npass > 1 to detremine reflections from input layer.
    nout : float, optional
        Refractive index of the output (top) surface (1. by default). Used
        in combination with npass > 1 to detremine reflections from output layer.
    npass: int, optional
        How many passes (iterations) to perform. For strongly reflecting elements
        this should be set to a higher value. If npass > 1, then input field data is
        overwritten and adds reflected light from the sample. (defaults to 1)
    nstep: int or 1D array_like of ints
        Specifies layer transmission computation steps (defaults to 1).
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
        msg = "{}/{}".format(i+1,npass)
        for j in range(n):
            print_progress(j,n,level = verbose_level, suffix = msg) 
            thickness = d[j]*(-1)**i
            thickness_eff = d_eff[j]*(-1)**i
            field = propagate_layer(field, ks, (thickness, epsv[j], epsa[j]),(thickness_eff, epsv_eff[j], epsa_eff[j]), 
                            beta = beta, phi = phi, nsteps = substeps[j], diffraction = diffraction,
                            betamax = betamax, out = out)
       
        print_progress(n,n,level = verbose_level, suffix = msg) 
        if npass > 1:
            if i%2 == 0:
                if i != npass -1:
                    field = diffract(field,dmat_out, window = window, out = field_in)
                    out = field_in
                    
            else:
                field = diffract(field,dmat_in, window = window, out = field_in)
                field = np.add(field0, field, out = field_in)
                #field0 = field_in.copy()  
                out = field_out

    return out, wavelengths, pixelsize


def propagate_layer(field,wavenumbers, layer_data, effective_data, beta = 0, phi=0,
                    nsteps = 1, diffraction = True, betamax = 0.9, out = None):

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

@cached_function
def diffraction_alphaffi_xy(shape, ks, epsv = (1,1,1), 
                            epsa = (0.,0.,0.), betamax = 0.9, out = None):

    ks = np.asarray(ks)
    ks = abs(ks)
    #shape = fieldv.shape[-2:]
    #eps = uniaxial_order(0.,eps0)
    beta, phi = betaphi(shape,ks)
    #m = tukey(beta,0.1)
    #mask0 = (beta>0.9) & (beta < 1.1)
    mask0 = (beta >=betamax)
    #mask = np.empty(mask0.shape + (4,), mask0.dtype)
    #for i in range(4):
    #    mask[...,i] = mask0   

            
    alpha, f, fi = alphaffi_xy(beta,phi,epsa,epsv, out = out) 
    fi[mask0] = 0.
    f[mask0] = 0.
    alpha[mask0] = 0.
    out = (alpha,f,fi)

    #np.multiply(f,m[...,None,None],f)
    #np.multiply(fi,m[...,None,None],fi)
    #return mask, alpha,f,fi
    
    return out


def layer_matrices(shape, ks, eps = (1,1,1), layer = (0.,0.,0.), betamax = 0.9):
    ks = np.asarray(ks)
    ks = abs(ks)
    #shape = fieldv.shape[-2:]
    #eps = uniaxial_order(0.,eps0)
    beta, phi = betaphi(shape,ks)
    #m = tukey(beta,0.1)
    #mask0 = (beta>0.9) & (beta < 1.1)
    mask0 = (beta >=betamax)
    #mask = np.empty(mask0.shape + (4,), mask0.dtype)
    #for i in range(4):
    #    mask[...,i] = mask0    
    
 
            
    alpha, f, fi = alphaffi_xy(beta,phi,layer,eps) 
    fi[mask0] = 0.
    f[mask0] = 0.
    alpha[mask0] = 0.
    
    #np.multiply(f,m[...,None,None],f)
    #np.multiply(fi,m[...,None,None],fi)
    #return mask, alpha,f,fi
    return alpha,f,fi

  
def phase_matrix(alpha, kd, mode = None, mask = None, out = None):
    kd = np.asarray(kd)
    if mode == "t":
        out = phasem_t(alpha ,kd[...,None,None], out = out)
    elif mode == "r":
        out = phasem_r(alpha, kd[...,None,None], out = out)
    else:
        out = phasem(alpha,kd[...,None,None], out = out)  
    if mask is not None:
        out[mask] = 0.
    return out  


@cached_function
def diffraction_matrix(shape, ks,  d = 1., epsv = (1,1,1), epsa = (0,0,0.), mode = "b", betamax = 0.9, out = None):
    ks = np.asarray(ks)
    alpha, f, fi = diffraction_alphaffi_xy(shape, ks, epsv = epsv, epsa = epsa, betamax = betamax)
    kd = np.asarray(ks) * d
    pmat = phase_matrix(alpha, kd , mode = mode)
    return dotmdm(f,pmat,fi,out = out) 

@cached_function
def correction_matrix(beta,phi,ks, d=1., epsv = (1,1,1), epsa = (0,0,0.), out = None):
    alpha, f, fi = alphaffi_xy(beta,phi,epsa,epsv)  
    kd = -np.asarray(ks)*d
    pmat = phase_matrix(alpha, kd)  
    return dotmdm(f,pmat,fi, out = out)

@cached_function
def corrected_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = 0.9, out = None):
    dmat = diffraction_matrix(shape, ks, d, epsv, epsa, betamax = betamax)
    cmat = correction_matrix(beta, phi, ks, d, epsv, epsa)
    if d > 0:
        return dotmm(dmat,cmat, out = None)
    else:
        return dotmm(cmat, dmat, out = None)

@cached_function
def projection_matrix(shape, ks, epsv = (1,1,1),epsa = (0,0,0.), mode = "t", betamax = 0.9, out = None):
    alpha, f, fi = diffraction_alphaffi_xy(shape, ks, epsv = epsv, epsa = epsa, betamax = betamax)
    mask = None
    kd = np.zeros_like(ks)
    pmat = phase_matrix(alpha, kd , mode = mode, mask = mask)
    return dotmdm(f,pmat,fi,out = out)   
  
def diffract(fieldv, dmat, window = None, out = None): 
    f = fft2(fieldv, out = out)
    f2 = dotmf(dmat, f ,out = f)
    out = ifft2(f2, out = out)
    if window is not None:
        out = np.multiply(out,window,out = out)
    return out

def transmitted_field(field, wavenumbers, n = 1, betamax = 0.9, out = None):
    eps = refind2eps([n]*3)
    pmat = projection_matrix(field.shape[-2:], wavenumbers, epsv = eps, epsa = (0.,0.,0.), mode = "t", betamax = betamax)
    return diffract(field, pmat, out = out) 

def reflected_field(field, wavenumbers, n = 1, betamax = 0.9, out = None):
    eps = refind2eps([n]*3)
    pmat = projection_matrix(field.shape[-2:], wavenumbers, epsv = eps, epsa = (0.,0.,0.), mode = "r", betamax = betamax)
    return diffract(field, pmat, out = out) 


    
__all__ = ["transmit_field", "transmit_field_old"]
