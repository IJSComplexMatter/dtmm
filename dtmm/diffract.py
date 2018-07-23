#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:28:25 2018

@author: andrej

"""
from __future__ import absolute_import, print_function, division

from dtmm.conf import cached_function, BETAMAX, FDTYPE, CDTYPE
from dtmm.wave import betaphi
from dtmm.window import tukey
from dtmm.data import refind2eps
from dtmm.tmm import alphaffi_xy, phasem,phasem_r, phasem_t
from dtmm.linalg import dotmdm, dotmf
from dtmm.fft import fft2, ifft2
import numpy as np


DIFRACTION_PARAMETERS = ("distance", "mode")#, "refind")


@cached_function
def diffraction_alphaffi_xy(shape, ks, epsv = (1,1,1), 
                            epsa = (0.,0.,0.), betamax = BETAMAX, out = None):

    ks = np.asarray(ks)
    ks = abs(ks)
    #shape = fieldv.shape[-2:]
    #eps = uniaxial_order(0.,eps0)
    beta, phi = betaphi(shape,ks)
    m = tukey(beta,0.,betamax)
    #mask0 = (beta>0.9) & (beta < 1.1)
    mask0 = (beta >= betamax)#betamax)
    #mask = np.empty(mask0.shape + (4,), mask0.dtype)
    #for i in range(4):
    #    mask[...,i] = mask0   

            
    alpha, f, fi = alphaffi_xy(beta,phi,epsa,epsv, out = out) 
    fi[mask0] = 0.
    f[mask0] = 0.
    alpha[mask0] = 0.
    out = (alpha,f,fi)

    np.multiply(f,m[...,None,None],f)
    np.multiply(fi,m[...,None,None],fi)
    #return mask, alpha,f,fi
    
    return out


def layer_matrices(shape, ks, eps = (1,1,1), layer = (0.,0.,0.), betamax = BETAMAX):
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
    kd = np.asarray(kd, dtype = FDTYPE)
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
def diffraction_matrix(shape, ks,  d = 1., epsv = (1,1,1), epsa = (0,0,0.), mode = "b", betamax = BETAMAX, out = None):
    ks = np.asarray(ks, dtype = FDTYPE)
    epsv = np.asarray(epsv, dtype = CDTYPE)
    epsa = np.asarray(epsa, dtype = FDTYPE)
    alpha, f, fi = diffraction_alphaffi_xy(shape, ks, epsv = epsv, epsa = epsa, betamax = betamax)
    kd =ks * d
    pmat = phase_matrix(alpha, kd , mode = mode)
    return dotmdm(f,pmat,fi,out = out) 

@cached_function
def projection_matrix(shape, ks, epsv = (1,1,1),epsa = (0,0,0.), mode = "t", betamax = BETAMAX, out = None):
    """Computes a reciprocial field projection matrix.
    """
    ks = np.asarray(ks, dtype = FDTYPE)
    epsv = np.asarray(epsv, dtype = CDTYPE)
    epsa = np.asarray(epsa, dtype = FDTYPE)    
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

def diffracted_field(field, wavenumbers, d = 0.,n = 1, mode = "t", betamax = BETAMAX, out = None):
    eps = refind2eps([n]*3)
    pmat = diffraction_matrix(field.shape[-2:], wavenumbers, d = d, epsv = eps, epsa = (0.,0.,0.), mode = mode, betamax = betamax)
    return diffract(field, pmat, out = out) 

def transmitted_field(field, wavenumbers, n = 1, betamax = BETAMAX, out = None):
    eps = refind2eps([n]*3)
    pmat = projection_matrix(field.shape[-2:], wavenumbers, epsv = eps, epsa = (0.,0.,0.), mode = "t", betamax = betamax)
    return diffract(field, pmat, out = out) 

def reflected_field(field, wavenumbers, n = 1, betamax = BETAMAX, out = None):
    eps = refind2eps([n]*3)
    pmat = projection_matrix(field.shape[-2:], wavenumbers, epsv = eps, epsa = (0.,0.,0.), mode = "r", betamax = betamax)
    return diffract(field, pmat, out = out) 


    
__all__ = []
