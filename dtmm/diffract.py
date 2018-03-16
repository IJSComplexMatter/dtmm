#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:28:25 2018

@author: andrej

"""
from __future__ import absolute_import, print_function, division

from dtmm.conf import FDTYPE,DTMMConfig
from dtmm.wave import betaphi
from dtmm.dirdata import uniaxial_order, refind2eps
from dtmm.field import alphaffi_xy, phasem,phasem_r, phasem_t
from dtmm.linalg import ftransmit, dotmdm, dotmf, dotmm
from dtmm.print_tools import print_progress
from dtmm.fft import fft2, ifft2
import numpy as np


DIFRACTION_PARAMETERS = ("distance", "mode")#, "refind")

def propagate_full(fieldv,ks,stack,eps, out = None):

    fieldv = np.asarray(fieldv)
    if out is None:
        out = np.empty_like(fieldv)
    else:
        if not isinstance(out, np.ndarray) or out.shape != fieldv.shape or out.dtype != fieldv.dtype:
            raise TypeError("Output array invalid!")
           
    ks = np.asarray(ks)
    stack = np.asarray(stack)
    shape = fieldv.shape[-2:]
        
    beta, phi = betaphi(shape, ks)
    mask = (beta <= 1.)
    
    xindices, yindices = np.indices(shape)
    xindices, yindices = xindices[mask], yindices[mask]

    n = len(stack)
    
    nk = len(xindices)
    
    verbose_level = DTMMConfig.verbose
    for i,layer in enumerate(stack):
        print_progress(i,n,level = verbose_level)
        ffield = fft2(fieldv)
        fieldv[...]=0.
        for j in range(nk):
            ii,jj = xindices[j], yindices[j]
            fieldv = eigenwave(shape, amplitude = ffield[ii,jj])
            alpha, f, fi = alphaffi_xy(beta[ii,jj],phi[ii,jj],layer,eps)
            fieldv += ftransmit(f,alpha,fi, fieldv, ks) 
    print_progress(n,n,level = verbose_level)
    return fieldv


def propagate(fieldv,ks,stack,eps, beta0 = 0., phi0 = 0., eps0 = None, layer0 = None,
              diffraction = True, mode = "t", out = None):

    fieldv = np.asarray(fieldv)
    if out is None:
        out = np.empty_like(fieldv)
    else:
        if not isinstance(out, np.ndarray) or out.shape != fieldv.shape or out.dtype != fieldv.dtype:
            raise TypeError("Output array invalid!")
           
    ks = np.asarray(ks)
    stack = np.asarray(stack)
    if eps0 is None:
        eps0 = uniaxial_order(0.,eps)
        eps0 = eps0.mean(axis = tuple(range(eps0.ndim-1)))
    else:
        eps0 = np.asarray(eps0)
        assert eps0.ndim == 1
    if layer0 is None:
        layer0 = np.array((0.,0.,0.))
    else:
        layer0 = np.asarray(layer0)
        assert layer0.ndim == 1
    shape = fieldv.shape[-2:]
        
    dmat = diffraction_matrix(shape, ks, eps = eps0, layer = layer0, d = 1., mode = mode)
    
    #modify diffriction matrix to remove accumulated average phase shift
    alpha, f, fi = alphaffi_xy(beta0,phi0,layer0,eps0)
    if mode == "t":
        pmat = phasem_t(alpha, -ks)   
    else:
        pmat = phasem(alpha, -ks) 
    #phase back shift matrix
    cmat = dotmdm(f,pmat,fi)
    #compute corrected diffraction matrix
    dmat = dotmm(dmat,cmat[:,None,None,:,:],dmat)
    
    n = len(stack)
    
    verbose_level = DTMMConfig.verbose
    for i,layer in enumerate(stack):
        print_progress(i,n,level = verbose_level)
        alpha, f, fi = alphaffi_xy(beta0,phi0,layer,eps)
        if diffraction:
            fieldv = diffract(fieldv, dmat, out = out)
        fieldv = ftransmit(f,alpha,fi, fieldv, ks, out = out) 
    print_progress(n,n,level = verbose_level)
    return fieldv

def layer_matrices(shape, ks, eps = (1,1,1), layer = (0.,0.,0.)):
    ks = np.asarray(ks)
    #shape = fieldv.shape[-2:]
    #eps = uniaxial_order(0.,eps0)
    beta, phi = betaphi(shape,ks)
    mask0 = (beta>=1)
    
    mask = np.empty(mask0.shape + (4,), mask0.dtype)
    for i in range(4):
        mask[...,i] = mask0    
            
    alpha, f, fi = alphaffi_xy(beta,phi,layer,eps) 
    
    return mask, alpha,f,fi

def phase_matrix(alpha, kd, mode = "b", mask = None, out = None):
    kd = np.asarray(kd)
    if mode == "t":
        out = phasem_t(alpha ,kd[...,np.newaxis,np.newaxis], out = out)
    elif mode == "r":
        out = phasem_r(alpha, kd[...,np.newaxis,np.newaxis], out = out)
    else:
        out = phasem(alpha,kd[...,np.newaxis,np.newaxis], out = out)  
    if mask is not None:
        out[mask] = 0.
    return out  

def diffraction_matrix(shape, ks,  d = 1., eps = (1,1,1),layer = (0.,0.,0.), mode = "b", out = None):
    ks = np.asarray(ks)
    mask, alpha, f, fi = layer_matrices(shape, ks, eps = eps, layer = layer)
    kd = ks * d
    pmat = phase_matrix(alpha, kd , mode = mode, mask = mask, out = alpha)
    return dotmdm(f,pmat,fi,out = out) 

def projection_matrix(shape, ks, eps = (1,1,1),layer = (0.,0.,0.), mode = "t", out = None):
    mask, alpha, f, fi = layer_matrices(shape, ks, eps = eps, layer = layer)
    kd = np.zeros_like(ks)
    pmat = phase_matrix(alpha, kd , mode = mode, mask = mask, out = alpha)
    return dotmdm(f,pmat,fi,out = out)   
  
def diffract(fieldv, tmat, out = None): 
    f = fft2(fieldv, out = out)
    dotmf(tmat, f ,out = f)
    return ifft2(f, out = out)

def transmitted_field(field, wavenumbers, refind = 1, out = None):
    eps = refind2eps([refind]*3)
    pmat = projection_matrix(field.shape[-2:], wavenumbers, eps = eps, layer = (0.,0.,0.), mode = "t")
    return diffract(field, pmat, out = out) 

def reflected_field(field, wavenumbers, refind = 1, out = None):
    eps = refind2eps([refind]*3)
    pmat = projection_matrix(field.shape[-2:], wavenumbers, eps = eps, layer = (0.,0.,0.), mode = "r")
    return diffract(field, pmat, out = out) 

class FieldDiffract(object):
    """Field diffraction object"""
    def __init__(self, shape, ks, refind = 1, mode = "b"):
        self.init(shape, ks, refind = refind, mode = mode)
        
    def init(self, shape, ks, refind = 1, mode = "b"):
        ks = np.asarray(ks, dtype = FDTYPE)
        eps = refind2eps([refind]*3)
        self.ks = ks
        self._d = 0.
        self.kd = np.zeros_like(ks)
        self.mask, self.alpha, self.f, self.fi = layer_matrices(shape, ks, eps = eps)
        self.phasem = np.empty_like(self.alpha)
        self.tmat = np.empty_like(self.f)
        self.mode = mode #triggers tmat calculation
        
    @property    
    def mode(self):
        return self._mode
    
    @mode.setter         
    def mode(self, mode):
        self.phasem = phase_matrix(self.alpha, self.kd, mode = mode, mask = self.mask, out = self.phasem)
        self._mode = mode   
        dotmdm(self.f,self.phasem,self.fi,out = self.tmat)  
        
    @property     
    def distance(self):
        return self._d
    
    @distance.setter    
    def distance(self,d):
        if self._d != d:
            self._d = d
            np.multiply(self.ks,d,out = self.kd)
            self.mode = self._mode
            
#    @property     
#    def refind(self):
#        return self._refind
#    
#    @refind.setter    
#    def refind(self,refind):
#        if self._refind != refind:
#            self._refind = refind
            
    def set_parameters(self, **kwargs):
        """Sets difraction parameters"""
        for key, value in kwargs.items():
            if key in DIFRACTION_PARAMETERS:
                setattr(self, key, value) 
            else:
                raise TypeError("Unexpected keyword argument '{}'".format(key))

    def get_parameters(self):
        """Returns difraction parameters as dict"""
        return {name : getattr(self,name) for name in DIFRACTION_PARAMETERS}  
                    
    def calculate(self, field, out = None):
        return diffract(field, self.tmat, out)
    
__all__ = ["FieldDiffract","propagate"]
