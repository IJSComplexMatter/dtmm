#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diffraction matrices used in the difraction step of the layer propagation functions
"""

from __future__ import absolute_import, print_function, division

from dtmm.conf import cached_function, BETAMAX,FDTYPE,CDTYPE
from dtmm.tmm import alphaffi, alphaEEi, alphaf,  E_mat, phase_mat
from dtmm.linalg import dotmdm, dotmm,  inv
from dtmm.diffract import diffraction_alphaffi, E_diffraction_matrix, phase_matrix, diffraction_alphaf

import numpy as np
from dtmm.diffract import field_diffraction_matrix

@cached_function
def correction_matrix(beta,phi,ks, d=1., epsv = (1,1,1), epsa = (0,0,0.), out = None):
    alpha, f, fi = alphaffi(beta,phi,epsv,epsa)  
    kd = -np.asarray(ks)*d
    pmat = phase_matrix(alpha, kd)  
    return dotmdm(f,pmat,fi, out = out)

@cached_function
def field_correction_matrix(beta,phi,ks, d=1., epsv = (1,1,1), epsa = (0,0,0.), out = None):
    alpha, f, fi = alphaffi(beta,phi,epsv,epsa)  
    kd = -np.asarray(ks)*d
    pmat = phase_matrix(alpha, kd)  
    return dotmdm(f,pmat,fi, out = out)

@cached_function
def E_correction_matrix(beta,phi,ks, d=1., epsv = (1,1,1), epsa = (0,0,0.), mode = +1, out = None):
    alpha, j, ji = alphaEEi(beta,phi,epsv,epsa, mode = mode)  
    kd = -np.asarray(ks)*d
    pmat = phase_matrix(alpha, kd)  
    return dotmdm(j,pmat,ji, out = out)

@cached_function
def corrected_E_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), mode = +1, betamax = BETAMAX, out = None):
    dmat = E_diffraction_matrix(shape, ks, d, epsv, epsa, mode = mode, betamax = betamax)
    cmat = E_correction_matrix(beta, phi, ks, d/2., epsv, epsa, mode = mode)
    return dotmm(cmat,dotmm(dmat,cmat, out = out), out = out)
    
 
@cached_function
def second_E_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), mode = +1, betamax = BETAMAX, out = None):
    dmat = E_diffraction_matrix(shape, ks, d, epsv, epsa, mode = mode,betamax = betamax)
    cmat = E_correction_matrix(beta, phi, ks, d, epsv, epsa, mode = mode)
    return dotmm(dmat,cmat, out = None)

@cached_function
def first_E_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), mode = +1, betamax = BETAMAX, out = None):
    dmat = E_diffraction_matrix(shape, ks, d, epsv, epsa, mode = mode, betamax = betamax)
    cmat = E_correction_matrix(beta, phi, ks, d, epsv, epsa, mode = mode)
    return dotmm(cmat,dmat, out = None)


@cached_function
def corrected_Epn_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
    ep = corrected_E_diffraction_matrix(shape, ks, beta,phi, d= d,
                                 epsv = epsv, epsa = epsa, mode = +1, betamax = betamax)
    en = corrected_E_diffraction_matrix(shape, ks, beta,phi, d= d,
                                 epsv = epsv, epsa = epsa, mode = -1, betamax = betamax)
    return ep, en
    
@cached_function
def corrected_field_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
 
    dmat = field_diffraction_matrix(shape, ks, d, epsv, epsa, betamax = betamax)
    cmat = correction_matrix(beta, phi, ks, d/2, epsv, epsa)
    dmat = dotmm(dmat,cmat, out = out)
    return dotmm(cmat,dmat, out = dmat) 

@cached_function
def first_field_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
    dmat = field_diffraction_matrix(shape, ks, d, epsv, epsa, betamax = betamax)
    cmat = field_correction_matrix(beta, phi, ks, d, epsv, epsa)
    return dotmm(dmat,cmat, out = None)

@cached_function
def second_field_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
    dmat = field_diffraction_matrix(shape, ks, d, epsv, epsa, betamax = betamax)
    cmat = field_correction_matrix(beta, phi, ks, d, epsv, epsa)
    return dotmm(cmat,dmat, out = None)

@cached_function
def first_Epn_diffraction_matrix(shape, ks,  d = 1., epsv = (1,1,1), epsa = (0,0,0.),  betamax = BETAMAX, out = None):
    ks = np.asarray(ks, dtype = FDTYPE)
    epsv = np.asarray(epsv, dtype = CDTYPE)
    epsa = np.asarray(epsa, dtype = FDTYPE)
    alpha, f, fi= diffraction_alphaffi(shape, ks, epsv = epsv, epsa = epsa, betamax = betamax)
    kd = ks * d
    e = E_mat(f, mode = None)
    pmat = phase_mat(alpha, kd[...,None,None])
    return dotmdm(e,pmat,fi,out = out) 

@cached_function
def second_Epn_diffraction_matrix(shape, ks,  d = 1., epsv = (1,1,1), epsa = (0,0,0.),  betamax = BETAMAX, out = None):
    ks = np.asarray(ks, dtype = FDTYPE)
    epsv = np.asarray(epsv, dtype = CDTYPE)
    epsa = np.asarray(epsa, dtype = FDTYPE)
    alpha, f = diffraction_alphaf(shape, ks, epsv = epsv, epsa = epsa, betamax = betamax)
    kd = ks * d
    e = E_mat(f, mode = None)
    ei = inv(e)
    pmat = phase_mat(alpha, kd[...,None,None])
    return dotmdm(f,pmat,ei,out = out) 

@cached_function
def Epn_correction_matrix(beta,phi,ks, d=1., epsv = (1,1,1), epsa = (0,0,0.), out = None):
    alpha, f = alphaf(beta,phi,epsv,epsa)  
    kd = -np.asarray(ks)*d
    pmat = phase_mat(alpha, kd[...,None,None]) 
    e = E_mat(f, mode = None)
    ei = inv(e)
    return dotmdm(e,pmat,ei, out = out)

@cached_function
def first_corrected_Epn_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
    dmat = first_Epn_diffraction_matrix(shape, ks, d, epsv, epsa, betamax = betamax)
    cmat = Epn_correction_matrix(beta, phi, ks, d, epsv, epsa)
    return dotmm(cmat,dmat, out = None)

@cached_function
def second_corrected_Epn_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
    dmat = second_Epn_diffraction_matrix(shape, ks, d, epsv, epsa, betamax = betamax)
    cmat = Epn_correction_matrix(beta, phi, ks, d, epsv, epsa)
    return dotmm(dmat, cmat, out = None)

