#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:53:08 2018

@author: andrej
"""

from dtmm.conf import FDTYPE,NFDTYPE, NCDTYPE, NUMBA_TARGET
import numba as nb
import numpy as np

def polarizer_matrix(angle, out = None):
    """Return jones matrix for polarizer. Angle is the polarizer angle.
    
    Broadcasting rules apply.
    """
    angle = np.asarray(angle)
    shape = angle.shape + (2,2)
    if out is None:
        out = np.empty(shape = shape, dtype = FDTYPE)
    else:
        assert out.shape == shape 
    c = np.cos(angle)
    s = np.sin(angle)
    cs = c*s
    out[...,0,0] = c*c
    out[...,1,0] = cs
    out[...,0,1] = cs
    out[...,1,1] = s*s
    return out

@nb.guvectorize([(NFDTYPE[:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],"(m,m),(n,k,l)->(n,k,l)", target = NUMBA_TARGET)
def apply_polarizer_matrix(pmat, field, out):
    """Multiplies 2x2 jones polarizer matrix with 4 x n x m field array"""
    for i in range(field.shape[1]):
        for j in range(field.shape[2]):
            Ex = field[0,i,j] * pmat[0,0] + field[2,i,j] * pmat[0,1]
            Hy = field[1,i,j] * pmat[0,0] - field[3,i,j] * pmat[0,1]
            Ey = field[0,i,j] * pmat[1,0] + field[2,i,j] * pmat[1,1]
            Hx = -field[1,i,j] * pmat[1,0] + field[3,i,j] * pmat[1,1]
            out[0,i,j] = Ex
            out[1,i,j] = Hy
            out[2,i,j] = Ey
            out[3,i,j] = Hx

__all__ = ["polarizer_matrix", "apply_polarizer_matrix"]
    