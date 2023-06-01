"""Functions for field magnification.
"""
from dtmm.conf import cached_function, BETAMAX, FDTYPE, CDTYPE
import numpy as np
from dtmm.linalg import dotmm

from dtmm.diffract import E_diffraction_alphaEEi, E_diffraction_alphaE, diffraction_alphaffi, diffraction_alphaf

@cached_function
def field_magnification_matrix(shape, ks,  m = 1., epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
    ks = np.asarray(ks, dtype = FDTYPE)
    ksm = ks * m
    epsv = np.asarray(epsv, dtype = CDTYPE)
    epsa = np.asarray(epsa, dtype = FDTYPE)
    _,_, fi = diffraction_alphaffi(shape, ks, epsv = epsv, epsa = epsa, betamax = betamax)
    _, f = E_diffraction_alphaE(shape, ksm, epsv = epsv, epsa = epsa, betamax = betamax)
    return dotmm(f,fi,out = out) 

@cached_function
def E_magnification_matrix(shape, ks,  m = 1., epsv = (1,1,1), epsa = (0,0,0.), mode = +1, betamax = BETAMAX, out = None):
    ks = np.asarray(ks, dtype = FDTYPE)
    ksm = ks * m
    epsv = np.asarray(epsv, dtype = CDTYPE)
    epsa = np.asarray(epsa, dtype = FDTYPE)
    _,_, ji = E_diffraction_alphaEEi(shape, ks, epsv = epsv, epsa = epsa, mode = mode, betamax = betamax)
    _, j = E_diffraction_alphaE(shape, ksm, epsv = epsv, epsa = epsa, mode = mode, betamax = betamax)
    return dotmm(j,ji,out = out) 
