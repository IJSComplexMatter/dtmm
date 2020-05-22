"""
4x4 and 2x2 transfer matrix method functions. 
"""

from __future__ import absolute_import, print_function, division

import numpy as np

from dtmm.conf import NCDTYPE,NFDTYPE, CDTYPE, FDTYPE, NUMBA_TARGET, BETAMAX, \
                        NUMBA_PARALLEL, NUMBA_CACHE, NUMBA_FASTMATH, DTMMConfig
from dtmm.rotation import  _calc_rotations_uniaxial, _calc_rotations, _rotate_diagonal_tensor
from dtmm.linalg import _dotr2m, dotmdm, dotmm, inv, dotmv, _dotr2v
from dtmm.data import refind2eps
from dtmm.rotation import rotation_vector2
from dtmm.print_tools import print_progress

from dtmm.jones import polarizer as polarizer2x2
from dtmm.jones import as4x4, jonesvec

import numba as nb
from numba import prange
import time

if NUMBA_PARALLEL == False:
    prange = range

sqrt = np.sqrt

@nb.njit([(NFDTYPE,NCDTYPE[:],NCDTYPE[:,:])])                                                                
def _auxiliary_matrix(beta,eps,Lm):
    "Computes all elements of the auxiliary matrix of shape 4x4."
    eps2m = 1./eps[2]
    eps4eps2m = eps[4]*eps2m
    eps5eps2m = eps[5]*eps2m
    
    Lm[0,0] = (-beta*eps4eps2m)
    Lm[0,1] = 1.-beta*beta*eps2m
    Lm[0,2] = (-beta*eps5eps2m)
    Lm[0,3] = 0.
    Lm[1,0] = eps[0]- eps[4]*eps4eps2m
    Lm[1,1] = Lm[0,0]
    Lm[1,2] = eps[3]- eps[5]*eps4eps2m
    Lm[1,3] = 0.
    Lm[2,0] = 0.
    Lm[2,1] = 0.
    Lm[2,2] = 0.
    Lm[2,3] = -1. 
    Lm[3,0] = (-1.0*Lm[1,2])
    Lm[3,1] = (-1.0*Lm[0,2])
    Lm[3,2] = beta * beta + eps[5]*eps5eps2m - eps[1]  
    Lm[3,3] = 0.  

@nb.njit([(NFDTYPE,NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _alphaf_iso(beta,eps0,alpha,F):
    #n = eps0[0]**0.5
    aout = sqrt(eps0[0]-beta**2)
    if aout != 0.:
        gpout = eps0[0]/aout
        gsout = -aout
        alpha[0] = aout
        alpha[1] = -aout
        alpha[2] = aout
        alpha[3] = -aout 
        F[0,0] = 0.5 
        F[0,1] = 0.5
        F[0,2] = 0.
        F[0,3] = 0.
        F[1,0] = 0.5 * gpout 
        F[1,1] = -0.5 * gpout 
        F[1,2] = 0.
        F[1,3] = 0.
        F[2,0] = 0.
        F[2,1] = 0.
        F[2,2] = 0.5 
        F[2,3] = 0.5
        F[3,0] = 0.
        F[3,1] = 0.
        F[3,2] = 0.5 * gsout 
        F[3,3] = -0.5 * gsout 
    else:
        
        F[...]=0.
        alpha[...] = 0.

@nb.njit([(NFDTYPE,NCDTYPE[:],NFDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _alphaf_uniaxial(beta,eps0,R,alpha,F): 

    #uniaxial case
    ct = R[2,2]
    st = -R[2,0] 
    st2 = st * st
    ct2 = ct * ct
    
    sf = -R[0,1]
    cf = R[1,1]

    eps11 = eps0[0]

    
    delta = eps0[2] -  eps11
    if beta == 0.: #same as calculation for beta !=0, except faster... no multiplying with zeros
        ev02 =  eps11 
        evs = sqrt(ev02)
        u = eps11 + delta * ct2
        w = eps11 * (ev02 + delta)
        sq = sqrt(u*w)/u
        evpp = sq
        evpm = -sq
        
    else: #can also be used for beta=0... just slower
        ev02 =  eps11 - beta * beta
        evs = sqrt(ev02)
        
        u = eps11 + delta * ct2
        gama = beta * cf
        v = gama * delta * 2 * st * ct
        w = gama * gama * (delta * st2)- eps11 * (ev02 + delta)
        
        sq = sqrt(v*v-4*u*w)/2/u
        v = v/2/u
        
        evpp = -v + sq
        evpm = -v - sq
        

    alpha[0] = evpp
    alpha[1] = evpm
    alpha[2] = evs
    alpha[3] = -evs    


    if beta == 0.:

        eps11sf = eps11 * sf
        evssf = evs*sf
        evscf = evs*cf
        eps11cf = eps11*cf
        
        F[0,2] = evssf
        F[1,2] = eps11sf
        F[2,2] = -evscf
        F[3,2] = eps11cf
        
        F[0,3] = -evssf
        F[1,3] = eps11sf 
        F[2,3] = evscf 
        F[3,3] = eps11cf
        
        F[0,0] = cf
        F[1,0] = evpp *cf
        F[2,0] = sf
        F[3,0] = -evpp *sf 
        
        F[0,1] = cf
        F[1,1] = evpm *cf
        F[2,1] = sf
        F[3,1] = -evpm *sf    
        
    else:
        sfst = (R[1,2])
        cfst = (R[0,2])                   
                                    
        ctbeta = ct * beta
        ctbetaeps11 = ctbeta / eps11
        eps11sfst = eps11 * sfst
        evssfst = evs*sfst
        evscfst = evs*cfst
        evsctbeta = evs*ctbeta
        ev02cfst = ev02*cfst
        ev02cfsteps11 = ev02cfst/eps11
      
        F[0,2] = -evssfst 
        F[1,2] = -eps11sfst
        F[2,2] = evscfst - ctbeta
        F[3,2] = evsctbeta - ev02cfst
  
        F[0,3] = -evssfst
        F[1,3] = eps11sfst
        F[2,3] = evscfst + ctbeta
        F[3,3] = ev02cfst + evsctbeta
        
        F[0,0] = -evpp*ctbetaeps11 + ev02cfsteps11
        F[1,0] = evpp *cfst - ctbeta
        F[2,0] = sfst
        F[3,0] = -evpp *sfst
        
        F[0,1] = -evpm*ctbetaeps11 + ev02cfsteps11
        F[1,1] = evpm *cfst - ctbeta
        F[2,1] = sfst
        F[3,1] = -evpm *sfst 
        
    #normalize base vectors
    for j in range(4):
        tmp = 0.
        for i in range(4):
            tmp += F[i,j].real * F[i,j].real + F[i,j].imag * F[i,j].imag
        
        tmp = tmp ** 0.5
        F[0,j] = F[0,j]/tmp 
        F[1,j] = F[1,j]/tmp 
        F[2,j] = F[2,j]/tmp 
        F[3,j] = F[3,j]/tmp 

@nb.njit([NFDTYPE(NCDTYPE[:])], cache = NUMBA_CACHE)
def _poynting(field):
    tmp1 = (field[0].real * field[1].real + field[0].imag * field[1].imag)
    tmp2 = (field[2].real * field[3].real + field[2].imag * field[3].imag)
    return tmp1-tmp2 

@nb.njit([(NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])], cache = NUMBA_CACHE)
def _copy_sorted(alpha,fmat, out_alpha, out_fmat):
    i = 0
    j = 1
    for k in range(4):
        assert i < 4
        assert j < 4
        p = _poynting(fmat[:,k])
        if p == 1j:
            if alpha[k].imag >= 0.:
                out_alpha[i] = alpha[k]
                out_fmat[:,i] = fmat[:,k]
                i = i + 2
                
            else:
                out_alpha[j] = alpha[k]
                out_fmat[:,j] = fmat[:,k] 
                j = j + 2
        else:
            if p >= 0.:
                out_alpha[i] = alpha[k]
                out_fmat[:,i] = fmat[:,k]
                i = i + 2
            else:
                out_alpha[j] = alpha[k]
                out_fmat[:,j] = fmat[:,k] 
                j = j + 2
            
@nb.guvectorize([(NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NFDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:])],
                 "(),(),(m),(l),(k),(n)->(n),(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _alphaf_vec(beta,phi,rv,epsv,epsa,dummy,alpha,F):
    #F is a 4x4 matrix... we can use 3x3 part for Rotation matrix and F[3] for eps  temporary data
    
    #isotropic case
    if (epsv[0] == epsv[1] and epsv[1]==epsv[2]):
        eps = F[3] 
        #_uniaxial_order(0.,epsv,eps) #store caluclated eps values in Fi[3]
        _alphaf_iso(beta[0],epsv,alpha,F)
        _dotr2m(rv,F,F)
    #uniaxial
    elif (epsv[0] == epsv[1]):
        R = F.real
        eps = F[3] 
        #_uniaxial_order(1.,epsv,eps)
        _calc_rotations_uniaxial(phi[0],epsa,R) #store rotation matrix in Fi.real[0:3,0:3]
        _alphaf_uniaxial(beta[0],epsv,R,alpha,F)
        _dotr2m(rv,F,F)
    else:#biaxial case
        R = F.real 
        eps = F.ravel() #reuse F memory (eps is length 6 1D array)
        _calc_rotations(phi[0],epsa,R) #store rotation matrix in Fi.real[0:3,0:3]
        _rotate_diagonal_tensor(R,epsv,eps)
        _auxiliary_matrix(beta[0],eps,F) #calculate Lm matrix and put it to F
        alpha0,F0 = np.linalg.eig(F)
        _copy_sorted(alpha0,F0,alpha,F)#copy data and sort it
        _dotr2m(rv,F,F)
        
#dummy arrays for gufuncs    
_dummy_array = np.empty((4,),CDTYPE)
_dummy_array2 = np.empty((9,),CDTYPE)
_dummy_EH = np.empty((2,),CDTYPE)
    

def _alphaf(beta,phi,epsv,epsa,out = None):
    rv = rotation_vector2(phi) 
    return _alphaf_vec(beta,phi,rv,epsv,epsa,_dummy_array, out = out)

def _default_beta_phi(beta,phi):
    """Checks the validity of beta, phi arguments and sets default values if needed"""
    beta = np.asarray(beta, FDTYPE) if beta is not None else np.asarray(0., FDTYPE)
    phi = np.asarray(phi, FDTYPE) if phi is not None else np.asarray(0., FDTYPE)
    return beta, phi

def _default_epsv_epsa(epsv,epsa):
    """Checks the validity of epsv, epsa arguments and sets default values if needed"""
    epsv = np.asarray(epsv, CDTYPE) if epsv is not None else np.asarray((1.,1.,1.), CDTYPE)
    epsa = np.asarray(epsa, FDTYPE) if epsa is not None else np.asarray((0.,0.,0.), FDTYPE)
    assert epsv.shape[-1] >= 3
    assert epsa.shape[-1] >= 3
    return epsv, epsa

def _as_field_vec(fvec):
    fvec = np.asarray(fvec, dtype = CDTYPE)
    assert fvec.shape[-1] == 4
    return fvec

def alphaf(beta=None,phi=None,epsv=None,epsa=None,out = None):
    """Computes alpha and field arrays (eigen values and eigen vectors arrays).
    Broadcasting rules apply.
    
    Parameters
    ----------
    beta : float, optional
       The beta parameter of the field (defaults to 0.)
    phi : float, optional
       The phi parameter of the field (defaults to 0.)
    epsv : array-like, optional
       Dielectric tensor eigenvalues array (defaults to unity).
    epsa : array_like, optional
       Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
    out : ndarray, optional
       Output array
       
    Returns
    -------
    alpha, field arrays : (ndarray, ndarray)
        Eigen values and eigen vectors arrays 
    """
    beta, phi = _default_beta_phi(beta,phi)
    epsv, epsa = _default_epsv_epsa(epsv, epsa)
    rv = rotation_vector2(phi) 
    if out is None:
        return _alphaf_vec(beta,phi,rv,epsv,epsa,_dummy_array)
    else:
        return _alphaf_vec(beta,phi,rv,epsv,epsa,_dummy_array, out = out)

def alphaffi(beta=None,phi=None,epsv=None,epsa=None,out = None):
    """Computes alpha and field arrays (eigen values and eigen vectors arrays)
    and inverse of the field array. See :func:`alphaf` for details
    
    Broadcasting rules apply.
       
    Returns
    -------
    alpha, field, ifield  : (ndarray, ndarray, ndarray)
        Eigen values and eigen vectors arrays and its inverse
        
    This is equivalent to
    
    >>> alpha,field = alphaf(0,0, [2,2,2], [0.,0.,0.])
    >>> ifield = inv(field)
    """
    if out is not None:
        a,f,fi = out
        alphaf(beta,phi,epsv,epsa, out = (a,f))
        inv(f,fi)
    else:
        a,f = alphaf(beta,phi,epsv,epsa)
        fi = inv(f)
    return a,f,fi
 
def alphaE(beta,phi,epsv,epsa, mode = +1, out = None):
    alpha,f = alphaf(beta,phi,epsv,epsa)
    e = E_mat(f,mode = mode, copy = False)
    if mode == 1:
        alpha = alpha[...,::2]
    else:
        alpha = alpha[...,1::2]
    if out is not None:
        out[0][...] = alpha
        out[1][...] = e
    else:
        out = alpha.copy(), e.copy()
    return out

def alphaEEi(beta,phi,epsv,epsa, mode = +1, out = None):
    if out is None:
        alpha,E = alphaE(beta,phi,epsv,epsa, mode = mode)
        return alpha, E, inv(E)
    else:
        alpha, E, Ei = out
        alpha, E = alphaE(beta,phi,epsv,epsa, mode = mode, out = (alpha, E))
        Ei = inv(E,out = Ei)
        return alpha, E, Ei 

from dtmm.conf import _numba_0_39_or_greater

if _numba_0_39_or_greater:
    @nb.vectorize([NCDTYPE(NCDTYPE,NFDTYPE)],
        target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)       
    def _phase_mat_vec(alpha,kd):
        return np.exp(NCDTYPE(1j)*kd*alpha)

    def phasem(alpha,kd,out = None):
        kd = np.asarray(kd,FDTYPE)[...,None]
        out = _phase_mat_vec(alpha,kd,out)
        #if out.shape[-1] == 4:
        #    out[...,1::2]=0.
        return out
else:
    @nb.guvectorize([(NCDTYPE[:],NFDTYPE[:], NCDTYPE[:])],
                    "(n),()->(n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)       
    def _phase_mat_vec(alpha,kd,out):
        for i in range(alpha.shape[0]):
            out[i] = np.exp(NCDTYPE(1j)*kd[0]*(alpha[i]))
                    
    phasem = _phase_mat_vec

def phase_mat(alpha, kd, mode = None,  out = None):
    """Computes phase a 4x4 or 2x2 matrix from eigenvalue matrix alpha 
    and wavenumber"""
    kd = np.asarray(kd, dtype = FDTYPE)
    if out is None:
        if mode is None:
            b = np.broadcast(alpha,kd[...,None])
        else:
            b = np.broadcast(alpha[...,::2],kd[...,None])
        out = np.empty(b.shape, dtype = CDTYPE)
        
    if mode == +1:
        phasem(alpha[...,::2],kd, out = out)
    elif mode == -1:
        phasem(alpha[...,1::2],kd,out = out)
    elif mode is None:
        out = phasem(alpha,kd, out = out) 
    else:
        raise ValueError("Unknown propagation mode.")
    return out 

def iphase_mat(alpha, kd, cfact = 0.1, mode = "t",  out = None):
    """Computes incoherent 4x4 phase matrix from eigenvalue matrix alpha and wavenumber"""
    if mode == "t":
        np.add(alpha[...,1::2],alpha[...,1::2].real*2*1j*cfact, out = alpha[...,1::2])
    elif mode == "r":
        np.add(alpha[...,::2],-alpha[...,::2].real*2*1j*cfact, out = alpha[...,::2])
    else:
        raise ValueError("Unknown propagation mode.")    
    return phase_mat(alpha, kd, mode = None, out = out)


@nb.guvectorize([(NCDTYPE[:], NFDTYPE[:])],
                    "(n)->()", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)       
def poynting(fvec, out):
    """Calculates a z-component of the poynting vector from the field vector"""
    assert fvec.shape[0] == 4
    #tmp1 = (fvec[0].real * fvec[1].real + fvec[0].imag * fvec[1].imag)
    #tmp2 = (fvec[2].real * fvec[3].real + fvec[2].imag * fvec[3].imag)
    #out[0] = tmp1-tmp2   
    out[0] = _poynting(fvec)

#@nb.guvectorize([(NCDTYPE[:,:], NFDTYPE[:])],
#                    "(n,n)->(n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)       
#def fmat2poynting(fmat, out):
#    """Calculates a z-component of the poynting vector from the field vector"""
#    assert fmat.shape[0] == 4 and fmat.shape[1] == 4 
#    for i in range(4):
#        tmp1 = (fmat[0,i].real * fmat[1,i].real + fmat[0,i].imag * fmat[1,i].imag)
#        tmp2 = (fmat[2,i].real * fmat[3,i].real + fmat[2,i].imag * fmat[3,i].imag)
#        out[i] = tmp1-tmp2  
        
def fmat2poynting(fmat, out = None):
    """Calculates a z-component of the poynting vectors from the field matrix"""
    axes = list(range(fmat.ndim))
    n = axes.pop(-2)
    axes.append(n)
    fmat = fmat.transpose(*axes)
    return poynting(fmat, out = out)
    
@nb.guvectorize([(NCDTYPE[:,:], NCDTYPE[:,:])],
                    "(n,n)->(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)       
def normalize_f(fmat, out):
    """Normalizes column of field matrix so that fmat2poytning of the resulted
    matrix returns ones"""
    assert fmat.shape[0] == 4 and fmat.shape[1] == 4 
    for i in range(4):
        tmp1 = (fmat[0,i].real * fmat[1,i].real + fmat[0,i].imag * fmat[1,i].imag)
        tmp2 = (fmat[2,i].real * fmat[3,i].real + fmat[2,i].imag * fmat[3,i].imag)
        n = np.abs(tmp1-tmp2)**0.5 
        if n == 0.:
            n = 0.
        else:
            n = 1./n
        out[0,i] = fmat[0,i] * n 
        out[1,i] = fmat[1,i] * n
        out[2,i] = fmat[2,i] * n
        out[3,i] = fmat[3,i] * n


def intensity(fvec):
    """Calculates absolute value of the z-component of the poynting vector"""
    fvec = _as_field_vec(fvec)
    p = poynting(fvec)
    return np.abs(p)


def projection_mat(fmat, fmati = None, mode = +1):
    """Calculates projection matrix from the given field matrix. By multiplying
    the field with this matrix you obtain only the forward (mode = +1) or
    backward (mode = -1) propagating field,"""
    if fmati is None:
        fmati = inv(fmat)
    diag = np.zeros(fmat.shape[:-1],fmat.dtype)
    if mode == 1:
        diag[...,0::2] = 1.
    elif mode == -1:
        diag[...,1::2] = 1.
    else:
        raise ValueError("Unknown propagation mode.")     
    return dotmdm(fmat,diag,fmati)    


    
def EHz(fvec, beta = None, phi = None, epsv = None, epsa = None, out = None):
    """Constructs the z component of the electric and magnetic fields from the
    input field vector in a given medium. If epsv and epsa are not given, vacuum is assumed"""
    
    fvec = _as_field_vec(fvec)
    beta, phi = _default_beta_phi(beta, phi)
    epsv,epsa = _default_epsv_epsa(epsv, epsa)
    rv = rotation_vector2(-phi)
    
    return _EHz(fvec,beta,phi,rv, epsv,epsa,_dummy_EH,out)
    
@nb.guvectorize([(NCDTYPE[:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NFDTYPE[:],NCDTYPE[:],NCDTYPE[:])],
                 "(n),(),(),(m),(l),(k),(o)->(o)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _EHz(fvec, beta,phi,rv,epsv,epsa,dummy,out):
    eps = np.empty(shape = (6,), dtype = epsv.dtype)
    R = np.empty(shape = (3,3), dtype = rv.dtype)
    frot = np.empty_like(fvec)
    _dotr2v(rv,fvec,frot)
    _calc_rotations_uniaxial(phi[0],epsa,R)
    _rotate_diagonal_tensor(R,epsv,eps)
    out[0] = - (eps[4]*frot[0] + eps[5]*frot[2] + beta[0] * frot[1]) / eps[2]
    out[1] = beta[0] * frot[3]


def T_mat(fin, fout, fini = None, fouti = None, mode = +1):
    """Computes amplitude interface transmittance matrix.
    
    Parameters
    ----------
    fin : ndarray
        Input media field matrix
    fout : ndarray
        Output media field matrix
    fini : ndarray, optional
        Input media inverse of the field. 
    """
    if fini is None:
        fini = inv(fin)
    if fouti is None:
        fouti = inv(fout)
    Sf = dotmm(fini,fout)
    Sb = dotmm(fouti,fin)
    out = np.zeros_like(Sf)
    if mode == +1:
        out[...,::2,::2] = inv(Sf[...,::2,::2])
        out[...,1::2,1::2] = Sb[...,1::2,1::2]
        return out
    elif mode == -1:
        out[...,1::2,1::2] = inv(Sf[...,1::2,1::2])
        out[...,::2,::2] = (Sb[...,::2,::2])
        return out
    else:
        raise ValueError("Unknown propagation mode.")      
    
def S_mat(fin, fout, fini = None, overwrite_fin = False, mode = +1):
    if overwrite_fin == True:
        out = fin
    else:
        out = None
    if fini is None:
        fini = inv(fin, out = out)
    S = dotmm(fini,fout, out = out)
    if mode == +1:
        return S[...,::2,::2],S[...,1::2,0::2]
    elif mode == -1:
        return S[...,1::2,1::2],S[...,0::2,1::2]
    else:
        raise ValueError("Unknown propagation mode.")  
        
def transmission_mat(fin, fout, fini = None, mode = +1,out = None):
    A,B = S_mat(fin, fout, fini = fini, mode = mode)
    if mode == +1:
        A1 = fin[...,::2,::2]
        A2 = fout[...,::2,::2]
    elif mode == -1:
        A1 = fin[...,1::2,1::2]
        A2 = fout[...,1::2,1::2]
    else:
        raise ValueError("Unknown propagation mode.")        
    Ai = inv(A, out = out)
    A1i = inv(A1)
    return dotmm(dotmm(A2,Ai, out = Ai),A1i, out = Ai)
#
#def reflection_mat(fin, fout, fini = None, mode = +1,out = None):
#    A,B = S_mat(fin, fout, fini = fini, mode = mode)
#    if mode == +1:
#        A1p = fin[...,::2,::2]
#        A1m = fin[...,::2,1::2]
#    elif mode == -1:
#        A1p = fin[...,1::2,1::2]
#        A1m = fin[...,1::2,::2]  
#    else:
#        raise ValueError("Unknown propagation mode.")
#    Ai = inv(A, out = out)
#    A1pi = inv(A1p)
#    return dotmm(dotmm(dotmm(A1m,B,out = Ai),Ai, out = Ai),A1pi, out = Ai)

def tr_mat(fin, fout, fini = None, overwrite_fin = False, mode = +1, out = None):
    if overwrite_fin == True:
        er = E_mat(fin, mode = mode * (-1), copy = True)
    else:
        er = E_mat(fin, mode = mode * (-1), copy = False)
    et = E_mat(fout, mode = mode, copy = False)
    eti,eri = Etri_mat(fin, fout, fini = fini, overwrite_fin = overwrite_fin, mode = mode, out = out)
    return dotmm(et,eti, out = eti), dotmm(er,eri, out = eri)

def t_mat(fin, fout, fini = None, overwrite_fin = False, mode = +1, out = None):
    eti = Eti_mat(fin, fout, fini = fini, overwrite_fin = overwrite_fin, mode = mode, out = out)
    et = E_mat(fout, mode = mode, copy = False)
    return dotmm(et,eti, out = eti)

def E_mat(fmat, mode = None, copy = True):
    if mode == +1:
        e = fmat[...,::2,::2]
    elif mode == -1:
        e = fmat[...,::2,1::2]
    elif mode is None:
        ep = fmat[...,::2,::2]
        en = fmat[...,::2,1::2]
        out = np.zeros_like(fmat)
        out[...,::2,::2] = ep
        out[...,1::2,1::2] = en
        return out 
    else:
        raise ValueError("Unknown propagation mode.")
    return e.copy() if copy else e  

def Eti_mat(fin, fout, fini = None, overwrite_fin = False, mode = +1, out = None):
    A = E_mat(fin, mode = mode, copy = False) 
    #Ai = inv(A, out = out)
    Ai = inv(A)
    St,Sr = S_mat(fin, fout, fini = fini, overwrite_fin = overwrite_fin, mode = mode)
    Sti = inv(St, out = St)
    return dotmm(Sti,Ai, out = out)




def Etri_mat(fin, fout, fini = None, overwrite_fin = False, mode = +1, out = None):
    out1, out2 = out if out is not None else (None, None)
    A = E_mat(fin, mode = mode, copy = False)
    Ai = inv(A, out = out1)  
    St,Sr = S_mat(fin, fout, fini = fini, overwrite_fin = overwrite_fin, mode = mode)
    Sti = inv(St, out = St)
    ei = dotmm(Sti,Ai,out = Ai)
    return ei, dotmm(Sr,ei, out = out2)
    
def E2H_mat(fmat, mode = +1, out = None):  
    if mode == +1:
        A = fmat[...,::2,::2]
        B = fmat[...,1::2,::2]
    elif mode == -1:
        A = fmat[...,::2,1::2]
        B = fmat[...,1::2,1::2]
    else:
        raise ValueError("Unknown propagation mode.") 
    Ai = inv(A, out = out)
    return dotmm(B,Ai, out = Ai)  

def f_iso(beta=0.,phi = 0.,n=1.):
    """Returns field matrix for isotropic layer of a given refractive index
    and beta, phi parameters"""
    epsv = refind2eps([n]*3)
    epsa = np.zeros(shape = (3,),dtype= FDTYPE)
    alpha, f = alphaf(beta,phi,epsv,epsa)    
    return f

def ffi_iso(n=1,beta=0.,phi = 0.):
    """Returns field matrix and inverse of the field matrix for isotropic layer 
    of a given refractive index and beta, phi parameters"""
    epsv = refind2eps([n]*3)
    epsa = np.zeros(shape = (3,),dtype= FDTYPE)
    alpha, f, fi = alphaffi(beta,phi,epsv,epsa)    
    return f,fi

def layer_mat(kd, epsv,epsa, beta = 0,phi = 0, cfact = 0.1, method = "4x4", fmatin = None, retfmat = False, out = None):
    """Computes characteristic matrix of a single layer M=F.P.Fi,
    
    Numpy broadcasting rules apply
    
    Parameters
    ----------
    kd : float
        A sequence of phase values (layer thickness times wavenumber in vacuum).
        len(kd) must match len(epsv) and len(epsa).
    epsv : array_like
        Epsilon eigenvalues.
    epsa : array_like
        Optical axes orientation angles (psi, theta, phi).
    beta : float
        Beta angle of input light.
    phi : float
        Phi angle of input light.
    cfact : float, optional
        Coherence factor, only used in combination with `4x4_2` method.
    method : str
        One of `4x4` (4x4 berreman - trasnmittance + reflectance), 
        `2x2` (2x2 jones - transmittance only), 
        `4x4_1` (4x4, single reflections - transmittance only),
        `2x2_1` (2x2, single reflections - transmittance only) 
        `4x4_2`(4x4, partially coherent reflections - transmittance only) 
    fmatin : ndarray, optional
        Used in compination with 2x2_1 method. Itspecifies the field matrix of 
        the input media in order to compute fresnel reflections. If not provided 
        it reverts to 2x2 with no reflections.
        
    out : ndarray, optional
    
    Returns
    -------
    cmat : ndarray
        Characteristic matrix of the layer.
    """    
    if method in ("2x2","2x2_1"):
        alpha,fmat = alphaf(beta,phi,epsv,epsa)
        f = E_mat(fmat, mode = +1, copy = False)
        if fmatin is not None and method == "2x2_1":
            fi = Eti_mat(fmatin, fmat, mode = +1)
        else:
            fi = inv(f)
        pmat = phase_mat(alpha[...,::2],kd)

    elif method in ("4x4","4x4_1","4x4_r","4x4_2"):
        alpha,fmat,fi = alphaffi(beta,phi,epsv,epsa)
        fmat = normalize_f(fmat)
        fi = inv(fmat)
        f = fmat
        pmat0 = phase_mat(alpha,-kd)
        if method ==  "4x4_r":
            alpha1 = alpha.copy()
            np.add(alpha1[...,1::2],alpha1[...,1::2].real*2j*cfact, out = alpha1[...,1::2])
            np.add(alpha[...,::2],-alpha[...,::2].real*2j*cfact, out = alpha[...,::2])
            pmat1 = phase_mat(alpha1,-kd)
            pmat2 = phase_mat(alpha,-kd)
            
            pmat = np.zeros(shape = pmat0.shape[:-1] + (8,8), dtype = pmat0.dtype)
            pmat[...,0,0] = pmat1[...,0]
            pmat[...,1,1] = pmat1[...,1]
            pmat[...,2,2] = pmat1[...,2]
            pmat[...,3,3] = pmat1[...,3]
            pmat[...,4,0] = pmat1[...,0]
            pmat[...,6,2] = pmat1[...,2]
            
            #pmat[...,4,4] = pmat2[...,0]
            #pmat[...,5,5] = pmat2[...,1]
            #pmat[...,6,6] = 0#pmat2[...,2]
            #pmat[...,7,7] = pmat2[...,3]
            pmat[...,5,1] = pmat2[...,1]
            pmat[...,7,3] = pmat2[...,3]
            #pmat[...,5,1] = ((1 - np.abs(pmat1[...,1])**2)**0.5)*pmat2[...,1]
            #pmat[...,7,3] = ((1 - np.abs(pmat1[...,3])**2)**0.5)*pmat2[...,3]
            
            f1 = np.zeros_like(pmat)
            f1[...,0:4,0:4] = f
            f1[...,4:8,4:8] = f
            
            f1i = inv(f1)
            #f1i = np.zeros_like(pmat)
            #f1i[...,0:4,0:4] = fi
            #f1i[...,4:8,4:8] = fi  
            
            return dotmm(f1,dotmm(pmat,f1i))
            
        elif method == "4x4_2":
            np.add(alpha[...,1::2],alpha[...,1::2].real*2*1j*cfact, out = alpha[...,1::2])

        pmat = phase_mat(alpha,-kd)
        if method == "4x4_1":
            pmat[...,1::2] = 0.
    else:
        raise ValueError("Unknown method!")
        
    out = dotmdm(f,pmat,fi,out = out) 
    
    if retfmat == False:
        return out   
    else:
        return fmat, out 

def layer3D_mat(kd, epsv,epsa, betamax = BETAMAX,  out = None):
    """Computes characteristic matrix of a single layer M=F.P.Fi,
    
    Numpy broadcasting rules apply
    
    Parameters
    ----------
    kd : float
        A sequence of phase values (layer thickness times wavenumber in vacuum).
        len(kd) must match len(epsv) and len(epsa).
    epsv : array_like
        Epsilon eigenvalues.
    epsa : array_like
        Optical axes orientation angles (psi, theta, phi).
    out : ndarray, optional
    
    Returns
    -------
    cmat : ndarray
        Characteristic matrix of the layer.
    """    
    if method in ("2x2","2x2_1"):
        alpha,fmat = alphaf(beta,phi,epsv,epsa)
        f = E_mat(fmat, mode = +1, copy = False)
        if fmatin is not None and method == "2x2_1":
            fi = Eti_mat(fmatin, fmat, mode = +1)
        else:
            fi = inv(f)
        pmat = phase_mat(alpha[...,::2],kd)

    elif method in ("4x4","4x4_1","4x4_r","4x4_2"):
        alpha,fmat,fi = alphaffi(beta,phi,epsv,epsa)
        f = fmat
        pmat0 = phase_mat(alpha,-kd)
        if method ==  "4x4_r":
            alpha1 = alpha.copy()
            np.add(alpha1[...,1::2],alpha1[...,1::2].real*2j*cfact, out = alpha1[...,1::2])
            np.add(alpha[...,::2],-alpha[...,::2].real*2j*cfact, out = alpha[...,::2])
            pmat1 = phase_mat(alpha1,-kd)
            pmat2 = phase_mat(alpha,-kd)
            
            pmat = np.zeros(shape = pmat0.shape[:-1] + (8,8), dtype = pmat0.dtype)
            pmat[...,0,0] = pmat1[...,0]
            pmat[...,1,1] = pmat1[...,1]
            pmat[...,2,2] = pmat1[...,2]
            pmat[...,3,3] = pmat1[...,3]
            pmat[...,4,4] = pmat2[...,0]
            pmat[...,5,5] = pmat2[...,1]
            pmat[...,6,6] = pmat2[...,2]
            pmat[...,7,7] = pmat2[...,3]
            pmat[...,5,1] = pmat0[...,1] - pmat1[...,1]
            pmat[...,7,3] = pmat0[...,3] - pmat1[...,3]
            
            f1 = np.zeros_like(pmat)
            f1[...,0:4,0:4] = f
            f1[...,4:8,4:8] = f
            
            f1i = np.zeros_like(pmat)
            f1i[...,0:4,0:4] = fi
            f1i[...,4:8,4:8] = fi  
            
            return dotmm(f1,dotmm(pmat,f1i))
            
        elif method == "4x4_2":
            np.add(alpha[...,1::2],alpha[...,1::2].real*2*1j*cfact, out = alpha[...,1::2])

        pmat = phase_mat(alpha,-kd)
        if method == "4x4_1":
            pmat[...,1::2] = 0.
    else:
        raise ValueError("Unknown method!")
        
    out = dotmdm(f,pmat,fi,out = out) 
    
    if retfmat == False:
        return out   
    else:
        return fmat, out 

def stack_mat(kd,epsv,epsa, beta = 0, phi = 0, cfact = 0.01, method = "4x4", out = None):
    """Computes a stack characteristic matrix M = M_1.M_2....M_n if method is
    4x4, 4x2(2x4) and a characteristic matrix M = M_n...M_2.M_1 if method is
    2x2.
    
    Note that this function calls :func:`layer_mat`, so numpy broadcasting 
    rules apply to kd[i], epsv[i], epsa[i], beta and phi. 
    
    Parameters
    ----------
    kd : array_like
        A sequence of phase values (layer thickness times wavenumber in vacuum).
        len(kd) must match len(epsv) and len(epsa).
    epsv : array_like
        A sequence of epsilon eigenvalues.
    epsa : array_like
        A sequence of optical axes orientation angles (psi, theta, phi).
    beta : float
        Beta angle of input light.
    phi : float
        Phi angle of input light.
    cfact : float
        Coherence factor, only used in combination with `4x4_r` and `4x4_2` methods.
    method : str
        One of `4x4` (4x4 berreman), `2x2` (2x2 jones), 
        `4x4_1` (4x4, single reflections),`2x2_1` (2x2, single reflections) 
        `4x4_r` (4x4, incoherent to compute reflection) or 
        `4x4_t`(4x4, incoherent to compute transmission) 
    out : ndarray, optional
    
    Returns
    -------
    cmat : ndarray
        Characteristic matrix of the stack.
    """
    t0 = time.time()
    mat = None
    tmat = None
    fmat = None
    n = len(kd)

    verbose_level = DTMMConfig.verbose
    if verbose_level > 1:
        print ("Building stack matrix.")
    for pi,i in enumerate(range(n)):
        print_progress(pi,n,level = verbose_level) 
        if method == "2x2_1":
            fmat, mat = layer_mat(kd[i],epsv[i],epsa[i],beta = beta, phi = phi, cfact = cfact, method = method, fmatin = fmat, out = mat, retfmat = True)
        else:
            mat = layer_mat(kd[i],epsv[i],epsa[i],beta = beta, phi = phi, cfact = cfact, method = method, out = mat)

        if pi == 0:
            if out is None:
                out = mat.copy()
            else:
                out[...] = mat
        else:
            if tmat is not None:
                dotmm(tmat,mat,mat)
            if method.startswith("2x2"):
                dotmm(mat,out,out)
            else:
                dotmm(out,mat,out)
    print_progress(n,n,level = verbose_level) 
    t = time.time()-t0
    if verbose_level >1:
        print("     Done in {:.2f} seconds!".format(t))  
    return out 

m1 = np.array([[1.,0,0,0],
         [0,1,0,0],
         [0,0,1,0],
         [0,0,0,1],
         [0,0,0,0],
         [0,0,0,0],
         [0,0,0,0],
         [0,0,0,0]])

m0 = np.array([[1.,0,0,0,0,0,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,1,0,0,0,0,0],
               [0,0,0,0,0,0,0,1]])    
    
def system_mat(cmat = None,fmatin = None, fmatout = None, fmatini = None, out = None):
    """Computes a system matrix from a characteristic matrix Fin-1.C.Fout"""
    if fmatini is None:
        if fmatin is None:
            fmatin = f_iso()
        fmatini = inv(fmatin)
    if fmatout is None:
        fmatout = fmatin
    if cmat is not None:
        if cmat.shape[-1] == 8:
            dotmm(fmatini,cmat[...,0:4,0:4],out = cmat[...,0:4,0:4])
            dotmm(cmat[...,0:4,0:4],fmatout,out = cmat[...,0:4,0:4])
            dotmm(fmatini,cmat[...,4:8,4:8],out = cmat[...,4:8,4:8])
            dotmm(cmat[...,4:8,4:8],fmatout,out = cmat[...,4:8,4:8])
            if out is None:
                out = np.empty_like(cmat[...,0:4,0:4])
            
            out[...] = dotmm(m0,dotmm(cmat,m1))
            return out

        else:
            out = dotmm(fmatini,cmat,out = out)
            return dotmm(out,fmatout,out = out)  
    else:
        return dotmm(fmatini,fmatout,out = out)

def reflection_mat(smat, out = None):
    """Computes a 4x4 reflection matrix.
    """
    m1 = np.zeros_like(smat)
    m2 = np.zeros_like(smat)
    m1[...,1,1] = 1.
    m1[...,3,3] = 1.
    m1[...,:,0] = -smat[...,:,0]
    m1[...,:,2] = -smat[...,:,2]
    m2[...,0,0] = -1.
    m2[...,2,2] = -1.
    m2[...,:,1] = smat[...,:,1]
    m2[...,:,3] = smat[...,:,3]
    m1 = inv(m1)
    return dotmm(m1,m2, out = out)

def reflection_mat(smat, out = None):
    """Computes a 4x4 reflection matrix.
    """
    m1 = np.zeros_like(smat)
    m2 = np.zeros_like(smat)
    #fill diagonals
    for i in range(smat.shape[-1]//2):
        m1[...,i*2+1,i*2+1] = 1.
        m2[...,i*2,i*2] = -1.
    m1[...,:,0::2] = -smat[...,:,0::2]
    m2[...,:,1::2] = smat[...,:,1::2]
    m1 = inv(m1)
    return dotmm(m1,m2, out = out)

def fvec2E(fvec, fmat = None, fmati = None, mode = +1, inplace = False):
    """Converts field vector to E vector. If inplace == True, also 
    makes input field forward or backward propagating. """
    if inplace == True:
        out  = fvec
    else:
        out = None
    if fmat is None:
        fmat = f_iso()
    pmat = projection_mat(fmat, fmati = fmati, mode = mode)
    if mode == +1:
        return dotmv(pmat,fvec, out = out)[...,::2]
    elif mode == -1:
        return dotmv(pmat,fvec, out = out)[...,1::2]
    else:
        raise ValueError("Unknown mode!")
        
    
def E2fvec(evec, fmat = None, mode = +1, out = None):
    """Converts E vector to field vector"""
    if fmat is None:
        fmat = f_iso()
    e2h = E2H_mat(fmat, mode = mode)
    hvec = dotmv(e2h, evec)
    if out is None:
        out = np.empty(shape = evec.shape[:-1] + (4,), dtype = evec.dtype)
    out[...,::2] = evec
    out[...,1::2] = hvec
    return out
    
def transmit2x2(evec_in, cmat,  tmatin = None, tmatout = None, evec_out = None):
    """Transmits E-field vector using 2x2 method.
    
    This functions takes an E-field vector that describes the input field and
    computes the output transmited field using the 2x2 characteristic matrix.
    """
    b = np.broadcast(evec_in[...,0][...,None,None],cmat)

    if evec_out is not None:
        evec_out[...] = 0 
    else:   
        evec_out = np.zeros(b.shape[:-2] + (2,), evec_in.dtype)
        
    if tmatin is not None:
        evec = dotmv(tmatin, evec_in, out = evec_out)
    else:
        evec = evec_in
    eout = dotmv(cmat, evec, out = evec_out)
    if tmatout is not None:
        eout = dotmv(tmatout, eout, out = evec_out)
    return evec_out

def transmit2x2(fvec_in, cmat, fmatout = None, tmatin = None, tmatout = None, fvec_out = None):
    """Transmits field vector using 2x2 method.
    
    This functions takes a field vector that describes the input field and
    computes the output transmited field using the 2x2 characteristic matrix.
    """
    b = np.broadcast(fvec_in[...,0][...,None,None],cmat)
    if fmatout is None:
        fmatout = f_iso(1,0,0) 
    if fvec_out is not None:
        fvec_out[...] = 0 
    else:   
        fvec_out = np.zeros(b.shape[:-2] + (4,), fvec_in.dtype)
        
    if tmatin is not None:
        evec = dotmv(tmatin, fvec_in[...,::2], out = fvec_out[...,::2])
    else:
        evec = fvec_in[...,::2]
    eout = dotmv(cmat, evec, out = fvec_out[...,::2])
    if tmatout is not None:
        eout = dotmv(tmatout, eout, out = fvec_out[...,::2])
    e2h = E2H_mat(fmatout, mode = +1)
    hout = dotmv(e2h, eout, out = fvec_out[...,1::2])
    return fvec_out


def transmit(fvec_in, cmat = None, fmatin = None, fmatout = None, fmatini = None, fmatouti = None, fvec_out = None):
    """Transmits field vector using 4x4 method.
    
    This functions takes a field vector that describes the input field and
    computes the output transmited field and also updates the input field 
    with the reflected waves.
    """
    b = np.broadcast(fvec_in[..., None],cmat[...,0:4,0:4], fmatin, fmatout)
    

    if fmatini is None:
        if fmatin is None:
            fmatin = f_iso()
        fmatini = inv(fmatin)
    if fmatin is None:
        fmatin = inv(fmatini)
    if fmatouti is None:
        if fmatout is None:
            fmatout = fmatin
            fmatouti = fmatini
        else:
            fmatouti = inv(fmatout)
    if fmatout is None:
        fmatout = inv(fmatouti)
        
    smat = system_mat(cmat = cmat,fmatini = fmatini, fmatout = fmatout)
    

    
    avec = dotmv(fmatini,fvec_in)
    
    
    a = np.zeros(b.shape[:-1], avec.dtype)
    a[...,0::2] = avec[...,0::2]
    avec = a.copy()#so that it broadcasts

    if fvec_out is not None:
        bvec = dotmv(fmatouti,fvec_out)
        a[...,1::2] = bvec[...,1::2] 
    else:
        bvec = np.zeros_like(avec)

    r = reflection_mat(smat)
    out = dotmv(r,a, out = fvec_out)
    
    avec[...,1::2] = out[...,1::2]
    bvec[...,::2] = out[...,::2]
        
    dotmv(fmatin,avec,out = fvec_in)    
    return dotmv(fmatout,bvec,out = out)

def transfer4x4(fvec_in, kd, epsv, epsa,  beta = 0., phi = 0., nin = 1., nout = 1., 
             method = "4x4", reflect_in = False, reflect_out = False, fvec_out = None):
    if method not in ("4x4", "4x4_1","4x4_r","4x4_2"):
        raise ValueError("Unknown method!")
        
        
    fveci = fvec_in
    fvecf = fvec_out
    
    fmatin = f_iso(n = nin, beta  = beta, phi = phi)
    fmatout = f_iso(n = nout, beta  = beta, phi = phi)
        
#    if reflect_in == True:
#        
#        #make fresnel reflection of the input (forward propagating) field
#        fin = f_iso(n = nin, beta  = beta, phi = phi)
#        alpha,fmatin = alphaf(beta = beta, phi = phi, epsv = epsv[0], epsa = epsa[0])
#        
#        t = T_mat(fin = fin, fout = fmatin, mode = +1)
#        fveci = dotmv(t,fveci, out = fveci)
#        
##        tmati = t_mat(fin,fmatin, mode = +1)
##        evec0 = fvec2E(fveci, fin, mode = +1)
##        evec = dotmv(tmati,evec0)
##        fveci = E2fvec(evec,fmatin, mode = +1)
##        fvec_in = E2fvec(evec0,fin, mode = +1, out = fvec_in)
#    else:
#        fmatin = f_iso(n = nin, beta  = beta, phi = phi)
#       
#    if reflect_out == True:
#        #make fresnel reflection of the output (backward propagating) field
#        alpha,fmatout = alphaf(beta = beta, phi = phi, epsv = epsv[-1], epsa = epsa[-1])
#        fout = f_iso(n = nout, beta  = beta, phi = phi)
#        if fvecf is not None:
#            t = T_mat(fin = fout, fout = fmatout, mode = -1)
#            fvecf = dotmv(t,fvecf)            
#            
##            tmatf = t_mat(fmatout,fout, mode = -1)
##            evec0 = fvec2E(fvecf, fout, mode = -1)
##            evec = dotmv(tmatf,evec) 
##            fvecf = E2fvec(evec,fmatout, mode = -1)
##            fvec_out = E2fvec(evec0,fout, mode = -1, out = fvec_out)
#    else:
#        fmatout = f_iso(n = nout, beta  = beta, phi = phi)
    
    cmat = stack_mat(kd, epsv, epsa, method = method)
    fvecf = transmit(fveci, cmat = cmat, fmatin = fmatin, fmatout = fmatout, fvec_out = fvecf)

#    if reflect_in == True:
#        #make fresnel reflection of the input (backward propagating) field
#        t = T_mat(fin = fmatin, fout = fin, mode = -1)
#        fveci = dotmv(t,fveci, out = fvec_in)        
#        
##        tmati = t_mat(fin,fmatin, mode = -1)
##        evec = fvec2E(fveci, mode = -1)
##        evec = dotmv(tmati,evec)
##        fveci = E2fvec(evec,fout, mode = -1) 
##        np.add(fvec_in, fveci, out = fvec_in)
#
#
#    if reflect_out == True:
#        #make fresnel reflection of the output (forward propagating) field
#        
#        t = T_mat(fin = fmatout, fout = fout, mode = 1)
#        fvecf = dotmv(t,fvecf, out = fvec_out)    
##        tmati = t_mat(fmatout,fout, mode = +1)
##        evec = fvec2E(fvecf, mode = +1)
##        evec = dotmv(tmati,evec)
##        fvecf = E2fvec(evec,fout, mode = +1) 
##        fvecf = np.add(fvec_out, fvecf, out = fvec_out)  

    return fvecf

def transfer2x2(evec, kd, epsv, epsa,  beta = 0., phi = 0., nin  = None, nout = None, 
             method = "2x2", out = None):
    if method not in ("2x2", "2x2_1"):
        raise ValueError("Unknown method!")
        
    if nin is not None:
       fin = f_iso(n = nin, beta  = beta, phi = phi)
       alpha,fout = alphaf(beta = beta, phi = phi, epsv = epsv[0], epsa = epsa[0])
       tmat = t_mat(fin,fout)
       evec = dotmv(tmat,evec)
    
    cmat = stack_mat(kd, epsv, epsa, method = method)
    evec = dotmv(cmat,evec, out = out)
    
    if nout is not None:
       fout = f_iso(n = nout, beta  = beta, phi = phi)
       alpha,fin = alphaf(beta = beta, phi = phi, epsv = epsv[-1], epsa = epsa[-1])
       tmat = t_mat(fin,fout)
       dotmv(tmat,evec, out = evec)  
       
    return evec

def transfer(fvec_in, kd, epsv, epsa,  beta = 0., phi = 0., nin = 1., nout = 1., 
             method = "2x2", reflect_in = False, reflect_out = False, fvec_out = None):
    """Transfer input field vector through a layered material specified by the propagation
    constand k*d, eps tensor (epsv, epsa) and input and output isotropic media.
    
    Parameters
    ----------
    fvec_in : ndarray
        Input field 4-vector
    kd : array-like
        A sequence of propagation constant for each of the layers (layer thickness * wave number)
    epsv : array-like
        eps eigenvalues sequence specifying the eps eigenvalues of each of the layers
    epsa : array-like
        eps euler angles sequence specifying the euler rotation angles of each of the layers     
    beta : float
        beta parameter of the input light
    phi : float
        The phi  parameter of the input light
    nin : float or complex
        Input media refractive index (1. by default)
    nout : float or complex
        Output media refractive index (1. by default)
    method : str
        Any of 4x4, 2x2, 2x2_1 or 4x4_1, 4x4_2, 4x4_r
    reflect_in : bool
        Defines how to treat reflections from the input media and the first layer.
        If specified it does an incoherent reflection from the first interface.
    reflect_out : bool
        Defines how to treat reflections from the last layer and the output media.
        If specified it does an incoherent reflection from the last interface.
                
    """
    
    if method.startswith("2x2"):
        fin = f_iso(n = nin, beta  = beta, phi = phi)
        fout = f_iso(n = nout, beta  = beta, phi = phi)
        evec =  fvec2E(fvec_in, fmat = fin, mode = +1)
        nin = nin if reflect_in == True else None
        nout = nout if reflect_out == True else None
        evec = transfer2x2(evec, kd, epsv, epsa,  beta = beta, phi = phi, nin  = nin, nout = nout, 
             method = method)
        return E2fvec(evec,fout, mode = +1, out = fvec_out) 
    else:
        return transfer4x4(fvec_in, kd, epsv, epsa,  beta = beta, phi = phi, nin = nin, nout = nout, 
             method = method,  reflect_in = reflect_in, reflect_out = reflect_out, fvec_out = fvec_out)        
       

transfer1d = transfer
    
def polarizer4x4(jones, fmat, out = None):
    """Returns a polarizer matrix from a given jones vector and a field matrix. 
    
    Numpy broadcasting rules apply.
    
    Parameters
    ----------
    jones : array_like
        A length two array describing the jones vector. Jones vector should
        be normalized.
    fmat : array_like
        A field matrix array of the isotropic medium.
    out : ndarray, optional
        Output array
    
    Examples
    --------
    >>> f = f_iso(n = 1.) 
    >>> jvec = dtmm.jones.jonesvec((1,0)) 
    >>> pol_mat = polarizer4x4(jvec, f) #x polarizer matrix
    
    """
    jonesmat = polarizer2x2(jones)
    fmat = normalize_f(fmat)
    fmati = inv(fmat)
    pmat = as4x4(jonesmat)    
    m = dotmm(fmat,dotmm(pmat,fmati, out = out), out = out)
    return m

def avec(jones = (1,0), amplitude = 1., mode = +1, out = None):
    """Constructs amplitude vector.
    
    Numpy broadcasting rules apply for jones, and amplitude parameters
    
    Parameters
    ----------
    jones : (int,int), optional
        Jones vector that defines the polarization state (1,0) x polarization
        by default. 
    amplitude : float, optional
        Amplitude, optional
    mode : int, optional
        Propagation mode, either +1 (default) for forward propagating mode, or
        -1 for backward propagating mode. 
    out : ndarray, optional
        Output array.
        
    Returns
    -------
    avec : ndarray
        Amplitude vector of shape (4,).
        
    Examples
    --------
    
    X polarized light with amplitude = 1
    >>> avec()
    array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
    
    X polarized light with amplitude 1 and y polarized light with amplitude 2.
    >>> avec(jones = ((1,0),(0,1)),amplitude = (1,2))
    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 2.+0.j, 0.+0.j]])
    """
    jones = np.asarray(jones)
    amplitude = np.asarray(amplitude)  
    c,s = jones[...,0], jones[...,1] 
    b = np.broadcast(c, amplitude)
    shape = b.shape + (4,)
    if out is None:
        out = np.empty(shape,CDTYPE)
    assert out.shape[-1] == 4
    if mode == +1:
        out[...,0] = c
        out[...,2] = s
        out[...,1] = 0.
        out[...,3] = 0.
    elif mode == -1:
        out[...,1] = c
        out[...,3] = s 
        out[...,0] = 0.
        out[...,2] = 0.
    else:
        raise ValueError("Unknown propagation mode.")
        
    out = np.multiply(out, amplitude[...,None] ,out = out)     
    return out


def fvec(fmat, jones = (1,0),  amplitude = 1., mode = +1, out = None):
    """Build field vector form a given polarization state, amplitude and mode.
    
    This function calls avec and then avec2fvec, see avec for details.
    
    Examples
    --------
    
    X polarized light traveling at beta = 0.4 and phi = 0.2 in medium with n = 1.5
    
    >>> fmat = f_iso(beta = 0.4, phi = 0.2, n = 1.5)
    >>> m = fvec(fmat, jones = jonesvec((1,0), phi = 0.2))
    
    This is equivalent to
    
    >>> a = avec(jones = jonesvec((1,0), phi = 0.2))
    >>> ma = avec2fvec(a,fmat)
    >>> np.allclose(ma,m)
    True
    """
    
    a = avec(jones, amplitude, mode, out)
    return avec2fvec(a, fmat, out = a)


def fvec2avec(fvec, fmat, normalize_fmat = True, out = None):
    """Converts field vector to amplitude vector
    
    Parameters
    ----------
    fvec : ndarray
        Input field vector
    fmat : ndarray
        Field matrix 
    normalize_fmat : bool, optional
        Setting this to false will not normalize the field matrix. In this case 
        user has to make sure that the normalization of the field matrix has 
        been performed prior to calling this function by calling normalize_f.
    out : ndarray, optional
        Output array
        
    Returns
    -------
    avec : ndarray
        Amplitude vector
    """
    if normalize_fmat == True:
        fmat = normalize_f(fmat)
    fmati = inv(fmat)
    return dotmv(fmati,fvec, out = out)

def avec2fvec(avec, fmat,normalize_fmat = True, out = None):
    """Converts amplitude vector to field vector
    
    Parameters
    ----------
    avec : ndarray
        Input amplitude vector
    fmat : ndarray
        Field matrix 
    normalize_fmat : bool, optional
        Setting this to false will not normalize the field matrix. In this case 
        user has to make sure that the normalization of the field matrix has 
        been performed prior to calling this function by calling normalize_f.
    out : ndarray, optional
        Output array
        
    Returns
    -------
    fvec : ndarray
        Field vector.
    """
    if normalize_fmat == True:
        fmat = normalize_f(fmat)
    return dotmv(fmat, avec, out = out)
    
__all__ = ["alphaf","alphaffi","phasem", "phase_mat", "fvec", "avec", "fvec2avec",
           "avec2fvec","f_iso"]

if __name__ == "__main__":
    import doctest
    doctest.testmod()