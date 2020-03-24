"""
4x4 and 2x2 transfer matrix method functions. 
"""

from __future__ import absolute_import, print_function, division

import numpy as np

from dtmm.conf import NCDTYPE,NFDTYPE, CDTYPE, FDTYPE, NUMBA_TARGET, \
                        NUMBA_PARALLEL, NUMBA_CACHE, NUMBA_FASTMATH, DTMMConfig
from dtmm.rotation import  _calc_rotations_uniaxial, _calc_rotations, _rotate_diagonal_tensor
from dtmm.linalg import _dotr2m, dotmdm, dotmm, inv, dotmv
from dtmm.data import refind2eps
from dtmm.rotation import rotation_vector2
from dtmm.print_tools import print_progress

from dtmm.jones import polarizer as polarizer2x2
from dtmm.jones import as4x4

import numba as nb
from numba import prange
import time

if not NUMBA_PARALLEL:
    prange = range

sqrt = np.sqrt


@nb.njit([(NFDTYPE, NCDTYPE[:], NCDTYPE[:, :])])
def _auxiliary_matrix(beta, eps, Lm):
    """
    Computes all elements of the auxiliary matrix of shape 4x4.
    Parameters
    ----------
    beta
    eps
    Lm

    Returns
    -------

    """
    eps2m = 1./eps[2]
    eps4eps2m = eps[4]*eps2m
    eps5eps2m = eps[5]*eps2m
    
    Lm[0, 0] = (-beta*eps4eps2m)
    Lm[0, 1] = 1.-beta*beta*eps2m
    Lm[0, 2] = (-beta*eps5eps2m)
    Lm[0, 3] = 0.
    Lm[1, 0] = eps[0] - eps[4]*eps4eps2m
    Lm[1, 1] = Lm[0, 0]
    Lm[1, 2] = eps[3] - eps[5]*eps4eps2m
    Lm[1, 3] = 0.
    Lm[2, 0] = 0.
    Lm[2, 1] = 0.
    Lm[2, 2] = 0.
    Lm[2, 3] = -1.
    Lm[3, 0] = (-1.0*Lm[1, 2])
    Lm[3, 1] = (-1.0*Lm[0, 2])
    Lm[3, 2] = beta * beta + eps[5]*eps5eps2m - eps[1]
    Lm[3, 3] = 0.


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
def _power(field):
    tmp1 = (field[0].real * field[1].real + field[0].imag * field[1].imag)
    tmp2 = (field[2].real * field[3].real + field[2].imag * field[3].imag)
    return tmp1-tmp2 

@nb.njit([(NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])], cache = NUMBA_CACHE)
def _copy_sorted(alpha,fmat, out_alpha, out_fmat):
    i = 0
    j = 1
    for k in range(4):
        p = _power(fmat[:,k])
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
    

def _alphaf(beta, phi, epsv, epsa, out=None):
    """

    Parameters
    ----------
    beta
    phi
    epsv
    epsa
    out

    Returns
    -------

    """
    rv = rotation_vector2(phi)
    return _alphaf_vec(beta, phi, rv, epsv, epsa, _dummy_array, out=out)


def alphaf(beta, phi, epsv, epsa, out=None):
    """
    Computes alpha and field arrays (eigen values and eigen vectors arrays).
    Broadcasting rules apply.

    Parameters
    ----------
    beta : float
       The beta parameter of the field
    phi : float
       The phi parameter of the field
    epsv : array-like
       Dielectric tensor eigenvalues array.
    epsa : array_like
       Euler rotation angles (psi, theta, phi)
    out : ndarray, optional
       Output array

    Returns
    -------
    alpha, field arrays : (ndarray, ndarray)
        Eigen values and eigen vectors arrays
    """
    # Convert angle to rotation vector
    rotation_vector = rotation_vector2(phi)

    # Convert values to numpy arrays
    beta = np.asarray(beta, FDTYPE)
    phi = np.asarray(phi, FDTYPE)
    epsv = np.asarray(epsv, CDTYPE)
    epsa = np.asarray(epsa, FDTYPE)

    return _alphaf_vec(beta, phi, rotation_vector, epsv, epsa, _dummy_array, out=out)


def alphaffi(beta,phi,epsv,epsa,out = None):
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
        _alphaf(beta,phi,epsv,epsa, out = (a,f))
        inv(f,fi)
    else:
        a,f = _alphaf(beta,phi,epsv,epsa)
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

_numba_0_39_or_greater = False    

try:        
    major, minor, patch = map(int,nb.__version__.split("."))
    if major == 0 and minor >=39:
        _numba_0_39_or_greater = True
except:
    pass

if _numba_0_39_or_greater:
    @nb.vectorize([NCDTYPE(NCDTYPE,NFDTYPE)],
        target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)       
    def _phase_mat_vec(alpha,kd):
        return np.exp(1j*kd*alpha)

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
            out[i] = np.exp(1j*kd[0]*(alpha[i]))
                    
    phasem = _phase_mat_vec

def phase_mat(alpha, kd, mode = None, out = None):
    """Computes phse matrix from eigenvalue matrix alpha and wavenumber"""
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

@nb.guvectorize([(NCDTYPE[:], NFDTYPE[:])],
                    "(n)->()", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)       
def poynting(fvec, out):
    """Calculates a z-component of the poynting vector from the field vector"""
    assert fvec.shape[0] == 4
    tmp1 = (fvec[0].real * fvec[1].real + fvec[0].imag * fvec[1].imag)
    tmp2 = (fvec[2].real * fvec[3].real + fvec[2].imag * fvec[3].imag)
    out[0] = tmp1-tmp2   

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
    """Calculates a z-component of the poynting vector from the field vector"""
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
    fvec = np.asarray(fvec)
    p = poynting(fvec)
    return np.abs(p)


def projection_mat(fmat, fmati = None, mode = +1):
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
    Ai = inv(A, out = out)
    St,Sr = S_mat(fin, fout, fini = fini, overwrite_fin = overwrite_fin, mode = mode)
    Sti = inv(St, out = St)
    return dotmm(Sti,Ai, out = Ai)

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

def f_iso(n,beta=0.,phi = 0.):
    """Returns field matrix for isotropic layer of a given refractive index
    and beta, phi parameters"""
    epsv = refind2eps([n]*3)
    epsa = np.zeros(shape = (3,),dtype= FDTYPE)
    alpha, f = alphaf(beta,phi,epsv,epsa)    
    return f

def ffi_iso(n,beta=0.,phi = 0.):
    """Returns field matrix and inverse of the field matrix for isotropic layer 
    of a given refractive index and beta, phi parameters"""
    epsv = refind2eps([n]*3)
    epsa = np.zeros(shape = (3,),dtype= FDTYPE)
    alpha, f, fi = alphaffi(beta,phi,epsv,epsa)    
    return f,fi

def layer_mat(kd, epsv,epsa, beta = 0,phi = 0, method = "4x4", out = None):
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
    method : str
        One of `4x4` (4x4 berreman), `2x2` (2x2 jones) or `4x2` (4x4 single reflections)
    out : ndarray, optional
    
    Returns
    -------
    cmat : ndarray
        Characteristic matrix of the layer.
    """    
    if method == "2x2":
        alpha,f,fi = alphaEEi(beta,phi,epsv,epsa)
        pmat = phase_mat(alpha,kd)
    else:
        alpha,f,fi = alphaffi(beta,phi,epsv,epsa)
        pmat = phase_mat(alpha,-kd)
        if method in ("4x2","2x4"):
            pmat[...,1::2] = 0.        
    return dotmdm(f,pmat,fi,out = out)    

def stack_mat(kd,epsv,epsa, beta = 0, phi = 0, method = "4x4", out = None):
    """Computes a stack characteristic matrix M = M_1.M_2....M_n if method is
    4x4, 4x2(2x4) and a characteristic matrix M = M_n...M_2.M_1 if method is
    2x2.
    
    Note that this function calls :func:`layer_mat`, so numpy broadcasting 
    rules apply to kd[i], epsv[i], epsa[], beta and phi. 
    
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
    method : str
        One of `4x4` (4x4 berreman), `2x2` (2x2 jones) or `4x2` (4x4 single reflections)
    out : ndarray, optional
    
    Returns
    -------
    cmat : ndarray
        Characteristic matrix of the stack.
    """
    t0 = time.time()
    mat = None
    n = len(kd)
    indices = range(n)
    if method == "2x2":
        indices = reversed(indices)
    verbose_level = DTMMConfig.verbose
    if verbose_level > 1:
        print ("Building stack matrix.")
    for pi,i in enumerate(range(n)):
        print_progress(pi,n,level = verbose_level) 
        mat = layer_mat(kd[i],epsv[i],epsa[i],beta = beta, phi = phi, method = method, out = mat)
        if pi == 0:
            if out is None:
                out = mat.copy()
            else:
                out[...] = mat
        else:
            dotmm(out,mat,out)
    print_progress(n,n,level = verbose_level) 
    t = time.time()-t0
    if verbose_level >1:
        print("     Done in {:.2f} seconds!".format(t))  
    return out 


def system_mat(cmat,fmatin = None, fmatout = None, fmatini = None, out = None):
    """Computes a system matrix from a characteristic matrix Fin-1.C.Fout"""
    if fmatini is None:
        if fmatin is None:
            fmatin = f_iso(1,0,0)
        fmatini = inv(fmatin)
    if fmatout is None:
        fmatout = fmatin
    out = dotmm(fmatini,cmat,out = out)
    return dotmm(out,fmatout,out = out)    

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

def transmit(fvec_in, cmat, fmatin = None, fmatout = None, fmatini = None, fmatouti = None, fvec_out = None):
    """Transmits field vector using 4x4 method.
    
    This functions takes a field vector that describes the input field and
    computes the output transmited field and also updates the input field 
    with the reflected waves.
    """
    b = np.broadcast(fvec_in[..., None],cmat, fmatin, fmatout)
    
    if fmatini is None:
        if fmatin is None:
            fmatin = f_iso(1,0,0)
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
        
    smat = system_mat(cmat,fmatini = fmatini, fmatout = fmatout)
        
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

def polarizer4x4(jones, fmat, out = None):
    """Returns a polarizer matrix from a given jones vector and field matrix. 
    
    Numpy broadcasting rules apply.
    
    Parameters
    ----------
    jones : array_like
        A length two array describing the jones vector. Jones vector should
        be normalized.
    fmat : array_like
        A field matrix array of the medium.
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

def field4(fmat, jones = (1,0),  amplitude = 1., mode = +1, out = None):
    """Build field vector form a given polarization state, amplitude and mode.
    Numpy broadcasting rules apply."""
    jones = np.asarray(jones)
    amplitude = np.asarray(amplitude)
    
    c,s = jones[...,0], jones[...,1] 
    b = np.broadcast(fmat[...,0,0], c, amplitude)
    shape = b.shape + (4,)
    
    fmat = normalize_f(fmat)
    
    fvec = np.zeros(shape,CDTYPE)
    if mode == +1:
        fvec[...,0] = c
        fvec[...,2] = s
    elif mode == -1:
        fvec[...,1] = c
        fvec[...,3] = s 
    else:
        raise ValueError("Unknown propagation mode.")
        
    fvec = dotmv(fmat,fvec, out = out)
    a = np.asarray(amplitude)[...,None]
    out = np.multiply(fvec, a ,out = fvec) 

    return out 
    
__all__ = ["alphaf","alphaffi","phasem", "phase_mat", "field4"]