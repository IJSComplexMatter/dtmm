"""
4x4 and 2x2 transfer matrix method functions. 
"""

from __future__ import absolute_import, print_function, division

import numpy as np

from dtmm.conf import NCDTYPE,NFDTYPE, CDTYPE, FDTYPE,NUDTYPE,  NUMBA_TARGET, \
                        NUMBA_PARALLEL, NUMBA_CACHE, NUMBA_FASTMATH, cached_function,DTMMConfig
from dtmm.rotation import  _calc_rotations_uniaxial, _calc_rotations, _rotate_diagonal_tensor
from dtmm.linalg import _dotr2m, dotmdm, dotmm, inv, dotmv
from dtmm.data import _uniaxial_order, refind2eps
from dtmm.rotation import rotation_vector2
from dtmm.print_tools import print_progress

import numba as nb
from numba import prange

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
    n = eps0[0]**0.5
    aout = sqrt(n**2-beta**2)
    if aout != 0.:
        gpout = n**2/aout
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
    

def alphaf(beta,phi,epsv,epsa,out = None):
    rv = rotation_vector2(phi) 
    return _alphaf_vec(beta,phi,rv,epsv,epsa,_dummy_array,out = out)

def alphaffi(beta,phi,epsv,epsa,out = None):
    if out is not None:
        a,f,fi = out
        alphaf(beta,phi,epsv,epsa, out = (a,f))
        inv(f,fi)
    else:
        a,f = alphaf(beta,phi,epsv,epsa)
        fi = inv(f)
    return a,f,fi
        
def alphajji(beta,phi,epsv,epsa, out = None):
    rv = rotation_vector2(phi) 
    alpha,f = _alphaf_vec(beta,phi,rv,epsv,epsa,_dummy_array)
    a = alpha[...,::2]
    j = f[...,::2,::2]
    ji = inv(j)
    if out is not None:
        aout, jout, jiout = out
        aout[...] = a
        jout[...] = j
        jiout[...] = ji
        return out
    else:
        return a,j,ji
 
def alphae(beta,phi,epsv,epsa, out = None):
    alpha,f = alphaf(beta,phi,epsv,epsa)
    e = E_mat(f)

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
        return np.exp(1j*kd*(alpha))

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
    kd = np.asarray(kd, dtype = FDTYPE)
    if out is None:
        b = np.broadcast(alpha,kd[...,None])
        out = np.empty(b.shape, dtype = CDTYPE)
    if mode == "forward":
        phasem(alpha[...,::2],kd, out = out[...,::2])
        out[...,1::2] = 0.
    elif mode == "backward":
        phasem(alpha[...,1::2],kd,out = out[...,1::2])
        out[...,::2] = 0.
    elif mode is None:
        out = phasem(alpha,kd, out = out) 
    else:
        raise ValueError("Unknown propagation mode.")
    return out  
       
def transmission_mat(fin, fout, fini = None, out = None):
    if fini is None:
        fini = inv(fin)
    S = dotmm(fini,fout)
    A1 = fin[...,::2,::2]
    A2 = fout[...,::2,::2]
    A = S[...,::2,::2]
    Ai = inv(A, out = out)
    A1i = inv(A1)
    return dotmm(dotmm(A2,Ai, out = Ai),A1i, out = Ai)

def E_mat(fmat, mode = "forward", copy = False):
    if mode == "forward":
        e = fmat[...,::2,::2]
    elif mode == "backward":
        e = fmat[...,::2,1::2]
    else:
        raise ValueError("Unknown propagation mode.")
    return e.copy() if copy else e     

def EEi_mat(fmat,mode = "forward", copy = False):
    e = E_mat(fmat, mode, copy)
    return e,inv(e)
    
def E2H_mat(fmat, out = None):  
    A = fmat[...,::2,::2]
    B = fmat[...,1::2,::2]
    Ai = inv(A, out = out)
    return dotmm(B,Ai, out = Ai)  

def ffi_iso(n,beta=0.,phi = 0.):
    epsv = refind2eps([n]*3)
    epsa = np.zeros(shape = (3,),dtype= FDTYPE)
    alpha, f, fi = alphaffi(beta,phi,epsa,epsv)    
    return f,fi

def layer_mat(k0, d, epsv,epsa, beta = 0,phi = 0, out = None):
    """Computes characteristic matrix M=F.P.Fi"""
    alpha,f,fi = alphaffi(beta,phi,epsa,epsv)
    kd = -k0*d
    pmat = phasem(alpha,kd)
    return dotmdm(f,pmat,fi,out = out)

def stack_mat(k0,stack, beta = 0, phi = 0, out = None):
    """Computes a stack characteristic matrix M = M_n...M_2.M_1"""
    d,epsv,epsa = stack
    mat = None
    n = len(d)
    verbose_level = DTMMConfig.verbose
    for pi,i in enumerate(reversed(range(len(d)))):
        print_progress(pi,n,level = verbose_level) 
        mat = layer_mat(k0,d[i],epsv[i],epsa[i],beta = beta, phi = phi, out = mat)
        if pi == 0:
            if out is None:
                out = mat.copy()
            else:
                out[...] = mat
        else:
            dotmm(out,mat,out)
    print_progress(n,n,level = verbose_level) 
    return out 

def system_mat(cmat,beta=0.,phi = 0.,nin=1.,nout = 1., out = None):
    """Computes a system matrix from a characteristic matrix Fin-1.C.Fout"""
    f,fi = ffi_iso(nin,beta, phi)
    out = dotmm(fi,cmat,out = out)
    f,fi = ffi_iso(nout,beta, phi)
    return dotmm(out,f,out = out)    

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

    


def transmit(fvec_in, cmat, beta = 0, phi = 0, nin = 1, nout = 1, fvec_out = None):
    """Transmits field vector.
    
    This functions takes a field vector that describes the input field and
    computes the output transmited field and also updates the input field 
    with the reflected waves.
    """
    smat = system_mat(cmat, beta = beta, phi = phi, nin = nin, nout = nout)
    f1,f1i = ffi_iso(nin,beta, phi)
    f2,f2i = ffi_iso(nout,beta, phi)
    
    avec = dotmv(f1i,fvec_in)
    a = np.zeros_like(avec)
    a[...,0] = avec[...,0]
    a[...,2] = avec[...,2]

    if fvec_out is not None:
        bvec = dotmv(f2i,fvec_out)
        a[...,1::2] = bvec[...,1::2] 
    else:
        bvec = np.zeros_like(avec)

    r = reflection_mat(smat)
    out = dotmv(r,a, out = fvec_out)
    
    avec[...,1::2] = out[...,1::2]
    bvec[...,::2] = out[...,::2]
        
    dotmv(f1,avec,out = fvec_in)    
    return dotmv(f2,bvec,out = out)


__all__ = ["alphaffi","phasem"]