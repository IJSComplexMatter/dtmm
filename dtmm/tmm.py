"""
4x4 transfer matrix method functions. 
"""

from __future__ import absolute_import, print_function, division

import numpy as np

from dtmm.conf import NCDTYPE,NFDTYPE, CDTYPE, FDTYPE,NUDTYPE,  NUMBA_TARGET, \
                        NUMBA_PARALLEL, NUMBA_CACHE, NUMBA_FASTMATH, cached_function
from dtmm.rotation import  _calc_rotations_uniaxial, _calc_rotations, _rotate_diagonal_tensor
from dtmm.linalg import _inv4x4, _dotmr2, _dotr2m, dotmdm, dotmm, inv, dotmv
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
    "Computes all non-zero elements of the auxiliary matrix of shape 4x4."
    eps2m = 1./eps[2]
    eps4eps2m = eps[4]*eps2m
    eps5eps2m = eps[5]*eps2m
    
    Lm[0,0] = (-beta*eps4eps2m)
    Lm[0,1] = 1.-beta*beta*eps2m#z0-z0*beta*beta/eps[2,2]
    Lm[0,2] = (-beta*eps5eps2m)
    Lm[0,3] = 0.
    Lm[1,0] = eps[0]- eps[4]*eps4eps2m#eps[0,0]/z0- eps[0,2]*eps[2,0]/z0/eps[2,2]
    Lm[1,1] = Lm[0,0]
    Lm[1,2] = eps[3]- eps[5]*eps4eps2m#eps[0,1]/z0- eps[0,2]*eps[2,1]/z0/eps[2,2]
    Lm[1,3] = 0.
    Lm[2,0] = 0.
    Lm[2,1] = 0.
    Lm[2,2] = 0.
    Lm[2,3] = -1. #(-z0)
    Lm[3,0] = (-1.0*Lm[1,2])
    Lm[3,1] = (-1.0*Lm[0,2])
    Lm[3,2] = beta * beta + eps[5]*eps5eps2m - eps[1]  #beta * beta / z0 + eps[1,2]*eps[2,1]/eps[2,2]/z0- eps[1,1]/z0  
    Lm[3,3] = 0.  

@nb.njit([(NCDTYPE,NCDTYPE[:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def __alpha_iso(aout, alpha):
    alpha[0] = aout
    alpha[1] = -aout
    alpha[2] = aout
    alpha[3] = -aout  
    
#@nb.njit([(NCDTYPE,NCDTYPE[:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
#def __alpha_iso2(aout, alpha):
#    alpha[0] = aout
#    alpha[1] = aout 

@nb.njit([(NCDTYPE,NCDTYPE,NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def __fmat_iso(gpout, gsout, F):
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

#@nb.njit([(NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
#def __unity2(F):
#    F[0,0] = 1. 
#    F[0,1] = 0.
#    F[1,0] = 0.
#    F[1,1] = 1. 
    
@nb.njit([(NCDTYPE,NCDTYPE,NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def __fimat_iso(gpout, gsout, Fi):    
    Fi[0,0] = 1. 
    Fi[1,0] = 1.
    Fi[2,0] = 0.
    Fi[3,0] = 0.
    Fi[0,1] = 1. / gpout 
    Fi[1,1] = -1. / gpout 
    Fi[2,1] = 0.
    Fi[3,1] = 0.
    Fi[0,2] = 0.
    Fi[1,2] = 0.
    Fi[2,2] = 1.
    Fi[3,2] = 1.
    Fi[0,3] = 0.
    Fi[1,3] = 0.
    Fi[2,3] = 1. / gsout 
    Fi[3,3] = -1 / gsout  

@nb.njit([(NFDTYPE,NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _alphaffi_iso(beta,eps0,alpha,F,Fi):
    n = eps0[0]**0.5
    aout = sqrt(n**2-beta**2)
    if aout != 0.:
        gpout = n**2/aout
        gsout = -aout
        __alpha_iso(aout, alpha)
        __fmat_iso(gpout,gsout,F)
        __fimat_iso(gpout,gsout,Fi)
    else:
        F[...]=0.
        Fi[...] = 0.
        alpha[...] = 0.

#@nb.njit([(NFDTYPE,NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
#def _alphaffi_iso2(beta,eps0,alpha,F,Fi):
#    n = eps0[0]**0.5
#    aout = sqrt(n**2-beta**2)
#    __alpha_iso2(aout, alpha)
#    __unity2(F)
#    __unity2(Fi)

@nb.njit([(NFDTYPE,NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _alphaf_iso(beta,eps0,alpha,F):
    n = eps0[0]**0.5
    aout = sqrt(n**2-beta**2)
    if aout != 0.:
        gpout = n**2/aout
        gsout = -aout
        __alpha_iso(aout, alpha)
        __fmat_iso(gpout,gsout,F)
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
        
        #evp =  csqrt(-1.*ev02)*(0.+1j)
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
        
        F[0,3] = -evssf#*(-1) 
        F[1,3] = eps11sf#*(-1) 
        F[2,3] = evscf #*(-1) 
        F[3,3] = eps11cf#*(-1) 
        
        F[0,0] = -cf
        F[1,0] = -evpp *cf
        F[2,0] = -sf
        F[3,0] = evpp *sf 
        
        F[0,1] = -cf
        F[1,1] = -evpm *cf
        F[2,1] = -sf
        F[3,1] = evpm *sf    
        
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
        F[2,2] = (evscfst - ctbeta)
        F[3,2] = (evsctbeta - ev02cfst)
  
        F[0,3] = evssfst
        F[1,3] = (-eps11sfst)
        F[2,3] = (-evscfst - ctbeta)
        F[3,3] = (-ev02cfst-evsctbeta)
        
        F[0,0] = (-evpp*ctbetaeps11 + ev02cfsteps11)#*(-1)
        F[1,0] = (evpp *cfst - ctbeta)#*(-1)
        F[2,0] = (sfst)#*(-1)
        F[3,0] = -evpp *sfst#   *(-1) 
        
        F[0,1] = (-evpm*ctbetaeps11 + ev02cfsteps11)
        F[1,1] = (evpm *cfst - ctbeta)
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
        

@nb.njit()    
def _is_isotropic(eps):
    return (eps[0] == eps[1] and eps[1]==eps[2])

@nb.njit()     
def _is_uniaxial(eps):
    return (eps[0] == eps[1])

@nb.njit([(NCDTYPE[:,:],NCDTYPE[:,:])])
def _copy_4x4(ain,aout):
    for i in range(4):
        for j in range(4):
            aout[i,j] = ain[i,j]

@nb.njit([(NCDTYPE[:],NCDTYPE[:])])
def _copy_4(ain,aout):
    for i in range(4):
        aout[i] = ain[i]
        
#@nb.njit([(NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])])        
#def _copy_sorted(alphain, fin, alphaout, fout):
#    first, second, third, fourth = 0,1,2,3
#    if alphain[0].real <0:
#        first, second = second, first
#    if alphain[1].real >=0:
#        
#
#        
#    alphaout[i] = alphain[0]
#    for j in range(4):
#        aout[i,j] = ain[i,j]
#
#    if alphain[1].real >=0:
#        if i == 0:
#            i =
#    else:
#        i = 1
#        
#    alphaout[i] = alphain[0]
#    for j in range(4):
#        aout[i,j] = ain[i,j]
    

@nb.guvectorize([(NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])],
                 "(),(),(l),(k),(n)->(n),(n,n),(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE)
def _alphaffi_vec(beta,phi,epsa,epsv,dummy,alpha,F,Fi):
    #select the fastest algorithm
    
    if _is_isotropic(epsv):
        eps = F[3] 
        _uniaxial_order(0.,epsv,eps) #store caluclated eps values in Fi[3]
        _alphaffi_iso(beta[0],eps,alpha,F,Fi)
    elif _is_uniaxial(epsv):
        R = F.real
        eps = F[3] 
        _uniaxial_order(1.,epsv,eps)
        _calc_rotations_uniaxial(phi[0],epsa,R) #store rotation matrix in Fi.real[0:3,0:3]
        _alphaf_uniaxial(beta[0],eps,R,alpha,F)
        _inv4x4(F,Fi)
    else:#biaxial case
        R = Fi.real 
        eps = F.ravel() #reuse tmp memory
        _calc_rotations(phi[0],epsa,R) #store rotation matrix in Fi.real[0:3,0:3]
        _rotate_diagonal_tensor(R,epsv,eps)
        _auxiliary_matrix(beta[0],eps,Fi) #calculate Lm matrix and put it to Fi
        alpha0,F0 = np.linalg.eig(Fi)
        _copy_4(alpha0,alpha)#copy data
        _copy_4x4(F0,F)#copy data
        _inv4x4(F,Fi)   
        
#@nb.njit([(NFDTYPE,NCDTYPE[:,:])])
#def _rm4(phi,out):
#    c = np.cos(phi)
#    s = np.sin(phi)
#    out[...] = 0.
#    out[0,0] = c
#    out[0,2] = -s
#    out[1,1] = c
#    out[1,3] = s
#    out[2,0] = s
#    out[2,2] = c
#    out[3,1] = -s
#    out[3,3] = c

@nb.guvectorize([(NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])],
                 "(),(),(m),(l),(k),(n)->(n),(n,n),(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _alphaffi_xy_vec(beta,phi,rv, epsa,epsv,dummy,alpha,F,Fi):
    #Fi is a 4x4 matrix... we can use 3x3 part for Rotation matrix and Fi[3] for eps  temporary data
    
    if _is_isotropic(epsv):
        eps = F[3] 
        #_uniaxial_order(0.,epsv,eps) #store caluclated eps values in Fi[3]
        _alphaffi_iso(beta[0],epsv,alpha,F,Fi)
        _dotr2m(rv,F,F)
        _dotmr2(Fi,rv,Fi)
    elif _is_uniaxial(epsv):
        R = F.real
        eps = F[3] 
        #_uniaxial_order(1.,epsv,eps)
        _calc_rotations_uniaxial(phi[0],epsa,R) #store rotation matrix in Fi.real[0:3,0:3]
        _alphaf_uniaxial(beta[0],epsv,R,alpha,F)
        _dotr2m(rv,F,F)
        _inv4x4(F,Fi)
    else:#biaxial case
        R = Fi.real 
        eps = F.ravel() #reuse F memory (eps is length 6 1D array)
        _calc_rotations(phi[0],epsa,R) #store rotation matrix in Fi.real[0:3,0:3]
        _rotate_diagonal_tensor(R,epsv,eps)
        _auxiliary_matrix(beta[0],eps,Fi) #calculate Lm matrix and put it to Fi
        #_rm4(-phi[0],F)
        #_dotmm(Fi,F,Fi)
        #_rm4(phi[0],F)
        #_dotmm(F,Fi,Fi)
        alpha0,F0 = np.linalg.eig(Fi)
        _copy_4(alpha0,alpha)#copy data
        #_copy_4x4(F0,F)#copy data
        #F0 = np.linalg.inv(F)
        #_copy_4x4(F0,Fi)#copy data
        _dotr2m(rv,F0,F)
        _inv4x4(F,Fi)    

@nb.guvectorize([(NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:])],
                 "(),(),(m),(l),(k),(n)->(n),(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _alphaf_xy_vec(beta,phi,rv, epsa,epsv,dummy,alpha,F):
    #Fi is a 4x4 matrix... we can use 3x3 part for Rotation matrix and Fi[3] for eps  temporary data
    
    if _is_isotropic(epsv):
        eps = F[3] 
        #_uniaxial_order(0.,epsv,eps) #store caluclated eps values in Fi[3]
        _alphaf_iso(beta[0],epsv,alpha,F)
        _dotr2m(rv,F,F)
    elif _is_uniaxial(epsv):
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
        _copy_4(alpha0,alpha)#copy data
        _dotr2m(rv,F0,F)
        
    
_dummy_array = np.empty((4,),CDTYPE)
_dummy_array2 = np.empty((9,),CDTYPE)
    
#def alpha_F(beta,eps0,R,*args,**kw):
#    return _alpha_F_vec(beta,eps0,R,_dummy_array,*args,**kw)

def alphaffi(beta,phi,element,eps0,*args,**kw):
    return _alphaffi_vec(beta,phi,element,eps0,_dummy_array,*args,**kw)

#def alphaffi_xy_2(beta,phi,element,eps0,*args,**kw):
#    return _alphaffi_xy_vec_iso(beta,phi,eps0,_dummy_array,*args,**kw)

#@cached_function
def alphaffi_xy(beta,phi,element,eps0,*args,**kw):
    rv = rotation_vector2(phi) 
    return _alphaffi_xy_vec(beta,phi,rv,element,eps0,_dummy_array,*args,**kw)

#
#def field_matrices(beta,phi,epsv, epsa,  with_inverse = False, out = None):
#    rv = rotation_vector2(phi) 
#    if with_inverse:
#        return _alphaffi_xy_vec(beta,phi,rv,epsa,epsv,_dummy_array, out = out)    
#    else:
#        return _alphaf_xy_vec(beta,phi,rv,epsa,epsv,_dummy_array, out = out)    
#    
#def E_matrices(beta,phi,epsv, epsa,  with_inverse = False, out = None):
#    alpha, f = field_matrices(beta,phi,epsv, epsa)
#    a = alpha[...,::2]
#    j = f[...,::2,::2]
#    ji = inv(j)
#    if out is not None:
#        aout, jout, jiout = out
#        aout[...] = a
#        jout[...] = j
#        jiout[...] = ji
#        return out
#    else:
#        return a,j,ji    

def alphaf_xy(beta,phi,element,eps0,*args,**kw):
    rv = rotation_vector2(phi) 
    return _alphaf_xy_vec(beta,phi,rv,element,eps0,_dummy_array,*args,**kw)

#def fmat2jonesmat(fmat, copy = True):
#    """Converts a 4x4 field matrix to 2x2 jones matrix"""
#    j = f[...,::2,::2]
#    if copy == True:
#        return j.copy()
#    else:
#        return j
    
    
def alphajji_xy(beta,phi,element,eps0, out = None):
    rv = rotation_vector2(phi) 
    alpha,f = _alphaf_xy_vec(beta,phi,rv,element,eps0,_dummy_array)
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
    

#def alphaffi_xy_iso(beta,phi,element,eps0,*args,**kw):
#    rv = rotation_vector2(phi) #+ np.random.randn(2)
#    #print (rv)
#    #x,y = rv[...,0], rv[...,1]
#    #rv[...,0] = y
#    #rv[...,1] = x
#    return _alphaffi_xy_vec_iso2(beta,phi,rv,element,eps0,_dummy_array,*args,**kw)
#


#def alphaffi_xy2(beta,phi,element,eps0,mask,*args,**kw):
#    rv = rotation_vector2(phi)
#    #x,y = rv[...,0], rv[...,1]
#    #rv[...,0] = y
#    #rv[...,1] = x
#    return _alphaffi_xy_vec2(beta,phi,rv,element,eps0,mask,_dummy_array,*args,**kw)

def alphaffi0(beta,phi,n = 1.,*args,**kw):
    element = np.array([0.,0.,0.], dtype = FDTYPE)
    eps0 = np.array([n]*3, dtype = FDTYPE)
    return _alphaffi_vec(beta,phi,element,eps0,_dummy_array,*args,**kw)

#def deltaffi(beta,phi,element,eps0,*args,**kw):
#    return _deltaffi_vec(beta,phi,element,eps0,_dummy_array,*args,**kw)    
#
#def alpha_FFi(beta,eps0,R,*args,**kw):
#    return _alpha_FFi_vec(beta,eps0,R,_dummy_array,*args,**kw)

def field_matrix(beta,phi,n = 1):
    eps = np.array([n,n,n],dtype = CDTYPE)**2
    element = (0.,np.pi/2,0.)
    alpha,F,Fi = alphaffi(beta,phi,element,eps)
    return F

def ifield_matrix(beta,phi,n = 1):
    eps = np.array([n,n,n],dtype = CDTYPE)**2
    element = (0.,np.pi/2,0.)
    alpha,F,Fi = alphaffi(beta,phi,element,eps)
    return Fi

#@nb.guvectorize([(NCDTYPE[:,:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NCDTYPE[:,:,:])],"(n,m),(),(l),(), (k)->(k,n,m)")
#def _field_vector(beam, k0, pol, n, dummy, out):
#    beta, phi = mean_betaphi(beam, k0)
#    F = field_matrix(beta,n)
#    if k0 > 0:
#        out[1] = 0.
#        out[3] = 0.
#        out[0] = beam*pol[0]
#        out[2] = beam*pol[1]
#    else:
#        out[0] = 0.
#        out[2] = 0.
#        out[1] = beam*pol[0]
#        out[3] = beam*pol[1]        
#    dotm1f(F, out, out)
    
    
@nb.guvectorize([(NCDTYPE[:],NFDTYPE[:], NCDTYPE[:])],
                "(n),()->(n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def phasem_t(alpha,kd,out):
    out[0] = np.exp(1j*kd[0]*(alpha[0].real))
    out[1] = 0.
    out[2] = np.exp(1j*kd[0]*(alpha[2].real)) 
    out[3] = 0.
    
@nb.guvectorize([(NCDTYPE[:],NFDTYPE[:], NCDTYPE[:])],
                "(n),()->(n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def phasem_r(alpha,kd,out):
    out[0] = 0.
    out[1] = np.exp(1j*kd[0]*(alpha[1].real))  
    out[2] = 0.
    out[3] = np.exp(1j*kd[0]*(alpha[3].real))  
    
@nb.guvectorize([(NCDTYPE[:],NFDTYPE[:], NCDTYPE[:])],
                "(n),()->(n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)       
def phase_mat(alpha,kd,out):
##    f0 = 1j*kd[0]*(alpha[0].real)
##    f1 = 1j*kd[0]*(alpha[1].real)
##    f2 = 1j*kd[0]*(alpha[2].real)
##    f3 = 1j*kd[0]*(alpha[3].real)
    for i in range(alpha.shape[0]):
        out[i] = np.exp(1j*kd[0]*(alpha[i]))
        
phasem = phase_mat
#
##    f0 = 1j*kd[0]*(alpha[0])
##    f1 = 1j*kd[0]*(alpha[1])
##    f2 = 1j*kd[0]*(alpha[2])
##    f3 = 1j*kd[0]*(alpha[3])
##    
##    out[0] = np.exp(f0)
##    out[1] = np.exp(f1)
##    out[2] = np.exp(f2)
##    out[3] = np.exp(f3)

#@nb.vectorize([NCDTYPE(NCDTYPE,NFDTYPE)], target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)       
#def _phasem(alpha,kd):
#    return np.exp(1j*kd*(alpha))
#
#def phasem(alpha,kd,**kwds):
#    kd = np.asarray(kd, dtype = FDTYPE)[...,None]
#    return _phasem(alpha,kd,**kwds)
        
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

def ffi_iso(n,beta=0.,phi = 0.):
    epsv = refind2eps([n]*3)
    epsa = np.zeros(shape = (3,),dtype= FDTYPE)
    alpha, f, fi = alphaffi_xy(beta,phi,epsa,epsv)    
    return f,fi

def layer_mat(k0, d, epsv,epsa, beta = 0,phi = 0, out = None):
    """Computes characteristic matrix F.P.Fi"""
    alpha,f,fi = alphaffi_xy(beta,phi,epsa,epsv)
    kd = -k0*d
    pmat = phasem(alpha,kd)
    return dotmdm(f,pmat,fi,out = out)

def stack_mat(k0,stack, beta = 0, phi = 0, out = None):
    d,epsv,epsa = stack
    mat = None
    n = len(d)
    verbose_level = 1
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
    return out 

def system_mat(cmat,beta=0.,phi = 0.,nin=1.,nout = 1., out = None):
    f,fi = ffi_iso(nin,beta, phi)
    out = dotmm(fi,cmat,out = out)
    f,fi = ffi_iso(nout,beta, phi)
    return dotmm(out,f,out = out)    

def reflection_mat(smat, out = None):
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

def transmit(fvec, cmat, beta = 0, phi = 0, nin = 1, nout = 1, out = None):
    smat = system_mat(cmat, beta = beta, phi = phi, nin = nin, nout = nout)
    f1,f1i = ffi_iso(nin,beta, phi)
    f2,f2i = ffi_iso(nout,beta, phi)
    
    avec = dotmv(f1i,fvec)
    a = np.zeros_like(avec)
    a[...,0] = avec[...,0]
    a[...,2] = avec[...,2]

    if out is not None:
        bvec = dotmv(f2i,out)
        a[...,1::2] = bvec[...,1::2] 
    else:
        bvec = np.zeros_like(avec)

    r = reflection_mat(smat)
    out = dotmv(r,a, out = out)
    
    avec[...,1::2] = out[...,1::2]
    bvec[...,::2] = out[...,::2]
        
    dotmv(f1,avec,out = fvec)    
    return dotmv(f2,bvec,out = out)


__all__ = ["alphaffi_xy","alphaffi","phasem_t", "phasem_r","phasem"]