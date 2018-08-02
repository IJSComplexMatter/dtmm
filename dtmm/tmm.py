# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:44:09 2017

@author: andrej

E-H field functions

"""

from __future__ import absolute_import, print_function, division

import numpy as np

from dtmm.conf import NCDTYPE,NFDTYPE, CDTYPE, NUDTYPE,  NUMBA_TARGET, NUMBA_PARALLEL, NUMBA_CACHE, NUMBA_FASTMATH
#from dtmm.wave import mean_betaphi, betaphi
from dtmm.rotation import  _calc_rotations_uniaxial, _calc_rotations, _rotate_diagonal_tensor
from dtmm.linalg import _inv4x4, _dotmr2, _dotr2m, _dotmm
from dtmm.data import _uniaxial_order
from dtmm.rotation import rotation_vector2



import numba as nb
from numba import prange

if NUMBA_PARALLEL == False:
    prange = range

sqrt = np.sqrt


#def calc_Lm(eps,beta, output = None):
#    """Creates Lm matrix from a given eps tensor of shape = (6,) and a given beta parameter.
#    If output is given it must be initialized to zero values. This function only fills non-zero values for speed...
#     
#    >>> R = rotation_matrix(0.12,0.245,0.78)
#    >>> eps0 = np.array([1.3,1.4,1.5], dtype = 'complex')
#    >>> eps = rotate_diagonal_tensor(R, eps0) 
#    >>> beta = 0.2
#    >>> out = calc_Lm(eps,beta)
#    
#    the same can be calculated from:
#        
#    >>> eps = tensor_to_matrix(eps)
#    
#    >>> Lm = np.zeros((4,4),dtype = CDTYPE)
#    >>> Lm[0,0] = (-beta*eps[2,0]/eps[2,2])
#    >>> Lm[0,1] = 1.-beta*beta/eps[2,2]#z0-z0*beta*beta/eps[2,2]
#    >>> Lm[0,2] = (-beta*eps[2,1]/eps[2,2])
#    >>> Lm[1,0] = eps[0,0]- eps[0,2]*eps[2,0]/eps[2,2]#eps[0,0]/z0- eps[0,2]*eps[2,0]/z0/eps[2,2]
#    >>> Lm[1,1] = Lm[0,0]
#    >>> Lm[1,2] = eps[0,1]- eps[0,2]*eps[2,1]/eps[2,2]#eps[0,1]/z0- eps[0,2]*eps[2,1]/z0/eps[2,2]
#    >>> Lm[2,3] = -1. #(-z0)
#    >>> Lm[3,0] = (-1.0*Lm[1,2])
#    >>> Lm[3,1] = (-1.0*Lm[0,2])
#    >>> Lm[3,2] = beta * beta + eps[1,2]*eps[2,1]/eps[2,2]- eps[1,1]  #beta * beta / z0 + eps[1,2]*eps[2,1]/eps[2,2]/z0- eps[1,1]/z0  
#    
#    >>> np.allclose(Lm, out)
#    True
#    """
#    if output is None:
#        output = np.zeros((4,4),CDTYPE) # output must be zero-valued 
#    output = _output_matrix(output,(4,4),CDTYPE)
#    eps = _input_matrix(eps,(6,),CDTYPE)
#    _calc_Lm(eps,beta,output)
#    return output

@nb.njit([(NFDTYPE,NCDTYPE[:],NCDTYPE[:,:])])                                                                
def _auxiliary_matrix(beta,eps,Lm):
    "Computes all non-zero elements of the auxiliary matrix of shape 4x4."
    eps2m = 1/eps[2]
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

@nb.njit([(NFDTYPE,NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _alphaffi_iso(beta,eps0,alpha,F,Fi):
    n = eps0[0]**0.5
    aout = sqrt(n**2-beta**2)
    if aout != 0:
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
    else:
        F[...]=0.
        Fi[...] = 0.
        alpha[...] = 0.

@nb.njit([(NFDTYPE,NCDTYPE[:],NFDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _alpha_F(beta,eps0,R,alpha,F): 

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

            

@nb.guvectorize([(NFDTYPE[:],NCDTYPE[:],NFDTYPE[:,:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:])],
                 "(),(n),(m,m),(k)->(k),(k,k)", target = NUMBA_TARGET, cache = NUMBA_CACHE)
def _alpha_F_vec(beta,eps0,R,dummy,alpha,F):
    _alpha_F(beta[0],eps0,R,alpha,F)


@nb.guvectorize([(NFDTYPE[:],NCDTYPE[:],NFDTYPE[:,:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])],
                 "(),(n),(m,m),(k)->(k),(k,k),(k,k)", target = NUMBA_TARGET, cache = NUMBA_CACHE)
def _alpha_FFi_vec(beta,eps0,R,dummy,alpha,F,Fi):
    _alpha_F(beta[0],eps0,R,alpha,F)
    _inv4x4(F,Fi)
    
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

@nb.guvectorize([(NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])],
                 "(),(),(l),(k),(n)->(n),(n,n),(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE)
def _alphaffi_vec(beta,phi,epsa,epsv,dummy,alpha,F,Fi):
    #select the fastest algorithm
    
    if _is_isotropic(epsv):
        eps = Fi[3] 
        _uniaxial_order(0.,epsv,eps) #store caluclated eps values in Fi[3]
        _alphaffi_iso(beta[0],eps,alpha,F,Fi)
    elif _is_uniaxial(epsv):
        R = Fi.real
        eps = Fi[3] 
        _uniaxial_order(1.,epsv,eps)
        _calc_rotations_uniaxial(phi[0],epsa,R) #store rotation matrix in Fi.real[0:3,0:3]
        _alpha_F(beta[0],eps,R,alpha,F)
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
        eps = Fi[3] 
        _uniaxial_order(0.,epsv,eps) #store caluclated eps values in Fi[3]
        _alphaffi_iso(beta[0],eps,alpha,F,Fi)
        _dotr2m(rv,F,F)
        _dotmr2(Fi,rv,Fi)
    elif _is_uniaxial(epsv):
        R = Fi.real
        eps = Fi[3] 
        _uniaxial_order(1.,epsv,eps)
        _calc_rotations_uniaxial(phi[0],epsa,R) #store rotation matrix in Fi.real[0:3,0:3]
        _alpha_F(beta[0],eps,R,alpha,F)
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
        
@nb.guvectorize([(NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])],
                 "(),(),(m),(l),(k),(n)->(n),(n,n),(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE)
def _alphaffi_xy_vec_iso2(beta,phi,rv, element,eps0,dummy,alpha,F,Fi):
    #Fi is a 4x4 matrix... we can use 3x3 part for Rotation matrix and Fi[3] for eps  temporary data
    R = Fi.real 
    eps = Fi[3] 
    _uniaxial_order(0.,eps0,eps) #store caluclated eps values in Fi[3]
    _calc_rotations_uniaxial(phi[0],element,R) #store rotation matrix in Fi.real[0:3,0:3]
    _alphaffi_iso(beta[0],eps,alpha,F,Fi)
    _dotr2m(rv,F,F)
    _dotmr2(Fi,rv,Fi)

@nb.guvectorize([(NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])],
                 "(),(),(k),(n)->(n),(n,n),(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE)
def _alphaffi_xy_vec_iso(beta,phi,eps0,dummy,alpha,F,Fi):
    #Fi is a 4x4 matrix... we can use 3x3 part for Rotation matrix and Fi[3] for eps  temporary data
    eps = Fi[3] 
    _uniaxial_order(0.,eps0,eps) #store caluclated eps values in Fi[3]
    _alphaffi_iso(beta[0],eps,alpha,F,Fi)


@nb.guvectorize([(NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:,:],NUDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])],
                 "(),(),(m),(l),(k,o),(),(n)->(n),(n,n),(n,n)", target = NUMBA_TARGET, cache = NUMBA_CACHE)
def _alphaffi_xy_vec2(beta,phi,rv, element,eps0,mask,dummy,alpha,F,Fi):
    #Fi is a 4x4 matrix... we can use 3x3 part for Rotation matrix and Fi[3] for eps  temporary data
    R = Fi.real 
    eps = Fi[3] 
    i = mask[0]
    assert i < eps0.shape[0]
    _uniaxial_order(1.,eps0[i],eps) #store caluclated eps values in Fi[3]
    _calc_rotations_uniaxial(phi[0],element,R) #store rotation matrix in Fi.real[0:3,0:3]
    _alpha_F(beta[0],eps,R,alpha,F)
    _dotmr2(F,rv,F)
    _inv4x4(F,Fi)
    assert 1==0


   
#@nb.jit([(NCDTYPE[:],NCDTYPE[:],NFDTYPE[:,:],NCDTYPE[:])])
#def _delta(alpha,eps,R,out):
#    ct = R[2,2] #cos theta is always in R[2,2]
#    neff = eps[0]**0.5
#    #neff = 2.1906**0.5
#    alpha0 = ct * neff
#    out[0] = alpha[0] - alpha0.real
#    out[1] = alpha[1] + alpha0.real
#    out[2] = alpha[2] - alpha0.real
#    out[3] = alpha[3] + alpha0.real
#    #print(alpha0)
#    
#    

#@nb.guvectorize([(NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])],"(),(),(l),(k),(n)->(n),(n,n),(n,n)", target = "parallel")
#def _deltaffi_vec(beta,phi,element,eps0,dummy,delta,F,Fi):
#    #Fi is a 4x4 matrix... we can use 3x3 part for Rotation matrix and Fi[3] for eps  temporary data
#    R = Fi.real 
#    eps = Fi[3]
#    if _is_isotropic(eps0):
#        _calc_rotations_isotropic(phi[0],R)
#    else:
#        _calc_rotations_uniaxial(phi[0],element,R) #store rotation matrix in Fi.real[0:3,0:3]
#    _uniaxial_order(element[0],eps0,eps) #store caluclated eps values in Fi[3]
#    _alpha_F(beta[0],eps,R,delta,F)
#    _uniaxial_order(0.,eps0,eps) #calculate effective refractive index
#    _delta(delta,eps,R,delta)
#    _inv4x4(F,Fi)
#    
_dummy_array = np.empty((4,),CDTYPE)

_dummy_array2 = np.empty((9,),CDTYPE)
    
def alpha_F(beta,eps0,R,*args,**kw):
    return _alpha_F_vec(beta,eps0,R,_dummy_array,*args,**kw)

def alphaffi(beta,phi,element,eps0,*args,**kw):
    return _alphaffi_vec(beta,phi,element,eps0,_dummy_array,*args,**kw)

def alphaffi_xy_2(beta,phi,element,eps0,*args,**kw):
    return _alphaffi_xy_vec_iso(beta,phi,eps0,_dummy_array,*args,**kw)

#@cached_function
def alphaffi_xy(beta,phi,element,eps0,*args,**kw):
    rv = rotation_vector2(phi) 
    return _alphaffi_xy_vec(beta,phi,rv,element,eps0,_dummy_array,*args,**kw)


def alphaffi_xy_iso(beta,phi,element,eps0,*args,**kw):
    rv = rotation_vector2(phi) #+ np.random.randn(2)
    #print (rv)
    #x,y = rv[...,0], rv[...,1]
    #rv[...,0] = y
    #rv[...,1] = x
    return _alphaffi_xy_vec_iso2(beta,phi,rv,element,eps0,_dummy_array,*args,**kw)



def alphaffi_xy2(beta,phi,element,eps0,mask,*args,**kw):
    rv = rotation_vector2(phi)
    #x,y = rv[...,0], rv[...,1]
    #rv[...,0] = y
    #rv[...,1] = x
    return _alphaffi_xy_vec2(beta,phi,rv,element,eps0,mask,_dummy_array,*args,**kw)

def alphaffi0(beta,phi,n = 1.,*args,**kw):
    element = [0.,0.,0.]
    eps0 = [float(n)**2]*3
    return _alphaffi_vec(beta,phi,element,eps0,_dummy_array,*args,**kw)

def deltaffi(beta,phi,element,eps0,*args,**kw):
    return _deltaffi_vec(beta,phi,element,eps0,_dummy_array,*args,**kw)    

def alpha_FFi(beta,eps0,R,*args,**kw):
    return _alpha_FFi_vec(beta,eps0,R,_dummy_array,*args,**kw)

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
def phasem(alpha,kd,out):
#    f0 = 1j*kd[0]*(alpha[0].real)
#    f1 = 1j*kd[0]*(alpha[1].real)
#    f2 = 1j*kd[0]*(alpha[2].real)
#    f3 = 1j*kd[0]*(alpha[3].real)

    f0 = 1j*kd[0]*(alpha[0])
    f1 = 1j*kd[0]*(alpha[1])
    f2 = 1j*kd[0]*(alpha[2])
    f3 = 1j*kd[0]*(alpha[3])
    
    out[0] = np.exp(f0)
    out[1] = np.exp(f1)
    out[2] = np.exp(f2)
    out[3] = np.exp(f3)

#def jonesvec(pol):
#    """Returns normalized jones vector from an input length 2 vector. 
#    Numpy broadcasting rules apply.
#    
#    >>> jonesvec((1,1j))
#    
#    """
#    pol = np.asarray(pol)
#    assert pol.shape[-1] == 2
#    norm = (pol[...,0] * pol[...,0].conj() + pol[...,1] * pol[...,1].conj())**0.5
#    return pol/norm[...,np.newaxis]

    

__all__ = ["alphaffi_xy","alphaffi","phasem_t", "phasem_r","phasem"]