# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:44:09 2017

@author: andrej

E-H field functions

"""

from __future__ import absolute_import, print_function, division

import numpy as np

from dtmm.conf import NCDTYPE,NFDTYPE, CDTYPE,  NUMBA_TARGET, NUMBA_PARALLEL
#from dtmm.wave import mean_betaphi, betaphi
from dtmm.rotation import  _calc_rotations_uniaxial
from dtmm.linalg import _inv4x4, _dotmr2
from dtmm.dirdata import _uniaxial_order
from dtmm.rotation import rotation_vector2



import numba as nb
from numba import prange

if NUMBA_PARALLEL == False:
    prange = range

sqrt = np.sqrt

@nb.njit([(NFDTYPE,NCDTYPE[:],NFDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])])
def _alpha_F(beta,eps0,R,alpha,F): 
#    if eps0[0] == eps0[1] and eps0[2] == eps0[1]:
##        n = eps0[0]**0.5
##        aout = sqrt(n**2-beta**2)
##        gpout = n**2/aout
##        gsout = -aout
##    
##        alpha[0] = aout
##        alpha[1] = -aout
##        alpha[2] = aout
##        alpha[3] = -aout  
##        
##        F[0,0] = 0.5 
##        F[0,1] = 0.5
##        F[0,2] = 0.
##        F[0,3] = 0.
##        F[1,0] = 0.5 * gpout 
##        F[1,1] = -0.5 * gpout 
##        F[1,2] = 0.
##        F[1,3] = 0.
##        F[2,0] = 0.
##        F[2,1] = 0.
##        F[2,2] = 0.5 
##        F[2,3] = 0.5
##        F[3,0] = 0.
##        F[3,1] = 0.
##        F[3,2] = 0.5 * gsout 
##        F[3,3] = -0.5 * gsout   
#        
#        #iso case
#        eps11 = eps0[0]
#        ev02 =  eps11 - beta * beta
#        evs = sqrt(ev02)
#
#        alpha[0] = evs
#        alpha[1] = -evs
#        alpha[2] = evs
#        alpha[3] = -evs  
#        
#        sf = -R[0,1]
#        cf = R[1,1]
#
#        eps11sf = eps11 * sf
#        evssf = evs*sf
#        evscf = evs*cf
#        ev02cf = ev02*cf
#        ev02cfeps11 = ev02cf/eps11            
#            
#        F[0,2] = evssf
#        F[1,2] = eps11sf
#        F[2,2] = -evscf
#        F[3,2] = ev02cf
#        
#        F[0,3] = (-evssf)
#        F[1,3] = eps11sf
#        F[2,3] = evscf
#        F[3,3] = ev02cf 
#        
#        F[0,0] = (- ev02cfeps11)
#        F[1,0] = (-evscf)
#        F[2,0] = (-sf)
#        F[3,0] = evssf    
#        
#        F[0,1] = (- ev02cfeps11)
#        F[1,1] = (-evscf)
#        F[2,1] = (-sf)
#        F[3,1] = evssf      
#    
#    else:

    
    #uniaxial case
    ct = R[2,2]
    st = R[2,0]
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
        
        evpp = v + sq
        evpm = v - sq

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
        
        F[0,0] = -cf
        F[1,0] = -evpp *cf
        F[2,0] = -sf
        F[3,0] = evpp *sf    
        
        F[0,1] = -cf
        F[1,1] = -evpm *cf
        F[2,1] = -sf
        F[3,1] = evpm *sf    
        
    else:
        sfst = (-R[1,2])
        cfst = (-R[0,2])                   
                                    
        ctbeta = ct * beta
        ctbetaeps11 = ctbeta / eps11
        eps11sfst = eps11 * sfst
        evssfst = evs*sfst
        evscfst = evs*cfst
        evsctbeta = evs*ctbeta
        ev02cfst = ev02*cfst
        ev02cfsteps11 = ev02cfst/eps11
        
        F[0,2] = evssfst
        F[1,2] = eps11sfst
        F[2,2] = (-evscfst - ctbeta)
        F[3,2] = evsctbeta + ev02cfst
        
        F[0,3] = (-evssfst)
        F[1,3] = eps11sfst
        F[2,3] = evscfst - ctbeta
        F[3,3] = ev02cfst-evsctbeta 
        
        F[0,0] = (-evpp*ctbetaeps11 - ev02cfsteps11)
        F[1,0] = (-evpp *cfst - ctbeta)
        F[2,0] = (-sfst)
        F[3,0] = evpp *sfst    
        
        F[0,1] = (-evpm*ctbetaeps11 - ev02cfsteps11)
        F[1,1] = (-evpm *cfst - ctbeta)
        F[2,1] = (-sfst)
        F[3,1] = evpm *sfst   
        
 
    
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
                 "(),(n),(m,m),(k)->(k),(k,k)", target = NUMBA_TARGET)
def _alpha_F_vec(beta,eps0,R,dummy,alpha,F):
    _alpha_F(beta[0],eps0,R,alpha,F)


@nb.guvectorize([(NFDTYPE[:],NCDTYPE[:],NFDTYPE[:,:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])],
                 "(),(n),(m,m),(k)->(k),(k,k),(k,k)", target = NUMBA_TARGET)
def _alpha_FFi_vec(beta,eps0,R,dummy,alpha,F,Fi):
    _alpha_F(beta[0],eps0,R,alpha,F)
    _inv4x4(F,Fi)
    
@nb.jit()    
def _is_isotropic(eps):
    return (eps[0] == eps[1] and eps[1]==eps[2])

@nb.guvectorize([(NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])],
                 "(),(),(l),(k),(n)->(n),(n,n),(n,n)", target = NUMBA_TARGET)
def _alphaffi_vec(beta,phi,element,eps0,dummy,alpha,F,Fi):
    #Fi is a 4x4 matrix... we can use 3x3 part for Rotation matrix and Fi[3] for eps  temporary data
    R = Fi.real 
    eps = Fi[3] 
    _uniaxial_order(element[0],eps0,eps) #store caluclated eps values in Fi[3]
    #if _is_isotropic(eps):
    #    _calc_rotations_isotropic(phi[0],R)
    #else:
    _calc_rotations_uniaxial(phi[0],element,R) #store rotation matrix in Fi.real[0:3,0:3]
    _alpha_F(beta[0],eps,R,alpha,F)
    _inv4x4(F,Fi)

@nb.guvectorize([(NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])],
                 "(),(),(m),(l),(k),(n)->(n),(n,n),(n,n)", target = NUMBA_TARGET)
def _alphaffi_xy_vec(beta,phi,rv, element,eps0,dummy,alpha,F,Fi):
    #Fi is a 4x4 matrix... we can use 3x3 part for Rotation matrix and Fi[3] for eps  temporary data
    R = Fi.real 
    eps = Fi[3] 
    _uniaxial_order(element[0],eps0,eps) #store caluclated eps values in Fi[3]
    _calc_rotations_uniaxial(phi[0],element,R) #store rotation matrix in Fi.real[0:3,0:3]
    _alpha_F(beta[0],eps,R,alpha,F)
    _dotmr2(F,rv,F)
    _inv4x4(F,Fi)

   
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

def alphaffi_xy(beta,phi,element,eps0,*args,**kw):
    rv = rotation_vector2(phi)
    return _alphaffi_xy_vec(beta,phi,rv,element,eps0,_dummy_array,*args,**kw)

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
                "(n),()->(n)", target = NUMBA_TARGET)    
def phasem_t(alpha,kd,out):
    out[0] = np.exp(1j*kd[0]*(alpha[0].real))
    out[1] = 0.
    out[2] = np.exp(1j*kd[0]*(alpha[2].real)) 
    out[3] = 0.
    
@nb.guvectorize([(NCDTYPE[:],NFDTYPE[:], NCDTYPE[:])],
                "(n),()->(n)", target = NUMBA_TARGET)    
def phasem_r(alpha,kd,out):
    out[0] = 0.
    out[1] = np.exp(1j*kd[0]*(alpha[1].real))  
    out[2] = 0.
    out[3] = np.exp(1j*kd[0]*(alpha[3].real))  
    
@nb.guvectorize([(NCDTYPE[:],NFDTYPE[:], NCDTYPE[:])],
                "(n),()->(n)", target = NUMBA_TARGET)    
def phasem(alpha,kd,out):
    for i in range(4):
        f = 1j*kd[0]*(alpha[i].real)
        out[i] = np.exp(f)
        
fphasem = phasem    


#@nb.guvectorize([(NCDTYPE[:],NFDTYPE[:], NCDTYPE[:])],
#                "(n),()->(n)", target = NUMBA_TARGET)    
#def bphasem(alpha,kd,out):
#    for i in range(4):
#        f = -1j*kd[0]*alpha[i]
#        out[i] = np.exp(f)

def jonesvec(pol):
    """Returns normalized jones vector from an input length 2 vector. 
    Numpy broadcasting rules apply.
    
    >>> jonesvec((1,1j))
    
    """
    pol = np.asarray(pol)
    assert pol.shape[-1] == 2
    norm = (pol[...,0] * pol[...,0].conj() + pol[...,1] * pol[...,1].conj())**0.5
    return pol/norm[...,np.newaxis]


#@nb.guvectorize([(NCDTYPE[:,:],NFDTYPE[:],NCDTYPE[:],NFDTYPE[:],NCDTYPE[:],NCDTYPE[:,:,:])],"(n,m),(),(l),(), (k)->(k,n,m)")
#def _beam2field(beam, k0, pol, n, dummy, out):
#    fbeam = spfft.fft2(beam)
#    beta, phi = betaphi(beam.shape, k0[0])
#    f = ifield_matrix(beta,phi,n[0])
#    out[0] = fbeam * pol[0]
#    out[2] = fbeam * pol[1]
##    
#    if k0 > 0:
#        b1 = f[...,1,0]*out[0] + f[...,1,2]*out[2]
#        b2 = f[...,3,0]*out[0] + f[...,3,2]*out[2]
##        
#        det = f[...,1,3] * f[...,3,1] - f[...,1,1]*f[...,3,3]
##        
#        out[1] = f[...,3,3] * b1 / det - f[...,1,3] * b2 / det
#        out[3] = f[...,1,1] * b2 / det - f[...,3,1] * b1 / det
#    else:
#        b1 = f[:,:,0,0]*out[0] + f[:,:,0,2]*out[2]
#        b2 = f[:,:,2,0]*out[0] + f[:,:,2,2]*out[2]
##        
#        det = f[:,:,0,3] * f[:,:,2,1] - f[:,:,0,1]*f[:,:,2,3]
#        
#        out[1] = f[:,:,2,3] * b1 / det - f[:,:,0,3] * b2 / det
#        out[3] = f[:,:,0,1] * b2 / det - f[:,:,2,1] * b1 / det     
##
#    spfft.ifft2(out, overwrite_x = True)
## 
#@nb.guvectorize([(NCDTYPE[:,:,:],NFDTYPE[:],NCDTYPE[:],NFDTYPE[:],NCDTYPE[:,:])],"(k,n,m),(),(l),()->(n,m)")
#def _field2beam(field, k0, pol, n, out):
#    ffield = spfft.fft2(field)
#  
#    beta, phi = betaphi(out.shape, k0[0])
#    mask0 = (beta>=1)
#    
#    mask = np.empty(mask0.shape[:-2] + (4,)+mask0.shape[-2:] , mask0.dtype)
#    for i in range(4):
#        mask[...,i,:,:] = mask0 
#    ffield[mask] = 0.
#    
#    alpha, f, fi = alphaffi0(beta,phi,n[0])
#    dotmf(fi,ffield, ffield)
#    if k0 > 0:
#        ffield[1,...] = 0.
#        ffield[3,...] = 0.
#    else:
#        ffield[0,...] = 0.
#        ffield[2,...] = 0.
#    dotmf(f,ffield, ffield)    
#    
#        
#    spfft.ifft2(ffield, overwrite_x = True)
##    
#    out[...] = ffield[0,...]*pol[0] + ffield[2,...]*pol[1] 
# 


@nb.njit([(NCDTYPE[:,:,:,:],NFDTYPE[:,:,:])], parallel = NUMBA_PARALLEL)
def _field2specter(field, out):
    
    for j in prange(field.shape[2]):
        for i in range(field.shape[0]):
            for k in range(field.shape[3]):
                tmp1 = (field[i,0,j,k].real * field[i,1,j,k].real + field[i,0,j,k].imag * field[i,1,j,k].imag)
                tmp2 = (field[i,2,j,k].real * field[i,3,j,k].real + field[i,2,j,k].imag * field[i,3,j,k].imag)
                out[j,k,i] = tmp1-tmp2 

@nb.njit([(NCDTYPE[:,:,:,:,:],NFDTYPE[:,:,:])], parallel = NUMBA_PARALLEL)
def _field2spectersum(field, out):
    for n in prange(field.shape[0]):
        for j in range(field.shape[3]):
            for i in range(field.shape[1]):
                for k in range(field.shape[4]):
                    tmp1 = (field[n,i,0,j,k].real * field[n,i,1,j,k].real + field[n,i,0,j,k].imag * field[n,i,1,j,k].imag)
                    tmp2 = (field[n,i,2,j,k].real * field[n,i,3,j,k].real + field[n,i,2,j,k].imag * field[n,i,3,j,k].imag)
                    if n == 0:
                        out[j,k,i] = tmp1 - tmp2
                    else:
                        out[j,k,i] += (tmp1 -tmp2)


@nb.guvectorize([(NCDTYPE[:,:,:,:],NFDTYPE[:,:,:])],"(w,k,n,m)->(n,m,w)", target = "cpu")
def field2specter(field, out):
    _field2specter(field, out)

@nb.guvectorize([(NCDTYPE[:,:,:,:,:],NFDTYPE[:,:,:])],"(l,w,k,n,m)->(n,m,w)", target = "cpu")
def field2spectersum(field, out):
    _field2spectersum(field, out)    


#@nb.guvectorize([(NCDTYPE[:,:,:],NFDTYPE[:],NFDTYPE[:],NCDTYPE[:,:,:])],"(k,n,m),(),()->(k,n,m)")
#def transmitted_field(field, k0, n, out):
#    """Returns the transmitted part of the field with a given wavenumber and refractive index"""
#    if out is not field:
#        out[...] = field
#    
#    spfft.fft2(out, overwrite_x = True)
#  
#    beta, phi = betaphi(out.shape[1:], k0[0])
#    mask0 = (beta>=1)
#    
#    mask = np.empty(out.shape , mask0.dtype)
#    for i in range(4):
#        mask[i,:,:] = mask0
#    out[mask] = 0.
#    
#    alpha, f, fi = alphaffi0(beta,phi,n[0])
#    dotmf(fi,out, out)
#    out[1,...] = 0.
#    out[3,...] = 0.
#
#    dotmf(f,out,out)    
#    
#    spfft.ifft2(out, overwrite_x = True)
    




#def field2waves(field, k0, n = 1, out = None):
#    """Converts E-H field to plane waves (p+,p-,s+,s-)  assuming a homogeneous layer"""
#    assert field.ndim >= 3 and field.shape[-3] == 4 
#    rays = spfft.fft2(field, axes = (-3,-2))
#    beta, phi = betaphi(rays.shape[-2:], k0)
#    f = ifield_matrix(beta,phi,n)
#    return dotmf(f, rays, out)
#
#def waves2field(waves, k0, n = 1, out = None):
#    """Converts plane waves (p+,p-,s+,s-) too E-H field assuming a homogeneous layer"""
#    assert waves.ndim >= 3 and waves.shape[-3] == 4
#    beta, phi = betaphi(waves.shape[-2:], k0)
#    f = field_matrix(beta,phi,n)
#    out = dotmf(f, waves, out)
#    return spfft.ifft2(out, axes = (-3,-2), overwrite_x = True)
#    
#
#def beam2field(beam,k0,pol = (1,0), n = 1,*args,**kw): 
#    return _beam2field(beam,k0,pol,n,_dummy_array,*args,**kw)
#
#def field2beam(field,k0,pol = (1,0), n = 1,*args,**kw): 
#    """Computes scalar beam profile from a given field vector."""
#    return _field2beam(field,k0,pol,n,*args,**kw)
#
#@nb.njit([(NCDTYPE[:,:,:],NFDTYPE[:,:,:])],parallel = True)
#def _beam2specter(beam, out):
#    for i in nb.prange(beam.shape[0]):
#        for j in range(beam.shape[1]):
#            for k in range(beam.shape[2]):
#                out[j,k,i] = beam[i,j,k].real**2+beam[i,j,k].imag**2
#                
#@nb.guvectorize([(NCDTYPE[:,:,:],NFDTYPE[:,:,:])],"(k,n,m)->(n,m,k)")
#def beam2specter(beam, out):
#    _beam2specter(beam, out)
#        

#def field_vector(beam,k0,pol=(1,0),n = 1,*args,**kw):
#    return _field_vector(beam,k0,pol,n,_dummy_array,*args,**kw)
#    

__all__ = ["alphaffi_xy","alphaffi", "jonesvec","phasem_t", "phasem_r","phasem","field2specter"]