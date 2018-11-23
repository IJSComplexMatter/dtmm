"""
Diffraction functions
"""
from __future__ import absolute_import, print_function, division

from dtmm.conf import cached_function, BETAMAX, FDTYPE, CDTYPE
from dtmm.wave import betaphi
from dtmm.window import tukey
from dtmm.data import refind2eps
from dtmm.tmm import phase_mat, polarizer4x4, alphaffi, alphaf,  transmission_mat, alphaEEi, tr_mat, t_mat
from dtmm.linalg import dotmdm, dotmf
from dtmm.fft import fft2, ifft2
from dtmm.jones import jonesvec
import numpy as np


DIFRACTION_PARAMETERS = ("distance", "mode")#, "refind")

@cached_function
def diffraction_alphaf(shape, ks, epsv = (1.,1.,1.), 
                            epsa = (0.,0.,0.), betamax = BETAMAX, out = None):

    ks = np.asarray(ks)
    ks = abs(ks)
    beta, phi = betaphi(shape,ks)
    epsv = np.asarray(epsv, CDTYPE)
    epsa = np.asarray(epsa, FDTYPE)


    alpha, f= alphaf(beta,phi,epsv,epsa,out = out) 

    out = (alpha,f)
    
#    try:
#        b1,b2 = betamax
#        a = (b2-b1)/(b2)
#        m = tukey(beta,a,b2)
#        
#        print(b1)
#        m = (b1-beta.clip(b1,betamax))/(betamax-b1)+1
#        
#        np.multiply(f,m[...,None,None],f)
#    except:
#        pass

    mask0 = (beta >= betamax)
    
    f[mask0] = 0.
    alpha[mask0] = 0.

    return out


#@cached_function
#def diffraction_polarizer(shape, ks, jones = (1,0), epsv = (1.,1.,1.), 
#                            epsa = (0.,0.,0.), betamax = BETAMAX, out = None):
#    ks = np.asarray(ks)
#    ks = abs(ks)
#    epsv = np.asarray(epsv, CDTYPE)
#    epsa = np.asarray(epsa, FDTYPE)
#    beta, phi = betaphi(shape,ks)
#    alpha, f = diffraction_alphaf(shape, ks, epsv = epsv, 
#                            epsa = epsa, betamax = betamax)
#
#    beta, phi = betaphi(shape,ks)
#    jones = jonesvec(jones, phi)
#    pmat = polarizer4x4(jones, f)
#    return pmat


@cached_function
def diffraction_alphaffi(shape, ks, epsv = (1.,1.,1.), 
                            epsa = (0.,0.,0.), betamax = BETAMAX, out = None):


    ks = np.asarray(ks)
    ks = abs(ks)
    #shape = fieldv.shape[-2:]
    #eps = uniaxial_order(0.,eps0)
    beta, phi = betaphi(shape,ks)
    #m = tukey(beta,0.3,betamax)
    #mask0 = (beta>0.9) & (beta < 1.1)
    

    
    #mask = np.empty(mask0.shape + (4,), mask0.dtype)
    #for i in range(4):
    #    mask[...,i] = mask0   

     
    alpha, f, fi = alphaffi(beta,phi,epsv,epsa,out = out) 

    out = (alpha,f,fi)
    
    try:
        b1,betamax = betamax
        a = (betamax-b1)/(betamax)
        m = tukey(beta,a,betamax)
        #print(b1)
        #m = (b1-beta.clip(b1,betamax))/(betamax-b1)+1
        #np.multiply(alpha,m[...,None,None],alpha)
        np.multiply(f,m[...,None,None],f)
        #np.multiply(fi,m[...,None,None],fi)
    except:
        pass
    mask0 = (beta >= betamax)#betamax)
    fi[mask0] = 0.
    f[mask0] = 0.
    alpha[mask0] = 0.
    #return mask, alpha,f,fi
    
    return out


@cached_function
def E_diffraction_alphaEEi(shape, ks, epsv = (1,1,1), 
                            epsa = (0.,0.,0.), mode = +1, betamax = BETAMAX, out = None):

    ks = np.asarray(ks)
    ks = abs(ks)
    #shape = fieldv.shape[-2:]
    #eps = uniaxial_order(0.,eps0)
    beta, phi = betaphi(shape,ks)
    #m = tukey(beta,0.,betamax)
    #mask0 = (beta>0.9) & (beta < 1.1)
    mask0 = (beta >= betamax)#betamax)
    #mask = np.empty(mask0.shape + (4,), mask0.dtype)
    #for i in range(4):
    #    mask[...,i] = mask0   

            
    alpha, j, ji = alphaEEi(beta,phi,epsv,epsa, mode = mode, out = out) 
    ji[mask0] = 0.
    j[mask0] = 0.
    alpha[mask0] = 0.
    out = (alpha,j,ji)

    #np.multiply(f,m[...,None,None],f)
    #np.multiply(fi,m[...,None,None],fi)
    #return mask, alpha,f,fi
    
    return out

  
def phase_matrix(alpha, kd, mode = None, mask = None, out = None):
    kd = np.asarray(kd, dtype = FDTYPE)
    out = phase_mat(alpha,kd[...,None,None], out = out)  
    if mode == "t" or mode == +1:
        out[...,1::2] = 0.
    elif mode == "r" or mode == -1:
        out[...,::2] = 0.
    if mask is not None:
        out[mask] = 0.
    return out  

@cached_function
def field_diffraction_matrix(shape, ks,  d = 1., epsv = (1,1,1), epsa = (0,0,0.), mode = "b", betamax = BETAMAX, out = None):
    ks = np.asarray(ks, dtype = FDTYPE)
    epsv = np.asarray(epsv, dtype = CDTYPE)
    epsa = np.asarray(epsa, dtype = FDTYPE)
    alpha, f, fi = diffraction_alphaffi(shape, ks, epsv = epsv, epsa = epsa, betamax = betamax)
    kd =ks * d
    pmat = phase_matrix(alpha, kd , mode = mode)
    return dotmdm(f,pmat,fi,out = out) 

@cached_function
def E_diffraction_matrix(shape, ks,  d = 1., epsv = (1,1,1), epsa = (0,0,0.), mode = +1, betamax = BETAMAX, out = None):
    ks = np.asarray(ks, dtype = FDTYPE)
    epsv = np.asarray(epsv, dtype = CDTYPE)
    epsa = np.asarray(epsa, dtype = FDTYPE)
    alpha, j, ji = E_diffraction_alphaEEi(shape, ks, epsv = epsv, epsa = epsa, mode = mode, betamax = betamax)
    kd =ks * d
    pmat = phase_matrix(alpha, kd)
    return dotmdm(j,pmat,ji,out = out) 


#@cached_function
#def jones_transmission_matrix(shape, ks, epsv_in = (1.,1.,1.), epsa_in = (0.,0.,0.),
#                            epsv_out = (1.,1.,1.), epsa_out = (0.,0.,0.), mode = +1, betamax = BETAMAX, out = None):
#    
#    
#    alpha, fin,fini = diffraction_alphaffi(shape, ks, epsv = epsv_in, 
#                            epsa = epsa_in, betamax = betamax)
#    
#    alpha, fout,fouti = diffraction_alphaffi(shape, ks, epsv = epsv_out, 
#                            epsa = epsa_out, betamax = betamax)
#    
#    return transmission_mat(fin, fout, fini = fini, mode = mode, out = out)
#
@cached_function
def E_tr_matrix(shape, ks, epsv_in = (1.,1.,1.), epsa_in = (0.,0.,0.),
                            epsv_out = (1.,1.,1.), epsa_out = (0.,0.,0.), mode = +1, betamax = BETAMAX, out = None):
    
    
    alpha, fin,fini = diffraction_alphaffi(shape, ks, epsv = epsv_in, 
                            epsa = epsa_in, betamax = betamax)
    
    alpha, fout,fouti = diffraction_alphaffi(shape, ks, epsv = epsv_out, 
                            epsa = epsa_out, betamax = betamax)
    
    return tr_mat(fin, fout, fini = fini, mode = mode, out = out)
#
#@cached_function
#def jones_t_matrix(shape, ks, epsv_in = (1.,1.,1.), epsa_in = (0.,0.,0.),
#                            epsv_out = (1.,1.,1.), epsa_out = (0.,0.,0.), mode = +1, betamax = BETAMAX, out = None):
#    
#    
#    alpha, fin,fini = diffraction_alphaffi(shape, ks, epsv = epsv_in, 
#                            epsa = epsa_in, betamax = betamax)
#    
#    alpha, fout,fouti = diffraction_alphaffi(shape, ks, epsv = epsv_out, 
#                            epsa = epsa_out, betamax = betamax)
#    
#    return t_mat(fin, fout, fini = fini, mode = mode, out = out)
        

@cached_function
def projection_matrix(shape, ks, epsv = (1,1,1),epsa = (0,0,0.), mode = +1, betamax = BETAMAX, out = None):
    """Computes a reciprocial field projection matrix.
    """
    ks = np.asarray(ks, dtype = FDTYPE)
    epsv = np.asarray(epsv, dtype = CDTYPE)
    epsa = np.asarray(epsa, dtype = FDTYPE)    
    alpha, f, fi = diffraction_alphaffi(shape, ks, epsv = epsv, epsa = epsa, betamax = betamax)
    kd = np.zeros_like(ks)
    pmat = phase_matrix(alpha, kd , mode = mode)
    return dotmdm(f,pmat,fi,out = out)   
 
  
def diffract(fieldv, dmat, window = None, out = None): 
    f = fft2(fieldv, out = out)
    f2 = dotmf(dmat, f ,out = f)
    out = ifft2(f2, out = out)
    if window is not None:
        out = np.multiply(out,window,out = out)
    return out

def diffracted_field(field, wavenumbers, d = 0.,n = 1, mode = "t", betamax = BETAMAX, out = None):
    eps = refind2eps([n]*3)
    pmat = field_diffraction_matrix(field.shape[-2:], wavenumbers, d = d, epsv = eps, epsa = (0.,0.,0.), mode = mode, betamax = betamax)
    return diffract(field, pmat, out = out) 

def transmitted_field(field, wavenumbers, n = 1, betamax = BETAMAX, out = None):
    eps = refind2eps([n]*3)
    pmat = projection_matrix(field.shape[-2:], wavenumbers, epsv = eps, epsa = (0.,0.,0.), mode = "t", betamax = betamax)
    return diffract(field, pmat, out = out) 

def reflected_field(field, wavenumbers, n = 1, betamax = BETAMAX, out = None):
    eps = refind2eps([n]*3)
    pmat = projection_matrix(field.shape[-2:], wavenumbers, epsv = eps, epsa = (0.,0.,0.), mode = "r", betamax = betamax)
    return diffract(field, pmat, out = out) 

__all__ = []
