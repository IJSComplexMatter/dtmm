"""
Diffraction calculation functions.


"""
from __future__ import absolute_import, print_function, division

from dtmm.conf import cached_function, BETAMAX, FDTYPE, CDTYPE
from dtmm.wave import betaphi
from dtmm.data import refind2eps
from dtmm.tmm import phase_mat,  alphaffi, alphaf,  alphaEEi, tr_mat, alphaE
from dtmm.linalg import dotmdm, dotmf
from dtmm.fft import fft2, ifft2


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
    
    mask0 = (beta >= betamax)
    
    
    alpha, f= alphaf(beta,phi,epsv,epsa,out = out) 

    out = (alpha,f)
    
    f[mask0] = 0.
    alpha[mask0] = 0.
    

    return out



@cached_function
def diffraction_alphaffi(shape, ks, epsv = (1.,1.,1.), 
                            epsa = (0.,0.,0.), betamax = BETAMAX, out = None):


    ks = np.asarray(ks)
    ks = abs(ks)
    beta, phi = betaphi(shape,ks)
    
    mask0 = (beta >= betamax)


    alpha, f, fi = alphaffi(beta,phi,epsv,epsa, out = out) 


    out = (alpha,f,fi)
    
#    try:
#        b1,betamax = betamax
#        a = (betamax-b1)/(betamax)
#        m = tukey(beta,a,betamax)
#        np.multiply(f,m[...,None,None],f)
#    except:
#        pass

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
    beta, phi = betaphi(shape,ks)

    mask0 = (beta >= betamax)#betamax)
            
    alpha, j, ji = alphaEEi(beta,phi,epsv,epsa, mode = mode, out = out) 
    ji[mask0] = 0.
    j[mask0] = 0.
    alpha[mask0] = 0.
    out = (alpha,j,ji)
    return out


@cached_function
def E_diffraction_alphaE(shape, ks, epsv = (1,1,1), 
                            epsa = (0.,0.,0.), mode = +1, betamax = BETAMAX, out = None):

    ks = np.asarray(ks)
    ks = abs(ks)
    beta, phi = betaphi(shape,ks)

    mask0 = (beta >= betamax)#betamax)
            
    alpha, j = alphaE(beta,phi,epsv,epsa, mode = mode, out = out) 
    j[mask0] = 0.
    alpha[mask0] = 0.
    out = (alpha,j)
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
    """Build field diffraction matrix. 
    """
    
    ks = np.asarray(ks, dtype = FDTYPE)
    epsv = np.asarray(epsv, dtype = CDTYPE)
    epsa = np.asarray(epsa, dtype = FDTYPE)
    alpha, f, fi = diffraction_alphaffi(shape, ks, epsv = epsv, epsa = epsa, betamax = betamax)
    kd =ks * d
    pmat = phase_matrix(alpha, kd , mode = mode)
    
    return dotmdm(f,pmat,fi,out = out) 

#@cached_function
def field_thick_cover_diffraction_matrix(shape, ks,  d = 1., epsv = (1,1,1), epsa = (0,0,0.), d_cover = 0, epsv_cover = (1.,1.,1.), epsa_cover = (0.,0.,0.), mode = "b", betamax = BETAMAX, out = None):
    """Build field diffraction matrix. 
    """
    
    ks = np.asarray(ks, dtype = FDTYPE)
    epsv = np.asarray(epsv, dtype = CDTYPE)
    epsa = np.asarray(epsa, dtype = FDTYPE)
    alpha, f, fi = diffraction_alphaffi(shape, ks, epsv = epsv, epsa = epsa, betamax = betamax)
    alpha0, f0 = diffraction_alphaf(shape, ks, epsv = epsv_cover ,epsa = epsa_cover, betamax = betamax)
    
    
    alphac = alpha0 - alpha / 1.5
    alphac = alphac - alphac[...,0,0,:][...,None,None,:]
    
    #offset = (alphac.mean(axis = (-2,-3)))[...,None,None,:]
    
    #alphac = alphac - offset
    
    
    
    kd = ks * d_cover
    
    pmatc = phase_matrix(alphac, kd , mode = mode)

    kd = ks * d
    
    pmat = phase_matrix(alpha, kd , mode = mode)
    
    pmat = pmat * pmatc
    
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

@cached_function
def E_cover_diffraction_matrix(shape, ks,  n = 1., d_cover = 0, n_cover = 1.5, mode = +1, betamax = BETAMAX, out = None):
    ks = np.asarray(ks, dtype = FDTYPE)
    epsv = np.asarray(refind2eps((n,)*3),CDTYPE)
    epsa = np.asarray((0.,0.,0.), dtype = FDTYPE)
    epsv_cover = np.asarray(refind2eps((n_cover,)*3),CDTYPE)
    epsa_cover = np.asarray((0.,0.,0.), dtype = FDTYPE)    
    alpha, j = E_diffraction_alphaE(shape, ks, epsv = epsv, epsa = epsa, mode = mode, betamax = betamax)
    alpha0, j0, j0i = E_diffraction_alphaEEi(shape, ks, epsv = epsv_cover, epsa = epsa_cover, mode = mode, betamax = betamax)

    alphac = alpha0 - alpha * n / n_cover
    alphac = alphac - alphac[...,0,0,:][...,None,None,:]     
    
    kd =ks * d_cover
    pmat = phase_matrix(alphac, kd)
    return dotmdm(j0,pmat,j0i,out = out) 



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
    
    return tr_mat(fin, fout, fmatini = fini, mode = mode, out = out)
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
 
  
def diffract(field, dmat, window = None, input_fft = False, output_fft = False, out = None):
    """Takes input field vector and diffracts it
    
    Parameters
    ----------
    field : (...,4,:,:) array
        Input field array.
    dmat : array
        Diffraction matrix. Use :func:`field_diffraction_matrix`to create one
    window : array, optional
        A window function applied to the result
    input_fft : bool
        Specifies whether input field array is the Fourier transform of the field.
        By default, input is assumed to be in real space.
    output_fft : bool
        Specifies whether the computed field array is the Fourier transform.
        By default, output is assumed to be in real space.
    out : ndarray, optional
        If specified, store the results here.
        
    Returns
    -------
    field : (...,4,:,:) array
        Diffracted field in real space (if output_fft == False )or in fft spaace
        (if output_fft == True).
            
    """
    if not input_fft:   
        field = fft2(field, out = out)
        out = field
    if dmat is not None:
        out = dotmf(dmat, field ,out = out)
    else:
        #make it work for dmat = None input... so just copy data
        if out is None:
            out = field.copy()
        else:
            if out is not field:
                out[...] = field
    if not output_fft:
        out = ifft2(out, out = out)
    if window is not None:
        if output_fft:
            raise ValueError("Cannot use window function if ouput is fft field.")
        out = np.multiply(out,window,out = out)
    return out

def diffracted_field(field, wavenumbers, d = 0.,n = 1, mode = "t", betamax = BETAMAX, out = None):
    eps = refind2eps([n]*3)
    pmat = field_diffraction_matrix(field.shape[-2:], wavenumbers, d = d, epsv = eps, epsa = (0.,0.,0.), mode = mode, betamax = betamax)
    return diffract(field, pmat, out = out) 
#
#def transmitted_field(field, wavenumbers, n = 1, betamax = BETAMAX, out = None):
#    eps = refind2eps([n]*3)
#    pmat = projection_matrix(field.shape[-2:], wavenumbers, epsv = eps, epsa = (0.,0.,0.), mode = "t", betamax = betamax)
#    return diffract(field, pmat, out = out) 
#
#def reflected_field(field, wavenumbers, n = 1, betamax = BETAMAX, out = None):
#    eps = refind2eps([n]*3)
#    pmat = projection_matrix(field.shape[-2:], wavenumbers, epsv = eps, epsa = (0.,0.,0.), mode = "r", betamax = betamax)
#    return diffract(field, pmat, out = out) 

__all__ = []
