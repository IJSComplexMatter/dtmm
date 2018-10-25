"""
Main top level calculation functions for light propagation through optical data.
"""
from __future__ import absolute_import, print_function, division

from dtmm.conf import DTMMConfig, cached_function, BETAMAX, FDTYPE, cached_result
from dtmm.wave import k0, eigenwave, betaphi, betaxy, mean_betaphi2, mean_betaphi, mean_betaphi3, mean_betaphi4
from dtmm.window import blackman
from dtmm.data import uniaxial_order, refind2eps, validate_optical_data
from dtmm.tmm import alphaffi, phasem,  alphajji,alphaEEi, alphaf, transmission_mat, E2H_mat, E_mat, Eti_mat, phase_mat, Etri_mat
from dtmm.linalg import dotmdm, dotmm, dotmf, dotmdmf, inv, dotmd
from dtmm.print_tools import print_progress
from dtmm.diffract import diffraction_alphaffi, projection_matrix, diffract, phase_matrix, \
                jones_diffraction_matrix, jones_transmission_matrix, jones_tr_matrix, jones_t_matrix
from dtmm.field import field2intensity, select_fftfield, fft_window
from dtmm.fft import fft2, ifft2
from dtmm.jones import polarizer, apply_jones_matrix, jonesvec
import numpy as np
from dtmm.diffract import diffraction_matrix
from dtmm.diffract import diffraction_matrix as field_diffraction_matrix

#norm flags
DTMM_NORM_FFT = 1<<0 #normalize in fft mode
DTMM_NORM_REF = 1<<1 #normalize using reference field

def _isotropic_effective_data(data):
    d, material, angles = data
    n = len(d)
    epseff = uniaxial_order(0.,material).mean(axis = (0,1,2))
    epseff = np.broadcast_to(epseff,(n,3)).copy()#better to make copy.. to make it c contiguous
    aeff = np.array((0.,0.,0.))
    aeff = np.broadcast_to(aeff,(n,3)).copy()#better to make copy.. to make it c contiguous
    return validate_optical_data((d,epseff,aeff), homogeneous = True)

def _validate_betaphi(beta,phi, extendeddim = 0):
    if beta is None or phi is None:
        raise ValueError("Both beta and phi must be defined!")
    beta = np.asarray(beta)
    phi = np.asarray(phi)  
    
    if beta.ndim != phi.ndim:
        raise ValueError("Beta nad phi should have same dimensions!")
    
    if beta.ndim == 1:
        if len(beta) != len(phi):
            raise ValueError("Beta nad phi should have same length!")
        #make arrays broadcastable to field by adding extra dimensions
        for i in range(extendeddim):
            beta = beta[...,None]
            phi = phi[...,None]
    elif beta.ndim != 0:
        raise ValueError("Only length 1 or scalar values are supported for beta and phi")
    return beta, phi

@cached_function
def correction_matrix(beta,phi,ks, d=1., epsv = (1,1,1), epsa = (0,0,0.), out = None):
    alpha, f, fi = alphaffi(beta,phi,epsv,epsa)  
    kd = -np.asarray(ks)*d
    pmat = phase_matrix(alpha, kd)  
    return dotmdm(f,pmat,fi, out = out)

@cached_function
def field_correction_matrix(beta,phi,ks, d=1., epsv = (1,1,1), epsa = (0,0,0.), out = None):
    alpha, f, fi = alphaffi(beta,phi,epsv,epsa)  
    kd = -np.asarray(ks)*d
    pmat = phase_matrix(alpha, kd)  
    return dotmdm(f,pmat,fi, out = out)

@cached_function
def jones_correction_matrix(beta,phi,ks, d=1., epsv = (1,1,1), epsa = (0,0,0.), mode = +1, out = None):
    alpha, j, ji = alphaEEi(beta,phi,epsv,epsa, mode = mode)  
    kd = -np.asarray(ks)*d
    pmat = phase_matrix(alpha, kd)  
    return dotmdm(j,pmat,ji, out = out)

@cached_function
def corrected_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
 
    dmat = field_diffraction_matrix(shape, ks, d, epsv, epsa, betamax = betamax)
    cmat = correction_matrix(beta, phi, ks, d, epsv, epsa)
    if d > 0:
        return dotmm(dmat,cmat, out = None) #why out none?!! check this!
    else:
        return dotmm(cmat, dmat, out = None)
    

@cached_function
def corrected_jones_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), mode = +1, betamax = BETAMAX, out = None):
    dmat = jones_diffraction_matrix(shape, ks, d, epsv, epsa, mode = mode, betamax = betamax)
    cmat = jones_correction_matrix(beta, phi, ks, d, epsv, epsa, mode = mode)
    if d > 0:
        return dotmm(dmat,cmat, out = None)
    else:
        return dotmm(cmat, dmat, out = None)
    
    
@cached_function
def corrected_E_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), mode = +1, betamax = BETAMAX, out = None):
    dmat = jones_diffraction_matrix(shape, ks, d, epsv, epsa, mode = mode, betamax = betamax)
    cmat = jones_correction_matrix(beta, phi, ks, d/2., epsv, epsa, mode = mode)
    return dotmm(cmat,dotmm(dmat,cmat, out = out), out = out)

@cached_function
def corrected_field_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
 
    dmat = field_diffraction_matrix(shape, ks, d, epsv, epsa, betamax = betamax)
    cmat = correction_matrix(beta, phi, ks, d/2, epsv, epsa)
    dmat = dotmm(dmat,cmat, out = out)
    return dotmm(cmat,dmat, out = dmat) 

    
@cached_function
def first_jones_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), mode = +1, betamax = BETAMAX, out = None):
    dmat = jones_diffraction_matrix(shape, ks, d, epsv, epsa, mode = mode,betamax = betamax)
    cmat = jones_correction_matrix(beta, phi, ks, d, epsv, epsa, mode = mode)
    return dotmm(dmat,cmat, out = None)

@cached_function
def second_jones_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), mode = +1, betamax = BETAMAX, out = None):
    dmat = jones_diffraction_matrix(shape, ks, d, epsv, epsa, mode = mode, betamax = betamax)
    cmat = jones_correction_matrix(beta, phi, ks, d, epsv, epsa, mode = mode)
    return dotmm(cmat,dmat, out = None)

@cached_function
def first_field_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
    dmat = field_diffraction_matrix(shape, ks, d, epsv, epsa, betamax = betamax)
    cmat = field_correction_matrix(beta, phi, ks, d, epsv, epsa)
    return dotmm(dmat,cmat, out = None)

@cached_function
def second_field_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
    dmat = field_diffraction_matrix(shape, ks, d, epsv, epsa, betamax = betamax)
    cmat = field_correction_matrix(beta, phi, ks, d, epsv, epsa)
    return dotmm(cmat,dmat, out = None)

@cached_function
def interface_field_diffraction_matrix(shape, ks, beta,phi, d1=1., d2 = 1.,
                                 epsv1 = (1,1,1), epsa1 = (0,0,0.), 
                                 epsv2 = (1,1,1), epsa2 = (0,0,0.), 
                                 betamax = BETAMAX, out = None):
    
    dmat2 = second_field_diffraction_matrix(shape, ks, beta, phi,d = d2/2.,
                                               epsv = epsv2,epsa = epsa2,
                                               betamax = betamax)

    dmat1 = first_field_diffraction_matrix(shape, ks, beta, phi,d = d1/2.,
                                               epsv = epsv1,epsa = epsa1,
                                               betamax = betamax)
    return dotmm(dmat2,dmat1)

@cached_function
def interface_jones_diffraction_matrix(shape, ks, beta,phi, d1=1., d2 = 1.,
                                 epsv1 = (1,1,1), epsa1 = (0,0,0.), 
                                 epsv2 = (1,1,1), epsa2 = (0,0,0.), 
                                 mode = +1, betamax = BETAMAX, out = None):
    
    dmat2 = second_jones_diffraction_matrix(shape, ks, beta, phi,d = d2/2.,
                                               epsv = epsv2,epsa = epsa2, mode = mode,
                                               betamax = betamax)

    dmat1 = first_jones_diffraction_matrix(shape, ks, beta, phi,d = d1/2.,
                                               epsv = epsv1,epsa = epsa1, mode = mode,
                                               betamax = betamax)

    return dotmm(dmat2,dmat1)


def fft_windows(betax, betay, n, betax_off = 0., betay_off = 0., betamax = BETAMAX, out = None):

    d = 2*betamax/(n-1)
    xoffset = np.mod(betax_off, d)
    xoffsetm = np.mod(betax_off, -d) 
    mask = np.abs(xoffset) > np.abs(xoffsetm)
    try:
        xoffset[mask] = xoffsetm[mask]
    except TypeError: #scalar
        if mask:
            xoffset = xoffsetm
    
    yoffset = np.mod(betay_off, d)
    yoffsetm = np.mod(betay_off, -d) 
    mask = np.abs(yoffset) > np.abs(yoffsetm)
    try:
        yoffset[mask] = yoffsetm[mask]
    except TypeError:
        if mask:
            yoffset = yoffsetm
    
    ax = np.linspace(-betamax, betamax, n)
    ay = np.linspace(-betamax, betamax, n)
    step = ax[1]-ax[0]
    for i,bx in enumerate(ax):
        if i == 0:
            xtyp = -1
        elif i == n-1:
            xtyp = 1
        else:
            xtyp = 0
        for j,by in enumerate(ay):
            index = i*n+j
            if out is None:
                _out = None
            else:
                _out = out[index]
            
            if j == 0:
                ytyp = -1
            elif j == n-1:
                ytyp = 1
            else:
                ytyp = 0
            
            fmask = fft_window(betax,betay,bx+xoffset,by+yoffset,step,step,xtyp,ytyp,betamax, out = _out) 
            if out is None:
                out = np.empty((n*n,)+fmask.shape, fmask.dtype)
                out[0] = fmask
        
    return out

def fft_betaxy(shape, k0):
    bx,by = betaxy(shape[-2:], np.asarray(k0,FDTYPE)[...,None])
    return bx,by #np.broadcast_to(bx,shape),np.broadcast_to(by,shape)

def fft_betaxy_mean(betax, betay, fft_windows):
    axis = tuple(range(-betax.ndim,0))
    bx = betax*fft_windows
    by = betay*fft_windows
    norm = fft_windows.sum(axis = axis)
    bx = bx.sum(axis = axis) 
    by = by.sum(axis = axis) 
    mask = (norm>0)
    return np.divide(bx,norm, where = mask, out = bx), np.divide(by,norm, where = mask, out = by)

def betaxy2betaphi(bx,by):
    beta = (bx**2+by**2)**0.5
    phi = np.arctan2(by,bx)
    return beta, phi

@cached_result
def fft_mask(shape, k0, n, betax_off = 0., betay_off = 0., betamax = BETAMAX):
    
    betax, betay = fft_betaxy(shape, k0)
    windows = fft_windows(betax, betay, n, betax_off = betax_off, betay_off = betay_off, betamax = betamax)
    bxm, bym = fft_betaxy_mean(betax, betay, windows)
#    if betax.ndim == len(shape)+1:
#        bxm = bxm.mean(axis = tuple(range(2,bxm.ndim)))
#        bym = bym.mean(axis = tuple(range(2,bym.ndim)))
#        1/0
    return windows, betaxy2betaphi(bxm,bym)

#def _fftmask(shape, k0, n, betax = 0, betay = 0, betamax = BETAMAX):
#    d = 2*betamax/(n-1)
#    xoffset = np.mod(betax, d)
#    yoffset = np.mod(betay, d)
#    ax = np.linspace(-betamax-d, betamax+d, n+2)
#    ay = np.linspace(-betamax-d, betamax+d, n+2)
#    step = ax[1]-ax[0]
#    betax, betay = betaxy(shape, k0)
#    for bx in ax:
#        for by in ay:
#            fmask = fft_window(betax,betay,bx+xoffset,by+yoffset,step,step,betamax)
#            norm = fmask.mean()
#            if norm != 0.:                      
#                bxmean = (betax*fmask).mean()/norm
#                bymean = (betay*fmask).mean()/norm
#                yield fmask, bxmean, bymean
#
#
#
#
#@cached_result                
#def fftmask(shape, k0, n, betax = 0, betay = 0, betamax = BETAMAX):
#    data = [out for out in _fftmask(shape, k0, n, betax = betax, betay = betay, betamax = betamax)]       
#    betax = np.asarray([f[1] for f in data],FDTYPE)
#    betay = np.asarray([f[2] for f in data],FDTYPE)
#    fmask = [f[0] for f in data]
#    out = np.stack(fmask)
#    return out, (betax**2+betay**2)**0.5, np.arctan2(betay,betax)


def normalize_field(field, intensity_in, intensity_out, out = None):
    m = intensity_out == 0.
    intensity_out[m] = 1.
    intensity_in[m] = 0.
    fact = (intensity_in/intensity_out)[...,None,:,:]
    fact[fact<=-1] = -1
    fact[fact>=1] = 1
    fact = np.abs(fact)
    return np.multiply(field,fact, out = out) 


def diffract_normalized_fft(field, dmat, window = None, ref = None, out = None):
    if ref is not None:
        fref = fft2(ref, out = out)
        intensity1 = field2intensity(fref)
        f1 = fft2(field, out = out)
    else:
        f1 = fft2(field, out = out)
        intensity1 = field2intensity(f1)
    f2 = dotmf(dmat, f1 ,out = f1)
    intensity2 = field2intensity(f2)
    f3 = normalize_field(f2, intensity1, intensity2, out = f2)
    out = ifft2(f3, out = out)
    if window is not None:
        out = np.multiply(out,window,out = out)
    return out  

def diffract_normalized_local(field, dmat, window = None, ref = None, out = None):
    lmat = polarizer(jonesvec((1,1j)))
    rmat = polarizer(jonesvec((1,-1j)))
    
    #lmat = polarizer(jonesvec((1,0)))
    #rmat = polarizer(jonesvec((0,1)))    
    if ref is not None:
        intensity1l = field2intensity(apply_jones_matrix(lmat, ref))
        intensity1r = field2intensity(apply_jones_matrix(rmat, ref))
    else:
        intensity1l = field2intensity(apply_jones_matrix(lmat, field))
        intensity1r = field2intensity(apply_jones_matrix(rmat, field))
    f1 = fft2(field, out = out)
    
    f2 = dotmf(dmat, f1 ,out = f1)
    out = ifft2(f2, out = out)
    outl = apply_jones_matrix(lmat, out)
    outr = apply_jones_matrix(rmat, out)
    intensity2l = field2intensity(outl)
    intensity2r = field2intensity(outr)
    #out = normalize_field(out, intensity1, intensity2, out = out)
    outl = normalize_field(outl, intensity1l, intensity2l)
    outr = normalize_field(outr, intensity1r, intensity2r)
    np.add(outl,outr,out)
    if window is not None:
        out = np.multiply(out,window,out = out)
    return out 

def normalize_field_total(field, i1, i2, out = None):
    m = i2 == 0.
    i2[m] = 1.
    i1[m] = 0.
    fact = (i1/i2)
    fact[fact<=-1] = -1
    fact[fact>=1] = 1
    fact = np.abs(fact)
    fact = fact[...,None,None,None]
    return np.multiply(field,fact, out = out) 

def total_intensity(field):
    """Calculates total intensity of the field"""
    i = field2intensity(field)
    return i.sum(tuple(range(i.ndim))[-2:])#sum over pixels

def diffract_normalized_total(field, dmat, window = None, ref = None, out = None):
    if ref is not None:
        i1 = total_intensity(ref)
    else:
        i1 = total_intensity(field)
    f1 = fft2(field, out = out)
    
    f2 = dotmf(dmat, f1 ,out = f1)
    out = ifft2(f2, out = out)
    i2 = total_intensity(out)
    
    out = normalize_field_total(out, i1, i2, out = out)
    
    if window is not None:
        out = np.multiply(out,window,out = out)
    return out    

def _projected_field(field, wavenumbers, mode, n = 1, betamax = BETAMAX, norm = None, ref = None, out = None):
    eps = refind2eps([n]*3)
    pmat = projection_matrix(field.shape[-2:], wavenumbers, epsv = eps, epsa = (0.,0.,0.), mode = mode, betamax = betamax)

    if norm is None:
        return diffract(field, pmat, out = out) 
    elif norm == "fft":
        return diffract_normalized_fft(field, pmat, ref = ref,out = out)
    elif norm == "local":
        return diffract_normalized_local(field, pmat, ref = ref, out = out)
    elif norm == "total":
        return diffract_normalized_total(field, pmat, ref = ref, out = out)    
    else:
        raise ValueError("Unsupported normalization '{}'".format(norm))

def _projected_field2(field, wavenumbers, mode, n = 1, betamax = BETAMAX, norm = None, out = None):
    eps = refind2eps([n]*3)
    pmat = projection_matrix(field.shape[-2:], wavenumbers, epsv = eps, epsa = (0.,0.,0.), mode = mode, betamax = betamax)

    if norm is None:
        return diffract(field, pmat, out = out) 
    else:
        return diffract_normalized_local(field, pmat, out = out)
    
#def jones2H(jones,beta = 0., phi = 0., n = 1., out = None):
#    eps = refind2eps([n]*3)
#    layer = np.asarray((0.,0.,0.), dtype = FDTYPE)
#    alpha, f = alphaf(beta,phi,layer,eps) 
#    A = f[...,::2,::2]
#    B = f[...,1::2,::2]
#    Ai = inv(A)
#    D = dotmm(B,Ai)
#    return dotm1f(D,jones,out)

def jones2H(jones, wavenumbers, n = 1., betamax = BETAMAX, mode = +1, out = None):  
    eps = refind2eps([n]*3)
    shape = jones.shape[-2:]
    layer = np.asarray((0.,0.,0.), dtype = FDTYPE)
    alpha, f, fi = diffraction_alphaffi(shape, wavenumbers, epsv = eps, 
                            epsa = layer, betamax = betamax)
#    A = f[...,::2,::2]
#    B = f[...,1::2,::2]
#    Ai = inv(A)
#    D = dotmm(B,Ai)  
    D = E2H_mat(f, mode = mode)
    return diffract(jones, D, out = out) 


def transmitted_field(field, wavenumbers, n = 1, betamax = BETAMAX, norm = None,  ref = None, out = None):
    """Computes transmitted (forward propagating) part of the field.
    
    Parameters
    ----------
    field : ndarray
        Input field array
    wavenumbers : array_like
        Wavenumbers of the field
    n : float, optional
        Refractive index of the media (1 by default)
    betamax : float, optional
        Betamax perameter used.
    norm : str, optional
        If provided, transmitted field is normalized according to the selected
        mode. Possible values are ['fft', 'local', 'total']
    ref : ndarray, optional
        Reference field that is used to calculate normalization. If not provided,
        reflected waves of the input field are used for calculation.
    out : ndarray, optinal
        Output array
        
    Returns
    -------
    out : ndarray
       Transmitted field.
    """
        
    return _projected_field(np.asarray(field), wavenumbers, "t", n = n, 
                            betamax = betamax, norm = norm, ref = ref, out = out) 
    
def reflected_field(field, wavenumbers, n = 1, betamax = BETAMAX, norm = None, t = None, out = None):
    """Computes reflected (backward propagating) part of the field.
    
    Parameters
    ----------
    field : ndarray
        Input field array
    wavenumbers : array_like
        Wavenumbers of the field
    n : float, optional
        Refractive index of the media (1 by default)
    betamax : float
        Betamax perameter used.
    norm : bool, optional
        Whether to normalize field so that power spectrum of the output field
        remains the same as that of the input field.
    out : ndarray, optinal
        Output array
        
    Returns
    -------
    out : ndarray
       Reflected field.
    """
    return _projected_field2(np.asarray(field), wavenumbers, "r", n = n, betamax = betamax, norm = norm, out = out) 
    

def normalize_input_field(field, wavenumbers, rfield, n=1, betamax = BETAMAX, out = None):
    i1 = field2intensity(rfield)
    i1 = i1.sum(tuple(range(i1.ndim))[-2:])
    r = reflected_field(field, wavenumbers, n = n, betamax = betamax)
    t = field -r
    i2 = field2intensity(t)
    i2 = i2.sum(tuple(range(i2.ndim))[-2:])
    fact = (i1/i2)**0.5
    fact = fact[...,None,None,None]
    print (fact.max(),fact.min())
    r = r* fact
    return np.add(rfield, r, out = out)

def transfer_field(field_data, optical_data, beta = None, phi = None, nin = 1., nout = 1.,  
           npass = 1, nstep=1, diffraction = 1, reflection = 1, method = "2x2", 
           norm = DTMM_NORM_FFT, betamax = BETAMAX, split = False, 
           eff_data = None, ret_bulk = False):
    """Tranfers input field data through optical data.
    
    This function calculates transmitted field and possibly (when npass > 1) 
    updates input field with reflected waves. 
    
    
    Parameters
    ----------
    field_data : Field data tuple
        Input field data tuple
    optical_data : Optical data tuple
        Optical data tuple through which input field is transfered.
    beta : float or 1D array_like of floats
        Beta parameter of the input field. If it is a 1D array, beta[i] is the
        beta parameter of the field_data[0][i] field array.
    phi : float or 1D array_like of floats
        Phi angle of the input light field. If it is a 1D array, phi[i] is the
        phi parameter of the field_data[0][i] field array.
    nin : float, optional
        Refractive index of the input (bottom) surface (1. by default). Used
        in combination with npass > 1 to determine reflections from input layer,
        or in combination with reflection = True to include Fresnel reflection
        from the input surface.
    nout : float, optional
        Refractive index of the output (top) surface (1. by default). Used
        in combination with npass > 1 to determine reflections from output layer,
        or in combination with reflection = True to include Fresnel reflection
        from the output surface.
    npass: int, optional
        How many passes (iterations) to perform. For strongly reflecting elements
        this should be set to a higher value. If npass > 1, then input field data is
        overwritten and adds reflected light from the sample (defaults to 1).
    nstep: int or 1D array_like of ints
        Specifies layer propagation computation steps (defaults to 1). For thick 
        layers you may want to increase this number. If layer thickness is greater
        than pixel size, you should increase this number.
    diffraction : bool or int, optional
        Defines how diffraction is calculated. Setting this to False or 0 will 
        disable diffraction calculation. Diffraction is enabled by default.
        If specified as an integer, it defines diffraction calculation quality.
        1 for simple (fast) calculation, higher numbers increase accuracy 
        and decrease computation speed. You can set it to np.inf or -1 for max 
        (full) diffraction calculation and very slow computation.
    reflection : bool or int, optional
        Specifies reflection calculation mode for '2x2' method. It can be either
        0 or False for no reflections, 1 (default) for reflections in fft space 
        (from effective layers), or 2 for reflections in real space 
        (from individual layers). See documentation for details. 
    method : str, optional
        Specifies which method to use, either '2x2' (default) or '4x4'.
    norm : int, optional
        Normalization mode used when calculating multiple reflections with 
        npass > 1 and 4x4 method. Possible values are 0, 1, 2, default value is 1.
    split: bool, optional
        In multi-ray computation this option specifies whether to split 
        computation over single rays to consume less temporary memory storage.
        For large multi-ray datasets this option should be set.
    eff_data : Optical data tuple or None
        Optical data tuple of homogeneous layers through which light is diffracted
        in the diffraction calculation when diffraction >= 1. If not provided, 
        an effective data is build from optical_data by taking an average 
        isotropic refractive index of the material.
    ret_bulk : bool, optional
        Whether to return bulk field instead of the transfered field (default).
    """
    verbose_level = DTMMConfig.verbose
    if verbose_level >0:
        print("Transferring input field.")   
    if split == False:
        if method  == "4x4":
            return transfer_4x4(field_data, optical_data, beta = beta, 
                           phi = phi, eff_data = eff_data, nin = nin, nout = nout, npass = npass,nstep=nstep,
                      diffraction = diffraction, reflection = reflection,norm = norm,
                      betamax = betamax, ret_bulk = ret_bulk)
        return transfer_2x2(field_data, optical_data, beta = beta, 
                   phi = phi, eff_data = eff_data, nin = nin, nout = nout, npass = npass,nstep=nstep,
              diffraction = diffraction, reflection = reflection, betamax = betamax, ret_bulk = ret_bulk)
        
    else:#split input data by rays and compute ray-by-ray
        
        field_in,wavelengths,pixelsize = field_data
        if ret_bulk == True:
            #must have a length of optical data + 2 extra layers
            field_out = np.empty(shape = (len(optical_data[0])+2,)+field_in.shape, dtype = field_in.dtype)     
        else:
            field_out = np.empty_like(field_in) 

        nrays = len(beta)
        for i, bp in enumerate(zip(beta,phi)):
            if verbose_level >0:
                print("Ray {}/{}".format(i+1,nrays))
        
            field_data = (field_in[i],wavelengths, pixelsize)
            beta, phi = bp
            
            if ret_bulk == True:
                out = field_out[:,i]
            else:
                out = field_out[i]
            if method  == "4x4":
                 transfer_4x4(field_data, optical_data, beta = beta, 
                       phi = phi, eff_data = eff_data, nin = nin, nout = nout, npass = npass,nstep=nstep,
                  diffraction = diffraction, reflection = reflection,norm = norm,
                  betamax = betamax, out = out, ret_bulk = ret_bulk)
            else:
                transfer_2x2(field_data, optical_data, beta = beta, 
                   phi = phi, eff_data = eff_data, nin = nin, nout = nout, npass = npass,nstep=nstep,
              diffraction = diffraction, reflection = reflection, betamax = betamax, out = out, ret_bulk = ret_bulk)
        
            
        return field_out, wavelengths, pixelsize
        

def transfer_4x4(field_data, optical_data, beta = 0., 
                   phi = 0., eff_data = None, nin = 1., nout = 1., npass = 1,nstep=1,
              diffraction = True, reflection = 1, norm = DTMM_NORM_FFT,
              betamax = BETAMAX, ret_bulk = False, out = None):
    """Tranfers input field data through optical data. See transfer_field.
    """
    verbose_level = DTMMConfig.verbose
    if verbose_level >1:
        print(" * Initializing.")
        
    calc_reference = bool(norm & DTMM_NORM_REF)
    
    if (norm & DTMM_NORM_FFT):
        norm = "fft"
    else:
        if calc_reference:
            norm = "local"
        else:
            norm = "total"
    
    #define optical data
    d, epsv, epsa = validate_optical_data(optical_data)
       
    layers, eff_layers = _layers_list(optical_data, eff_data, nin, nout, nstep)
            
    #define input field data
    field_in, wavelengths, pixelsize = field_data
    
    #define constants 
    ks = k0(wavelengths, pixelsize)
    
    n = len(layers)

    if beta is None and phi is None:
        ray_tracing = False
        beta, phi = field2betaphi(field_in,ks)
    else:
        ray_tracing = False
    if diffraction != 1:
        ray_tracing = False
    beta, phi = _validate_betaphi(beta,phi,extendeddim = field_in.ndim-2)
    
    #define output field
    if out is None:
        if ret_bulk == True:
            bulk_out = np.zeros((n,)+field_in.shape, field_in.dtype)
            bulk_out[0] = field_in
            field_in = bulk_out[0]
            field_out = bulk_out[-1]
        else:
            bulk_out = None
            field_out = np.zeros_like(field_in)   
    else:
        out[...] = 0.
        if ret_bulk == True:
            bulk_out = out
            bulk_out[0] = field_in
            field_in = bulk_out[0]
            field_out = bulk_out[-1]
        else:
            bulk_out = None
            field_out = out
    indices = list(range(1,n-1))
    #if npass > 1:
    #    field0 = field_in.copy()
 
    field0 = field_in.copy()
    field = field_in.copy()

    
    field_in[...] = 0.
            
    
    if norm == 2:
        #make sure we take only the forward propagating part of the field
        transmitted_field(field, ks, n = nin, betamax = betamax, out = field)

    if calc_reference:
        ref = field.copy()
    else:
        ref = None    
    
    i0 = field2intensity(transmitted_field(field0, ks, n = nin, betamax = betamax))
    i0 = i0.sum(tuple(range(i0.ndim))[-2:]) 
    
    if reflection != 2 and 0<= diffraction and diffraction < np.inf:
        work_in_fft = True
    else:
        work_in_fft = False
        
    if work_in_fft:
        field = fft2(field,out = field)
    _reuse = False
    
    for i in range(npass):
        if verbose_level > 0:
            prefix = " * Pass {:2d}/{}".format(i+1,npass)
            suffix = ""
        else:
            prefix = ""
            suffix = "{}/{}".format(i+1,npass)

        _bulk_out = bulk_out
        direction = (-1)**i 
        _nstep, (thickness,ev,ea)  = layers[indices[0]]
        
        for pindex, jout in enumerate(indices):
            print_progress(pindex,n,level = verbose_level, suffix = suffix, prefix = prefix) 
            
            nstep, (thickness,ev,ea) = layers[jout]
            output_layer = (thickness*direction,ev,ea)

            _nstep, output_layer_eff = eff_layers[jout]

            d,e,a = output_layer_eff
            output_layer_eff = d*direction, e,a 
            
            if ray_tracing == True:
                if work_in_fft:
                    beta, phi = field2betaphi(ifft2(field),ks)
                else:
                    beta, phi = field2betaphi(field,ks)
                beta, phi = _validate_betaphi(beta,phi,extendeddim = field_in.ndim-2)
            
            if calc_reference and i%2 == 0:
                ref2, refl = propagate_2x2_effective_2(ref[...,::2,:,:], ks, None, output_layer ,None, output_layer_eff, 
                            beta = beta, phi = phi, nsteps = nstep, diffraction = diffraction, reflection = 0, 
                            betamax = betamax,mode = direction, out = ref[...,::2,:,:])
                
            if bulk_out is not None:
                out_field = _bulk_out[jout]
            else:
                out_field = field
            
            if diffraction >= 0 and diffraction < np.inf:
                if reflection ==2:
                    field = propagate_4x4_effective_2(field, ks, output_layer,output_layer_eff, 
                                beta = beta, phi = phi, nsteps = nstep, diffraction = diffraction, 
                                betamax = betamax, out = out_field, _reuse = _reuse)
                else:
                    field = propagate_4x4_effective_1(field, ks, output_layer,output_layer_eff, 
                                beta = beta, phi = phi, nsteps = nstep, diffraction = diffraction, 
                                betamax = betamax, out = out_field, _reuse = _reuse)                    
            else:
                field = propagate_4x4_full(field, ks, output_layer, 
                            nsteps = nstep, 
                            betamax = betamax, out = out_field)
            _reuse = True
        if ref is not None:
            ref[...,1::2,:,:] = jones2H(ref2,ks,betamax = betamax, n = nout)
        print_progress(n,n,level = verbose_level, suffix = suffix, prefix = prefix) 
        
        indices.reverse()
        
        if work_in_fft == True:
            field = ifft2(field, out = field)
            
        if npass > 1:
            if i%2 == 0:
                if i != npass -1:
                    if verbose_level > 1:
                        print(" * Normalizing transmissions.")
                    if calc_reference:
                        np.multiply(ref, (nin/nout)**0.5,ref)
                    field = transmitted_field(field, ks, n = nout, betamax = betamax, norm = norm, ref = ref, out = field)
#                    if window is not None:
#                        field = np.multiply(field,window,field)
                np.add(field_out, field, field_out)

                field = field_out.copy()
                
                
            else:
                field_in[...] = field
                if i != npass -1:
                    if verbose_level > 1:
                        print(" * Normalizing reflections.")
                    field = transmitted_field(field, ks, n = nin, betamax = betamax, norm = None, ref = None, out = field)
                    i0f = total_intensity(field)
                    fact = ((i0/i0f))
                    fact = fact[...,None,None,None]
                    np.multiply(field,fact, out = field) 
                    np.multiply(field_out,fact,out = field_out) 
                    np.multiply(field_in,fact,out = field_in) 
                    np.subtract(field0,field, out = field)
                    np.add(field_in,field,field_in)
                    if calc_reference:
                        ref = field.copy()

            if work_in_fft == True:
                field = fft2(field, out = field)
                        
        else:
            field_out[...] = field
            field_in[...] = field0
            
    if ret_bulk == True:
        if work_in_fft:
            ifft2(bulk_out[1:-1],out =bulk_out[1:-1])
        return bulk_out, wavelengths, pixelsize
    else:
        return field_out, wavelengths, pixelsize


def transfer_E_through_interface(field, wavenumbers, layer1, layer2, beta = 0., phi = 0.,
                                 diffraction = True, reflections = False, betamax = BETAMAX, mode = +1, input_fft = False, out = None):
    shape = field.shape[-2:]
    d1, epsv1, epsa1 = layer1
    d2, epsv2, epsa2 = layer2
    dmat = None
    rmat = None
    if diffraction == True:
        dmat = interface_jones_diffraction_matrix(shape, wavenumbers, beta,phi, d1=d1,
                d2 = d2, epsv2 = epsv2, epsa2 = epsa2,
                 epsv1 = epsv1, epsa1 = epsa1, mode = mode, betamax = betamax)
    if reflections == True:
        rmat = jones_transmission_matrix(shape, wavenumbers, epsv_in = epsv1, epsa_in = epsa1,
                                epsv_out = epsv2, epsa_out = epsa2, mode = mode, betamax = betamax)
    if rmat is not None:
        if dmat is not None:
            #dmat = dotmm(rmat,dmat)
            dmat = dotmm(dmat,rmat)
        else:
            dmat = rmat
    if input_fft == True:
        field = dotmf(dmat,field, out = out)
        return ifft2(field, out = out)
    else:
        return diffract(field, dmat, out = out)


def transfer_field_through_interface(field, wavenumbers, layer1, layer2, beta = 0., phi = 0.,
                                  betamax = BETAMAX, input_fft = False, out = None):
    shape = field.shape[-2:]
    d1, epsv1, epsa1 = layer1
    d2, epsv2, epsa2 = layer2
    dmat = interface_field_diffraction_matrix(shape, wavenumbers, beta,phi, d1=d1,
            d2 = d2, epsv2 = epsv2, epsa2 = epsa2,
             epsv1 = epsv1, epsa1 = epsa1, betamax = betamax)
    if input_fft == True:
        field = dotmf(dmat,field, out = out)
        return ifft2(field, out = out)
        
    else:
        return diffract(field, dmat, out = out)
    

def layer_alphaf(layer, beta, phi):
    d, epsv, epsa = layer
    out = alphaffi(beta,phi,epsv,epsa)
    layer_alphaf.out = out
    
    
layer_alphaf.out = None

def _dotmdmf(a,b,c,d,out = None):
    return dotmdmf(a,b,c,d,out = out)
    
def _transfer_ray_4x4_2(field, wavenumbers, layer,  beta = 0, phi=0,
                    nsteps = 1, dmat = None,
                    out = None, _reuse = False):
    
    func = _transfer_ray_4x4_2
    if _reuse == False:
        func.out_affi = None
        func.out_phase = None
    
    d, epsv, epsa = layer
    
    if dmat is None:
        kd = wavenumbers*d
    else:
        kd = wavenumbers*d/2    

    func.out_affi = alphaffi(beta,phi,epsv,epsa, out = func.out_affi)
    alpha, f, fi = func.out_affi
    func.out_phase = phasem(alpha,kd[...,None,None], out = func.out_phase)
    p = func.out_phase
  
    for j in range(nsteps):
        if dmat is None:
            field = dotmdmf(f,p,fi,field, out = out)  
        else:
            field = _dotmdmf(f,p,fi,field, out = out) 
            field = diffract(field, dmat, out = field)
            field = dotmdmf(f,p,fi,field, out = field)              
         
    return field


def _transfer_ray_4x4_1(field, wavenumbers, layer, effective_layer, beta = 0, phi=0,
                    nsteps = 1, 
                    betamax = BETAMAX, out = None, _reuse = False):
    
    func = _transfer_ray_4x4_1
    if _reuse == False:
        func.out_affi = None
        func.out_phase = None
    
    d, epsv, epsa = layer
    d_eff, epsv_eff, epsa_eff = effective_layer
    kd = wavenumbers*d 

    func.out_affi = alphaffi(beta,phi,epsv,epsa, out = func.out_affi)
    alpha, f, fi = func.out_affi
    func.out_phase = phasem(alpha,kd[...,None,None], out = func.out_phase)
    p = func.out_phase

    dmat1 = second_field_diffraction_matrix(field.shape[-2:], wavenumbers, beta, phi,d_eff/2, epsv = epsv_eff, 
                                        epsa =  epsa_eff,betamax = betamax) 
    dmat2 = first_field_diffraction_matrix(field.shape[-2:], wavenumbers, beta, phi,d_eff/2, epsv = epsv_eff, 
                                        epsa =  epsa_eff,betamax = betamax) 
    for j in range(nsteps):
        field = dotmf(dmat1,field, out = out)
        field = ifft2(field, out = field)
        field = dotmdmf(f,p,fi,field, out = field)  
        field = fft2(field, out = field)
        field = dotmf(dmat2,field, out = out)
                  
    return field


def _transfer_ray_2x2_1(fft_field, wavenumbers, layer, effective_layer_in,effective_layer_out, beta = 0, phi=0,
                    nsteps = 1, mode = +1, reflection = True, betamax = BETAMAX, refl = None, bulk = None, out = None):
    
    #fft_field = fft2(fft_field, out = out)
    shape = fft_field.shape[-2:]
    d_in, epsv_in,epsa_in = effective_layer_in     
    
    d_out, epsv_out,epsa_out = effective_layer_out    
    

    dmat1 = second_jones_diffraction_matrix(shape, wavenumbers, beta, phi,d_out/2, epsv = epsv_out, 
                                        epsa =  epsa_out, mode = mode, betamax = betamax) 
    dmat2 = first_jones_diffraction_matrix(shape, wavenumbers, beta, phi,d_out/2, epsv = epsv_out, 
                                        epsa =  epsa_out, mode = mode, betamax = betamax) 
    if reflection:
        tmat,rmat = jones_tr_matrix(shape, wavenumbers, epsv_in = epsv_in, epsa_in = epsa_in,
                            epsv_out = epsv_out, epsa_out = epsa_out, mode = mode, betamax = betamax)
    
    d, epsv, epsa = layer
    alpha, fmat = alphaf(beta,phi, epsv, epsa)
    
    e = E_mat(fmat, mode = mode) #2x2 E-only view of fmat
    ei = inv(e)
#    
    kd = wavenumbers * d   
    p = phase_mat(alpha,kd[...,None,None], mode = mode)


    for j in range(nsteps):
        if j == 0 and reflection:
            #reflect only at the beginning
            if refl is not None:
                trans = refl.copy()
                refl = dotmf(rmat, fft_field, out = refl)
                fft_field = dotmf(tmat, fft_field, out = out)
                fft_field = np.add(fft_field,trans, out = fft_field)

                fft_field = dotmf(dmat1, fft_field, out = fft_field)
            else:
                fft_field = dotmf(tmat, fft_field, out = out)
                fft_field = dotmf(dmat1, fft_field, out = fft_field)
                out = fft_field
        else:
            fft_field = dotmf(dmat1, fft_field, out = out)
            out = fft_field
        field = ifft2(fft_field, out = out)
        field = dotmdmf(e,p,ei,field, out = field)
        fft_field = fft2(field, out = field)
        fft_field = dotmf(dmat2, fft_field, out = fft_field)
    #return fft_field, refl  
    
    #out = ifft2(fft_field, out = out)
   
    if bulk is not None:
        field = ifft2(fft_field)
        e2h = E2H_mat(fmat, mode = mode)
        bulk[...,1::2,:,:] +=  dotmf(e2h, field)
        bulk[...,::2,:,:] += field
    
    return fft_field, refl



def _transfer_ray_2x2_2(field, wavenumbers, in_layer, out_layer, effective_layer, beta = 0, phi=0,
                    nsteps = 1, mode = +1,  diffraction = True, reflection = True, betamax = BETAMAX, refl = None, bulk = None, out = None):
    if in_layer is not None:
        d, epsv,epsa = in_layer        
        alpha, fmat_in = alphaf(beta,phi, epsv, epsa)
    d, epsv,epsa = out_layer        
    alpha, fmat = alphaf(beta,phi, epsv, epsa)    
    e = E_mat(fmat, mode = mode) #2x2 E-only view of fmat
#    if refl is not None:
#        ein = E_mat(fmat_in, mode = mode * (-1)) #reverse direction
#
#    if reflection == 0:
#        kd = wavenumbers * d
#    else:   
    kd = wavenumbers * d /2  
    p = phase_mat(alpha,kd[...,None,None], mode = mode)
    shape = field.shape[-2:]
    d_eff, epsv_eff, epsa_eff = effective_layer
    
    ei0 = inv(e)
    ei = ei0
    
    if diffraction:
        dmat = corrected_E_diffraction_matrix(shape, wavenumbers, beta,phi, d = d_eff,
                                 epsv = epsv_eff, epsa = epsa_eff, mode = mode, betamax = betamax)
    for j in range(nsteps):
        #reflect only at the beginning
        if j == 0 and reflection != 0:
            #if we need to track reflections (multipass)
            if refl is not None:
                ei,eri = Etri_mat(fmat_in, fmat, mode = mode)
                
                trans = refl.copy()
                refl = dotmf(eri, field, out = refl)
                
                field = dotmf(ei,field, out = out)
                field = np.add(field,trans, out = out)
                if d != 0.:                
                    field = dotmf(dotmd(e,p),field, out = out)
                else:
                    field = dotmf(e,field, out = out)
                
#                rmat = dotmm(ein, eri, out = eri)
#                trans = refl.copy()
#                refl = dotmf(rmat, field, out = refl)     
#                ei0 = inv(e)
#                field = dotmf(ei,field, out = out)
#                field = dotmf(e,field) + trans
#                if d != 0.:
#                    field = dotmdmf(e,p,ei0,field, out = out)
                    
                ei = ei0
            else:
                ei = Eti_mat(fmat_in, fmat, mode = mode)
                field = dotmdmf(e,p,ei,field, out = out)  
                ei = ei0
        else:
            #no need to compute if d == 0!.. identity
            if d != 0.:
                field = dotmdmf(e,p,ei,field, out = out)  
        if diffraction:   
            field = diffract(field, dmat)
        #no need to compute if d == 0!.. identity
        if d != 0.:
            field = dotmdmf(e,p,ei,field, out = out) 
            
    if bulk is not None:
        e2h = E2H_mat(fmat, mode = mode)
        bulk[...,1::2,:,:] +=  dotmf(e2h, field)
        bulk[...,::2,:,:] += field
     

    return field, refl


def _layers_list(optical_data, eff_data, nin, nout, nstep):
    """Build optical data layers list and effective data layers list.
    It appends/prepends input and output layers. A layer consists of
    a tuple of (n, thickness, epsv, epsa) where n is number of sublayers"""
    d, epsv, epsa = validate_optical_data(optical_data)  
    n = len(d)
    substeps = np.broadcast_to(np.asarray(nstep),(n,))
    layers = [(n,(t/n, ev, ea)) for n,t,ev,ea in zip(substeps, d, epsv, epsa)]
    #add input and output layers
    layers.insert(0, (1,(0., np.broadcast_to(refind2eps([nin]*3), epsv[0].shape), np.broadcast_to(np.array((0.,0.,0.), dtype = FDTYPE), epsa[0].shape))))
    layers.append((1,(0., np.broadcast_to(refind2eps([nout]*3), epsv[0].shape), np.broadcast_to(np.array((0.,0.,0.), dtype = FDTYPE), epsa[0].shape))))
    if eff_data is None:
        d_eff, epsv_eff, epsa_eff = _isotropic_effective_data(optical_data)
    else:
        d_eff, epsv_eff, epsa_eff = validate_optical_data(eff_data, homogeneous = True)        
    eff_layers = [(n,(t/n, ev, ea)) for n,t,ev,ea in zip(substeps, d_eff, epsv_eff, epsa_eff)]
    eff_layers.insert(0, (1,(0., refind2eps([nin]*3), np.array((0.,0.,0.), dtype = FDTYPE))))
    eff_layers.append((1,(0., refind2eps([nout]*3), np.array((0.,0.,0.), dtype = FDTYPE))))
    return layers, eff_layers


def field2betaphi2(field_in,ks):
    beta, phi = mean_betaphi2(field_in ,ks)
    if field_in.ndim > 4:  #must have at least two polarization states or multi-ray input
        beta = beta.mean(axis = tuple(range(1,field_in.ndim-3))) #average all, but first (multu-ray) axis
        phi = phi.mean(axis = tuple(range(1,field_in.ndim-3)))
    else:
        beta = beta.mean() #mean over all axes - single ray case
        phi = phi.mean()
    return _validate_betaphi(beta,phi,extendeddim = field_in.ndim-2)

def field2betaphi(field_in,ks):
    b = blackman(field_in.shape[-2:])
    f = fft2(field_in*b) #filter it with blackman..
    betax, betay = betaxy(field_in.shape[-2:], ks)
    beta, phi = mean_betaphi4(f,betax,betay)
    if field_in.ndim > 4:  #must have at least two polarization states or multi-ray input
        beta = beta.mean(axis = tuple(range(1,field_in.ndim-3))) #average all, but first (multu-ray) axis
        phi = phi.mean(axis = tuple(range(1,field_in.ndim-3)))
    else:
        beta = beta.mean() #mean over all axes - single ray case
        phi = phi.mean()
    return beta, phi

def transfer_2x2(field_data, optical_data, beta = None, 
                   phi = None, eff_data = None, nin = 1., nout = 1., npass = 1,nstep=1,
              diffraction = True, reflection = True,
              betamax = BETAMAX, ret_bulk = False, out = None):
    """Tranfers input field data through optical data using the 2x2 method
    See transfer_field for documentation.
    """
    verbose_level = DTMMConfig.verbose
    if verbose_level >1:
        print(" * Initializing.")
    
    #create layers lists
    layers, eff_layers = _layers_list(optical_data, eff_data, nin, nout, nstep)
    #define input field data
    field_in, wavelengths, pixelsize = field_data
    #wavenumbers
    ks = k0(wavelengths, pixelsize)
    n = len(layers) - 1#number of interfaces
    
    if beta is None and phi is None:
        ray_tracing = False
        beta, phi = field2betaphi(field_in,ks)
    else:
        ray_tracing = False
    if diffraction != 1:
        ray_tracing = False
    beta, phi = _validate_betaphi(beta,phi,extendeddim = field_in.ndim-2)

    
    #define output field
    if out is None:
        if ret_bulk == True:
            bulk_out = np.zeros((n+1,)+field_in.shape, field_in.dtype)
            bulk_out[0] = field_in
            field_in = bulk_out[0]
            field_out = bulk_out[-1]
        else:
            bulk_out = None
            field_out = np.zeros_like(field_in)   
    else:
        out[...] = 0.
        if ret_bulk == True:
            bulk_out = out
            bulk_out[0] = field_in
            field_in = bulk_out[0]
            field_out = bulk_out[-1]
        else:
            bulk_out = None
            field_out = out
             
    indices = list(range(n))
        
    #make sure we take only the forward propagating part of the field
    field0 = transmitted_field(field_in, ks, n = nin, betamax = betamax)
    field = field0[...,::2,:,:].copy()
    
    if npass > 1:
        if reflection == 0:
            raise ValueError("Reflection mode `0` not compatible with npass > 1")
        field_in[...] = field0 #modify input field so that it has no back reflection
        #keep reference to reflected waves
        refl = np.zeros(shape = (n+1,) + field.shape[:-3] + (2,) + field.shape[-2:], dtype = field.dtype)
    else:
        #no need to store reflected waves
        refl = [None]*n
        
    if reflection == 1 and 0<= diffraction and diffraction < np.inf:
        work_in_fft = True
    else:
        work_in_fft = False
        
    if work_in_fft:
        field = fft2(field,out = field)

    for i in range(npass):
        if verbose_level > 0:
            prefix = " * Pass {:2d}/{}".format(i+1,npass)
            suffix = ""
        else:
            prefix = ""
            suffix = "{}/{}".format(i+1,npass)
            
        if i > 0:
            field[...] = 0.
       
        direction = (-1)**i 
        _nstep, (thickness,ev,ea)  = layers[indices[0]]

        for pindex, j in enumerate(indices):
            print_progress(pindex,n,level = verbose_level, suffix = suffix, prefix = prefix) 
            
            jin = j+(1-direction)//2
            jout = j+(1+direction)//2
            _nstep, (thickness,ev,ea) = layers[jin]
            input_layer = (thickness,ev,ea)
            nstep, (thickness,ev,ea) = layers[jout]
            output_layer = (thickness*direction,ev,ea)

            _nstep, input_layer_eff = eff_layers[jin]
            _nstep, output_layer_eff = eff_layers[jout]

            d,e,a = input_layer_eff
            input_layer_eff = d*direction, e,a
            d,e,a = output_layer_eff
            output_layer_eff = d*direction, e,a     
        
            
            if jout == 0:
                bulk = field_in
            elif jout == len(indices):
                bulk = field_out
            else:
                if bulk_out is None:
                    bulk = None
                else:
                    bulk = bulk_out[jout]
                    
            if ray_tracing == True:
                if work_in_fft:
                    beta, phi = field2betaphi(ifft2(field),ks)
                else:
                    beta, phi = field2betaphi(field,ks)
                beta, phi = _validate_betaphi(beta,phi,extendeddim = field_in.ndim-2)
                    
            if diffraction >= 0 and diffraction < np.inf:


                if reflection == 1 :
                    field, refli = propagate_2x2_effective_1(field, ks, input_layer, output_layer ,input_layer_eff, output_layer_eff, 
                            beta = beta, phi = phi, nsteps = nstep, diffraction = diffraction, reflection = reflection, 
                            betamax = betamax,mode = direction, refl = refl[j], bulk = bulk)
                else:
                    field, refli = propagate_2x2_effective_2(field, ks, input_layer, output_layer ,input_layer_eff, output_layer_eff, 
                            beta = beta, phi = phi, nsteps = nstep, diffraction = diffraction, reflection = reflection, 
                            betamax = betamax,mode = direction, refl = refl[j], bulk = bulk)
                

            else:
                field, refli = propagate_2x2_full(field, ks, output_layer, input_layer = input_layer, 
                    nsteps = 1,  reflection = reflection, mode = direction,
                    betamax = betamax, refl = refl[j], bulk = bulk)

        print_progress(n,n,level = verbose_level, suffix = suffix, prefix = prefix) 
        
        indices.reverse()

    if ret_bulk == True:
        return bulk_out, wavelengths, pixelsize
    else:
        return field_out, wavelengths, pixelsize  
    
    
def propagate_4x4_effective_2(field, wavenumbers, layer, effective_layer, beta = 0, phi=0,
                    nsteps = 1, diffraction = True, 
                    betamax = BETAMAX,out = None, _reuse = False):
    
    d_eff, epsv_eff, epsa_eff = effective_layer
    
    if diffraction <= 1:
        
        if diffraction != 0:
            dmat = corrected_field_diffraction_matrix(field.shape[-2:], wavenumbers, beta,phi, d=d_eff,
                                 epsv = epsv_eff, epsa = epsa_eff, betamax = betamax)
        else:
            dmat = None
        
        return _transfer_ray_4x4_2(field, wavenumbers, layer,
                                beta = beta, phi = phi, nsteps =  nsteps, dmat = dmat,
                                out = out, _reuse = _reuse)
    else:
        fout = 0.
        field = fft2(field)

        try: 
            beta_shape = beta.shape
            beta = beta[...,0]
            phi = phi[...,0]
        except IndexError:
            beta_shape = ()
            
        windows, (betas, phis) = fft_mask(field.shape, wavenumbers, int(diffraction), 
                 betax_off = beta*np.cos(phi), betay_off = beta*np.sin(phi), betamax = betamax)    
        
        for window, b, p  in zip(windows, betas, phis):
            fpart = field * window
            fpart_re = ifft2(fpart)
            
            b = b.reshape(beta_shape)
            p = p.reshape(beta_shape)
            
            if diffraction != 0:
                dmat = corrected_field_diffraction_matrix(field.shape[-2:], wavenumbers, b,p, d=d_eff,
                                     epsv = epsv_eff, epsa = epsa_eff)
            else:
                dmat = None

            _out =  _transfer_ray_4x4_2(fpart_re, wavenumbers, layer, 
                                beta = b, phi = p, nsteps =  nsteps,
                                dmat = dmat,_reuse = _reuse)                       
            fout += _out

        if out is not None:
            out[...] = fout
        else:
            out = fout
        return out

def propagate_4x4_effective_1(field, wavenumbers, layer, effective_layer, beta = 0, phi=0,
                    nsteps = 1, diffraction = True, 
                    betamax = BETAMAX,out = None,_reuse = False ):
    if diffraction == 1:
        return _transfer_ray_4x4_1(field, wavenumbers, layer,effective_layer, 
                                beta = beta, phi = phi, nsteps =  nsteps, 
                                betamax = betamax,  out = out,_reuse = _reuse)
    else:
        fout = np.zeros_like(out)
        _out = None

        try: 
            beta_shape = beta.shape
            beta = beta[...,0]
            phi = phi[...,0]
        except IndexError:
            beta_shape = ()
            
        windows, (betas, phis) = fft_mask(field.shape, wavenumbers, int(diffraction), 
                 betax_off = beta*np.cos(phi), betay_off = beta*np.sin(phi), betamax = betamax)    

        for window, b, p  in zip(windows, betas, phis):
            fpart = np.multiply(field, window, out = _out)

            _out =  _transfer_ray_4x4_1(fpart, wavenumbers, layer, effective_layer,
                                beta = b.reshape(beta_shape), phi = p.reshape(beta_shape), nsteps =  nsteps,
                                betamax = betamax, out = _out,_reuse = _reuse)                       
            fout = np.add(fout, _out, out = fout)

        if out is not None:
            out[...] = fout
        else:
            out = fout
        return out

        
def propagate_2x2_effective_1(field, wavenumbers, layer_in, layer_out, effective_layer_in, 
                            effective_layer_out, beta = 0, phi = 0,
                            nsteps = 1, diffraction = True, reflection = True, 
                            betamax = BETAMAX,  mode = +1, 
                            refl = None, bulk = None, out = None):
    
    if diffraction <= 1:
        return _transfer_ray_2x2_1(field, wavenumbers, layer_out, effective_layer_in, effective_layer_out,
                                beta = beta, phi = phi, nsteps =  nsteps,reflection = reflection,
                                betamax = betamax, mode = mode, refl = refl, bulk = bulk, out = out)            
    else:
        fout = 0.

        if refl is not None:
            _refl = 0.
        else:
            _refl = None
        try: 
            beta_shape = beta.shape
            beta = beta[...,0]
            phi = phi[...,0]
        except IndexError:
            beta_shape = ()
            
        windows, (betas, phis) = fft_mask(field.shape, wavenumbers, int(diffraction), 
                 betax_off = beta*np.cos(phi), betay_off = beta*np.sin(phi), betamax = betamax)    

        for window, b, p  in zip(windows, betas, phis):
            fpart = field * window
            
            if refl is not None:
                reflpart = refl * window
            else:
                reflpart = None
            _out, __refl = _transfer_ray_2x2_1(fpart, wavenumbers,layer_out, 
                                                effective_layer_in, effective_layer_out,
                            beta = b.reshape(beta_shape), phi = p.reshape(beta_shape), 
                            nsteps =  nsteps,reflection = reflection,
                            betamax = betamax, mode = mode, bulk = bulk,
                            out = out,  refl = reflpart)
             
            fout += _out
            if refl is not None and reflection != 0:
                _refl += __refl

    if out is not None:
        out[...] = fout
    else:
        out = fout
    if refl is not None:
        refl[...] = _refl
    return out, refl   

def propagate_2x2_effective_2(field, wavenumbers, layer_in, layer_out, effective_layer_in, 
                            effective_layer_out, beta = 0, phi = 0,
                            nsteps = 1, diffraction = True, reflection = True, 
                            betamax = BETAMAX,  mode = +1, 
                            refl = None, bulk = None, out = None):
    
    if diffraction <= 1:
        return _transfer_ray_2x2_2(field, wavenumbers, layer_in, layer_out, effective_layer_out,
                                beta = beta, phi = phi, nsteps =  nsteps,diffraction = diffraction,
                                reflection = reflection,
                                betamax = betamax, mode = mode, refl = refl, bulk = bulk, out = out)            
    else:
        fout = 0.
        field = fft2(field)

        if refl is not None:
            _refl = 0.
        else:
            _refl = None
        try: 
            beta_shape = beta.shape
            beta = beta[...,0]
            phi = phi[...,0]
        except IndexError:
            beta_shape = ()
            
        windows, (betas, phis) = fft_mask(field.shape, wavenumbers, int(diffraction), 
                 betax_off = beta*np.cos(phi), betay_off = beta*np.sin(phi), betamax = betamax)    

        for window, b, p  in zip(windows, betas, phis):
            fpart = field * window
            fpart_re = ifft2(fpart)
            
            if refl is not None:
                reflpart = refl * window
                reflpart_re = ifft2(reflpart)
            else:
                reflpart_re = None
            _out, __refl = _transfer_ray_2x2_2(fpart_re, wavenumbers, layer_in, layer_out, effective_layer_out,
                                beta = b.reshape(beta_shape), phi = p.reshape(beta_shape), nsteps =  nsteps,
                                diffraction = diffraction,betamax = betamax, reflection = reflection,
                                mode = mode, bulk = bulk,out = out,  refl = reflpart_re)                       
    
            fout += _out
            if refl is not None and reflection != 0:
                _refl += __refl

    if out is not None:
        out[...] = fout
    else:
        out = fout
    if refl is not None:
        refl[...] = _refl

    return out, refl
 
    
def propagate_2x2_full(field, wavenumbers, layer, input_layer = None, 
                    nsteps = 1,  mode = +1, reflection = True,
                    betamax = BETAMAX, refl = None, bulk = None, out = None):

    shape = field.shape[-2:]
   
    d, epsv, epsa = layer
    if input_layer is not None:
        d_in, epsv_in, epsa_in = input_layer

    kd = wavenumbers*d/nsteps
    
    if out is None:
        out = np.empty_like(field)
        
    ii,jj = np.meshgrid(range(shape[0]), range(shape[1]),copy = False, indexing = "ij") 
    
    for step in range(nsteps):
        for i in range(len(wavenumbers)):
            ffield = fft2(field[...,i,:,:,:])
            ofield = np.zeros_like(out[...,i,:,:,:])

            b,p = betaphi(shape,wavenumbers[i])
            mask = b < betamax
            
            amplitude = ffield[...,mask]
            
            betas = b[mask]
            phis = p[mask]
            iind = ii[mask]
            jind = jj[mask]
            
            if bulk is not None:
                obulk = bulk[...,i,:,:,:]
            
            if refl is not None:
                tampl = fft2(refl[...,i,:,:,:])[...,mask]
                orefl = refl[...,i,:,:,:]
                orefl[...] = 0.
              
            for bp in sorted(zip(range(len(betas)),betas,phis,iind,jind),key = lambda el : el[1], reverse = False):     
                #for j,bp in enumerate(zip(betas,phis,iind,jind)):     
                              
                j, beta, phi, ieig, jeig = bp

                out_af = alphaf(beta,phi,epsv,epsa)
                alpha,fmat_out = out_af 
                e = E_mat(fmat_out, mode = mode)
                ei0 = inv(e)
                ei = ei0                        
                pm = phase_mat(alpha,kd[i,None,None],mode = mode)
                w = eigenwave(amplitude.shape[:-1]+shape, ieig,jeig, amplitude = amplitude[...,j])
                if step == 0 and reflection != False:
                    alphain, fmat_in = alphaf(beta,phi,epsv_in,epsa_in)
                    if refl is not None:
                        ei,eri = Etri_mat(fmat_in, fmat_out, mode = mode)
                        ein =  E_mat(fmat_in, mode = -1*mode)
                        t = eigenwave(amplitude.shape[:-1]+shape, ieig,jeig, amplitude = tampl[...,j])
                        r = dotmf(eri, w)
                        r = dotmf(ein,r, out = r)
                        np.add(orefl,r,orefl)
                    
                        w = dotmf(ei, w, out = w)
                        t = dotmf(ei0,t, out = t)
                        w = np.add(t,w,out = w)
  
                    else:
                        ei = Eti_mat(fmat_in, fmat_out, mode = mode)
                        w = dotmf(ei, w, out = w)
                    w = dotmf(dotmd(e,pm),w, out = w)
                    np.add(ofield,w,ofield)                         
                    
                else:
                    w = dotmdmf(e,pm,ei,w, out = w)
                    np.add(ofield,w,ofield)

                if bulk is not None:
                    e2h = E2H_mat(fmat_out, mode = mode)
                    obulk[...,1::2,:,:] +=  dotmf(e2h, w)
                    obulk[...,::2,:,:] += w
                
            out[...,i,:,:,:] = ofield
                        

                    
        field = out
    return out, refl

    
def propagate_4x4_full(field, wavenumbers, layer, 
                    nsteps = 1,  betamax = BETAMAX, out = None):

    shape = field.shape[-2:]

    d, epsv, epsa = layer

    kd = wavenumbers*d/nsteps
    
    if out is None:
        out = np.empty_like(field)
    
    out_af = None
    pm = None
    
    ii,jj = np.meshgrid(range(shape[0]), range(shape[1]),copy = False, indexing = "ij") 
    
    for step in range(nsteps):
        for i in range(len(wavenumbers)):
            ffield = fft2(field[...,i,:,:,:])
            ofield = np.zeros_like(out[...,i,:,:,:])
            b,p = betaphi(shape,wavenumbers[i])
            mask = b < betamax
            
            amplitude = ffield[...,mask]
            
            betas = b[mask]
            phis = p[mask]
            iind = ii[mask]
            jind = jj[mask]
              
            for bp in sorted(zip(range(len(betas)),betas,phis,iind,jind),key = lambda el : el[1], reverse = False):     
                        
                #for j,bp in enumerate(zip(betas,phis,iind,jind)):     
                              
                j, beta, phi, ieig, jeig = bp

                out_af = alphaffi(beta,phi,epsv,epsa, out = out_af)
                alpha,f,fi = out_af                      
                        
                pm = phasem(alpha,kd[i], out = pm)
                w = eigenwave(amplitude.shape[:-1]+shape, ieig,jeig, amplitude = amplitude[...,j])
                w = dotmdmf(f,pm,fi,w, out = w)
                np.add(ofield,w,ofield)
                
            out[...,i,:,:,:] = ofield
        field = out
    return out


__all__ = ["transfer_field", "transmitted_field", "reflected_field"]
