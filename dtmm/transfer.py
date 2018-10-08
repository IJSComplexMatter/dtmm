"""
Main top level calculation functions for light propagation through optical data.
"""
from __future__ import absolute_import, print_function, division

from dtmm.conf import DTMMConfig, cached_function, BETAMAX, FDTYPE
from dtmm.wave import k0, eigenwave, betaphi, betaxy

from dtmm.data import uniaxial_order, refind2eps, validate_optical_data
from dtmm.tmm import alphaffi, phasem,  alphajji, alphaf, transmission_mat, E2H_mat
from dtmm.linalg import dotmdm, dotmm, dotmf, dotmdmf, inv
from dtmm.print_tools import print_progress
from dtmm.diffract import diffraction_alphaffi, projection_matrix, diffract, phase_matrix, \
                jones_diffraction_matrix, jones_transmission_matrix
from dtmm.field import field2intensity, select_fftfield
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
def jones_correction_matrix(beta,phi,ks, d=1., epsv = (1,1,1), epsa = (0,0,0.), out = None):
    alpha, j, ji = alphajji(beta,phi,epsv,epsa)  
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
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
    dmat = jones_diffraction_matrix(shape, ks, d, epsv, epsa, betamax = betamax)
    cmat = jones_correction_matrix(beta, phi, ks, d, epsv, epsa)
    if d > 0:
        return dotmm(dmat,cmat, out = None)
    else:
        return dotmm(cmat, dmat, out = None)

    
@cached_function
def first_jones_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
    dmat = jones_diffraction_matrix(shape, ks, d, epsv, epsa, betamax = betamax)
    cmat = jones_correction_matrix(beta, phi, ks, d, epsv, epsa)
    return dotmm(dmat,cmat, out = None)

@cached_function
def second_jones_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
    dmat = jones_diffraction_matrix(shape, ks, d, epsv, epsa, betamax = betamax)
    cmat = jones_correction_matrix(beta, phi, ks, d, epsv, epsa)
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
                                 betamax = BETAMAX, out = None):
    
    dmat2 = second_jones_diffraction_matrix(shape, ks, beta, phi,d = d2/2.,
                                               epsv = epsv2,epsa = epsa2,
                                               betamax = betamax)

    dmat1 = first_jones_diffraction_matrix(shape, ks, beta, phi,d = d1/2.,
                                               epsv = epsv1,epsa = epsa1,
                                               betamax = betamax)

    return dotmm(dmat2,dmat1)


    
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

def jones2H(jones, wavenumbers, n = 1., betamax = BETAMAX, out = None):  
    eps = refind2eps([n]*3)
    shape = jones.shape[-2:]
    layer = np.asarray((0.,0.,0.), dtype = FDTYPE)
    alpha, f, fi = diffraction_alphaffi(shape, wavenumbers, epsv = eps, 
                            epsa = layer, betamax = betamax)
#    A = f[...,::2,::2]
#    B = f[...,1::2,::2]
#    Ai = inv(A)
#    D = dotmm(B,Ai)  
    D = E2H_mat(f)
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

def transfer_field(field_data, optical_data, beta = 0., phi = 0., nin = 1., nout = 1.,  
           npass = 1,nstep=1, diffraction = True, reflections = True, interference = False, 
           norm = DTMM_NORM_FFT, window = None, betamax = BETAMAX, split = False, 
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
        or in combination with reflections = True to include Fresnel reflection
        from the input surface.
    nout : float, optional
        Refractive index of the output (top) surface (1. by default). Used
        in combination with npass > 1 to determine reflections from output layer,
        or in combination with reflections = True to include Fresnel reflection
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
    reflections : bool, optional
        Whether to include reflections from effective layers or not. Only effective
        when interference is disabled.
    interference : bool, optional
        Whether to enable interference. Note that interference is automatically 
        enabled with npass > 1.
    norm : int, optional
        Normalization mode used when calculating multiple reflections with 
        npass > 1. Possible values are 0, 1, 2, 3, default value is 1.
    window: array or None
        If specified, computed field data is multiplied with this window after 
        each pass.
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
        return _transfer_field(field_data, optical_data, beta = beta, 
                       phi = phi, eff_data = eff_data, nin = nin, nout = nout, npass = npass,nstep=nstep,
                  diffraction = diffraction, reflections = reflections, interference = interference, norm = norm,
                  window = window, betamax = betamax, ret_bulk = ret_bulk)
    else:#split input data by rays and compute ray-by-ray
        
        field_in,wavelengths,pixelsize = field_data
        if ret_bulk == True:
            field_out = np.empty(shape = (len(optical_data[0]),)+field_in.shape)     
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
            _transfer_field(field_data, optical_data, beta = beta, 
                       phi = phi, eff_data = eff_data, nin = nin, nout = nout, npass = npass,nstep=nstep,
                  diffraction = diffraction, reflections = reflections, interference = interference, norm = norm,
                  window = window, betamax = betamax, out = out, ret_bulk = ret_bulk)
            
        return field_out, wavelengths, pixelsize
        

def _transfer_field(field_data, optical_data, beta = 0., 
                   phi = 0., eff_data = None, nin = 1., nout = 1., npass = 1,nstep=1,
              diffraction = True, reflections = True, interference = False, quality = 0, norm = DTMM_NORM_FFT,
              window = None, betamax = BETAMAX, method = "effective",ret_bulk = False, out = None):
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
        
    #define effective optical data
    if eff_data is None:
        d_eff, epsv_eff, epsa_eff = _isotropic_effective_data((d, epsv, epsa))
    else:
        d_eff, epsv_eff, epsa_eff = validate_optical_data(eff_data, homogeneous = True)
        
    #define input field data
    field_in, wavelengths, pixelsize = field_data
    
    #define constants 
    ks = k0(wavelengths, pixelsize)
    n = len(d)
    substeps = np.broadcast_to(np.asarray(nstep),(n,))
    
    #define input ray directions. Either a scalar or 1D array
    beta0, phi0 = _validate_betaphi(beta,phi,extendeddim = field_in.ndim-4)
    beta, phi = _validate_betaphi(beta,phi,extendeddim = field_in.ndim-2)
    
    #define output field
    if out is None:
        if ret_bulk == True:
            bulk_out = np.zeros((n,)+field_in.shape, field_in.dtype)
            field_out = bulk_out[-1]
        else:
            bulk_out = None
            field_out = np.zeros_like(field_in)   
    else:
        out[...] = 0.
        if ret_bulk == True:
            bulk_out = out
            field_out = out[-1]
        else:
            bulk_out = None
            field_out = out
    
    #if npass > 1:
    #    field0 = field_in.copy()
 
    field0 = field_in.copy()
    field = field_in.copy()

    
    field_in[...] = 0.
            
    indices = list(range(n))
    interference = True if npass > 1 else interference
    mode = "t" if interference == False else None
    
    if mode == "t" or (norm == 2 and interference == True):
        #make sure we take only the forward propagating part of the field
        transmitted_field(field, ks, n = nin, betamax = betamax, out = field)

    if calc_reference:
        ref = field.copy()
    else:
        ref = None    
    
    i0 = field2intensity(transmitted_field(field0, ks, n = nin, betamax = betamax))
    i0 = i0.sum(tuple(range(i0.ndim))[-2:]) 

    out_affi = None #tmp data
    out_phase = None
    
    if interference == True:
        reflections = False #disables fresnel reflections.. 
        
    #initial input layer
    

    for i in range(npass):
        if verbose_level > 0:
            prefix = " * Pass {:2d}/{}".format(i+1,npass)
            suffix = ""
        else:
            prefix = ""
            suffix = "{}/{}".format(i+1,npass)

        if mode == "t":
            
            field = field[...,::2,:,:]
            if bulk_out is not None:
                _bulk_out = bulk_out[...,::2,:,:]
            else:
                _bulk_out = None
        else:
            _bulk_out = bulk_out
        direction = (-1)**i 
        
        #input layer is for calculation of back propagation step... and fresnel reflections
        #thickness of the layer should be zero here!  
        if direction == 1:
            input_layer = (0., refind2eps([nin]*3), np.array((0.,0.,0.), dtype = FDTYPE))
        else:
            input_layer = (0., refind2eps([nout]*3), np.array((0.,0.,0.), dtype = FDTYPE))
        output_layer = None
        
        for pindex, j in enumerate(indices):
            print_progress(pindex,n,level = verbose_level, suffix = suffix, prefix = prefix) 
            
            thickness = d[j]*direction
            thickness_eff = d_eff[j]*direction

            if pindex == n-1:
                if direction == 1:
                    output_layer = (0., refind2eps([nout]*3), np.array((0.,0.,0.), dtype = FDTYPE))
                else:
                    output_layer = (0., refind2eps([nin]*3), np.array((0.,0.,0.), dtype = FDTYPE))
                    
            if calc_reference and i%2 == 0 and interference == True:
                ref2 = propagate_field_effective(ref[...,::2,:,:], ks, (thickness, epsv[j], epsa[j]),(thickness_eff, epsv_eff[j], epsa_eff[j]), 
                            beta = beta, phi = phi, nsteps = substeps[j], diffraction = diffraction, reflections = False, mode = "t", 
                            input_layer = input_layer,output_layer = output_layer ,
                            betamax = betamax, ret_affi = False, out = ref[...,::2,:,:])
                

            if bulk_out is not None:
                out_field = _bulk_out[j]
            else:
                out_field = field
            
            if diffraction >= 0 and diffraction < np.inf:
                out_affi,out_phase,field = propagate_field_effective(field, ks, (thickness, epsv[j], epsa[j]),(thickness_eff, epsv_eff[j], epsa_eff[j]), 
                            beta = beta, phi = phi, nsteps = substeps[j], diffraction = diffraction, reflections = reflections, 
                            mode = mode, input_layer = input_layer, output_layer = output_layer ,
                            betamax = betamax, ret_affi = True, out = out_field, out_affi = out_affi, out_phase = out_phase)
            else:
                field = propagate_field_full(field, ks, (thickness, epsv[j], epsa[j]), 
                            nsteps = substeps[j], mode = mode,
                            betamax = betamax, out = out_field)
            if bulk_out is not None and mode == "t":

#FIXME: currently H field is computed if eff_data is isotropic... make it work for anisotropic!!
                bulk_out[j,...,1::2,:,:] = jones2H(field,ks,betamax = betamax, n = np.abs(epsv_eff[j,0])**0.5)
            

            #now set input layer to this layer... for calculation of diffraction correction step in the next layer.
            input_layer = (thickness_eff/substeps[j], epsv_eff[j], epsa_eff[j])
        
        if ref is not None:
            ref[...,1::2,:,:] = jones2H(ref2,ks,betamax = betamax, n = nout)
        print_progress(n,n,level = verbose_level, suffix = suffix, prefix = prefix) 
        
        indices.reverse()
        if npass > 1:
            if i%2 == 0:
                if i != npass -1:
                    if verbose_level > 1:
                        print(" * Normalizing transmissions.")
                    if calc_reference:
                        np.multiply(ref, (nin/nout)**0.5,ref)
                    field = transmitted_field(field, ks, n = nout, betamax = betamax, norm = norm, ref = ref, out = field)
                    if window is not None:
                        field = np.multiply(field,window,field)
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
                        
        else:
            if mode == "t":
                field_out[...,::2,:,:] = field
                #field_out[...,1::2,:,:] = jones2H(field,beta = beta0, phi = phi0, n = nout)
                field_out[...,1::2,:,:] = jones2H(field,ks,betamax = betamax, n = nout)
                

                    
                
                if reflections == False:
                    #assure conservation of energy...
                    np.multiply(field_out, (nin/nout)**0.5,field_out)
            else:
                field_out[...] = field
            field_in[...] = field0
            
    if ret_bulk == True:
        return bulk_out, wavelengths, pixelsize
    else:
        return field_out, wavelengths, pixelsize


def transfer_E_through_interface(field, wavenumbers, layer1, layer2, beta = 0., phi = 0.,
                                 diffraction = True, reflections = False, betamax = BETAMAX, input_fft = False, out = None):
    shape = field.shape[-2:]
    d1, epsv1, epsa1 = layer1
    d2, epsv2, epsa2 = layer2
    dmat = None
    rmat = None
    if diffraction == True:
        dmat = interface_jones_diffraction_matrix(shape, wavenumbers, beta,phi, d1=d1,
                d2 = d2, epsv2 = epsv2, epsa2 = epsa2,
                 epsv1 = epsv1, epsa1 = epsa1, betamax = betamax)
    if reflections == True:
        rmat = jones_transmission_matrix(shape, wavenumbers, epsv_in = epsv1, epsa_in = epsa1,
                                epsv_out = epsv2, epsa_out = epsa2, betamax = betamax)
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
    

    
def _propagate_ray(field, wavenumbers, layer, effective_layer, beta = 0, phi=0,
                    nsteps = 1, diffraction = True, reflections = True, mode = None, input_layer = None,output_layer = None,
                    betamax = BETAMAX, ret_affi = False, out_affi = None, out_phase = None,out = None,input_fft = False):
    
    d, epsv, epsa = layer
    d_eff, epsv_eff, epsa_eff = effective_layer
    kd = wavenumbers*d/nsteps
    d_eff = d_eff/nsteps
    

    d_in, epsv_in, epsa_in = input_layer #d_in of the previous layer has already been determined corectly (with nstep in the previous run)
    
    if mode == "t":
        alpha, f, fi = alphajji(beta,phi,epsv,epsa, out = out_affi)
    else:
        alpha, f, fi = alphaffi(beta,phi,epsv,epsa, out = out_affi)

    p = phasem(alpha,kd[...,None,None], out = out_phase)
    
    
    for j in range(nsteps):
        if mode == "t":
            #diffract & reflect, or reflect only
            if diffraction == True or reflections == True:
                field = transfer_E_through_interface(field, wavenumbers, (d_in, epsv_in, epsa_in),(d_eff, epsv_eff, epsa_eff) , beta = beta, phi = phi,
                                 diffraction = diffraction, reflections = reflections,  betamax = betamax, input_fft = input_fft, out = out)
        
        else:
            if diffraction == True:
                field = transfer_field_through_interface(field, wavenumbers, (d_in, epsv_in, epsa_in),(d_eff, epsv_eff, epsa_eff) , beta = beta, phi = phi,
                                  betamax = betamax, input_fft = input_fft, out = out)
                
        d_in, epsv_in, epsa_in = d_eff, epsv_eff, epsa_eff #make this layer as input layer if nsteps > 1
        field = dotmdmf(f,p,fi,field, out = out)  
        
    if output_layer is not None:
        d_out, epsv_out, epsa_out = output_layer
        if mode == "t":
            if diffraction == True or reflections == True:
                field = transfer_E_through_interface(field, wavenumbers, (d_in, epsv_in, epsa_in),(d_out, epsv_out, epsa_out) , beta = beta, phi = phi,
                             diffraction = diffraction, reflections = reflections,  betamax = betamax, input_fft = False, out = field)
        else:
            if diffraction == True:
                field = transfer_field_through_interface(field, wavenumbers, (d_in, epsv_in, epsa_in),(d_out, epsv_out, epsa_out) , beta = beta, phi = phi,
                             betamax = betamax, input_fft = False , out = out)
                
                
    out = field
            
    if ret_affi == True:
        return (alpha, f, fi), out_phase, out
    else:
        return out
    
    
def propagate_field_effective(field, wavenumbers, layer, effective_layer, beta = 0, phi=0,
                    nsteps = 1, diffraction = True, reflections = True, mode = None, quality = 1, input_layer = None,output_layer = None,
                    betamax = BETAMAX, ret_affi = False, out_affi = None, out_phase = None,out = None):
    
    if diffraction <= 1:
        return _propagate_ray(field, wavenumbers, layer,effective_layer, 
                                beta = beta, phi = phi, nsteps =  nsteps, diffraction = diffraction, reflections = reflections, 
                                mode = mode, input_layer = input_layer, output_layer = output_layer ,
                                betamax = betamax, ret_affi = ret_affi, out = out, out_affi = out_affi, out_phase = out_phase)
                        
    fout = 0.
    fftbetax, fftbetay = betaxy(field.shape[-2:], wavenumbers)
    fftbeta, fftphi = betaphi(field.shape[-2:], wavenumbers)
    
    betaxs = np.linspace(-betamax,betamax,int(diffraction))#np.array([-0.8,-0.4,0,0.4,0.8])#np.hstack((np.arange(0.,betamax,step),np.arange(-step,-betamax,-step)))
    betays = betaxs.copy()
    step = betaxs[1]-betaxs[0]
    
    ffield = fft2(field)
   
    for betax in betaxs:
        for betay in betays:
            fpart, bx, by = select_fftfield(ffield, fftbetax,fftbetay,betax,betay, step,step, betamax)
            b = (bx.mean()**2+by.mean()**2)**0.5
            p = np.arctan2(by.mean(),bx.mean())
            fout += _propagate_ray(fpart, wavenumbers, layer, effective_layer, beta = b.mean(), phi=p.mean(),
                        nsteps = nsteps, diffraction = True, reflections = reflections, mode = mode,
                        input_layer = input_layer,output_layer = output_layer,
                        betamax = betamax, input_fft = True) 
    if out is not None:
        out[...] = fout
    else:
        out = fout
    if ret_affi == True:
        return None, None, out
    else:
        return out
#
#def transmit_jones(field, wavenumbers, layer_in, layer_out, betamax = BETAMAX, out = None):
#    shape = field.shape[-2:]
#    d, epsv_in, epsa_in = layer_in
#    d, epsv_out, epsa_out = layer_out
#    dmat = jones_transmission_matrix(shape, wavenumbers, epsv_in = epsv_in, epsa_in = epsa_in,
#                                                 epsv_out = epsv_out, epsa_out = epsa_out, betamax = betamax)
#    return diffract(field, dmat, out = field)
#      
def propagate_field_full(field, wavenumbers, layer, input_layer = None, output_layer = None,
                    nsteps = 1,  mode = None,
                    betamax = BETAMAX, out = None):

    shape = field.shape[-2:]
    n = field.shape[-3]
   
    d, epsv, epsa = layer
    if input_layer is not None:
        d_in, epsv_in, epsa_in = input_layer
    if output_layer is not None:
        d_out, epsv_out, epsa_out = output_layer
    kd = wavenumbers*d/nsteps
    
    if out is None:
        out = np.empty_like(field)
    
    out_af_in = None
    out_af = None
    pm = None
    tmat = None
    
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
                if n == 4:
                    out_af = alphaffi(beta,phi,epsv,epsa, out = out_af)
                    alpha,f,fi = out_af
                else:
                    out_af = alphaf(beta,phi,epsv,epsa, out = out_af)
                    alpha,fout = out_af 
                    alpha = alpha[...,::2]
                    f = fout[...,::2,::2]
                    fi = inv(f)
                    if step == 0: #do this only first time
                        if input_layer is not None:
                            alphain, fin = alphaf(beta,phi,epsv_in,epsa_in, out = out_af_in)
                            tmat = transmission_mat(fin,f, out = tmat)
                            dotmm(fi,tmat,out = fi)
                    if step == nsteps - 1: #we are done.. 
                        if output_layer is not None:
                            alphaout, fout = alphaf(beta,phi,epsv_out,epsa_out, out = out_af_in)
                            tmat = transmission_mat(f,fout, out = tmat)
                            dotmm(tmat,f,out = f)                           
                        
                pm = phasem(alpha,kd[i], out = pm)
                w = eigenwave(amplitude.shape[:-1]+shape, ieig,jeig, amplitude = amplitude[...,j])
                w = dotmdmf(f,pm,fi,w, out = w)
                np.add(ofield,w,ofield)
                
            out[...,i,:,:,:] = ofield
        field = out
    return out


__all__ = ["transfer_field", "transmitted_field", "reflected_field"]
