"""
Main top level calculation functions for light propagation through optical data.
"""
from __future__ import absolute_import, print_function, division

from dtmm.conf import DTMMConfig, cached_function, BETAMAX
from dtmm.wave import k0, eigenwave, betaphi

from dtmm.data import uniaxial_order, refind2eps, validate_optical_data
from dtmm.tmm import alphaffi_xy, phasem, phasem_t
from dtmm.linalg import dotmdm, dotmm, dotmf, dotmdmf
from dtmm.print_tools import print_progress
from dtmm.diffract import projection_matrix, diffract, phase_matrix, diffraction_matrix
from dtmm.field import field2intensity
from dtmm.fft import fft2, ifft2
import numpy as np

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
    alpha, f, fi = alphaffi_xy(beta,phi,epsa,epsv)  
    kd = -np.asarray(ks)*d
    pmat = phase_matrix(alpha, kd)  
    return dotmdm(f,pmat,fi, out = out)

@cached_function
def corrected_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
    dmat = diffraction_matrix(shape, ks, d, epsv, epsa, betamax = betamax)
    cmat = correction_matrix(beta, phi, ks, d, epsv, epsa)
    if d > 0:
        return dotmm(dmat,cmat, out = None)
    else:
        return dotmm(cmat, dmat, out = None)
    
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
    if ref is not None:
        intensity1 = field2intensity(ref)
    else:
        intensity1 = field2intensity(field)
    f1 = fft2(field, out = out)
    
    f2 = dotmf(dmat, f1 ,out = f1)
    out = ifft2(f2, out = out)
    intensity2 = field2intensity(out)
    out = normalize_field(out, intensity1, intensity2, out = out)
    
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


def transfer_field_old(field_data, optical_data, beta = 0., phi = 0., 
                   eff_data = None, nin = 1., nout = 1., npass = 1,nstep=1,
              diffraction = True,  interference = False, window = None, betamax = BETAMAX):
    """Tranfers input field data through optical data.
    
    This function calculates transmitted field and possibly (when npass > 1) 
    updates input field to include reflected waves. 
    
    
    Parameters
    ----------
    field_data : Field data tuple
        Input field data tuple
    optical_data : Optical data tuple
        Optical data tuple through which input field is transmitted.
    beta : float or 1D array_like of floats
        Beta parameter of the input field. If it is a 1D array, beta[i] is the
        beta parameter of the field_data[0][i] field array.
    phi : float or 1D array_like of floats
        Phi angle of the input light field. If it is a 1D array, phi[i] is the
        phi parameter of the field_data[0][i] field array.
    eff_data : Optical data tuple or None
        Optical data tuple of homogeneous layers through which light is diffracted
        in the diffraction calculation. If not provided, an effective data is
        build from optical_data by taking an average isotropic refractive index
        of the material.
    nin : float, optional
        Refractive index of the input (bottom) surface (1. by default). Used
        in combination with npass > 1 to determine reflections from input layer.
    nout : float, optional
        Refractive index of the output (top) surface (1. by default). Used
        in combination with npass > 1 to determine reflections from output layer.
    npass: int, optional
        How many passes (iterations) to perform. For strongly reflecting elements
        this should be set to a higher value. If npass > 1, then input field data is
        overwritten and adds reflected light from the sample. (defaults to 1)
    nstep: int or 1D array_like of ints
        Specifies layer propagation computation steps (defaults to 1). For thick layers
        you may want to increase this number.
    diffraction : bool, optional
        Whether to perform difraction caclulation or not. Setting this to False 
        will dissable diffraction calculation (standard 4x4 method). Diffraction
        is enabled by default.
    interference : bool, optional
        Whether to enable interference (disabled by default, enabled also when 
        npass > 1)
    window: array or None
        Additional window function that is multiplied after each layer propagation step.
        Computed field data is multiplied with this window after each layer.

    """
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
    beta, phi = _validate_betaphi(beta,phi,extendeddim = field_in.ndim-2)
    
    #define output field
    field_out = np.zeros_like(field_in)
    
    if npass > 1:
        field0 = field_in.copy()
        field1 = field_out.copy()
        #i0 = field2intensity(field0)
        i0 = field2intensity(transmitted_field(field0, ks, n = nin, betamax = betamax))
        i0 = i0.sum(tuple(range(i0.ndim))[-2:])
        
    field = field_in
    out = field_out
    out_affi = None
    
    interference = True if npass > 1 else interference
    mode = "t" if interference == False else None
    
    verbose_level = DTMMConfig.verbose
    indices = list(range(n))
    for i in range(npass):
        msg = "{}/{}".format(i+1,npass)
        for pindex, j in enumerate(indices):
            print_progress(pindex,n,level = verbose_level, suffix = msg) 
            thickness = d[j]*(-1)**i
            thickness_eff = d_eff[j]*(-1)**i
            p = phi 
            out_affi, field = propagate_field(field, ks, (thickness, epsv[j], epsa[j]),(thickness_eff, epsv_eff[j], epsa_eff[j]), 
                            beta = beta, phi = p, nsteps = substeps[j], diffraction = diffraction, mode = mode,
                            betamax = betamax, ret_affi = True, out_affi = out_affi, out = out)
      
        print_progress(n,n,level = verbose_level, suffix = msg) 
        
        indices.reverse()
        if npass > 1:
            if i%2 == 0:
                if i != npass -1:
                    field = np.subtract(field,field1,out = field)  
                    field = transmitted_field(field, ks, n = nout, betamax = betamax, norm = "fft", out = field_out)
                    #if window is not None:
                    #    out = np.multiply(field,window,out = field)
                    field = np.add(field1, field, out = field_out)

                    out = field_in 
                    field1 = field_out.copy()
            else:
                if i != npass -1:
                    i1 = field2intensity(transmitted_field(field, ks, n = nin, betamax = betamax))
                    i1 = i1.sum(tuple(range(i1.ndim))[-2:])
                    fact = (i0/i1)[...,None,None,None]
                    field = np.multiply(field,fact,out = field) 
                    field1 = np.multiply(field1,fact,out = field1) 
                    field_out = np.multiply(field_out,fact,out = field_out) 
                    field = np.subtract(field,field0,out = field_in)     
                    field = reflected_field(field, ks, n = nin, betamax = betamax, norm = "local", out = field_in)
                    if window is not None:
                        field = np.multiply(field,window,out = field)
                    field = np.add(field0, field, out = field_in)
                    field0 = field_in.copy()
                    out = field_out

    return field_out, wavelengths, pixelsize

def transfer_field(field_data, optical_data, beta = 0., phi = 0., nin = 1., nout = 1.,  
           npass = 1,nstep=1,diffraction = True, interference = True, norm = DTMM_NORM_FFT,
              window = None, betamax = BETAMAX, split = False, method = "effective", eff_data = None):
    """Tranfers input field data through optical data.
    
    This function calculates transmitted field and possibly (when npass > 1) 
    updates input field to include reflected waves. 
    
    
    Parameters
    ----------
    field_data : Field data tuple
        Input field data tuple
    optical_data : Optical data tuple
        Optical data tuple through which input field is transmitted.
    beta : float or 1D array_like of floats
        Beta parameter of the input field. If it is a 1D array, beta[i] is the
        beta parameter of the field_data[0][i] field array.
    phi : float or 1D array_like of floats
        Phi angle of the input light field. If it is a 1D array, phi[i] is the
        phi parameter of the field_data[0][i] field array.
    nin : float, optional
        Refractive index of the input (bottom) surface (1. by default). Used
        in combination with npass > 1 to determine reflections from input layer.
    nout : float, optional
        Refractive index of the output (top) surface (1. by default). Used
        in combination with npass > 1 to determine reflections from output layer.
    npass: int, optional
        How many passes (iterations) to perform. For strongly reflecting elements
        this should be set to a higher value. If npass > 1, then input field data is
        overwritten and adds reflected light from the sample. (defaults to 1)
    nstep: int or 1D array_like of ints
        Specifies layer propagation computation steps (defaults to 1). For thick layers
        you may want to increase this number.
    diffraction : bool, optional
        Whether to perform difraction caclulation or not. Setting this to False 
        will dissable diffraction calculation (standard 4x4 method). Diffraction
        is enabled by default.
    interference : bool, optional
        Whether to enable interference. Interference is automatically enabled with 
        npass > 1.
    norm : int, optional
        Normalization mode used when calculating multiple reflections. Possible values
        are 0, 1, 2, 3, default value is 1.
    window: array or None
        If specified, computed field data is multiplied with this window after 
        each pass.
    split: bool, optional
        In multi-ray computation this option specifies whether to split 
        computation over single rays to consume less temporary memory storage.
        For large multi-ray datasets this option should be set.
    method : str, optional
        A transfer method used. Either 'effective' or 'full'
    eff_data : Optical data tuple or None
        Optical data tuple of homogeneous layers through which light is diffracted
        in the diffraction calculation when method == 'effective'. If not provided, 
        an effective data is build from optical_data by taking an average 
        isotropic refractive index of the material.
    """
    verbose_level = DTMMConfig.verbose
    if verbose_level >0:
        print("Transferring input field")    
    if split == False:
        return _transfer_field(field_data, optical_data, beta = beta, 
                       phi = phi, eff_data = eff_data, nin = nin, nout = nout, npass = npass,nstep=nstep,
                  diffraction = diffraction, interference = interference, norm = norm,
                  window = window, betamax = betamax, method = method)
    else:#split input data and compotu sequencially
        

        field_in,wavelengths,pixelsize = field_data
        field_out = np.empty_like(field_in) 

        nrays = len(beta)
        for i, bp in enumerate(zip(beta,phi)):
            if verbose_level >0:
                print("Ray {}/{}".format(i+1,nrays))
        
            field_data = (field_in[i],wavelengths, pixelsize)
            beta, phi = bp
            out = field_out[i]
            _transfer_field(field_data, optical_data, beta = beta, 
                       phi = phi, eff_data = eff_data, nin = nin, nout = nout, npass = npass,nstep=nstep,
                  diffraction = diffraction, interference = interference, norm = norm,
                  window = window, betamax = betamax, out = out)
        return field_out,wavelengths,pixelsize
        

def _transfer_field(field_data, optical_data, beta = 0., 
                   phi = 0., eff_data = None, nin = 1., nout = 1., npass = 1,nstep=1,
              diffraction = True, interference = True, norm = DTMM_NORM_FFT,
              window = None, betamax = BETAMAX, method = "effective",out = None):
    """Tranfers input field data through optical data. See transfer_field.
    """
    verbose_level = DTMMConfig.verbose
    if verbose_level >1:
        print("   Initializing...")
        
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
    beta, phi = _validate_betaphi(beta,phi,extendeddim = field_in.ndim-2)
    
    #define output field
    if out is None:
        field_out = np.zeros_like(field_in)
    else:
        field_out = out
        field_out[:] = 0.
    
    #if npass > 1:
    #    field0 = field_in.copy()
 
    field0 = field_in.copy()
    field = field_in.copy()
    if calc_reference:
        ref = field0.copy()
    else:
        ref = None
    
    field_in[...] = 0.
            
    indices = list(range(n))
    interference = True if npass > 1 else interference
    mode = "t" if interference == False else None
    
    i0 = field2intensity(transmitted_field(field0, ks, n = nin, betamax = betamax))
    i0 = i0.sum(tuple(range(i0.ndim))[-2:]) 
    
    out_affi = None #tmp data
    out_phase = None

    for i in range(npass):
        if verbose_level > 1:
            prefix = "   Transferring"
        else:
            prefix = ""
            #print("Transferring field.")
        msg = "{}/{}".format(i+1,npass)
        i0i = total_intensity(transmitted_field(field, ks, n = nin, betamax = betamax))
        for pindex, j in enumerate(indices):
            print_progress(pindex,n,level = verbose_level, suffix = msg, prefix = prefix) 
            thickness = d[j]*(-1)**i
            thickness_eff = d_eff[j]*(-1)**i
            if calc_reference and i%2 == 0 and interference == True:
                out_affi,out_phase,ref = propagate_field_effective(ref, ks, (thickness, epsv[j], epsa[j]),(thickness_eff, epsv_eff[j], epsa_eff[j]), 
                            beta = beta, phi = phi, nsteps = substeps[j], diffraction = diffraction, mode = "t",
                            betamax = betamax, ret_affi = True, out = ref, out_affi = out_affi, out_phase = out_phase)
            if method == "effective":
                out_affi,out_phase,field = propagate_field_effective(field, ks, (thickness, epsv[j], epsa[j]),(thickness_eff, epsv_eff[j], epsa_eff[j]), 
                            beta = beta, phi = phi, nsteps = substeps[j], diffraction = diffraction, mode = mode,
                            betamax = betamax, ret_affi = True, out = field, out_affi = out_affi, out_phase = out_phase)

            else:
                out_affi,out_phase,field = propagate_field_full(field, ks, (thickness, epsv[j], epsa[j]), 
                            nsteps = substeps[j], mode = mode,
                            betamax = betamax, ret_affi = True, out = field, out_affi = out_affi, out_phase = out_phase)


      
        print_progress(n,n,level = verbose_level, suffix = msg, prefix = prefix) 
        
        indices.reverse()
        if npass > 1:
            if i%2 == 0:
                if i != npass -1:
                    if verbose_level > 1:
                        print("   Normalizing transmissions...")
                    if calc_reference:
                        #normalize reference field, so that total intesity equals total intensity of input light
                        ref = transmitted_field(ref, ks, n = nin, betamax = betamax, out = ref)
                        i0tmp = field2intensity(ref)
                        i0s = i0tmp.sum(tuple(range(i0tmp.ndim))[-2:])
                        fact = (i0i/i0s)
                        fact = fact[...,None,None,None]
                        ref = np.multiply(fact,ref, out = ref)
                    
                    field = transmitted_field(field, ks, n = nout, betamax = betamax, norm = norm, ref = ref, out = field)
                    if window is not None:
                        field = np.multiply(field,window,field)
                np.add(field_out, field, field_out)
                field = field_out.copy()
            else:
                field_in[...] = field
                if i != npass -1:
                    if verbose_level > 1:
                        print("   Normalizing reflections...")
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
            field_out[...] = field
            field_in[...] = field0
            

    return field_out, wavelengths, pixelsize


def propagate_field_effective(field, wavenumbers, layer, effective_layer, beta = 0, phi=0,
                    nsteps = 1, diffraction = True, mode = None,
                    betamax = BETAMAX, ret_affi = False, out_affi = None, out_phase = None,out = None):
    
    shape = field.shape[-2:]
    d, epsv, epsa = layer
    d_eff, epsv_eff, epsa_eff = effective_layer
    kd = wavenumbers*d/nsteps
    d_eff = d_eff/nsteps
    
    alpha, f, fi = alphaffi_xy(beta,phi,epsa,epsv, out = out_affi)
    
    if diffraction == True:
        dmat = corrected_diffraction_matrix(shape, wavenumbers, beta,phi, d=d_eff,
                         epsv = epsv_eff, epsa = epsa_eff, betamax = betamax)
    p = phasem(alpha,kd[...,None,None], out = out_phase)
    if mode == "t":
        pt = phasem_t(alpha,kd[...,None,None])
    for j in range(nsteps):
        if diffraction == True:
            if d > 0:
                field = diffract(field, dmat, out = out)
                if mode == "t":
                    field = dotmdmf(f,pt,fi,field, out = field)
                    #field = ftransmit(f,alpha,fi, field, kd, out = field)
                else:
                    field = dotmdmf(f,p,fi,field, out = field)
                    #field = transmit(f,alpha,fi, field, kd, out = field)
            else:
                field = dotmdmf(f,p,fi,field, out = field)
                #field = transmit(f,alpha,fi, field, kd, out = out)  
                field = diffract(field, dmat, out = field)
        else:
            field = dotmdmf(f,p,fi,field, out = out)
            #field = transmit(f,alpha,fi, field, kd, out = out) 
        out = field
    if ret_affi == True:
        return (alpha, f, fi), out_phase, out
    else:
        return out


def propagate_field_full(field, wavenumbers, layer, 
                    nsteps = 1,  mode = None,
                    betamax = BETAMAX, ret_affi = False, out_affi = None, out_phase = None,out = None):

    shape = field.shape[-2:]
    d, epsv, epsa = layer
    kd = wavenumbers*d/nsteps
    
    if out is None:
        out = np.empty_like(field)
    
        
    shape = field.shape[-2:]
    
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
            
            for j, bp in enumerate(zip(betas,phis)):     
                beta, phi = bp
                alpha, f, fi = alphaffi_xy(beta,phi,epsa,epsv, out = out_affi)
                p = phasem(alpha,kd[i], out = out_phase)
                w = eigenwave(amplitude.shape[:-1]+shape, iind[j],jind[j], amplitude = amplitude[...,j])
                w = dotmdmf(f,p,fi,w, out = w)
                np.add(ofield,w,ofield)
            out[...,i,:,:,:] = ofield
        field = out

    if ret_affi == True:
        return (alpha, f, fi), out_phase, out
    else:
        return out


__all__ = ["transfer_field", "transmitted_field", "reflected_field"]
