"""
Main top level calculation functions for light propagation through optical data.


"""
from __future__ import absolute_import, print_function, division
import time
from dtmm.conf import DTMMConfig,  BETAMAX, SMOOTH, FDTYPE
from dtmm.wave import k0
from dtmm.data import uniaxial_order, refind2eps, validate_optical_data
from dtmm.tmm import E2H_mat, projection_mat, alphaf
from dtmm.tmm3d import transfer3d
from dtmm.linalg import  dotmf, dotmv
from dtmm.print_tools import print_progress
from dtmm.diffract import diffract, projection_matrix, diffraction_alphaffi
from dtmm.field import field2intensity, field2betaphi, field2fvec
from dtmm.fft import fft2, ifft2
from dtmm.jones import jonesvec, polarizer
from dtmm.jones4 import ray_jonesmat4x4, normal_polarizer
from dtmm.data import effective_data
import numpy as np
from dtmm.denoise import denoise_fftfield, denoise_field

from dtmm.propagate_4x4 import propagate_4x4_full, propagate_4x4_effective_1,\
    propagate_4x4_effective_2,propagate_4x4_effective_3,propagate_4x4_effective_4
from dtmm.propagate_2x2 import propagate_2x2_full, propagate_2x2_effective_1,propagate_2x2_effective_2

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
    beta = np.asarray(beta, FDTYPE)
    phi = np.asarray(phi, FDTYPE)  
    
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
    else:
        for i in range(extendeddim+1):
            beta = beta[...,None]
            phi = phi[...,None]        
    return beta, phi

def normalize_field(field, intensity_in, intensity_out, out = None):
    m = intensity_out == 0.
    intensity_out[m] = 1.
    intensity_in[m] = 0.
    fact = (intensity_in/intensity_out)[...,None,:,:]
    fact[fact<=-1] = -1
    fact[fact>=1] = 1
    fact = np.abs(fact)
    return np.multiply(field,fact, out = out) 


def project_normalized_fft(field, dmat, window = None, ref = None, out = None):
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

def transmitted_field_direct(field, beta, phi, n = 1.):
    ev = refind2eps([n]*3)
    ea = np.zeros_like(ev)
    alpha, fmat = alphaf(beta,phi, ev, ea)
    pmat = projection_mat(fmat)
    field0 = np.empty_like(field)
    dotmv(pmat,field2fvec(field), out = field2fvec(field0))
    return field0

def project_normalized_local(field, dmat, window = None, ref = None, out = None):
    f1 = fft2(field) 
    f2 = dotmf(dmat, f1 ,out = f1)
    f = ifft2(f2, out = f2)
    jmat = polarizer(jonesvec(field2fvec(f[...,::2,:,:])))
    #pmat1 = ray_jonesmat4x4(jmat)
    pmat1 = normal_polarizer(jonesvec(field2fvec(f[...,::2,:,:])))
    pmat2 = -pmat1
    pmat2[...,0,0] += 1
    pmat2[...,1,1] += 1 #pmat1 + pmat2 = identity by definition
    pmat2[...,2,2] += 1 #pmat1 + pmat2 = identity by definition
    pmat2[...,3,3] += 1 #pmat1 + pmat2 = identity by definition
    
    if ref is not None:
        intensity1 = field2intensity(dotmf(pmat1, ref))
        ref = dotmf(pmat2, ref)
        
    else:
        intensity1 = field2intensity(dotmf(pmat1, field))
        ref = dotmf(pmat2, field)

    intensity2 = field2intensity(f)
    
    f = normalize_field(f, intensity1, intensity2)
    
    out = np.add(f,ref,out = out)
    if window is not None:
        out = np.multiply(out,window,out = out)
    return out 

def transpose(field):
    """transposes field from shape (..., k,n,m) to (...,n,m,k). Inverse of
    itranspose_field"""
    taxis = list(range(field.ndim))
    taxis.append(taxis.pop(-3))
    return field.transpose(taxis) 

def normal_polarizer(jones = (1,0)):
    """A 4x4 polarizer for normal incidence light. It works reasonably well also
    for off-axis light, but it introduces weak reflections and depolarization.
    
    For off-axis planewaves you should use ray_polarizer instead of this."""
    p = polarizer(jonesvec(jones))
    pmat = np.zeros(shape = p.shape[:-2] + (4,4), dtype = p.dtype)
    pmat[...,::2,::2] = p
    pmat[...,1,1] = p[...,0,0]
    pmat[...,3,3] = p[...,1,1]
    pmat[...,1,3] = -p[...,0,1]
    pmat[...,3,1] = -p[...,1,0]
    return pmat

def project_normalized_local(field, dmat, window = None, ref = None, out = None):
    f1 = fft2(field) 
    f2 = dotmf(dmat, f1 ,out = f1)
    f = ifft2(f2, out = f2)
    pmat1 = normal_polarizer(jonesvec(transpose(f[...,::2,:,:])))
    pmat2 = -pmat1
    pmat2[...,0,0] += 1
    pmat2[...,1,1] += 1 #pmat1 + pmat2 = identity by definition
    pmat2[...,2,2] += 1 #pmat1 + pmat2 = identity by definition
    pmat2[...,3,3] += 1 #pmat1 + pmat2 = identity by definition
    
    if ref is not None:
        intensity1 = field2intensity(dotmf(pmat1, ref))
        ref = dotmf(pmat2, ref)
        
    else:
        intensity1 = field2intensity(dotmf(pmat1, field))
        ref = dotmf(pmat2, field)

    intensity2 = field2intensity(f)
    
    f = normalize_field(f, intensity1, intensity2)
    
    out = np.add(f,ref,out = out)
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
    """Calculates total intensity of the field. 
    Computes intesity and sums over pixels."""
    i = field2intensity(field)
    return i.sum(tuple(range(i.ndim))[-2:])#sum over pixels

def project_normalized_total(field, dmat, window = None, ref = None, out = None):
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

def normalize_total(field, dmat, window = None, ref = None, out = None):
    if ref is not None:
        i1 = total_intensity(ref)
    else:
        i1 = total_intensity(field)
        
    f2 = dotmf(dmat, field, out = out)
    i2 = total_intensity(out)
    
    out = normalize_field_total(out, i1, i2, out = f2)
    
    if window is not None:
        out = np.multiply(out,window,out = out)
    return out  

def _projected_field(field, wavenumbers, mode, n = 1, betamax = BETAMAX,  out = None):
    eps = refind2eps([n]*3)
    pmat = projection_matrix(field.shape[-2:], wavenumbers, epsv = eps, epsa = (0.,0.,0.), mode = mode, betamax = betamax)
    return diffract(field, pmat, out = out) 

    
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


def transmitted_field(field, wavenumbers, n = 1, betamax = BETAMAX, out = None):
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
    out : ndarray, optinal
        Output array
        
    Returns
    -------
    out : ndarray
       Transmitted field.
    """
        
    return _projected_field(np.asarray(field), wavenumbers, +1, n = n, 
                            betamax = betamax, out = out) 
    
def reflected_field(field, wavenumbers, n = 1, betamax = BETAMAX,  out = None):
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
    return _projected_field(np.asarray(field), wavenumbers, -1, n = n, betamax = betamax, out = out) 
    

def transfer_field(field_data, optical_data, beta = None, phi = None, nin = 1., nout = 1.,  
           npass = 1, nstep=1, diffraction = 1, reflection = None, method = "2x2", 
           multiray = False,
           norm = DTMM_NORM_FFT, betamax = BETAMAX, smooth = SMOOTH, split_rays = False,
           split_diffraction = False,split_wavelengths = False,
           eff_data = None, ret_bulk = False, out = None):
    """Tranfers input field data through optical data.
    
    This function calculates transmitted field and possibly (when npass > 1) 
    updates input field with reflected waves. 
    
    
    Parameters
    ----------
    field_data : Field data tuple
        Input field data tuple
    optical_data : Optical data tuple
        Optical data tuple through which input field is transfered.
    beta : float or 1D array_like of floats, optional
        Beta parameter of the input field. If it is a 1D array, beta[i] is the
        beta parameter of the field_data[0][i] field array.f not provided, beta
        is caluclated from input data (see also multiray option).
    phi : float or 1D array_like of floats, optional
        Phi angle of the input light field. If it is a 1D array, phi[i] is the
        phi parameter of the field_data[0][i] field array. If not provided, phi
        is caluclated from input data (see also multiray option).
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
    reflection : bool or int or None, optional
        Reflection calculation mode for '2x2' method. It can be either
        0 or False for no reflections, 1 (default) for reflections in fft space 
        (from effective layers), or 2 for reflections in real space 
        (from individual layers). If this argument is not provided it is 
        automatically set to 0 if npass == 1 and 1 if npass > 1 and diffraction
        == 1 and to 2 if npass > 1 and diffraction > 1. See documentation for details. 
    method : str, optional
        Specifies which method to use, either '2x2' (default) or '4x4'.
    multiray : bool
        If specified it defines if first axis of the input data is treated as multiray data
        or not. If beta and phi are not set, you must define this if your data
        is multiray so that beta and phi values are correctly determined.
    norm : int, optional
        Normalization mode used when calculating multiple reflections with 
        npass > 1 and 4x4 method. Possible values are 0, 1, 2, default value is 1.
    smooth : float, optional
        Smoothing parameter when calculating multiple reflections with 
        npass > 1 and 4x4 method. Possible values are values above 0.Setting this
        to higher values > 1 removes noise but reduces convergence speed. Setting
        this to < 0.1 increases convergence, but it increases noise. 
    split_diffraction : bool, optional
        In diffraction > 1 calculation this option specifies whether to split 
        computation over single beam to consume less temporary memory storage.
        For large diffraction values this option should be set.
    split_rays: bool, optional
        In multi-ray computation this option specifies whether to split 
        computation over single rays to consume less temporary memory storage.
        For large multi-ray datasets this option should be set.
    eff_data : Optical data tuple or symmetry, optional
        Optical data tuple of homogeneous layers through which light is diffracted
        in the diffraction calculation when diffraction >= 1. If not provided, 
        an effective data is build from optical_data by taking the mean value 
        of the epsilon tensor. You can also provide the symmetry argument, e.g.
        'isotropic', 'uniaxial' or 'biaxial', or a list of these values specifying
        the symmetry of each of the layers. This argument is passed directly to 
        the :func:`.data.effective_data` function.
    ret_bulk : bool, optional
        Whether to return bulk field instead of the transfered field (default).
    out : ndarray, optional
        Output array.
    
    """
    t0 = time.time()
    verbose_level = DTMMConfig.verbose
    
    
    if method == "4x4" and npass > 1 and diffraction == False:
        import warnings
        warnings.warn("The 4x4 method with diffraction disabled is not yet supported\
                      for input fields with beta >0. Use 2x2 method insted.")
        
    if npass == -1 or npass == np.inf:
        method = "4x4"
        diffraction = np.inf
        reflection = 2
        
    #choose best/supported reflection mode based on other arguments
    if reflection is None:
        reflection = 0 if method == "2x2" else 1
        if method == "4x4" and diffraction == 0:
            reflection = 2
        if npass > 1:
            reflection = 1
            if diffraction > 1 and method == "2x2":
                reflection = 2
                
    if verbose_level > 0:
        print("Transferring input field.")    
    if verbose_level > 1:
        print("------------------------------------")
        print(" $ calculation method: {}".format(method))  
        print(" $ reflection mode: {}".format(reflection)) 
        print(" $ diffraction mode: {}".format(diffraction))  
        print(" $ number of substeps: {}".format(nstep))     
        print(" $ input refractive index: {}".format(nin))   
        print(" $ output refractive index: {}".format(nout)) 
        print("------------------------------------")
        
    
    
    field_in,wavelengths,pixelsize = field_data

#    if out is None:
#        if ret_bulk == True:
#            #must have a length of optical data + 2 extra layers
#            field_out = np.empty(shape = (len(optical_data[0])+2,)+field_in.shape, dtype = field_in.dtype)     
#        else:
#            field_out = np.empty_like(field_in) 
#    else:
#        field_out = out
    
    splitted_wavelengths = split_wavelengths == True and not isinstance(field_in, tuple) and ret_bulk == False

    
    if splitted_wavelengths:
        if out is None:
            out_field = np.empty_like(field_in) 
        else:
            out_field = out
        out = [out_field[...,i,:,:,:] for i in range(len(wavelengths))]
        field_in = tuple((field_in[...,i,:,:,:] for i in range(len(wavelengths))))

    if isinstance(field_in, tuple):
        nwavelengths = len(field_in)
        if out is None:
            out = [None for i in range(len(field_in))]
        for i,(f,w,o) in enumerate(zip(field_in, wavelengths, out)):
            if verbose_level >0:
                print("Wavelength {}/{}".format(i+1,nwavelengths))
            field_data = f,w,pixelsize
            o = _transfer_field(field_data, optical_data, beta, phi, nin, nout,  
           npass , nstep, diffraction, reflection , method, 
           multiray, norm, betamax, smooth, split_rays,
           split_diffraction ,
           eff_data, ret_bulk, o) 
            out[i] = o
        out = tuple(out)
    else:
    
        out = _transfer_field(field_data, optical_data, beta, phi, nin, nout,  
               npass , nstep, diffraction, reflection , method, 
               multiray, norm, betamax, smooth, split_rays,
               split_diffraction ,
               eff_data, ret_bulk, out)   

    t = time.time()-t0
    if verbose_level >1:
        print("------------------------------------")
        print("   Done in {:.2f} seconds!".format(t))  
        print("------------------------------------")
    
    if splitted_wavelengths:
        return out_field, wavelengths, pixelsize
    else:
        return out

def _transfer_field(field_data, optical_data, beta, phi, nin, nout,  
           npass , nstep, diffraction, reflection , method, 
           multiray, norm, betamax, smooth, split_rays,
           split_diffraction ,
           eff_data, ret_bulk, out):
    verbose_level = DTMMConfig.verbose
 
    if split_rays == False:
        if method  == "4x4":
            if npass == -1 or npass == np.inf:
                out = transfer3d(field_data, optical_data,nin = nin, nout =nout, betamax = betamax)
            else:
                out = transfer_4x4(field_data, optical_data, beta = beta, 
                           phi = phi, eff_data = eff_data, nin = nin, nout = nout, npass = npass,nstep=nstep,
                      diffraction = diffraction, reflection = reflection, multiray = multiray,norm = norm, smooth = smooth,
                      betamax = betamax, ret_bulk = ret_bulk, out = out)
        else:
            out = transfer_2x2(field_data, optical_data, beta = beta, 
                   phi = phi, eff_data = eff_data, nin = nin, nout = nout, npass = npass,nstep=nstep,
              diffraction = diffraction,  multiray = multiray,split_diffraction = split_diffraction,reflection = reflection, betamax = betamax, ret_bulk = ret_bulk, out = out)
        
    else:#split input data by rays and compute ray-by-ray
        
        field_in,wavelengths,pixelsize = field_data
        if out is None:
            if ret_bulk == True:
                #must have a length of optical data + 2 extra layers
                field_out = np.empty(shape = (len(optical_data[0])+2,)+field_in.shape, dtype = field_in.dtype)     
            else:
                field_out = np.empty_like(field_in) 
        else:
            field_out = out
        nrays = len(field_in)
        if beta is None:
            beta = [None] * nrays
        if phi is None:
            phi = [None] * nrays
        multiray = False
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
                  diffraction = diffraction, reflection = reflection,multiray = multiray,norm = norm, smooth = smooth,
                  betamax = betamax, out = out, ret_bulk = ret_bulk)
            else:
                transfer_2x2(field_data, optical_data, beta = beta, 
                   phi = phi, eff_data = eff_data, nin = nin, nout = nout, npass = npass,nstep=nstep,
              diffraction = diffraction,multiray = multiray, split_diffraction = split_diffraction, reflection = reflection, betamax = betamax, out = out, ret_bulk = ret_bulk)
        
            
        out = field_out, wavelengths, pixelsize
    return out
      

def transfer_4x4(field_data, optical_data, beta = 0., 
                   phi = 0., eff_data = None, nin = 1., nout = 1., npass = 1,nstep=1,
              diffraction = True, reflection = 1, multiray = False,norm = DTMM_NORM_FFT, smooth = SMOOTH,
              betamax = BETAMAX, ret_bulk = False, out = None):
    """Transfers input field data through optical data. See transfer_field.
    """
    if reflection not in (1,2,3,4):
        raise ValueError("Invalid reflection. The 4x4 method is either reflection mode 1 or 2.")
    if smooth > 1.:
        pass
        #smooth =1.
    elif smooth < 0:
        smooth = 0.
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
        beta, phi = field2betaphi(field_in,ks, multiray)
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
        
        #out[...] = 0.
        if ret_bulk == True:
            bulk_out = out
            bulk_out[0] = field_in
            field_in = bulk_out[0]
            field_out = bulk_out[-1]
        else:
            bulk_out = None
            field_out = out
        #make sure we remove forward propagating waves
        reflected_field(field_out, ks, n = nin, betamax = betamax, out = field_out)
    indices = list(range(1,n-1))
    #if npass > 1:
    #    field0 = field_in.copy()
 
    #field0 = field_in.copy()
    field0 = transmitted_field(field_in, ks, n = nin, betamax = betamax)
    field = field_in.copy()

    
    #field_in[...] = 0.
            
    if norm == 2:
        #make sure we take only the forward propagating part of the field
        transmitted_field(field, ks, n = nin, betamax = min(betamax,nin), out = field)
        field_in[...] = field
        
    if calc_reference:
        ref = field.copy()
    else:
        ref = None    
    
    #i0 = field2intensity(transmitted_field(field0, ks, n = nin, betamax = betamax))
    i0 = field2intensity(field0)
    i0 = i0.sum(tuple(range(i0.ndim))[-2:]) 
    
    if reflection not in (2,4) and 0<= diffraction and diffraction < np.inf:
        work_in_fft = True
    else:
        work_in_fft = False
        
    if work_in_fft:
        field = fft2(field,out = field)
    _reuse = False
    tmpdata = {}
    
    #:projection matrices.. set when needed
    if npass > 1:
        pin_mat = projection_matrix(field.shape[-2:], ks,  epsv = refind2eps([nin]*3), mode = +1, betamax = betamax)
        pout_mat = projection_matrix(field.shape[-2:], ks,  epsv = refind2eps([nout]*3), mode = +1, betamax = betamax)

    
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
        
        if direction == 1:
            _betamax = betamax
        else:
            _betamax = betamax
        
        for pindex, j in enumerate(indices):
            print_progress(pindex,n,level = verbose_level, suffix = suffix, prefix = prefix) 
            
            nstep, (thickness,ev,ea) = layers[j]
            output_layer = (thickness*direction,ev,ea)

            _nstep, output_layer_eff = eff_layers[j]

            d,e,a = output_layer_eff
            output_layer_eff = d*direction, e,a 
            
            if ray_tracing == True:
                if work_in_fft:
                    beta, phi = field2betaphi(ifft2(field),ks, multiray)
                else:
                    beta, phi = field2betaphi(field,ks, multiray)
                beta, phi = _validate_betaphi(beta,phi,extendeddim = field_in.ndim-2)
            
            if calc_reference and i%2 == 0:
                _nstep, (thickness_in,ev_in,ea_in) = layers[j-1]
                input_layer = (thickness_in,ev_in,ea_in)
                ref2, refl = propagate_2x2_effective_2(ref[...,::2,:,:], ks, input_layer, output_layer ,None, output_layer_eff, 
                            beta = beta, phi = phi, nsteps = nstep, diffraction = diffraction, reflection = 0, 
                            betamax = _betamax,mode = direction, out = ref[...,::2,:,:])
                
            if bulk_out is not None:
                out_field = _bulk_out[j]
            else:
                out_field = field
            
            if diffraction >= 0 and diffraction < np.inf:
                if reflection == 4:
                    field = propagate_4x4_effective_4(field, ks, output_layer,output_layer_eff, 
                                beta = beta, phi = phi, nsteps = nstep, diffraction = diffraction, 
                                betamax = _betamax, out = out_field)  
                elif reflection == 3:
                    field = propagate_4x4_effective_3(field, ks, output_layer,output_layer_eff, 
                                beta = beta, phi = phi, nsteps = nstep, diffraction = diffraction, 
                                betamax = _betamax, out = out_field) 
                
                elif reflection ==2:
                    field = propagate_4x4_effective_2(field, ks, output_layer,output_layer_eff, 
                                beta = beta, phi = phi, nsteps = nstep, diffraction = diffraction, 
                                betamax = _betamax, out = out_field, tmpdata = tmpdata)
                else:
                    field = propagate_4x4_effective_1(field, ks, output_layer,output_layer_eff, 
                                beta = beta, phi = phi, nsteps = nstep, diffraction = diffraction, 
                                betamax = _betamax, out = out_field, _reuse = _reuse)                    
            else:
                field = propagate_4x4_full(field, ks, output_layer, 
                            nsteps = nstep, 
                            betamax = _betamax, out = out_field)
            _reuse = True
        if ref is not None:
            ref[...,1::2,:,:] = jones2H(ref2,ks,betamax = _betamax, n = nout)
        print_progress(n,n,level = verbose_level, suffix = suffix, prefix = prefix) 
        
        indices.reverse()
        
        if work_in_fft == True:
            field = ifft2(field)

        
        if npass > 1:
            #smooth = 1. - i/(npass-1.)
            #even passes - normalizing output field
            if i%2 == 0:
                if i == 0:
                    np.subtract(field,field_out, field)
                if i != npass -1:
                    if verbose_level > 1:
                        print(" * Normalizing transmissions.")
                    if calc_reference:
                        np.multiply(ref, (nin/nout)**0.5,ref)

                    sigma = smooth*(npass-i)/(npass)
                    
                    if norm == "fft":
                        field = project_normalized_fft(field, pout_mat, ref = ref, out = field)
                    elif norm == "local":
                        field = project_normalized_local(field, pout_mat, ref = ref, out = field)
                    elif norm == "total":
                        field = project_normalized_total(field, pout_mat, ref = ref, out = field)                      
                    
                    field = denoise_field(field, ks, nin, sigma, out = field)
                    
                np.add(field_out, field, field_out)
                field = field_out.copy()
                
            #odd passes - normalizeing input field   
            else:
                field_in[...] = field
                #if not the last pass...
                if i != npass -1:
                    if verbose_level > 1:
                        print(" * Normalizing reflections.")
                     
                    sigma = smooth*(npass-i)/(npass) 
                    
                    ffield = fft2(field, out = field)
                    ffield = dotmf(pin_mat, ffield, out = ffield)                                        
                    ffield = denoise_fftfield(ffield, ks, nin, sigma, out = ffield)
                    field = ifft2(field, out = ffield)
                    
                    i0f = total_intensity(field)
                    fact = ((i0/i0f))#**0.5)
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
            #field_in[...] = field0
    #denoise(field_out, ks, nout, smooth*10, out = field_out)           
        
    if ret_bulk == True:
        if work_in_fft:
            ifft2(bulk_out[1:-1],out =bulk_out[1:-1])
        return bulk_out, wavelengths, pixelsize
    else:
        return field_out, wavelengths, pixelsize


    
    
    #m = tukey(beta,0.3,betamax)
#    #mask0 = (beta>0.9) & (beta < 1.1)
#    
#
#    
#    #mask = np.empty(mask0.shape + (4,), mask0.dtype)
#    #for i in range(4):
#    #    mask[...,i] = mask0   
#
#     
#    alpha, f, fi = alphaffi(beta,phi,epsv,epsa,out = out) 
#
#    out = (alpha,f,fi)
#    
#    try:
#        b1,betamax = betamax
#        a = (betamax-b1)/(betamax)
#        m = tukey(beta,a,betamax)
#        #print(b1)
#        #m = (b1-beta.clip(b1,betamax))/(betamax-b1)+1
#        #np.multiply(alpha,m[...,None,None],alpha)
#        np.multiply(f,m[...,None,None],f)    
#    
#                                
#    t = field2intensity(field)
#    t2 = ndimage.gaussian_filter(t, sigma)
#    
#    f = (dtmp+np.abs(t2))/(dtmp+np.abs(t))
#    if out is not None:
#        out[...] = field*f[...,None,:,:]
#        return out
#    else:
#        return field*f[...,None,:,:]




def _layers_list(optical_data, eff_data, nin, nout, nstep):
    """Build optical data layers list and effective data layers list.
    It appends/prepends input and output layers. A layer consists of
    a tuple of (n, thickness, epsv, epsa) where n is number of sublayers"""
    d, epsv, epsa = validate_optical_data(optical_data)  
    
    if epsa is not None:
        substeps = np.broadcast_to(np.asarray(nstep),(len(d),))
        layers = [(n,(t/n, ev, ea)) for n,t,ev,ea in zip(substeps, d, epsv, epsa)]
        #add input and output layers
        layers.insert(0, (1,(0., np.broadcast_to(refind2eps([nin]*3), epsv[0].shape), np.broadcast_to(np.array((0.,0.,0.), dtype = FDTYPE), epsa[0].shape))))
        layers.append((1,(0., np.broadcast_to(refind2eps([nout]*3), epsv[0].shape), np.broadcast_to(np.array((0.,0.,0.), dtype = FDTYPE), epsa[0].shape))))

        try:
            d_eff, epsv_eff, epsa_eff = validate_optical_data(eff_data, homogeneous = True)        
        except (TypeError, ValueError):
            if eff_data is None:
                d_eff, epsv_eff, epsa_eff = _isotropic_effective_data(optical_data)
            else:
                d_eff, epsv_eff, epsa_eff = effective_data(optical_data, symmetry = eff_data)
                        
        eff_layers = [(n,(t/n, ev, ea)) for n,t,ev,ea in zip(substeps, d_eff, epsv_eff, epsa_eff)]
        eff_layers.insert(0, (1,(0., refind2eps([nin]*3), np.array((0.,0.,0.), dtype = FDTYPE))))
        eff_layers.append((1,(0., refind2eps([nout]*3), np.array((0.,0.,0.), dtype = FDTYPE))))
        return layers, eff_layers
    else:
        substeps = np.broadcast_to(np.asarray(nstep),(len(d),))
        layers = [(n,(t/n, ev, None)) for n,t,ev in zip(substeps, d, epsv)]
        #add input and output layers
        layers.insert(0, (1,(0., np.broadcast_to(refind2eps([nin,nin,nin,0,0,0]), epsv[0].shape), None)))
        layers.append((1,(0., np.broadcast_to(refind2eps([nout,nout,nout,0,0,0]), epsv[0].shape), None)))

        try:
            d_eff, epsv_eff, epsa_eff = validate_optical_data(eff_data, homogeneous = True)        
        except (TypeError,ValueError):
            if eff_data is None:
                d_eff, epsv_eff, epsa_eff = _isotropic_effective_data(optical_data)
            else:
                d_eff, epsv_eff, epsa_eff = effective_data(optical_data, symmetry = eff_data)
                    
        eff_layers = [(n,(t/n, ev, ea)) for n,t,ev,ea in zip(substeps, d_eff, epsv_eff, epsa_eff)]
        eff_layers.insert(0, (1,(0., refind2eps([nin]*3), np.array((0.,0.,0.), dtype = FDTYPE))))
        eff_layers.append((1,(0., refind2eps([nout]*3), np.array((0.,0.,0.), dtype = FDTYPE))))
        return layers, eff_layers       


def transfer_2x2(field_data, optical_data, beta = None, 
                   phi = None, eff_data = None, nin = 1., 
                   nout = 1., npass = 1,nstep=1,
              diffraction = True, reflection = True, multiray = False, split_diffraction = False,
              betamax = BETAMAX, ret_bulk = False, out = None):
    """Tranfers input field data through optical data using the 2x2 method
    See transfer_field for documentation.
    """
    if reflection not in (0,1,2):
        raise ValueError("Invalid reflection. The 2x2 method supports reflection mode 0,1 or 2.")
    
    
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
        beta, phi = field2betaphi(field_in,ks, multiray)
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
    if diffraction > 0:
        field0 = transmitted_field(field_in, ks, n = nin, betamax = betamax)
    else:
        field0 = transmitted_field_direct(field_in, beta, phi, n = nin)
    
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
        
    if reflection in (0,1) and 0< diffraction and diffraction < np.inf:
        work_in_fft = True
    elif reflection == 1:
        work_in_fft = True
    else:   
        work_in_fft = False
        
    if work_in_fft:
        field = fft2(field,out = field)
        
    tmpdata = {}

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
                    beta, phi = field2betaphi(ifft2(field),ks, multiray)
                else:
                    beta, phi = field2betaphi(field,ks, multiray)
                beta, phi = _validate_betaphi(beta,phi,extendeddim = field_in.ndim-2)

            if diffraction >= 0 and diffraction < np.inf:
     

                if work_in_fft:
                    field, refli = propagate_2x2_effective_1(field, ks, input_layer, output_layer ,input_layer_eff, output_layer_eff, 
                            beta = beta, phi = phi, nsteps = nstep, diffraction = diffraction, split_diffraction = split_diffraction, reflection = reflection, 
                            betamax = betamax,mode = direction, refl = refl[j], bulk = bulk, tmpdata = tmpdata)
                
                else:
                    field, refli = propagate_2x2_effective_2(field, ks, input_layer, output_layer ,input_layer_eff, output_layer_eff, 
                            beta = beta, phi = phi, nsteps = nstep, diffraction = diffraction, split_diffraction = split_diffraction, reflection = reflection, 
                            betamax = betamax,mode = direction, refl = refl[j], bulk = bulk, tmpdata = tmpdata)
                

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
    
 
__all__ = ["transfer_field", "transmitted_field", "reflected_field", "transfer_2x2", "transfer_4x4", "total_intensity"]
