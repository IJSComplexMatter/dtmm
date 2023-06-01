"""
Main top level calculation functions for light propagation through optical data.


"""
from __future__ import absolute_import, print_function, division
import time
from dtmm.conf import DTMMConfig, SMOOTH, FDTYPE, get_default_config_option
from dtmm.wave import k0, eigenmask
from dtmm.data import uniaxial_order, refind2eps, validate_optical_data, is_callable, validate_optical_block, material_dim
from dtmm.tmm import E2H_mat, projection_mat, alphaf,  fvec2avec, f_iso
from dtmm.solver import transfer3d
from dtmm.linalg import  dotmf, dotmv
from dtmm.print_tools import print_progress
from dtmm.diffract import diffract, projection_matrix, diffraction_alphaffi
from dtmm.field import field2intensity, field2betaphi, field2fvec, jones2field,field2jones, select_modes, set_modes
from dtmm.fft import fft2, ifft2
from dtmm.jones import jonesvec, polarizer
from dtmm.jones4 import ray_jonesmat4x4
from dtmm.data import sellmeier2eps, layered_data, effective_data
from dtmm.data import effective_block, is_optical_data_dispersive
import numpy as np
from dtmm.denoise import denoise_fftfield, denoise_field

from dtmm.propagate_4x4 import propagate_4x4_full, propagate_4x4_effective_1,\
    propagate_4x4_effective_2,propagate_4x4_effective_3,propagate_4x4_effective_4
from dtmm.propagate_2x2 import propagate_2x2_full, propagate_2x2_effective_1,propagate_2x2_effective_2

#norm flags
DTMM_NORM_FFT = 1<<0 #normalize in fft mode
DTMM_NORM_REF = 1<<1 #normalize using reference field
    
def _isotropic_effective_layer(d,epsv,epsa = None):
    epseff = uniaxial_order(0.,epsv).mean(axis = (0,1))
    aeff = np.array((0.,0.,0.))
    return (d,np.broadcast_to(epseff,epsv.shape),np.broadcast_to(aeff,epsv.shape))

def _isotropic_effective_data(data):
    d, material, angles = data
    n = len(d)
    epseff = uniaxial_order(0.,material).mean(axis = (0,1,2))
    epseff = np.broadcast_to(epseff,(n,3)).copy()#better to make copy.. to make it c contiguous
    aeff = np.array((0.,0.,0.))
    aeff = np.broadcast_to(aeff,(n,3)).copy()#better to make copy.. to make it c contiguous
    return [validate_optical_block((d,epseff,aeff))]

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

def project_normalized_local(field, dmat, nout, window = None, ref = None, out = None):
    f1 = fft2(field) 
    f2 = dotmf(dmat, f1 ,out = f1)
    f = ifft2(f2, out = f2)
    fvec = field2fvec(f)
    fmat =  f_iso(n = nout)
    #eigenfield amplitude
    a = fvec2avec(fvec, fmat)
    epsv = refind2eps((nout,nout,nout))
    #take the forward part and compute jones matrix
    jmat = polarizer(jonesvec(a[...,0::2]))
    #convert to 4x4 matrix
    pmat1 = ray_jonesmat4x4(jmat, epsv = epsv)

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

# def transpose(field):
#     """transposes field from shape (..., k,n,m) to (...,n,m,k). Inverse of
#     itranspose_field"""
#     taxis = list(range(field.ndim))
#     taxis.append(taxis.pop(-3))
#     return field.transpose(taxis) 

# def normal_polarizer(jones = (1,0)):
#     """A 4x4 polarizer for normal incidence light. It works reasonably well also
#     for off-axis light, but it introduces weak reflections and depolarization.
    
#     For off-axis planewaves you should use ray_polarizer instead of this."""
#     p = polarizer(jonesvec(jones))
#     pmat = np.zeros(shape = p.shape[:-2] + (4,4), dtype = p.dtype)
#     pmat[...,::2,::2] = p
#     pmat[...,1,1] = p[...,0,0]
#     pmat[...,3,3] = p[...,1,1]
#     pmat[...,1,3] = -p[...,0,1]
#     pmat[...,3,1] = -p[...,1,0]
#     return pmat

# def project_normalized_local(field, dmat, nout, window = None, ref = None, out = None):
#     f1 = fft2(field) 
#     f2 = dotmf(dmat, f1 ,out = f1)
#     f = ifft2(f2, out = f2)
#     pmat1 = normal_polarizer(jonesvec(transpose(f[...,::2,:,:])))
#     pmat2 = -pmat1
#     pmat2[...,0,0] += 1
#     pmat2[...,1,1] += 1 #pmat1 + pmat2 = identity by definition
#     pmat2[...,2,2] += 1 #pmat1 + pmat2 = identity by definition
#     pmat2[...,3,3] += 1 #pmat1 + pmat2 = identity by definition
    
#     if ref is not None:
#         intensity1 = field2intensity(dotmf(pmat1, ref))
#         ref = dotmf(pmat2, ref)
        
#     else:
#         intensity1 = field2intensity(dotmf(pmat1, field))
#         ref = dotmf(pmat2, field)

#     intensity2 = field2intensity(f)
    
#     f = normalize_field(f, intensity1, intensity2)
    
#     out = np.add(f,ref,out = out)
#     if window is not None:
#         out = np.multiply(out,window,out = out)
#     return out 

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

def _projected_field(field, wavenumbers, mode, n = 1, betamax = None,  out = None):
    betamax = get_default_config_option("betamax", betamax)
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

def jones2H(jones, wavenumbers, n = 1., betamax = None, mode = +1, out = None):  
    betamax = get_default_config_option("betamax", betamax)
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


def transmitted_field(field, wavenumbers, n = 1, betamax = None, out = None):
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
        Betamax perameter used. If not specified, default value is read from DTMMConf.
    out : ndarray, optinal
        Output array
        
    Returns
    -------
    out : ndarray
       Transmitted field.
    """
        
    return _projected_field(np.asarray(field), wavenumbers, +1, n = n, 
                            betamax = betamax, out = out) 
    
def reflected_field(field, wavenumbers, n = 1, betamax = None,  out = None):
    """Computes reflected (backward propagating) part of the field.
    
    Parameters
    ----------
    field : ndarray
        Input field array
    wavenumbers : array_like
        Wavenumbers of the field
    n : float, optional
        Refractive index of the media (1 by default)
    betamax : float, optional
        Betamax perameter used. If not specified, default value is read from DTMMConf.
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
    

def transfer_field(field_data, optical_data, beta = None, phi = None, nin = None, nout = None,  
           npass = None, nstep=1, diffraction = None, reflection = None, method = None, 
           multiray = False,
           norm = DTMM_NORM_FFT, betamax = None, smooth = SMOOTH, split_rays = False,
           split_diffraction = False,split_wavelengths = False, split_layers = False,
           create_matrix = False,
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
        Specifies which method to use, either '2x2' (default) or '4x4'. The "4x4"
        mathod is not yet fully supported. Some options may not work when set
        to the 4x4 method. The "4x4" method is still considered experimental, 
        so you should generally use "2x2", and only try "4x4" method if "2x2"
        fails to converge in multi-pass simulations.
    multiray : bool
        If specified it defines if first axis of the input data is treated as multiray data
        or not. If beta and phi are not set, you must define this if your data
        is multiray so that beta and phi values are correctly determined.
    norm : int, optional
        Normalization mode used when calculating multiple reflections with 
        npass > 1 and 4x4 method. Possible values are 0, 1, 2, default value is 1.
    betamax : float, optional
        Betamax perameter used. If not specified, default value is read from DTMMConf.
    smooth : float, optional
        Smoothing parameter when calculating multiple reflections with 
        npass > 1 and 4x4 method. Possible values are values above 0.Setting this
        to higher values > 1 removes noise but reduces convergence speed. Setting
        this to < 0.1 increases convergence, but it increases noise. 
    split_rays: bool, optional
        In multi-ray computation this option specifies whether to split 
        computation over single rays to consume less temporary memory storage.
        For large multi-ray datasets this option should be set.
    split_diffraction : bool, optional
        In diffraction > 1 calculation this option specifies whether to split 
        computation over single beam to consume less temporary memory storage.
        For large diffraction values this option should be set.
    split_wavelengths : bool, optional
        Specifies whether to treat data at each wavelength as an independent ray. 
        With off-axis propagation of eigenmodes with different beta values this
        should be set to true to assure proper determination of beta parameter
        for proper reflection calculation. Normally there will be very little
        difference, however, for small simulation volumes (compared to the 
        wavelength) and at large incidence angles, you should set this to True
        for simulations of multi-wavelength fields. For dispersive material,
        this option must be set to True.
    split_layers : bool, optional
        Specifieas whether to split layers. This affects how the diffraction
        propagation step, which is block-specific, handles the effective data.
        With split_layers set, each layer in the stack gets its own diffraction
        matrix, based on the effective parameters of the layer. This argument
        calls the :func:`.data.layered_data`. This also affects the eff_data
        argument.
    create_matrix : int, optional
        Either 0,1,2. Defines whether to compute transfer matrices for each
        of the block rather than performing field propagation. Setting this 
        to 1 will force creation of matrices for 1d layer only. Setting this to
        2 will force creation of matrices for 1d and 2d layers. Currently, 
        supported in 2x2 method only.
    eff_data : list or symmetry, optional
        Optical data list of homogeneous layers through which light is diffracted
        in the diffraction calculation when diffraction >= 1. If not provided, 
        an effective data is build from optical_data by taking the mean value 
        of the epsilon tensor. You can also provide the symmetry argument, e.g.
        'isotropic', 'uniaxial' or 'biaxial', or a list of these values specifying
        the symmetry of each of the blocks. This argument is passed directly to 
        the :func:`.data.effective_data` function.
    ret_bulk : bool, optional
        Whether to return bulk field instead of the transfered field (default).
    out : ndarray, optional
        Output array.
        
    Returns
    -------
    field_out : tuple or list
        If ret_bulk is False it returns a field data tuple. If ret_bulk is True
        it returns a list of field data tuples. Each element of the list
        represents bulk data for each of the blocks.
    """
    nin = get_default_config_option("nin",nin)
    nout = get_default_config_option("nout",nout)
    method = get_default_config_option("method",method)
    npass = get_default_config_option("npass",npass)
    eff_data = get_default_config_option("eff_data",eff_data)
    diffraction = get_default_config_option("diffraction",diffraction)
    reflection = get_default_config_option("reflection",reflection)
    betamax = get_default_config_option("betamax", betamax)
    
    if not isinstance(optical_data, list):
        # input is not optical data, but block data, so create alist of single-block data
        optical_data = [optical_data]
    
    
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

    eff_data_name = eff_data if eff_data in (0,1,2,"isotropic","uniaxial","biaxial") else "custom"
    

    field_in,wavelengths,pixelsize = field_data   
    shape = field_in.shape[-2:]
    
    if create_matrix == 0:
        solver = "`beam propagator`"
    elif create_matrix > 2:
        solver = "`full matrix solver`"
        if npass > 1 or method == "4x4":
            npass = np.inf
            diffraction = np.inf
            method = "4x4"
        elif npass == 1:
            if reflection > 0:
                diffraction = np.inf
                method = "4x4_1"
            else:
                diffraction = np.inf
                method = "2x2"
    else:
        if method != "2x2":
            raise ValueError("matrix solver is not available in 4x4, please use 2x2 method.")
        solver = "`hybrid solver`"

    if verbose_level > 1:
        print("------------------------------------")
        print(" $ solver: {}".format(solver))
        print(" $ calculation method: {}".format(method))  
        print(" $ create matrix: {}".format(create_matrix))  
        print(" $ reflection mode: {}".format(reflection)) 
        print(" $ diffraction mode: {}".format(diffraction))  
        print(" $ number of substeps: {}".format(nstep))     
        print(" $ input refractive index: {}".format(nin))   
        print(" $ output refractive index: {}".format(nout)) 
        print(" $ effective data mode: {}".format(eff_data_name))
        print(" $ max beta: {}".format(betamax)) 
        print(" $ field shape: {}".format(shape)) 
        print("------------------------------------")
        

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
        out = tuple((out_field[...,i,:,:,:] for i in range(len(wavelengths))))
        field_in = tuple((field_in[...,i,:,:,:] for i in range(len(wavelengths))))

    if isinstance(field_in, tuple):
        nwavelengths = len(field_in)
        if out is None:
             out = (None,) * len(field_in) 
        for i,(f,w,o) in enumerate(zip(field_in, wavelengths, out)):
            if verbose_level >0:
                print("Wavelength {}/{}".format(i+1,nwavelengths))
                
            # see below, for some reason it does not work to specify output
            # for this reason we copy inout data to makw it contiguous, just to make sure
            field_data = f.copy(),w,pixelsize
            
            _optical_data = validate_optical_data(optical_data, wavelength = w, shape = shape)
            if split_layers == True:
                _optical_data = layered_data(_optical_data) #make it list of layers
                _optical_data = validate_optical_data(_optical_data, shape = shape) #make it list of blocks

            if eff_data_name != "custom":
                _eff_data = [eff_data]* len(_optical_data)
            else:
                _eff_data = eff_data
                if not len(_eff_data) == len(_optical_data):
                    raise ValueError("eff_data length must match optical_data length")
                    
            # for some reason it does not work with pre-defined non-contiguous output array, 
            # so we set to None
            o = None
            
            o,w,p = _transfer_field(field_data, _optical_data, beta, phi, nin, nout,  
                npass , nstep, diffraction, reflection , method, 
                multiray, norm, betamax, smooth, split_rays,
                split_diffraction, create_matrix, _eff_data, ret_bulk, o) 
            # we need to copy because we used None as o
            # TODO: find out why it does not work without copying
            f[...] = field_data[0]
            out[i][...] = o
    else:
        _optical_data = validate_optical_data(optical_data, shape = shape)
        if split_layers == True:
            _optical_data = layered_data(_optical_data) #make it list of layers
            _optical_data = validate_optical_data(_optical_data, shape = shape) #make it list of blocks

        if eff_data_name != "custom":
            _eff_data = [eff_data] * len(_optical_data)
        else:
            _eff_data = eff_data
            if not len(_eff_data) == len(_optical_data):
                raise ValueError("eff_data length must match optical_data length")
                 
        if is_optical_data_dispersive(_optical_data):
            raise ValueError("You are using dispersive data, so you must use `split_wavelengths=True`")
        out = _transfer_field(field_data, _optical_data, beta, phi, nin, nout,  
               npass , nstep, diffraction, reflection , method, 
               multiray, norm, betamax, smooth, split_rays,
               split_diffraction , create_matrix,
               _eff_data, ret_bulk, out)   

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
           split_diffraction , create_matrix,
           eff_data, ret_bulk, out):
    verbose_level = DTMMConfig.verbose
 
    if split_rays == False:
        if method  == "4x4":
            if npass == -1 or npass == np.inf:
                out = transfer3d(field_data, optical_data, nin = nin, nout =nout, betamax = betamax)
            else:
                out = transfer_4x4(field_data, optical_data, beta = beta, 
                           phi = phi, eff_data = eff_data, nin = nin, nout = nout, npass = npass,nstep=nstep,
                      diffraction = diffraction, reflection = reflection, multiray = multiray,norm = norm, smooth = smooth,
                      betamax = betamax, ret_bulk = ret_bulk, out = out)
        else:
            out = transfer_2x2(field_data, optical_data, beta = beta, 
                   phi = phi, eff_data = eff_data, nin = nin, nout = nout, npass = npass,nstep=nstep,
              diffraction = diffraction,  multiray = multiray,split_diffraction = split_diffraction,
              create_matrix = create_matrix,
              reflection = reflection, betamax = betamax, ret_bulk = ret_bulk, out = out)
        
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
              diffraction = diffraction,multiray = multiray, split_diffraction = split_diffraction, reflection = reflection, betamax = betamax, out = out, ret_bulk = ret_bulk, create_matrix = create_matrix)
        
            
        out = field_out, wavelengths, pixelsize
    return out
      

def transfer_4x4(field_data, optical_data, beta = 0., 
                   phi = 0., eff_data = None, nin = 1., nout = 1., npass = 1,nstep=1,
              diffraction = True, reflection = 1, multiray = False,norm = DTMM_NORM_FFT, smooth = SMOOTH,
              betamax = None, ret_bulk = False, out = None):
    """Transfers input field data through optical data. See transfer_field.
    
    """
    betamax = get_default_config_option("betamax", betamax)
    
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
    #d, epsv, epsa = validate_optical_data(optical_data)
                   
    #define input field data
    field_in, wavelengths, pixelsize = field_data
    
    shape = field_in.shape[-2:]

    layers, eff_layers = _create_layers(optical_data, eff_data, nin, nout, nstep, shape)
    
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
        pass
        #make sure we take only the forward propagating part of the field
        #transmitted_field(field, ks, n = nin, betamax = min(betamax,nin), out = field)
        #field_in[...] = field
        
    if calc_reference:
        ref = transmitted_field(field, ks, n = nin, betamax = min(betamax,nin))
        #ref = field.copy()
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
    #tmpdata = {}
    
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
            print_progress(pindex,n, suffix = suffix, prefix = prefix) 
            
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
                                betamax = _betamax, out = out_field)
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
        print_progress(n,n, suffix = suffix, prefix = prefix) 
        
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
                        field = project_normalized_local(field, pout_mat, nout, ref = ref, out = field)
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




def _layers_list(optical_block, eff_data, nin, nout, nstep, shape, is_first = False, is_last = False):
    """Build optical data layers list and effective data layers list.
    It appends/prepends input and output layers. A layer consists of
    a tuple of (n, thickness, epsv, epsa) where n is number of sublayers"""
    d, epsv, epsa = optical_block
    
    if epsa is not None:
        substeps = np.broadcast_to(np.asarray(nstep),(len(d),))
        layers = [(n,(t/n, ev, ea)) for n,t,ev,ea in zip(substeps, d, epsv, epsa)]
        #add input and output layers if is first block or last block
        if is_first:
            layers.insert(0, (1,(0., np.broadcast_to(refind2eps([nin]*3), epsv[0].shape), np.broadcast_to(np.array((0.,0.,0.), dtype = FDTYPE), epsa[0].shape))))
        if is_last:
            layers.append((1,(0., np.broadcast_to(refind2eps([nout]*3), epsv[0].shape), np.broadcast_to(np.array((0.,0.,0.), dtype = FDTYPE), epsa[0].shape))))

        try:
            d_eff, epsv_eff, epsa_eff = validate_optical_block(eff_data, shape = (1,1))        
        except (TypeError, ValueError):
            if eff_data is None:
                d_eff, epsv_eff, epsa_eff = _isotropic_effective_data(optical_block)
            else:
                d_eff, epsv_eff, epsa_eff = effective_block(optical_block, symmetry = eff_data)        
        eff_layers = [(n,(t/n, ev, ea)) for n,t,ev,ea in zip(substeps, d_eff, epsv_eff, epsa_eff)]
        if is_first:
            eff_layers.insert(0, (1,(0., refind2eps([nin]*3), np.array((0.,0.,0.), dtype = FDTYPE))))
        if is_last:
            eff_layers.append((1,(0., refind2eps([nout]*3), np.array((0.,0.,0.), dtype = FDTYPE))))
        return layers, eff_layers
    else:
        substeps = np.broadcast_to(np.asarray(nstep),(len(d),))
        layers = [(n,(t/n, ev, None)) for n,t,ev in zip(substeps, d, epsv)]
        #add input and output layers
        if is_first:
            layers.insert(0, (1,(0., np.broadcast_to(refind2eps([nin,nin,nin,0,0,0]), epsv[0].shape), None)))
        if is_last:
            layers.append((1,(0., np.broadcast_to(refind2eps([nout,nout,nout,0,0,0]), epsv[0].shape), None)))

        try:
            d_eff, epsv_eff, epsa_eff = validate_optical_block(eff_data, shape = (1,1))        
        except (TypeError,ValueError):
            if eff_data is None:
                d_eff, epsv_eff, epsa_eff = _isotropic_effective_data(optical_block)
            else:
                d_eff, epsv_eff, epsa_eff = effective_block(optical_block, symmetry = eff_data)
                    
        eff_layers = [(n,(t/n, ev, ea)) for n,t,ev,ea in zip(substeps, d_eff, epsv_eff, epsa_eff)]
        if is_first:
            eff_layers.insert(0, (1,(0., refind2eps([nin]*3), np.array((0.,0.,0.), dtype = FDTYPE))))
        if is_last:
            eff_layers.append((1,(0., refind2eps([nout]*3), np.array((0.,0.,0.), dtype = FDTYPE))))
        return layers, eff_layers       

def _create_layers(optical_data, eff_data, nin, nout, nstep, shape):
    def is_first(i):
        return True if i == 0 else False
    
    def is_last(i):
        return True if i == len(optical_data)-1 else False
    
    if isinstance(optical_data, list):
        if isinstance(eff_data, str) or isinstance(eff_data, int):
            # one identifier for all layers
            builder =(_layers_list(d, eff_data, nin, nout, nstep, shape, is_first(i), is_last(i)) for i,d in enumerate(optical_data))
        else:
            # each group of layers must have its own effective data identifier.
            if len(eff_data) != len(optical_data):
                raise ValueError("Number of elements in eff_data must match length of optical_data.")
            builder =(_layers_list(d, e, nin, nout, nstep, shape, is_first(i), is_last(i)) for i,(d,e) in enumerate(zip(optical_data,eff_data)))
        
        layers = []
        eff_layers = []
        for layer, eff_layer in builder:
            layers += layer
            eff_layers += eff_layer
            
        return layers, eff_layers
    else:
        #legacy data format, check only
        validate_optical_data(optical_data,copy = False)  
        return _layers_list(optical_data, eff_data, nin, nout, nstep, shape, is_first = True, is_last = True)
            
def _block_layers_list(optical_block, nin = None, nout = None, nstep = 1):
    """Build optical data layers list.
    It appends/prepends input and output layers. A layer consists of
    a tuple of (n, thickness, epsv, epsa) where n is number of sublayers"""
    d, epsv, epsa = optical_block
        
    substeps = np.broadcast_to(np.asarray(nstep),(len(d),))

    layers = [(n,(t/n, ev, ea)) for n,t,ev,ea in zip(substeps, d, epsv, epsa)]

    first_layer = (0., np.broadcast_to(refind2eps([nin]*3), epsv[0].shape), np.broadcast_to(np.array((0.,0.,0.), dtype = FDTYPE), epsa[0].shape))        
    layers.insert(0, (1,first_layer))

    last_layer = (0., np.broadcast_to(refind2eps([nout]*3), epsv[0].shape).copy(), np.broadcast_to(np.array((0.,0.,0.), dtype = FDTYPE), epsa[0].shape).copy())
    layers.append((1,last_layer))
    
    return layers

def _block_eff_layers_list(eff_data,  nin = None, nout = None, nstep = 1):
    """Build  effective data layers list.
    It appends/prepends input and output layers. A layer consists of
    a tuple of (n, thickness, epsv, epsa) where n is number of sublayers"""
    d_eff, epsv_eff, epsa_eff = eff_data
    

    substeps = np.broadcast_to(np.asarray(nstep),(len(d_eff),))
    

    eff_layers = [(n,(t/n, ev, ea)) for n,t,ev,ea in zip(substeps, d_eff, epsv_eff, epsa_eff)]
  
    first_eff_layer = (0., refind2eps([nin]*3), np.array((0.,0.,0.), dtype = FDTYPE))
    eff_layers.insert(0, (1,first_eff_layer))    
    
    last_eff_layer = (0., refind2eps([nout]*3), np.array((0.,0.,0.), dtype = FDTYPE))            
    eff_layers.append((1,last_eff_layer))

    return eff_layers

def _iterate(block_layers, reverse = False):
    iterator = tuple(((i,data) for i,data in enumerate(block_layers)))
    if reverse == True:
        iterator = reversed(iterator)
    for out in iterator:
        yield out
        
def _iterate_layers(layers_data, eff_layers_data, reverse = False):
    nlayers = len(layers_data)
    assert nlayers == len(eff_layers_data)
    
    if reverse == True:
        # we are going to propagate in backward direction using forward-propagating 
        # algorithm, so we make thickness negative. The phase shift k*d will be negative
        # in the calculation.
        layers_data = [(n, (-d, epsv,epsa)) for (n, (d,epsv,epsa)) in layers_data]
        eff_layers_data = [(n, (-d, epsv,epsa)) for (n, (d,epsv,epsa)) in eff_layers_data]
    
    layer_indices = list(range(nlayers))
    # nlayers minus 1, number of layer interfaces
    interface_indices = list(range(nlayers-1))
    
    layers = list(zip(layer_indices, layers_data, eff_layers_data))

    if reverse == False:
        
        input_layers = layers[:-1]
        output_layers = layers[1:]
    else:
        interface_indices = interface_indices[-1::-1]
        input_layers = layers[-1:0:-1]
        output_layers = layers[-2::-1]        

    for out in zip(interface_indices, input_layers, output_layers):
        yield out

def _output_epsv(eff_data):
    #take last layer of the block as an output refractive indices
    return [epsv[-1] for (d, epsv, epsa) in eff_data]

def _output_epsa(eff_data):
    #take last layer of the block as an output refractive indices
    return [epsa[-1] for (d, epsv, epsa) in eff_data]

def _bulk_data_count(n_physical_layers):
    """Given the number of physical layers, determines the count for bulk data."""
    # must be: num of layers + num of block interfaces + 2
    # which is: num of layers + num blocks + 1
    return sum(n_physical_layers) + len(n_physical_layers) + 1

def _split_into_block_list(array, n_physical_layers):
    """Splits input array into overlapping arrays. Each array in the output
    list overlaps with previous/next array in the list by one element, except
    the first and the last elements of the list, which only have one overlapping element.
    So the first element of i-th array is the last element of the i-1 -th array.
    and the last element of i-th array is the first element of the 1+1 -th array.
    """
    if len(array)!= _bulk_data_count(n_physical_layers):
        raise ValueError("Array length is incompatible with specified physical layers count")
    n1 = 1 + np.asarray(n_physical_layers)
    n2 = n1 + 1                    
    offset = [0] + list(np.cumsum(n1)[0:-1])
    out = [array[o:o+n] for (o,n) in zip(offset, n2)]
    for (o,n) in zip(out, n2):
        if len(o) != n:
            raise ValueError("Cannot split input array into list. Invalid number of layers")
    return out
    
def build_transfer_matrices(block_data, shape, k0, nin = None, nout =None, betamax = None, dim = 1, method = "4x4"):
    betamax = get_default_config_option("betamax", betamax)
    nin = get_default_config_option("nin", nin)
    nout = get_default_config_option("nin", nout)
    
    d,epsv,epsa = block_data
    if material_dim(epsv, epsa) > dim:
        return None
    else:
        mask = eigenmask(shape,k0, betamax = betamax)
        return transfer_matrices3d(mask, k0, [(d, epsv, epsa)], nin = nin, nout = nout, method = method)
    
def transfer_2x2(field_data, optical_data, beta = None, 
                   phi = None, eff_data = None, nin = 1., 
                   nout = 1., npass = 1, nstep=1,
              diffraction = True, reflection = True, multiray = False, split_diffraction = False,
              betamax = None, ret_bulk = False, create_matrix = 0, out = None):
    """Tranfers input field data through optical data using the 2x2 field matrices.
    Optionally, it can create 2x2 or 4x4 transfer matrices, depending on the.
    See transfer_field for documentation.
    
    You should use :func:`transfer_field` instead of this.
    """
    betamax = get_default_config_option("betamax", betamax)
    
    if reflection not in (0,1,2):
        raise ValueError("Invalid reflection. The 2x2 method supports reflection mode 0,1 or 2.")
    
    verbose_level = DTMMConfig.verbose
    if verbose_level >1:
        print(" * Initializing.")
         
    try:
        eff_data = validate_optical_data(eff_data, shape = (1,1))        
    except (TypeError, ValueError):
        symmetry = 0 if eff_data is None else eff_data
        eff_data = effective_data(optical_data, symmetry = symmetry)   
    
    #: defines input field data
    field_in, wavelengths, pixelsize = field_data
    
    #: wavenumbers
    ks = k0(wavelengths, pixelsize)
    
    shape = field_in.shape[-2:]
    
    #: defines how many blocks of data we have
    nblocks = len(optical_data)
    
    epsv_out = _output_epsv(eff_data)
    #: mean block refractive index
    ns = [uniaxial_order(0,epsv)[0]**0.5 for epsv in epsv_out]

    #: input and output layer's refractive index for each of the blocks.     
    nins = [nin] + ns[:-1]
    nouts = ns[:-1] + [nout]
    
    if reflection > 0:
        if npass > 1:
            transfer_matrix_method = "4x4" 
        else:
            #single pass with single reflection
            transfer_matrix_method = "4x4_1"
    else:
        #no reflections, so use 2x2 matrices.
        transfer_matrix_method = "2x2"

    #:transfer matrices, wherever it is set to build the matrices.
    block_matrices = [build_transfer_matrices(block_data, shape, ks, nin, nout, betamax = betamax, dim = create_matrix, method = transfer_matrix_method) for (block_data, nin, nout) in zip(optical_data, nins,nouts)]
    
    #: defines number of physical layers per block
    # first determine number of physical layers
    nlayers = [len(block_data[0]) for block_data in optical_data]
    # now set number of physical layers to 0 if we have transfer matrices, else keep number of physcical layers.
    nlayers = [(n if m is None else 0) for (n,m) in zip(nlayers, block_matrices)] 
    
    #: number of sub-steps per block
    nsteps = [nstep] * nblocks
    
    #: virtual block layers used for computation
    block_layers = [_block_layers_list(block_data, nin = nin, nout = nout, nstep = nstep) for i,(block_data, nin, nout, nstep) in enumerate(zip(optical_data, nins,nouts, nsteps))]
    # virtual effective layers
    block_eff_layers = [_block_eff_layers_list(block_eff_data, nin = nin, nout = nout, nstep = nstep) for i,(block_eff_data, nin, nout, nstep) in enumerate(zip(eff_data, nins,nouts, nsteps))]
    
    #: number of total layers in each optical block. 
    block_nlayers = [len(optical_block[0])+2 for optical_block in optical_data]    
    # 2 extra layers are for the input- and output-coupling layers
    

    if beta is None and phi is None:
        ray_tracing = False
        beta, phi = field2betaphi(field_in,ks, multiray)
    else:
        ray_tracing = False
    if diffraction != 1:
        ray_tracing = False
    beta, phi = _validate_betaphi(beta,phi,extendeddim = field_in.ndim-2)

    #: define output field
    if out is None:
        if ret_bulk == True:
            # number of layers in bulk data. All physical layers + input and output layers
            nbulk = _bulk_data_count(nlayers)
            bulk_out = np.zeros((nbulk,)+field_in.shape, field_in.dtype) 
            bulk_out = _split_into_block_list(bulk_out, nlayers)
            bulk_out[0][0] = field_in
            field_in = [bulk_out[0][0]] + [None]*(nblocks-1)
            field_out = [None]*(nblocks-1) + [bulk_out[-1][-1]]

        else:
            bulk_out = None
            field_out = [None] * (nblocks-1) + [np.zeros_like(field_in)]
            field_in = [field_in] + [None] * (nblocks-1)
    else:
        if ret_bulk == True:
            nbulk = _bulk_data_count(nlayers)
            bulk_out = _split_into_block_list(out, nlayers)
            bulk_out[0][0] = field_in
            field_in = [bulk_out[0][0]] + [None]*(nblocks-1)
            field_out = [None]*(nblocks-1) + [bulk_out[-1][-1]]
        else:
            bulk_out = None
            field_out = [None] * (nblocks-1) + [out]
            field_in = [field_in] + [None] * (nblocks-1)
        
    # make sure we take only the forward propagating part of the field
    if diffraction > 0:
        field0 = transmitted_field(field_in[0], ks, n = nin, betamax = betamax)
    else:
        field0 = transmitted_field_direct(field_in[0], beta, phi, n = nin)
    
    field = field0[...,::2,:,:].copy()
    
    if npass > 1:
        if reflection == 0:
            raise ValueError("Reflection mode `0` not compatible with npass > 1")
        field_in[0][...] = field0 #modify input field so that it has no back reflection
        #keep reference to reflected waves
        nbulk = _bulk_data_count(nlayers)
        refl = np.zeros((nbulk,)+field.shape[:-3] + (2,) + field.shape[-2:], field.dtype) 
        refl = _split_into_block_list(refl, nlayers)        
    else:
        #no need to store reflected waves
        refl = [[None]*n for n in block_nlayers]
        
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
            
        if i > 0:
            field[...] = 0.
            
        direction = (-1)**i  
        # Each even pass has to be computed in reverse order
        reverse  = False if direction  == 1 else True
        
        # iteration over data blocks. iblock is current block index, 
        # layers_list and eff_layers_list are layers data and effective layers data lists
        # they are of length of len(block_data[0]) + 2, because we add to the stack 
        # one input and one output layers of thickness 0. 
        for iblock, (matrices, layers_list, eff_layers_list) in _iterate(zip(block_matrices,block_layers, block_eff_layers), reverse = reverse):
            if verbose_level > 0:
                prefix = " * Pass {:2d}/{}, Block {:2d}/{}".format(i+1,npass,iblock+1,nblocks)
                suffix = ""
            else:
                prefix = ""
                suffix = "{}/{};{}/{}".format(i+1,npass,iblock+1,nblocks)            
            
            # transfer field using pre-computed transfer matrices
            if matrices is not None:
                #determine input and output-coupling layer refractive index
                #(_, first_layer), (_, last_layer) = eff_layers_list[0], eff_layers_list[-1]
                #nin = float(uniaxial_order(0,first_layer[1])[0]**0.5)
                #nout = float(uniaxial_order(0,last_layer[1])[0]**0.5)
                nin = nins[iblock]
                nout = nouts[iblock]
                # reflection interface is always 0. optical block is considered as a single reflection interface
                interface = 0
                if direction == -1:
                    #roles of input/ output layers are reversed if propagatign backward
                    nin, nout = nout, nin
                    
                if direction == -1 and iblock == 0:
                    #we are propagating backward and have reached the input coupling layer.
                    bulk = field_in[iblock]
                # we are propagating forward and we have reached the output-coupling layer 
                elif direction == +1 and iblock == nblocks - 1:
                    bulk = field_out[iblock]
                else:
                    if bulk_out is None:
                        bulk = None
                    else:
                        if direction == +1:
                            bulk = bulk_out[iblock][-1]
                        else:
                            bulk = bulk_out[iblock][0]
                field = transfer_jones3d(field,ks,matrices, nin = nin, nout = nout, mode = direction, 
                                 input_fft = work_in_fft, output_fft = work_in_fft,
                                 refl = refl[iblock][interface], bulk = bulk, method = transfer_matrix_method)

            # propagate field    
            else:
                
                # defines how many interfaces we have
                n = block_nlayers[iblock] - 1
                
                #iterate over layers.
                for pindex, (interface, input_data, output_data) in enumerate(_iterate_layers(layers_list, eff_layers_list,reverse = reverse)):
                    
                    print_progress(pindex,n,suffix = suffix, prefix = prefix) 
                    
                    j = interface
                    jin, (_, input_layer), (_, input_layer_eff)  = input_data
                    jout, (nstep, output_layer), (_, output_layer_eff)  = output_data
                    
                    # if jout is zero, it means we are propagating reflected field backward
                    # and we have reached the input-coupling layer
                    if jout == 0:
                        bulk = field_in[iblock]
                    # if we have reached the output-coupling layer - propagating forward.
                    elif jout == block_nlayers[iblock]-1:
                        bulk = field_out[iblock]
                    else:
                        if bulk_out is None:
                            bulk = None
                        else:
                            bulk = bulk_out[iblock][jout]
                            
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
                                    betamax = betamax,mode = direction, refl = refl[iblock][j], bulk = bulk, tmpdata = tmpdata)
                        else:
                            field, refli = propagate_2x2_effective_2(field, ks, input_layer, output_layer ,input_layer_eff, output_layer_eff, 
                                    beta = beta, phi = phi, nsteps = nstep, diffraction = diffraction, split_diffraction = split_diffraction, reflection = reflection, 
                                    betamax = betamax,mode = direction, refl = refl[iblock][j], bulk = bulk, tmpdata = tmpdata)
                    else:   
                        
                        field, refli = propagate_2x2_full(field, ks, output_layer, input_layer = input_layer, 
                            nsteps = nstep,  reflection = reflection, mode = direction,
                            betamax = betamax, refl = refl[iblock][j], bulk = bulk)
        
                print_progress(n,n,suffix = suffix, prefix = prefix) 
            
    if ret_bulk == True:
        # if we are returning bulk data, we must create field data for each of the blocks.
        return tuple(((bulk, wavelengths, pixelsize) for bulk in bulk_out))
    else:
        return field_out[-1], wavelengths, pixelsize  




from dtmm import tmm3d
from dtmm.data import material_shape, optical_data_shape
         
def data_stack_mat3d(mask, k0, optical_data, method = "4x4"):
    verbose_level = DTMMConfig.verbose
    if verbose_level > 0:
        print("Computing optical data stack matrices.")
    
    def iterate_stack_matrices(mask, k0, optical_data, method):
        for i,block_data in enumerate(optical_data):
            if verbose_level > 0:
                print("Block {}/{}".format(i+1,len(optical_data)))
            d,epsv,epsa = block_data
            mat = tmm3d.stack_mat3d(k0, d, epsv, epsa, method = method, mask = mask) 
            yield mat
            
    data_shapes = tuple((material_shape(epsv,epsa) for (d,epsv,epsa) in optical_data))
    reverse = False if method.startswith("4x4") else True
    out = tmm3d.multiply_mat3d(iterate_stack_matrices(mask, k0, optical_data, method), mask = mask, data_shapes = data_shapes, reverse = reverse)
    
    return out

def _create_list(arg):
    if isinstance(arg,list):
        raise ValueError("Argument already a list")
    return [arg]

def transfer_matrices3d(mask, k0, optical_data, nin = 1., nout = 1., method = "4x4"):
   
    common_shape = optical_data_shape(optical_data)
    mat = data_stack_mat3d(mask, k0, optical_data,  method = method)
    
    #now build up matrices field matrices       
    fmatin = tmm3d.f_iso3d(mask, k0, n = nin, shape = common_shape)
    fmatout = tmm3d.f_iso3d(mask, k0, n = nout, shape = common_shape)
    
    #compute reflection/transmission matrices
    if method.startswith("4x4"):
        mat = tmm3d.system_mat3d(mat,fmatin,fmatout)
        mat = tmm3d.reflection_mat3d(mat)
    else:
        #2x2 method, transmit field without reflections
        mat = tmm3d.transmission_mat3d(mat)
        
    mask = tmm3d.split_mask3d(mask, common_shape)
    
    return mask, fmatin, fmatout, mat
   
def transfer_jones3d(jones_in, ks, matrices, nin = 1, nout = 1, mode = +1, method = "4x4", input_fft = False, output_fft = False, betamax = None, refl = None, bulk = None):
    betamax = get_default_config_option("betamax", betamax)
    
    masks, fmat_ins, fmat_outs, mats = matrices
    
    if method.startswith("4x4"):
        field_in = jones2field(jones_in, ks, epsv = refind2eps([nin]*3), mode = mode, input_fft = input_fft, output_fft = True,betamax = betamax)
    else:
        field_in = jones_in if input_fft else fft2(jones_in)
    field_out = np.zeros_like(field_in)
        
    if mode != +1:
        #input/output field roles for reflect3d are reversed, so swap them.
        field_in, field_out = field_out, field_in
        
    #swap so that we can iterate over wavelengths
    field_in_swapped = np.swapaxes(field_in, -4,0)
    field_out_swapped = np.swapaxes(field_out, -4,0) 
    
    #iterate over wavelengths    
    for wff_in, wff_out, mask_list,fin_list,fout_list,mat_list in zip(field_in_swapped, field_out_swapped, masks, fmat_ins, fmat_outs, mats):  
        if not isinstance(mask_list, list):
            #1d case, all are arrays, make them as lists so that we can iterate
            mask_list = _create_list(mask_list)
            fin_list = _create_list(fin_list)
            fout_list = _create_list(fout_list)
            mat_list = _create_list(mat_list)
            
        for mask, fmatin, fmatout, mat in zip(mask_list,fin_list, fout_list, mat_list):
            
            modes_in = select_modes(wff_in, mask)
            
            
            if method.startswith("4x4"):
                modes_out = select_modes(wff_out, mask)
                modes_out = tmm3d.reflect3d(modes_in, rmat = mat, fmatin = fmatin, fmatout = fmatout, fvecout = modes_out)
                set_modes(wff_in, mask, modes_in)
                set_modes(wff_out, mask, modes_out)
            else:
                modes_out = tmm3d.transmit3d(modes_in, tmat = mat, fmatin = fmatin, fmatout = fmatout)
                set_modes(wff_out, mask, modes_out)
            
    field_in = np.swapaxes(field_in_swapped, -4,0)
    field_out = np.swapaxes(field_out_swapped, -4,0)  
    
    if mode != +1:
        # we have done backward transform swap back
        field_in, field_out = field_out, field_in
    
    #if out is not None:
    #    out[...] = field_out[...,0::2,:,:]
    #    jones_out = out
    #else:
    if method.startswith("4x4"):
        jones_out = field_out[...,0::2,:,:]
    else:
        jones_out = field_out
        
    if refl is not None:
        jones_out += refl
        refl[...] = field_in[...,0::2,:,:] - jones_in

    if bulk is not None:
        bulk += jones2field(jones_out, ks, epsv = refind2eps([nout]*3), mode = mode, input_fft = True, 
                    output_fft = False, betamax = betamax)
    if output_fft == False:
        return ifft2(jones_out)
    else:
        return jones_out
    


    

__all__ = ["transfer_field", "transmitted_field", "reflected_field", "transfer_2x2", "transfer_4x4", "total_intensity"]
