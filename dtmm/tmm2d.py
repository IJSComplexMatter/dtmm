"""
4x4 transfer matrix method functions for 2d data. It can be used to transfer
2d plane waves over 2d or 1d data.

"""

import numpy as np
import math

from dtmm.conf import CDTYPE, DTMMConfig
from dtmm.linalg import dotmdm, inv, dotmv, bdotmm, bdotmd, bdotdm, dotmm
from dtmm.print_tools import print_progress

import dtmm.tmm as tmm
from dtmm.tmm import alphaffi, phase_mat, alphaf
from dtmm.data import validate_optical_layer, crop_fft,resize_epsva, symmetry
from dtmm.wave import eigenbetax1, eigenindices1, eigenmask1, eigenwave1, betaxy2beta, mask2betax1, mask2indices1,betaxy2phi, mask2order1
from dtmm.fft import mfft
from dtmm.data import material_shape
 


def _get_dimensions(epsv, epsa):
    if epsv.shape[-2] == 1 and epsa.shape[-2] == 1:
        dim = 1
    else:
        dim = 2
    return dim
    
def layer_mat2d(k0, d, epsv,epsa, mask = None, method = "4x4",  betay = 0., swap_axes = False, nsteps = 1, resize = 1):
    """Computes characteristic matrix of a single layer M=F.P.Fi,
    
    Numpy broadcasting rules apply.
    
    Parameters
    ----------
    k0 : float or sequence of floats
        A scalar or a vector of wavenumbers.
    d : float
        Layer thickness.
    epsv : ndarray
        Epsilon eigenvalues.
    epsa : ndarray
        Epsilon tensor rotation angles (psi, theta, phi).
    method : str, optional
        Either '4x4' (default), '4x4_1', or '2x2'.
    betay : float
        The beta value of the
    
    Returns
    -------
    cmat : ndarray
        Characteristic matrix of the layer.
    """
    dim = _get_dimensions(epsv, epsa) 
    d = np.asarray(d)

    if method not in ("2x2", "4x4","4x4_1"):
        raise ValueError("Unsupported method: '{}'".format(method))
        
    k0 = np.asarray(k0)
    
    if k0.ndim == 1:
        
        betay = np.asarray(betay)
        beta_shape = np.broadcast_shapes(betay.shape, k0.shape)
        betay = np.broadcast_to(betay,beta_shape)
        
        if mask is None:
            mask = [None] * len(k0)
        
        return tuple(layer_mat2d(k,d,epsv,epsa,mask = m,method = method, betay = by, swap_axes = swap_axes, nsteps= nsteps, resize = resize) for (k,m,by) in zip(k0,mask,betay))
            
    
    if mask is None:
        shape = material_shape(epsv,epsa, dim = 2)[0]
        mask = eigenmask1(shape, k0, betay)
        betax = eigenbetax1(shape, k0, betay)
        #indices = eigenindices1(shape, k0, betay)
        indices = mask2order1(mask)
    else:
        betax = mask2betax1(mask,k0)
        indices = mask2order1(mask)
        
    if dim == 2:
        return _layer_mat2d(k0,d,epsv,epsa, mask, betax,betay, indices, method,swap_axes,nsteps,resize)

    else:
        if swap_axes:  
            betas = betaxy2beta(betay,betax)
            phis = betaxy2phi(betay,betax) 
        else:
            betas = betaxy2beta(betax,betay)
            phis = betaxy2phi(betax,betay)   
        return tmm.layer_mat(k0*d,epsv,epsa, betas, phis, method = method)

# def _normalize_mat(mf,mask,f,j):
#     fi = inv(f)
        
#     mfm = dotmm(mf[mask,...],f)
   
#     delta = [tmm.intensity(mfm[...,i]).sum(0) for i in range(4)]
#     for i in range(4):
#         mfm[...,i] = (1/delta[i])**0.5*mfm[...,i] 
    
#     return dotmm(mfm,fi)
    
      
def _layer_mat2d(k0,d,epsv,epsa, mask, betaxs, betay,indices, method, swap_axes,nsteps, resize):   
    n = len(betaxs)
     
    power = int(math.log2(nsteps))
    steps = 2**power
    
    shape = material_shape(epsv,epsa, dim = 2)[0]
    
    kd = k0*d/steps
    
    imax =  indices.max(axis = 0)

    if resize == 1:
        new_shape = (imax*4)+1,
    elif resize == 2:
        new_shape = (imax*2)+1,
        
    #resize only if required and new_shape is actually smaller.
    if resize != 0 and new_shape[0] < shape:
        mask = crop_fft(mask, new_shape)
        sym = symmetry(epsv)
        epsv,epsa = resize_epsva(epsv, epsa, new_shape, symmetry = sym)
        shape = new_shape[0]
    
    if method.startswith("2x2"):
        out = np.empty(shape = (n, n, 2, 2), dtype = CDTYPE)
    else:
        out = np.empty(shape = (n, n, 4, 4), dtype = CDTYPE)
        if method == "4x4_1" and power > 0:
            out0 = np.zeros(shape = (n, n, 4, 4), dtype = CDTYPE)
            
    if swap_axes:  
        beta = betaxy2beta(betay,betaxs)
        phi = betaxy2phi(betay,betaxs) 
    else:
        beta = betaxy2beta(betaxs,betay)
        phi = betaxy2phi(betaxs,betay)   
        
    for j,(beta,phi) in enumerate(zip(beta, phi)):
        
        if method.startswith("2x2"):
            alpha,fmat = alphaf(beta,phi,epsv,epsa)
            f = tmm.E_mat(fmat, mode = +1, copy = False)
            fi = inv(f)
            pmat = phase_mat(alpha[...,::2],kd)
        else:
            alpha,f,fi = alphaffi(beta,phi,epsv,epsa)
            pmat = phase_mat(alpha,-kd)
            if method == "4x4_1":
                if power > 0:
                    alpha0,f0,f0i = alphaffi(beta,phi,(2.7,2.7,2.7),(0,0,0))
                    pmat0 = np.zeros_like(pmat[0])
                    pmat0[...,0::2] = 1.
                else:
                    pmat[...,1::2] = 0.
            elif method != "4x4":
                raise ValueError("Unsupported method.")
        #m is shape (...,4,4)
        m = dotmdm(f,pmat,fi)    
        
    
        wave = eigenwave1(shape, indices[j], amplitude = 1.)
        
        
        #wave is shape (...) make it broadcastable to (...,4,4)
        mw = m*wave[...,None,None]

        mf = mfft(mw, overwrite_x = True)

        mf = mf[mask,...]
        
        out[:,j,:,:] = mf
        
        if method == "4x4_1" and power > 0:
            m0 = dotmdm(f,pmat0,fi) 
            mw0 = m0*wave[...,None,None]
            mf0 = mfft(mw0, overwrite_x = True)
            mf0 = mf0[mask,...]

            out0[:,j,:,:] = mf0
        
        # if method == "4x4_1" and power > 0:
        #     m0 = dotmdm(f0,pmat0,f0i) 
        #     #mw0 = m0*wave[...,None,None]
        #     #mf0 = mfft(mw0, overwrite_x = True)
        #     #mf0 = m0[mask,...]

        #     out0[j,j,:,:] = m0

    for i in range(power):
        out = bdotmm(out,out)
    
    if method == "4x4_1" and power > 0:
        out = bdotmm(out,out0)
        
        
    return out

def dispersive_layer_mat2d(k0, d, epsv,epsa, mask = None, method = "4x4", betay = 0., swap_axes = False, nsteps = 1, resize = 1, wavelength = None):
    if wavelength is None:
        raise ValueError("`wavelength` is a required argument for dispersive layers.")
    k0 = np.asarray(k0) 
    wavelength = np.asarray(wavelength)
    if k0.shape != wavelength.shape:
        raise ValueError("Wrong length `wavelength` argument.")
    if k0.ndim == 0:
        return layer_mat2d(k0, *validate_optical_layer((d,epsv,epsa), wavelength = wavelength), \
                mask = mask, method = method, betay = betay, swap_axes = swap_axes, nsteps = nsteps, resize = resize)
    else:
        return tuple((layer_mat2d(k, *validate_optical_layer((d,epsv,epsa), wavelength = w), \
                mask = mask, method = method, betay = betay, swap_axes = swap_axes, nsteps = nsteps, resize = resize) for k,w in zip(k0, wavelength)))  

def stack_mat2d(k,d,epsv,epsa,mask = None, method = "4x4", betay = 0. ,swap_axes=False, nsteps = 1, resize = 1, wavelength = None):
     
    k = np.asarray(k)
    dim = _get_dimensions(epsv, epsa)
    
    def _iterate_wavelengths():
        for i in range(len(k)):
            verbose_level = DTMMConfig.verbose
            if verbose_level > 0:    
                print("Wavelength {}/{}".format(i+1,len(k)))
            m = None if mask is None else mask[i]
            if wavelength is None:
                wavelengths = [None] * len(k)
            yield stack_mat2d(k[i],d,epsv,epsa, method = method, mask = m, betay = betay[i], swap_axes = swap_axes, nsteps= nsteps, resize = resize, wavelength = wavelengths[i])
            
    n = len(d)
    
    try:
        nsteps= [int(i) for i in nsteps]
        if len(nsteps) != n:
            raise ValueError("Length of the `nsteps` argument must match length of `d`.")
    except TypeError:
        nsteps= [int(nsteps)] * n
        
    verbose_level = DTMMConfig.verbose
    if verbose_level > 1:
        print ("Building stack matrix in {}d.".format(dim))
        
    if k.ndim == 1:
        betay = np.asarray(betay)
        beta_shape = np.broadcast_shapes(betay.shape, k.shape)
        betay = np.broadcast_to(betay,beta_shape)
        return tuple(_iterate_wavelengths())
    
    for j in range(n):
        print_progress(j,n) 
        if wavelength is None:
            mat = layer_mat2d(k,d[j],epsv[j],epsa[j], mask = mask, method = method, betay = betay,swap_axes=swap_axes, resize = resize, nsteps= nsteps[j])
        else:
            mat = dispersive_layer_mat2d(k,d[j],epsv[j],epsa[j], mask = mask, method = method, betay = betay,swap_axes=swap_axes, nsteps= nsteps[j], resize = resize, wavelength = wavelength)
            
        if mat.ndim == 4:
        
        #if isinstance(mat, list):
            #2d case
            if j == 0:
                out = mat.copy()
                #out = [m.copy() for m in mat]
            else:
                if method.startswith("2x2"):
                    out = bdotmm(mat,out)
                    #out = [bdotmm(m,o) for m,o in zip(mat,out)]
                else:
                    out = bdotmm(out,mat)   
                    #out = [bdotmm(o,m) for m,o in zip(mat,out)]
        else:
            #1d case
            if j == 0:
                out = mat.copy()
            else:
                if method.startswith("2x2"):
                    out = dotmm(mat,out)
                else:
                    out = dotmm(out,mat)   
         
    print_progress(n,n) 
    return out

def f_iso2d(mask, k0, n = 1., shape = None, betay = 0, swap_axes = False):
    k0 = np.asarray(k0)
    
    if k0.ndim == 0:
        dim, = mask.shape # to test if valid mask shape
        if shape is None:
            shape = mask.shape[0]
        else:
            try:
                shape, = shape
            except TypeError:
                pass
            shape = int(shape)
        
        betay = 0 if betay is None else np.asarray(betay)
        
        betax = mask2betax1(mask,k0)
        if swap_axes:
            betax, betay = betay, betax
        
        beta = betaxy2beta(betax,betay)
        phi = betaxy2phi(betax,betay)

        fmat = tmm.f_iso(n = n, beta = beta, phi = phi)
        return fmat


    else:
        out = (f_iso2d(m, k, n = n, shape = shape, betay = betay, swap_axes = swap_axes) for m, k in zip(mask, k0))
        return tuple(out)

def _system_mat2d(fmatin, cmat, fmatout):
    """Computes a system matrix from a characteristic matrix Fin-1.C.Fout"""
    if cmat.ndim == 4:
        fini = inv(fmatin)
        out = bdotdm(fini,cmat)
        return bdotmd(out,fmatout)
    elif cmat.ndim == 3:
        return tmm.system_mat(cmat = cmat,fmatin = fmatin, fmatout = fmatout) 
    else:
        raise ValueError("Invalid matrix dim.")

def system_mat2d(cmat = None, fmatin = None, fmatout = None):
    """Computes a system matrix from a characteristic matrix Fin-1.C.Fout"""
    if cmat is None:
        raise ValueError("cmat is a required argument")
    if fmatin is None:
        raise ValueError("fmatin is a required argument.")
        
    if fmatout is None:
        fmatout = fmatin
    if isinstance(fmatin, tuple):
        if cmat is not None:
            out = (_system_mat2d(fi, c, fo) for fi,c,fo in zip(fmatin,cmat,fmatout))
        else:
            out = (_system_mat2d(fi, None, fo) for fi,fo in zip(fmatin,fmatout))
        return tuple(out)
    else:
        return _system_mat2d(fmatin, cmat, fmatout)

    
def _reflection_mat2d(smat):
    """Computes a 4x4 reflection matrix.
    """
    if smat.ndim == 4:
        shape = smat.shape[0:-4] + (smat.shape[-4] * 4,smat.shape[-4] * 4)  
        mat = np.moveaxis(smat, -2,-3)
        mat = mat.reshape(shape)
        return tmm.reflection_mat(mat)         
    elif smat.ndim == 3:
        return tmm.reflection_mat(smat)
    else:
        raise ValueError("Invalid `smat` dim")

def reflection_mat2d(smat):
    verbose_level = DTMMConfig.verbose
    if verbose_level > 1:
        print ("Building reflectance and transmittance matrix")    
    if isinstance(smat, tuple):
        out = []
        n = len(smat)
        for i,s in enumerate(smat):
            print_progress(i,n) 
            out.append(_reflection_mat2d(s))
        print_progress(n,n)     
        return tuple(out)
    else:
        return _reflection_mat2d(smat)

def _reflect2d(fvec_in, fmat_in, rmat, fmat_out, fvec_out = None):
    """Reflects/Transmits field vector using 4x4 method.
    
    This functions takes a field vector that describes the input field and
    computes the output transmited field and also updates the input field 
    with the reflected waves.
    """

    dim = 1 if rmat.shape[-1] == 4 else 2

    fmat_ini = inv(fmat_in)
        
    avec = dotmv(fmat_ini,fvec_in)

    a = np.zeros(avec.shape, avec.dtype)
    a[...,0::2] = avec[...,0::2]

    if fvec_out is not None:
        fmat_outi = inv(fmat_out)
        bvec = dotmv(fmat_outi,fvec_out)
        a[...,1::2] = bvec[...,1::2] 
    else:
        bvec = np.zeros_like(avec)
    if dim == 1:
        out = dotmv(rmat,a)
    else:
        av = a.reshape(a.shape[:-2] + (a.shape[-2]*a.shape[-1],))
        out = dotmv(rmat,av).reshape(a.shape)

    avec[...,1::2] = out[...,1::2]
    bvec[...,::2] = out[...,::2]
        
    dotmv(fmat_in,avec,out = fvec_in)   
    out = dotmv(fmat_out,bvec,out = out)

    return out

def reflect2d(fvecin,  rmat, fmatin, fmatout, fvecout = None):
    """Transmits/reflects field vector using 4x4 method.
    
    This functions takes a field vector that describes the input field and
    computes the output transmited field vector and also updates the input field 
    with the reflected waves.
    """

    verbose_level = DTMMConfig.verbose
    if verbose_level > 1:
        print ("Transmitting field.")   
    if isinstance(fvecin, tuple):
        n = len(fvecin)
        if fvecout is None:
            return tuple((_reflect2d(fvecin[i], fmatin[i], rmat[i], fmatout[i]) for i in range(n)))
        else:
            return tuple((_reflect2d(fvecin[i], fmatin[i], rmat[i], fmatout[i], fvecout[i]) for i in range(n)))
    else:
        return _reflect2d(fvecin, fmatin, rmat, fmatout, fvecout)
    
def _transmission_mat2d(cmat):
    """Computes a 2x2 transmission matrix.
    """
    if cmat.ndim == 4:
        shape = cmat.shape[0:-4] + (cmat.shape[-4] * 2,cmat.shape[-4] * 2)  
        mat = np.moveaxis(cmat, -2,-3)
        mat = mat.reshape(shape)
        return mat      
    elif cmat.ndim == 3:
        return cmat
    else:
        raise ValueError("Invalid `smat` dim")   
 
def transmission_mat2d(cmat):
    verbose_level = DTMMConfig.verbose
    if verbose_level > 1:
        print ("Building transmittance matrix")    
    if isinstance(cmat, tuple):
        out = []
        n = len(cmat)
        for i,s in enumerate(cmat):
            print_progress(i,n) 
            out.append(_transmission_mat2d(s))
        print_progress(n,n)     
        return tuple(out)
    else:
        return _transmission_mat2d(cmat)  
    
def _transmit2d(fvec_in, fmat_in, mat, fmat_out, fvec_out = None):
    """Transmits field vector using 2x2 matrix
    """
    dim = 1 if mat.shape[-1] == 4 else 2
    
    if fvec_in.shape[-1] == 4:
        a = tmm.fvec2E(fvec_in, fmat_in)
        out = None
    else:
        a = fvec_in
        out = fvec_out

    if dim == 1:
        out = dotmv(mat,a, out = out)
    else:
        new_shape = a.shape[:-2] + (a.shape[-2]*a.shape[-1],)
        av = a.reshape(new_shape)
        if out is not None:
            out = out.reshape(new_shape)
        out = dotmv(mat,av, out = out).reshape(a.shape)    
    
    if fvec_in.shape[-1] == 4:
        out = tmm.E2fvec(out,fmat_out,out = fvec_out)
    
    return out

def transmit2d(fvecin, tmat, fmatin,  fmatout, fvecout = None):
    """Transmits/reflects field vector using 2x2 method.
    
    This functions takes a field vector that describes the input field and
    computes the output transmited field vector and also updates the input field 
    with the reflected waves.
    """

    verbose_level = DTMMConfig.verbose
    if verbose_level > 1:
        print ("Transmitting field.")   
    if isinstance(fvecin, tuple):
        n = len(fvecin)
        if fvecout is None:
            return tuple((_transmit2d(fvecin[i], fmatin[i], tmat[i], fmatout[i]) for i in range(n)))
        else:
            return tuple((_transmit2d(fvecin[i], fmatin[i], tmat[i], fmatout[i], fvecout[i]) for i in range(n)))
    else:
        return _transmit2d(fvecin, fmatin, tmat, fmatout, fvecout)


def projection_mat2d(fmat, mode = +1):
    mode = int(mode)
    if isinstance(fmat, tuple):
        return tuple((projection_mat2d(m,mode = mode) for m in fmat))

    return tmm.projection_mat(fmat, mode =mode) 
    
def project2d(fvec, fmat, mode = +1):
    pmat = projection_mat2d(fmat, mode = mode)
    return dotmv(pmat,fvec)
    #return dotdv2d(pmat, fvec)


def dotmm2d(m1, m2):
    if isinstance(m1, tuple):
        return tuple((dotmm2d(_m1,_m2) for _m1, _m2 in zip(m1,m2)))
    if m1.ndim == 4:
        if m2.ndim == 4:
            return bdotmm(m1,m2)
        elif m2.ndim == 3:
            return bdotmd(m1,m2)
    elif m1.ndim == 3:
        if m2.ndim == 4:
            return bdotdm(m1,m2)
        elif m2.ndim == 3:
            return dotmm(m1,m2)
    #if we come to here, matrices are invalid
    raise ValueError("Invalid matrix shape")
    
def multiply_mat2d(matrices, reverse = False):
    mat0 = None
    for mat in matrices:
        if mat0 is not None:
            if reverse == False:       
                mat = dotmm2d(mat0, mat)
            else:
                mat = dotmm2d(mat, mat0)   
        mat0 = mat
    return mat    

def bdotmv(a,b):
    shape = a.shape[0:-4] + (a.shape[-4] * b.shape[-1],a.shape[-4] * b.shape[-1])  
    a = np.moveaxis(a, -2,-3)
    a = a.reshape(shape)
    bv = b.reshape(b.shape[:-2] + (b.shape[-2]*b.shape[-1],))
    return dotmv(a,bv).reshape(b.shape)

def dotmv2d(m, v):
    if isinstance(m, tuple):
        return tuple((dotmv2d(_m,_v) for _m, _v in zip(m,v)))
    if m.ndim == 4:
        return bdotmv(m,v) 
    elif m.ndim == 3:
        return dotmv(m,v)    
    #if we come to here, matrices are invalid
    raise ValueError("Invalid matrix shape")

def validate_modes2d(mask, modes):
    def validate_mode(mask, mode, i = None):
        mode = np.asarray(mode)
        if mode.ndim < 2:
            raise ValueError("mode {} must be a numpy array of dimension 2 or more.".format(i))
        elif mode.shape[-2:] != (mask.sum(),4):
            raise ValueError("mode {} must be of shape (...,{},4)".fromat(i))        
        return mode
    
    mask = np.asarray(mask)
    if not isinstance(mask, np.ndarray) or mask.ndim not in (1,2) or mask.dtype != bool:
        raise ValueError("`mask` must be a boolean numpy array of dimension 1 or 2.")
        
    if mask.ndim == 2:
        if not isinstance(modes, tuple) or len(modes) != len(mask):
            raise ValueError("`modes` must be a tuple of length {}".format(len(mask)))
        return mask, tuple((validate_mode(m,mode,i) for i, (m,mode) in enumerate(zip(mask,modes))))   

    else:
        return mask, validate_mode(mask, modes)
    
_f_iso = f_iso2d
_stack_mat = stack_mat2d
_layer_mat = layer_mat2d
_system_mat =  system_mat2d
_reflection_mat = reflection_mat2d
_transmission_mat = transmission_mat2d
_reflect = reflect2d
_transmit = transmit2d
_validate_modes = validate_modes2d
_dispersive_layer_mat = dispersive_layer_mat2d
_multiply_mat = multiply_mat2d
_dotmv = dotmv2d


