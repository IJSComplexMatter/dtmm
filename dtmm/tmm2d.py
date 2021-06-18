"""
4x4 transfer matrix method functions for 2d data. It can be used to transfer
2d plane waves over 2d or 1d data.



 
"""

from __future__ import absolute_import, print_function, division
import numpy as np

from dtmm.conf import  BETAMAX, CDTYPE, DTMMConfig
from dtmm.linalg import dotmdm, inv, dotmv, bdotmm, bdotmd, bdotdm, dotmm
from dtmm.print_tools import print_progress

import dtmm.tmm as tmm
from dtmm.tmm import alphaffi, phase_mat, alphaf

from dtmm.wave import eigenbetax1, eigenindices1, eigenmask1, eigenwave1, betaxy2beta, mask2betax1, mask2indices1,betaxy2phi
from dtmm.wave import k0 as wavenumber
from dtmm.field import field2modes1, modes2field1
from dtmm.fft import mfft

# def list_modes(modes):
#     return tuple(([m] for m in modes))

# def unlist_modes(modes):
#     return tuple((m[0] for m in modes))    

def _get_dimensions(epsv, epsa):
    if epsv.shape[-2] == 1 and epsa.shape[-2] == 1:
        dim = 1
    else:
        dim = 2
    return dim

def layer_mat2d(k0, d, epsv,epsa, mask = None, method = "4x4",  betay = 0., swap_axes = False, resolution_power = 0):
    """Computes characteristic matrix of a single layer M=F.P.Fi,
    
    Numpy broadcasting rules apply.
    
    Parameters
    ----------
    k0 : float or sequence of floats
        A scalar or a vector of wavenumbers
    d : array_like
        Layer thickness
    epsv : ndarray
        Epsilon eigenvalues.
    epsa : ndarray
        Optical axes orientation angles (psi, theta, phi).
    method : str, optional
        Either '4x4' (default), '4x4_1', or '2x2'.
    
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
        
        return tuple(layer_mat2d(k,d,epsv,epsa,mask = m,method = method, betay = by, swap_axes = swap_axes, resolution_power = resolution_power) for (k,m,by) in zip(k0,mask,betay))
            
    
    if mask is None:
        shape = epsv.shape[-2]
        mask = eigenmask1(shape, k0, betay)
        betax = eigenbetax1(shape, k0, betay)
        indices = eigenindices1(shape, k0, betay)
    else:
        betax = mask2betax1(mask,k0)
        indices = mask2indices1(mask)
        
    if dim == 2:
        return _layer_mat2d(k0,d,epsv,epsa, mask, betax,betay, indices, method,swap_axes,resolution_power)

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
    
      
def _layer_mat2d(k0,d,epsv,epsa, mask, betaxs, betay,indices, method, swap_axes,resolution_power):   
    n = len(betaxs)
     
    steps = 2**resolution_power
    
    shape = epsv.shape[-2]
    
    kd = k0*d/steps
    
    if method.startswith("2x2"):
        out = np.empty(shape = (n, n, 2, 2), dtype = CDTYPE)
    else:
        out = np.empty(shape = (n, n, 4, 4), dtype = CDTYPE)
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
                pmat[...,1::2] = 0.
            elif method != "4x4":
                raise ValueError("Unsupported method.")
                
    
        wave = eigenwave1(shape, indices[j], amplitude = 1.)
        
        #m is shape (...,4,4)
        m = dotmdm(f,pmat,fi) 
        
        #wave is shape (...) make it broadcastable to (...,4,4)
        mw = m*wave[...,None,None]

        mf = mfft(mw, overwrite_x = True)

        mf = mf[mask,...]
        
        out[:,j,:,:] = mf

    for i in range(resolution_power):
        out = bdotmm(out,out)
    return out

def stack_mat2d(k,d,epsv,epsa,mask = None, method = "4x4", betay = 0. ,swap_axes=False, resolution_power = 0):
     
    k = np.asarray(k)
    dim = _get_dimensions(epsv, epsa)
    
    def _iterate_wavelengths():
        for i in range(len(k)):
            verbose_level = DTMMConfig.verbose
            if verbose_level > 0:    
                print("Wavelength {}/{}".format(i+1,len(k)))
            m = None if mask is None else mask[i]
            yield stack_mat2d(k[i],d,epsv,epsa, method = method, mask = m, betay = betay[i], swap_axes = swap_axes, resolution_power = resolution_power)
            
    n = len(d)
    
    try:
        resolution_power = [int(i) for i in resolution_power]
        if len(resolution_power) != n:
            raise ValueError("Length of the `resolution_power` argument must match length of `d`.")
    except TypeError:
        resolution_power = [int(resolution_power)] * n
        
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
        mat = layer_mat2d(k,d[j],epsv[j],epsa[j], mask = mask, method = method, betay = betay,swap_axes=swap_axes, resolution_power = resolution_power[j])
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
            dim = int(shape) # to test if valid shape
        
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
    # #if isinstance(cmat, list):
    #     if len(cmat) == len(fmatin) and len(cmat) == len(fmatout):
    #         out = []
    #         for fin,c,fout in zip(fmatin,cmat,fmatout):
    #             fini = inv(fin)
    #             o = bdotdm(fini,c)
    #             out.append(bdotmd(o,fout))  
    #         return out
    #     else:
    #         raise ValueError("Wrong input data lengths")
    else:
        return tmm.system_mat(cmat = cmat,fmatin = fmatin, fmatout = fmatout)      

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
    
    # def iterate(smat):
    #     for mat in smat:
    #         shape = mat.shape[0:-4] + (mat.shape[-4] * 4,mat.shape[-4] * 4)  
    #         mat = np.moveaxis(mat, -2,-3)
    #         mat = mat.reshape(shape)
    #         yield tmm.reflection_mat(mat)
    
    #if isinstance(smat, list):
    #    return list(iterate(smat))
    if smat.ndim == 4:
        shape = smat.shape[0:-4] + (smat.shape[-4] * 4,smat.shape[-4] * 4)  
        mat = np.moveaxis(smat, -2,-3)
        mat = mat.reshape(shape)
        return tmm.reflection_mat(mat)        
    
    else:
        return tmm.reflection_mat(smat)

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
    # if isinstance(rmat, list):
    #     if not isinstance(fvec_in, list):
    #         raise ValueError("`rmat` is a list, so `fvecin` must be listed as well.")
    #     #2d and 3d case, we must iterate over modes
    #     if fvec_out is None:
    #         fvec_out = [None] * len(fvec_in)
    #     return [_reflect2d(*args) for args in zip(fvec_in,fmat_in,rmat,fmat_out,fvec_out)]
    
    #dim = 2 if isinstance(rmat, list) else 1
    #rmat = rmat[0]
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

def projection_mat2d(fmat, mode = +1):
    mode = int(mode)
    if isinstance(fmat, tuple):
        return tuple((projection_mat2d(m,mode = mode) for m in fmat))

    return tmm.projection_mat(fmat, mode =mode) 
    
def project2d(fvec, fmat, mode = +1):
    pmat = projection_mat2d(fmat, mode = mode)
    return dotmv(pmat,fvec)
    #return dotdv2d(pmat, fvec)

# def dotmm2d(m1, m2):
#     if isinstance(m1, tuple):
#         return tuple((dotmm2d(_m1,_m2) for _m1, _m2 in zip(m1,m2)))
#     if isinstance(m1, list):
#         if isinstance(m2, list):
#             return [bdotmm(a,b) for a,b in zip(m1,m2)]
#         else:
#             return [bdotmd(a,b) for a,b in zip(m1,[m2])]   
#     else:
#         if isinstance(m2, list):
#             return [bdotdm(a,b) for a,b in zip([m1],m2)]
#         else:
#             return dotmm(m1,m2)
        
# def bdotmv(a,b):
#     shape = a.shape[0:-4] + (a.shape[-4] * b.shape[-1],a.shape[-4] * b.shape[-1])  
#     a = np.moveaxis(a, -2,-3)
#     a = a.reshape(shape)
#     bv = b.reshape(b.shape[:-2] + (b.shape[-2]*b.shape[-1],))
#     return dotmv(a,bv).reshape(b.shape)

# def dotmv2d(m, v):
#     if isinstance(m, tuple):
#         return tuple((dotmv2d(_m,_v) for _m, _v in zip(m,v)))
#     if isinstance(m, list):
#         if isinstance(v, list):
#             return [bdotmv(a,b) for a,b in zip(m,v)]
#         else:
#             raise ValueError("Matrix is a list of matrices, so vector should be a ist of vectors.")
#     else:
#         return dotmv(m,v)
    
# def dotdv2d(m, v):
#     if isinstance(m, tuple):
#         return tuple((dotdv2d(_m,_v) for _m, _v in zip(m,v)))
#     if isinstance(m, list):
#         if isinstance(v, list):
#             return [dotmv(a,b) for a,b in zip(m,v)]
#         else:
#             raise ValueError("Matrix is a list of matrices, so vector should be a ist of vectors.")
#     else:
#         return dotmv(m,v)    

    
# def transfer_matrices2d(shape, k0, d, epsv, epsa, betay = 0, nin = 1, nout = 1, method = "4x4", betamax = BETAMAX, swap_axes = False):
#     mask = eigenmask1(shape, k0, betay, betamax)
    
#     fmatin = f_iso2d(mask = mask, betay = betay, k0 = k0, n=nin, swap_axes = swap_axes)
#     fmatout = f_iso2d(mask = mask, betay = betay, k0 = k0, n=nout,swap_axes = swap_axes)
    
#     mat = stack_mat2d(k0,d, epsv, epsa, betay = betay, mask = mask, method = method,swap_axes = swap_axes)
#     mat = system_mat2d(fmatin = fmatin, cmat = mat, fmatout = fmatout)
#     mat = reflection_mat2d(smat = mat)
    
#     return mask, fmatin, fmatout, mat   

# def transfer2d(field_data_in, optical_data, betay = 0., nin = 1., nout = 1., method = "4x4", betamax = BETAMAX, swap_axes = False, field_out = None):
    
#     f,w,p = field_data_in

#     d,epsv,epsa = optical_data
#     k0 = wavenumber(w, p)

#     if field_out is not None:
#         mask, fmode_out = field2modes1(field_out,k0, betay, betamax = betamax)
#     else:
#         fmode_out = None
    
#     mask, fmode_in = field2modes1(f,k0, betay, betamax = betamax)
    

#     mask, fmatin, fmatout, rmat = transfer_matrices2d(f.shape[-1], k0, d, epsv,epsa, betay = betay, nin = nin, nout = nout, method = method, betamax = betamax, swap_axes = swap_axes)
    
#     fmode_out = reflect2d(fmode_in, rmat = rmat, fmatin = fmatin, fmatout = fmatout, fvecout = fmode_out)
    
#     field_out = modes2field1(mask, fmode_out)
#     f[...] = modes2field1(mask, fmode_in)
    
#     return field_out,w,p


