"""
4x4 and 2x2 transfer matrix method functions for 3d calculation. 
"""

from __future__ import absolute_import, print_function, division

import numpy as np

from dtmm.conf import CDTYPE,DTMMConfig

from dtmm.linalg import dotmm, inv, dotmv,  bdotmm, bdotmd, bdotdm, dotmdm
from dtmm.print_tools import print_progress
from dtmm.data import material_shape, shape2dim, material_dim
import dtmm.tmm as tmm
import dtmm.tmm2d as tmm2d
from dtmm.tmm import alphaf, alphaffi, phase_mat
from dtmm.wave import eigenbeta, eigenphi,eigenindices, eigenmask, eigenwave,mask2beta, mask2phi, mask2indices, betax1
from dtmm.fft import mfft2


def _iterate_masks_2d(mask, shape, k0):
    x,y = shape
    xm,ym = mask.shape
    if y == 1:
        if x != xm:
            raise ValueError("Invalid shape")
        nmodes = ym
        betay = betax1(ym, k0)
        for i in range(nmodes):
            m = mask[:,i]
            if np.any(m):
                yield m, betay[i] 
    elif x == 1:
        if y != ym:
            raise ValueError("Invalid shape")
        nmodes = xm
        betay = betax1(xm, k0)
        for i in range(nmodes):
            m = mask[i,:]
            if np.any(m):
                yield m, betay[i] 
    else:
        raise ValueError("Invalid shape")

def split_mask3d(mask, shape):
    """For 2d or 3d data, given the material shape, it splits input mask into
    a list of masks. For 1d data no change is made."""
    
    if isinstance(mask, tuple) or mask.ndim == 3:
        out = (split_mask3d(m,shape) for m in mask)
        return tuple(out)
    
    x,y = shape
    xm,ym = mask.shape
    
    if x == 1 and y == 1:
        #1d case
        return mask
    
    if y == 1:
        if x != xm:
            raise ValueError("Invalid shape")
        nmodes = ym
        out = []
        for i in range(nmodes):
            m = mask[:,i]
            if np.any(m):
                mo = np.zeros_like(mask)
                mo[:,i] = m
                out.append(mo)
        return out
    elif x == 1:
        if y != ym:
            raise ValueError("Invalid shape")
        nmodes = xm
        out = []
        for i in range(nmodes):
            m = mask[i,:]
            if np.any(m):
                mo = np.zeros_like(mask)
                mo[i,:] = m
                out.append(mo)
        return out
    
    elif x == xm and y == ym:
        return [mask]
    else:
        raise ValueError("Invalid shape")
   

def mode_masks(mask, shape = None):
    if isinstance(mask, tuple) or mask.ndim == 3:
        out = (mode_masks(m,shape) for m in mask)
        return tuple(out)
    
    if shape is None:
        shape = mask.shape
    x,y = shape
    xm, ym = mask.shape
    if shape == (1,1):
        #1d case, no masks
        return []
    
    elif y == 1:
        nmodes = ym
        if x != xm:
            raise ValueError("Invalid shape")

        group_id = np.empty(mask.shape, int)
        for i in range(nmodes):
            group_id[:,i] = i
            
    elif x == 1:
        nmodes = xm
        
        if y != ym:
            raise ValueError("Invalid shape")

        group_id = np.empty(mask.shape, int)
        for i in range(nmodes):
            group_id[i,:] = i
    else:
        #3d case
        if x !=xm or y != ym:
            raise ValueError("Invalid shape")
        #all modes are dependent, no need to mask it, so indicate it with None mask
        return [None]
    
    masked_group_id = group_id[mask]
    group_masks = []
    for i in range(nmodes):
        m = masked_group_id == i
        if np.any(m):
            group_masks.append(m)
            
    return np.asarray(group_masks)
        
def group_modes(modes, mode_masks,axis = -2):
    if isinstance(modes, tuple):
        return tuple(group_modes(m, mm) for m,mm in zip(modes, mode_masks))
    
    if len(mode_masks) == 0:
        #1d case, modes are all independent, so no grouping
        return modes             
    else:
        #2d case and 3d case     
        return [modes[...,m,:] if m is not None else modes for m in mode_masks]

def _group_vector(vector, mode_masks):
    if isinstance(vector, tuple):
        return tuple(_group_vector(m, mm) for m,mm in zip(vector, mode_masks))
    
    if len(mode_masks) == 0:
        #1d case, modes are all independent, so no grouping
        return vector            
    else:
        #2d case and 3d case     
        return [vector[m] if m is not None else vector for m in mode_masks]

def ungroup_modes(modes, mode_masks):
    if isinstance(modes, tuple):
        return tuple(ungroup_modes(m, mm) for m,mm in zip(modes, mode_masks))
    if len(mode_masks) == 0:
        #nothing to do
        return modes
    else:
        if len(mode_masks) == 1 and mode_masks[0] is None:
            return modes[0]
        n, count = mode_masks.shape
        out_shape = modes[0].shape[:-2] + (count,) +  modes[0].shape[-1:] 
        out = np.empty(out_shape, modes[0].dtype)
        for mode, m in zip(modes,mode_masks):
            out[...,m,:] = mode
        return out
        
def layer_mat3d(k0, d, epsv,epsa, mask = None, method = "4x4", resolution_power = 0):
    """Computes characteristic matrix of a single layer M=F.P.Fi,
    
    Numpy broadcasting rules apply.
    
    Parameters
    ----------
    k0 : float or sequence of floats
        A scalar or a vector of wavenumbers
    d : array_like
        Layer thickness
    epsv : array_like
        Epsilon eigenvalues.
    epsa : array_like
        Optical axes orientation angles (psi, theta, phi).
    method : str, optional
        Either '4x4' (default), '4x4_1', or '2x2'.
    
    Returns
    -------
    cmat : ndarray
        Characteristic matrix of the layer.
    """
    dim = material_dim(epsv, epsa)

    if method not in ("4x4","4x4_1","2x2"):
        raise ValueError("Unsupported method: '{}'".format(method))
    k0 = np.asarray(k0)
    shape = material_shape(epsv, epsa)
    
    if mask is None:    
        mask = eigenmask(shape, k0)
        betas = eigenbeta(shape, k0)
        phis = eigenphi(shape, k0)
        indices = eigenindices(shape, k0)
    else:
        betas = mask2beta(mask,k0)
        phis = mask2phi(mask,k0)
        indices = mask2indices(mask,k0)
        
    if dim == 3:
        if k0.ndim == 0:
            return _layer_mat3d(k0,d,epsv,epsa, mask, betas, phis,indices, method,resolution_power)
        else:
            out = (_layer_mat3d(k0[i],d,epsv,epsa, mask[i],betas[i], phis[i],indices[i],method,resolution_power) for i in range(len(k0)))
            return tuple(out)
        
    elif dim == 2:
        if shape[0] == 1:
            epsv2 = epsv[...,0,:,:]
            epsa2 = epsa[...,0,:,:]
            swap_axes = False
        else:
            epsv2 = epsv[...,:,0,:]
            epsa2 = epsa[...,:,0,:] 
            swap_axes = True
        if k0.ndim == 0:
            return [tmm2d.layer_mat2d(k0, d, epsv2, epsa2, betay = betay, method = method, mask = m, swap_axes = swap_axes,resolution_power = resolution_power) for (m,betay) in _iterate_masks_2d(mask,shape,k0)]
        else:
            out =  ([tmm2d.layer_mat2d(k0[i], d, epsv2, epsa2, betay = betay, method = method, mask = m, swap_axes = swap_axes,resolution_power = resolution_power) for (m,betay) in _iterate_masks_2d(mask[i],shape,k0[i])] for i in range(len(k0)))
            return tuple(out)
    else:
        epsv1 = epsv[...,0,0,:]
        epsa1 = epsa[...,0,0,:] 
        if k0.ndim == 0:
            return tmm.layer_mat(k0*d,epsv1,epsa1, betas, phis, method = method)
        else:
            out = (tmm.layer_mat(k0[i]*d,epsv1,epsa1, betas[i], phis[i], method = method) for i in range(len(k0)))
            return tuple(out)

def _layer_mat3d(k0,d,epsv,epsa, mask, betas, phis,indices, method, resolution_power):   
    n = len(betas)
    steps = 2**resolution_power
    kd = k0*d/steps
    shape = mask.shape[-2:]
    if method.startswith("2x2"):
        out = np.empty(shape = (n, n, 2, 2), dtype = CDTYPE)
    else:
        out = np.empty(shape = (n, n, 4, 4), dtype = CDTYPE)

    for j,(beta,phi) in enumerate(zip(betas,phis)):    
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

        wave = eigenwave(shape, indices[j,0],indices[j,1], amplitude = 1.)

        m = dotmdm(f,pmat,fi) 
        mw = m*wave[...,None,None]
                
        mf = mfft2(mw, overwrite_x = True)
        
        #dd = np.linspace(0,1.,10)*d
        
        # dmat = 0.
        
        # for dm in dd:
        
        #     dmat = dmat + second_field_diffraction_matrix(shape, -k0, beta, phi,dm, 
        #                                   epsv = (1.5,1.5,1.5),
        #                             epsa = (0.,0.,0.), betamax = 1.4) /len(dd)
            
        # mf = dotmm(dmat,mf)
        

        mf = mf[mask,...]
        
        out[:,j,:,:] = mf
        
    for i in range(resolution_power):
        out = bdotmm(out,out)
        
    return [out]

def stack_mat3d(k,d,epsv,epsa, mask = None, method = "4x4", resolution_power = 0):
    k = np.asarray(k)
    dim = material_dim(epsv, epsa)
    verbose_level = DTMMConfig.verbose
    
    def _iterate_wavelengths():
        for i in range(len(k)):
            if verbose_level > 0:    
                print("Wavelength {}/{}".format(i+1,len(k)))
            m = None if mask is None else mask[i]
            yield stack_mat3d(k[i],d,epsv,epsa, method = method, mask = m, resolution_power = resolution_power)
            
    if k.ndim == 1:
        return tuple(_iterate_wavelengths())

    n = len(d)
    
    try:
        resolution_power = [int(i) for i in resolution_power]
        if len(resolution_power) != n:
            raise ValueError("Length of the `resolution_power` argument must match length of `d`.")
    except TypeError:
        resolution_power = [int(resolution_power)] * n
    
    prefix = ""
    
    if verbose_level > 1:
        print ("Building stack matrix in {}d.".format(dim))
        if mask is not None:
            prefix = "N modes: {}".format(mask.sum())
            
    for j in range(n):
        print_progress(j,n, prefix = prefix) 
        mat = layer_mat3d(k,d[j],epsv[j],epsa[j], mask = mask, method = method, resolution_power = resolution_power[j])
        if isinstance(mat, list):
            #2d and 3d case
            if j == 0:
                out = [m.copy() for m in mat]
            else:
                if method.startswith("2x2"):
                    out = [bdotmm(m,o) for m,o in zip(mat,out)]
                else:
                    out = [bdotmm(o,m) for m,o in zip(mat,out)]
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

def dotmm3d(m1, m2):
    if isinstance(m1, tuple):
        return tuple((dotmm3d(_m1,_m2) for _m1, _m2 in zip(m1,m2)))
    if isinstance(m1, list):
        if isinstance(m2, list):
            return [bdotmm(a,b) for a,b in zip(m1,m2)]
        else:
            return [bdotmd(a,b) for a,b in zip(m1,[m2])]   
    else:
        if isinstance(m2, list):
            return [bdotdm(a,b) for a,b in zip([m1],m2)]
        else:
            return dotmm(m1,m2)
        
def bdotmv(a,b):
    shape = a.shape[0:-4] + (a.shape[-4] * b.shape[-1],a.shape[-4] * b.shape[-1])  
    a = np.moveaxis(a, -2,-3)
    a = a.reshape(shape)
    bv = b.reshape(b.shape[:-2] + (b.shape[-2]*b.shape[-1],))
    return dotmv(a,bv).reshape(b.shape)

def dotmv3d(m, v):
    if isinstance(m, tuple):
        return tuple((dotmv3d(_m,_v) for _m, _v in zip(m,v)))
    if isinstance(m, list):
        if isinstance(v, list):
            return [bdotmv(a,b) for a,b in zip(m,v)]
        else:
            raise ValueError("Matrix is a list of matrices, so vector should be a ist of vectors.")
    else:
        return dotmv(m,v)

def _upscale1(mat,mask):
    #total number of modes
    n = mask.sum()
    out_shape = mat.shape[0:-4] + (n,n) + mat.shape[-2:]
    out = np.zeros(out_shape, mat.dtype)
    masked_mat = mat[...,mask,:,:] 
    #fill diagonals
    for i in range(n):
        out[...,i,i,:,:] = masked_mat[...,i,:,:] 
    return out    

def upscale1(mat, mask):
    """Upscales 1d matrix to 2d matrix
    """
    if isinstance(mat, tuple):
        return tuple((upscale1(_mat, _mask) for _mat, _mask in zip(mat,mask)))
    if len(mask) in (0,1):
        #no need to do anything, we can stay in 1d
        return mat
    else:
        return [_upscale1(mat,_mask) for _mask in mask]
         
def upscale2(mat, mask):
    """Upscales 2d matrix to 3d matrix
    """
    if isinstance(mat, tuple):
         return tuple((upscale2(_mat, _mask) for _mat, _mask in zip(mat,mask)))
    if not isinstance(mat, list):
        raise ValueError("Input matrix must be a list of matrices")
        
    #total number of modes
    n = len(mask[0])
    #take first matrix, just to test shapes and dtypes
    m0 = mat[0]
    out_shape = m0.shape[0:-4] + (n,n) + m0.shape[-2:]
    # upscaled matrix 
    out = np.zeros(out_shape, m0.dtype)

    for _mask, _mat in zip(mask, mat):
        tmp_shape = _mat.shape[0:-4] + (n,) + _mat.shape[-3:]
        tmp = np.zeros(tmp_shape, m0.dtype)
        tmp[...,_mask,:,:,:] = _mat 
        out[...,_mask,:,:] = tmp
    #must return a list for 3d data
    return [out]
        
def fmat3d(fmat):
    """Converts a sequence of 4x4 matrices to a single large matrix"""
    fmat = np.asarray(fmat)
    shape = fmat.shape
    n = shape[-3]
    out_shape = shape[0:-3] + (n*4,n*4)
    out = np.zeros(out_shape, fmat.dtype)
    for i in range(n):
        out[...,i*4:(i+1)*4,i*4:(i+1)*4] = fmat[...,i,:,:]
    return out


def f_iso3d(mask, k0, n = 1., shape = None):
    k0 = np.asarray(k0)

    if k0.ndim == 0:
        x,y = mask.shape # to test if valid mask shape
        if shape is None:
            shape = mask.shape
        else:
            x,y = shape # to test if valid shape
        beta = mask2beta(mask, k0)
        phi = mask2phi(mask, k0)
        
        m = mode_masks(mask, shape)
        
        if len(m) != 0:
            beta = _group_vector(beta, m)
            phi = _group_vector(phi, m)  
            fmat = [tmm.f_iso(n = n, beta = b, phi = p) for b,p in zip(beta,phi)]
            return fmat
        else:
            return tmm.f_iso(n = n, beta = beta, phi = phi)
    else:
        out = (f_iso3d(m, k, n = n, shape = shape) for m, k in zip(mask, k0))
        return tuple(out)

def f3d(mask, k0, epsv = (1,1,1), epsa = (0,0,0), shape = None):
    k0 = np.asarray(k0)
    if shape is None:
        shape = mask.shape[-2:]
    beta = mask2beta(mask, k0)
    phi = mask2phi(mask, k0)
    if k0.ndim == 0:
        fmat = tmm.f(beta,phi,epsv,epsa)
        return fmat
    else:
        fmat = (tmm.f(beta[i],phi[i],epsv,epsa) for i in range(len(k0)))
        return tuple(fmat)    

#from dtmm.tmm2d import system_mat2d as system_mat3d
#from dtmm.tmm2d import reflection_mat2d as reflection_mat3d

def _system_mat3d(fmatin, cmat, fmatout):
    """Computes a system matrix from a characteristic matrix Fin-1.C.Fout"""
    if isinstance(cmat, list):
        if len(cmat) == len(fmatin) and len(cmat) == len(fmatout):
            out = []
            for fin,c,fout in zip(fmatin,cmat,fmatout):
                fini = inv(fin)
                o = bdotdm(fini,c)
                out.append(bdotmd(o,fout))  
            return out
        else:
            raise ValueError("Wrong input data lengths")
    else:
        return tmm.system_mat(cmat = cmat,fmatin = fmatin, fmatout = fmatout)      

def system_mat3d(cmat = None, fmatin = None, fmatout = None):
    """Computes a system matrix from a characteristic matrix Fin-1.C.Fout"""
    if cmat is None:
        raise ValueError("cmat is a required argument")
    if fmatin is None:
        raise ValueError("fmatin is a required argument.")
        
    if fmatout is None:
        fmatout = fmatin
    if isinstance(fmatin, tuple):
        if cmat is not None:
            out = (_system_mat3d(fi, c, fo) for fi,c,fo in zip(fmatin,cmat,fmatout))
        else:
            out = (_system_mat3d(fi, None, fo) for fi,fo in zip(fmatin,fmatout))
        return tuple(out)
    else:
        return _system_mat3d(fmatin, cmat, fmatout)

    
def _reflection_mat3d(smat):
    """Computes a 4x4 reflection matrix.
    """
    def iterate(smat):
        for mat in smat:
            shape = mat.shape[0:-4] + (mat.shape[-4] * 4,mat.shape[-4] * 4)  
            mat = np.moveaxis(mat, -2,-3)
            mat = mat.reshape(shape)
            yield tmm.reflection_mat(mat)
    
    if isinstance(smat, list):
        return list(iterate(smat))
    else:
        return tmm.reflection_mat(smat)

def reflection_mat3d(smat):
    verbose_level = DTMMConfig.verbose
    if verbose_level > 1:
        print ("Building reflectance and transmittance matrix")    
    if isinstance(smat, tuple):
        out = []
        n = len(smat)
        for i,s in enumerate(smat):
            print_progress(i,n) 
            out.append(_reflection_mat3d(s))
        print_progress(n,n)     
        return tuple(out)
    else:
        return _reflection_mat3d(smat)


def _transmission_mat3d(cmat):
    """Computes a 2x2 transmission matrix.
    """
    def iterate(cmat):
        for mat in cmat:
            shape = mat.shape[0:-4] + (mat.shape[-4] * 2,mat.shape[-4] * 2)  
            mat = np.moveaxis(mat, -2,-3)
            mat = mat.reshape(shape)
            yield mat
    
    if isinstance(cmat, list):
        return list(iterate(cmat))
    else:
        return cmat
    
def transmission_mat3d(cmat):
    verbose_level = DTMMConfig.verbose
    if verbose_level > 1:
        print ("Building transmittance matrix")    
    if isinstance(cmat, tuple):
        out = []
        n = len(cmat)
        for i,s in enumerate(cmat):
            print_progress(i,n) 
            out.append(_transmission_mat3d(s))
        print_progress(n,n)     
        return tuple(out)
    else:
        return _transmission_mat3d(cmat)    

def _reflect3d(fvec_in, fmat_in, rmat, fmat_out, fvec_out = None):
    """Reflects/Transmits field vector using 4x4 method.
    
    This functions takes a field vector that describes the input field and
    computes the output transmited field and also updates the input field 
    with the reflected waves.
    """
    if isinstance(rmat, list):
        if not isinstance(fvec_in, list):
            raise ValueError("Invalid `fvecin`. Reflection matrix is a list of matrices, so `fvecin` shold be a list of field vectors.")
        #2d and 3d case, we must iterate over modes
        if fvec_out is None:
            fvec_out = [None] * len(fvec_in)
        return [_reflect3d(*args) for args in zip(fvec_in,fmat_in,rmat,fmat_out,fvec_out)]
    
    dim = 1 if rmat.shape[-1] == 4 else 3

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

def reflect3d(fvecin, rmat, fmatin, fmatout, fvecout = None):
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
            return tuple((_reflect3d(fvecin[i], fmatin[i], rmat[i], fmatout[i]) for i in range(n)))
        else:
            return tuple((_reflect3d(fvecin[i], fmatin[i], rmat[i], fmatout[i], fvecout[i]) for i in range(n)))
    else:
        return _reflect3d(fvecin, fmatin, rmat, fmatout, fvecout)

def _transmit3d(fvec_in, fmat_in, mat, fmat_out, fvec_out = None):
    """Transmits field vector using 2x2 matrix
    """
    if isinstance(mat, list):
        #2d and 3d case, we must iterate over modes
        if fvec_out is None:
            fvec_out = [None] * len(fvec_in)
        return [_transmit3d(*args) for args in zip(fvec_in,fmat_in,mat,fmat_out,fvec_out)]
    
    dim = 1 if mat.shape[-1] == 4 else 3
    
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

def transmit3d(fvecin, tmat, fmatin,  fmatout, fvecout = None):
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
            return tuple((_transmit3d(fvecin[i], fmatin[i], tmat[i], fmatout[i]) for i in range(n)))
        else:
            return tuple((_transmit3d(fvecin[i], fmatin[i], tmat[i], fmatout[i], fvecout[i]) for i in range(n)))
    else:
        return _transmit3d(fvecin, fmatin, tmat, fmatout, fvecout)
    
def multiply_mat3d(matrices, mask = None, data_shapes = None, reverse = False):
    if data_shapes is not None:
        #heterogeneous data shapes. We must first convert matrices to a common shape 
        if mask is None:
            raise ValueError("For heterogeneous data, ypu must provide mask array.")
        common_shape = (1,1)
        for data_shape in data_shapes:
            common_shape = tuple((max(x,y) for (x,y) in zip(common_shape, data_shape))) 
        
        common_dim = shape2dim(common_shape)
        if common_dim == 2:
            #this is a constant, so compute it in advance
            mask2d = mode_masks(mask, common_shape)     
        
        mat0 = None
        for mat, data_shape in zip(matrices, data_shapes):
            data_dim = shape2dim(data_shape)
            if data_dim < common_dim:
                if data_dim == 1:
                    #case common_dim == 3 or 2 
                    # from 1d to 2d
                    if common_dim == 2:  
                        mat = upscale1(mat, mask2d)
                    # from 1d to 3d is not needed.
                else:
                    #case common_dim == 3 with data_dim == 2
                    #mask2d may be different every time, so update 
                    mask2d = mode_masks(mask, data_shape) 
                    mat = upscale2(mat, mask2d)  
            if mat0 is not None:
                if reverse == False:       
                    mat = dotmm3d(mat0, mat)
                else:
                    mat = dotmm3d(mat, mat0)   
            mat0 = mat
        return mat
    else:
        #homogeneous data shapes. Just multiply.
        mat0 = None
        for mat in matrices:
            if mat0 is not None:
                if reverse == False:       
                    mat = dotmm3d(mat0, mat)
                else:
                    mat = dotmm3d(mat, mat0)   
            mat0 = mat
        return mat               

