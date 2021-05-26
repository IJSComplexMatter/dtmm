"""
4x4 transfer matrix method functions for 2d data
"""

from __future__ import absolute_import, print_function, division
import numpy as np

from dtmm.conf import  BETAMAX, CDTYPE, DTMMConfig
from dtmm.linalg import dotmdm, inv, dotmv, bdotmm, bdotmd, bdotdm
from dtmm.print_tools import print_progress

import dtmm.tmm as tmm
from dtmm.tmm import alphaffi, phase_mat

from dtmm.wave import eigenbetax1, eigenindices1, eigenmask1, eigenwave1, betaxy2beta, mask2betax1, mask2indices1,betaxy2phi
from dtmm.wave import k0 as wavenumber
from dtmm.field import field2modes1, modes2field1
from dtmm.fft import mfft

def layer_mat2d(k0, d, epsv,epsa, betay = 0., method = "4x4", mask = None):
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
        Either a 4x4 or 4x4_1
    
    Returns
    -------
    cmat : ndarray
        Characteristic matrix of the layer.
    """
    if method not in ("4x4","4x4_1"):
        raise ValueError("Unsupported method: '{}'".format(method))
    k0 = np.asarray(k0)
    shape = epsv.shape[-2]
    if mask is None:
        mask = eigenmask1(shape, k0)
        betas = eigenbetax1(shape, k0)
        indices = eigenindices1(shape, k0)
    else:
        betas = mask2betax1(mask,k0)
        indices = mask2indices1(mask)
    if k0.ndim == 0:
        return _layer_mat2d(k0,d,epsv,epsa, mask, betas,betay, indices, method)
    else:
        out = (_layer_mat2d(k0[i],d,epsv,epsa, mask[i],betas[i], betay, indices[i],method) for i in range(len(k0)))
        return tuple(out)


def _layer_mat2d(k0,d,epsv,epsa, mask, betaxs, betay,indices, method):   
    n = len(betaxs)
    kd = k0*d
    shape = epsv.shape[-2]
    out = np.empty(shape = (n, n, 4, 4), dtype = CDTYPE)
    beta = betaxy2beta(betaxs,betay)
    phi = betaxy2phi(betaxs,betay)
    for j,(beta,phi) in enumerate(zip(beta, phi)):   
        alpha,f,fi = alphaffi(beta,phi,epsv,epsa)
        pmat = phase_mat(alpha,-kd)
        if method == "4x4_1":
            pmat[...,1::2] = 0.
        elif method != "4x4":
            raise ValueError("Unsupported method!")

        wave = eigenwave1(shape, indices[j], amplitude = 1.)
        
        #m is shape (...,4,4)
        m = dotmdm(f,pmat,fi) 
        
        #wave is shape (...) make it broadcastable to (...,4,4)

        mw = m*wave[...,None,None]
        mf = mfft(mw, overwrite_x = True)
        mf = mf[mask,...]
        
        out[:,j,:,:] = mf

        #for i,mfj in enumerate(mf):
        #    out[i,j,:,:] = mfj

    return out

def stack_mat2d(k,d,epsv,epsa, betay = 0., method = "4x4" ,mask = None):
    n = len(d)
    indices = range(n)
    if method.startswith("2x2"):
        indices = reversed(indices)
    verbose_level = DTMMConfig.verbose
    if verbose_level > 1:
        print ("Building stack matrix.")
    for i in range(n):
        print_progress(i,n) 
        mat = layer_mat2d(k,d[i],epsv[i],epsa[i], betay = betay, mask = mask, method = method)

        if i == 0:
            if isinstance(mat, tuple):
                out = tuple((m.copy() for m in mat))
            else:
                out = mat.copy()
        else:
            if isinstance(mat, tuple):
                out = tuple((bdotmm(o,m) for o,m in zip(out,mat)))
            else:
                out = bdotmm(out,mat)

    print_progress(n,n) 

    return out 

def f_iso2d(shape, k0, n = 1., betay = 0, betamax = BETAMAX):
    k0 = np.asarray(k0)
    betax = eigenbetax1(shape,k0, betamax)
    
    if k0.ndim == 0:
        beta = betaxy2beta(betax,betay)
        phi = betaxy2phi(betax,betay)
        fmat = tmm.f_iso(n = n, beta = beta, phi = phi)
        return fmat
    else:
        beta = tuple((betaxy2beta(beta,betay) for beta in betax))
        phi = tuple((betaxy2phi(beta,betay) for beta in betax))
        fmat = (tmm.f_iso(n = n, beta = beta[i], phi = phi[i]) for i in range(len(k0)))
        return tuple(fmat)
    
def fi_iso2d(shape, k0, n = 1., betay = 0, betamax = BETAMAX):
    fmat = f_iso2d(shape, k0, n, betay, betamax)
    if isinstance(fmat, tuple):
        out = (inv(f) for f in fmat)
        return tuple(out)
    else:
        return inv(fmat)

def _system_mat2d(fmatin, cmat, fmatout):
    """Computes a system matrix from a characteristic matrix Fin-1.C.Fout"""
    fmatini = inv(fmatin)
    out = bdotdm(fmatini,cmat)
    return bdotmd(out,fmatout)  

def system_mat2d(fmatin, cmat, fmatout):
    """Computes a system matrix from a characteristic matrix Fin-1.C.Fout"""
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
    shape = smat.shape[0:-4] + (smat.shape[-4] * 4,smat.shape[-4] * 4)  
    smat = np.moveaxis(smat, -2,-3)
    smat = smat.reshape(shape)
    return tmm.reflection_mat(smat)

def reflection_mat2d(smat):
    verbose_level = DTMMConfig.verbose
    if verbose_level > 1:
        print ("Building reflectance and transmittance matrix.")    
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
    """Transmits/reflects field vector using 4x4 method.
    
    This functions takes a field vector that describes the input field and
    computes the output transmited field and also updates the input field 
    with the reflected waves.
    """
    
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

    av = a.reshape(a.shape[:-2] + (a.shape[-2]*a.shape[-1],))
    out = dotmv(rmat,av).reshape(a.shape)
    
    avec[...,1::2] = out[...,1::2]
    bvec[...,::2] = out[...,::2]
        
    dotmv(fmat_in,avec,out = fvec_in)    
    return dotmv(fmat_out,bvec,out = out)

def reflect2d(fvecin, fmatin, rmat, fmatout, fvecout = None):
    """Transmits/reflects field vector using 4x4 method.
    
    This functions takes a field vector that describes the input field and
    computes the output transmited field vector and also updates the input field 
    with the reflected waves.
    """
    verbose_level = DTMMConfig.verbose
    if verbose_level > 2:
        print ("Transmitting field.")   
    if isinstance(fvecin, tuple):
        n = len(fvecin)
        if fvecout is None:
            return tuple((_reflect2d(fvecin[i], fmatin[i], rmat[i], fmatout[i]) for i in range(n)))
        else:
            return tuple((_reflect2d(fvecin[i], fmatin[i], rmat[i], fmatout[i], fvecout[i]) for i in range(n)))
    else:
        return _reflect2d(fvecin, fmatin, rmat, fmatout, fvecout)

def transfer2d(field_data_in, optical_data, betay = 0., nin = 1., nout = 1., method = "4x4", betamax = BETAMAX, field_out = None):
    
    f,w,p = field_data_in
    shape = f.shape[-1]
    d,epsv,epsa = optical_data
    k0 = wavenumber(w, p)
    
    if field_out is not None:
        mask, fmode_out = field2modes1(field_out,k0, betamax = betamax)
    else:
        fmode_out = None
    
    mask, fmode_in = field2modes1(f,k0, betamax = betamax)
    
    fmatin = f_iso2d(shape = shape, betay = betay, k0 = k0, n=nin, betamax = betamax)
    fmatout = f_iso2d(shape = shape, betay = betay, k0 = k0, n=nout, betamax = betamax)
    
    cmat = stack_mat2d(k0,d, epsv, epsa, betay = betay, mask = mask, method = method)
    smat = system_mat2d(fmatin = fmatin, cmat = cmat, fmatout = fmatout)
    rmat = reflection_mat2d(smat)
    
    fmode_out = reflect2d(fmode_in, rmat = rmat, fmatin = fmatin, fmatout = fmatout, fvecout = fmode_out)
    
    field_out = modes2field1(mask, fmode_out)
    f[...] = modes2field1(mask, fmode_in)
    
    return field_out,w,p

        
    
