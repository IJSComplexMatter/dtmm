"""
4x4 and 2x2 transfer matrix method functions for 3d calculation. 
"""

from __future__ import absolute_import, print_function, division

import numpy as np

from dtmm.conf import CDTYPE,DTMMConfig, BETAMAX

from dtmm.linalg import dotmm, inv, dotmv,  bdotmm, bdotmd, bdotdm, dotmdm
from dtmm.print_tools import print_progress

import dtmm.tmm as tmm
from dtmm.tmm import alphaf, alphaffi, phase_mat
from dtmm.wave import eigenbeta, eigenphi,eigenindices, eigenmask, eigenwave,mask2beta, mask2phi, mask2indices
from dtmm.wave import k0 as wavenumber
from dtmm.field import field2modes, modes2field

from dtmm.fft import mfft2


# from dtmm.matrix import corrected_field_diffraction_matrix, second_field_diffraction_matrix,\
#                             first_field_diffraction_matrix, first_corrected_Epn_diffraction_matrix, \
#                             second_corrected_Epn_diffraction_matrix,\
#                             corrected_Epn_diffraction_matrix2


def layer_mat3d(k0, d, epsv,epsa, mask = None, method = "4x4"):
    """Computes characteristic matrix of a single layer M=F.P.Fi,
    
    Numpy broadcasting rules apply
    
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
        Either 2x2, 4x4 or 4x4_1
    
    Returns
    -------
    cmat : ndarray
        Characteristic matrix of the layer.
    """
    if method not in ("4x4","4x4_1","2x2"):
        raise ValueError("Unsupported method: '{}'".format(method))
    k0 = np.asarray(k0)
    shape = epsv.shape[-3],epsv.shape[-2]
    if mask is None:
        mask = eigenmask(shape, k0)
        betas = eigenbeta(shape, k0)
        phis = eigenphi(shape, k0)
        indices = eigenindices(shape, k0)
    else:
        betas = mask2beta(mask,k0)
        phis = mask2phi(mask,k0)
        indices = mask2indices(mask,k0)
    if k0.ndim == 0:
        return _layer_mat3d(k0,d,epsv,epsa, mask, betas, phis,indices, method)
    else:
        out = (_layer_mat3d(k0[i],d,epsv,epsa, mask[i],betas[i], phis[i],indices[i],method) for i in range(len(k0)))
        return tuple(out)

def _layer_mat3d(k0,d,epsv,epsa, mask, betas, phis,indices, method):   
    n = len(betas)
    kd = k0*d#/2.
    shape = epsv.shape[-3],epsv.shape[-2]
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
            if method != "4x4":
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


    return out

def stack_mat3d(k,d,epsv,epsa, method = "4x4" ,mask = None):
    n = len(d)
    verbose_level = DTMMConfig.verbose
    if verbose_level > 1:
        print ("Building stack matrix.")
    for i in range(n):
        print_progress(i,n) 
        mat = layer_mat3d(k,d[i],epsv[i],epsa[i], mask = mask, method = method)
        if i == 0:
            if isinstance(mat, tuple):
                out = tuple((m.copy() for m in mat))
            else:
                out = mat.copy()
        else:
            if isinstance(mat, tuple):
                if method.startswith("2x2"):
                    out = tuple((bdotmm(m,o) for o,m in zip(out,mat)))
                else:
                    out = tuple((bdotmm(o,m) for o,m in zip(out,mat)))
            else:
                if method.startswith("2x2"):
                    out = dotmm(mat,out)
                else:
                    out = dotmm(out,mat)
      
    print_progress(n,n) 

    return out 

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


def f_iso3d(shape, k0, n = 1., betamax = BETAMAX):
    k0 = np.asarray(k0)
    beta = eigenbeta(shape,k0, betamax)
    phi = eigenphi(shape,k0, betamax)
    if k0.ndim == 0:
        fmat = tmm.f_iso(n = n, beta = beta, phi = phi)
        return fmat
    else:
        fmat = (tmm.f_iso(n = n, beta = beta[i], phi = phi[i]) for i in range(len(k0)))
        return tuple(fmat)
    
def fi_iso3d(shape, k0, n = 1., betamax = BETAMAX):
    fmat = f_iso3d(shape, k0, n, betamax)
    if isinstance(fmat, tuple):
        out = (inv(f) for f in fmat)
        return tuple(out)
    else:
        return inv(fmat)

def _system_mat3d(fmatin, cmat, fmatout):
    """Computes a system matrix from a characteristic matrix Fin-1.C.Fout"""
    fmatini = inv(fmatin)
    out = bdotdm(fmatini,cmat)
    return bdotmd(out,fmatout)  

def system_mat3d(fmatin, cmat, fmatout):
    """Computes a system matrix from a characteristic matrix Fin-1.C.Fout"""
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
    shape = smat.shape[0:-4] + (smat.shape[-4] * 4,smat.shape[-4] * 4)  
    smat = np.moveaxis(smat, -2,-3)
    smat = smat.reshape(shape)
    return tmm.reflection_mat(smat)

def reflection_mat3d(smat):
    verbose_level = DTMMConfig.verbose
    if verbose_level > 1:
        print ("Building reflectance and transmittance matrix.")    
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


def _reflect3d(fvec_in, fmat_in, rmat, fmat_out, fvec_out = None):
    """Reflects/Transmits field vector using 4x4 method.
    
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


def reflect3d(fvecin, fmatin, rmat, fmatout, fvecout = None):
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
            return tuple((_reflect3d(fvecin[i], fmatin[i], rmat[i], fmatout[i]) for i in range(n)))
        else:
            return tuple((_reflect3d(fvecin[i], fmatin[i], rmat[i], fmatout[i], fvecout[i]) for i in range(n)))
    else:
        return _reflect3d(fvecin, fmatin, rmat, fmatout, fvecout)
    

def transfer3d(field_data_in, optical_data, nin = 1., nout = 1., method = "4x4", betamax = BETAMAX, field_out = None):
    
    f,w,p = field_data_in
    shape = f.shape[-2:]
    if isinstance(optical_data, list):
        raise ValueError("Heterogeneous optical data not supported. You must provide an optical data tuple.")
    d,epsv,epsa = optical_data
    k0 = wavenumber(w, p)
    
    if field_out is not None:
        mask, fmode_out = field2modes(field_out,k0, betamax = betamax)
    else:
        fmode_out = None
   
    mask, fmode_in = field2modes(f,k0, betamax = betamax)
    
    fmatin = f_iso3d(shape = shape, k0 = k0, n=nin, betamax = betamax)
    fmatout = f_iso3d(shape = shape, k0 = k0, n=nout, betamax = betamax)
    
    cmat = stack_mat3d(k0,d, epsv, epsa, mask = mask, method = method)
    smat = system_mat3d(fmatin = fmatin, cmat = cmat, fmatout = fmatout)
    rmat = reflection_mat3d(smat)
    
    fmode_out = reflect3d(fmode_in, rmat = rmat, fmatin = fmatin, fmatout = fmatout, fvecout = fmode_out)
    
    field_out = modes2field(mask, fmode_out)
    f = modes2field(mask, fmode_in, out = f)
    return field_out,w,p
    
