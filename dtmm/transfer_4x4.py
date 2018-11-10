"""
The 4x4 layer propagation functions
"""
from __future__ import absolute_import, print_function, division

from dtmm.conf import cached_function, BETAMAX,FDTYPE, cached_result, CDTYPE
from dtmm.wave import eigenwave, betaphi, betaxy
from dtmm.tmm import alphaffi, phasem,  alphaEEi, alphaf,  E_mat, phase_mat
from dtmm.linalg import dotmdm, dotmm, dotmf, dotmdmf, inv
from dtmm.diffract import diffraction_alphaffi, jones_diffraction_matrix, diffract, phase_matrix, diffraction_alphaf
from dtmm.field import fft_window
from dtmm.fft import fft2, ifft2
import numpy as np
from dtmm.diffract import diffraction_matrix as field_diffraction_matrix


E_diffraction_matrix = jones_diffraction_matrix

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
def E_correction_matrix(beta,phi,ks, d=1., epsv = (1,1,1), epsa = (0,0,0.), mode = +1, out = None):
    alpha, j, ji = alphaEEi(beta,phi,epsv,epsa, mode = mode)  
    kd = -np.asarray(ks)*d
    pmat = phase_matrix(alpha, kd)  
    return dotmdm(j,pmat,ji, out = out)

@cached_function
def corrected_E_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), mode = +1, betamax = BETAMAX, out = None):
    dmat = E_diffraction_matrix(shape, ks, d, epsv, epsa, mode = mode, betamax = betamax)
    cmat = E_correction_matrix(beta, phi, ks, d/2., epsv, epsa, mode = mode)
    return dotmm(cmat,dotmm(dmat,cmat, out = out), out = out)

@cached_function
def corrected_Epn_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
    ep = corrected_E_diffraction_matrix(shape, ks, beta,phi, d= d,
                                 epsv = epsv, epsa = epsa, mode = +1, betamax = betamax)
    en = corrected_E_diffraction_matrix(shape, ks, beta,phi, d= d,
                                 epsv = epsv, epsa = epsa, mode = -1, betamax = betamax)
    return ep, en
    
@cached_function
def corrected_field_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
 
    dmat = field_diffraction_matrix(shape, ks, d, epsv, epsa, betamax = betamax)
    cmat = correction_matrix(beta, phi, ks, d/2, epsv, epsa)
    dmat = dotmm(dmat,cmat, out = out)
    return dotmm(cmat,dmat, out = dmat) 

    
@cached_function
def second_E_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), mode = +1, betamax = BETAMAX, out = None):
    dmat = E_diffraction_matrix(shape, ks, d, epsv, epsa, mode = mode,betamax = betamax)
    cmat = E_correction_matrix(beta, phi, ks, d, epsv, epsa, mode = mode)
    return dotmm(dmat,cmat, out = None)

@cached_function
def first_E_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), mode = +1, betamax = BETAMAX, out = None):
    dmat = E_diffraction_matrix(shape, ks, d, epsv, epsa, mode = mode, betamax = betamax)
    cmat = E_correction_matrix(beta, phi, ks, d, epsv, epsa, mode = mode)
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
def first_Epn_diffraction_matrix(shape, ks,  d = 1., epsv = (1,1,1), epsa = (0,0,0.),  betamax = BETAMAX, out = None):
    ks = np.asarray(ks, dtype = FDTYPE)
    epsv = np.asarray(epsv, dtype = CDTYPE)
    epsa = np.asarray(epsa, dtype = FDTYPE)
    alpha, f, fi= diffraction_alphaffi(shape, ks, epsv = epsv, epsa = epsa, betamax = betamax)
    kd = ks * d
    e = E_mat(f, mode = None)
    pmat = phase_mat(alpha, kd[...,None,None])
    return dotmdm(e,pmat,fi,out = out) 

@cached_function
def second_Epn_diffraction_matrix(shape, ks,  d = 1., epsv = (1,1,1), epsa = (0,0,0.),  betamax = BETAMAX, out = None):
    ks = np.asarray(ks, dtype = FDTYPE)
    epsv = np.asarray(epsv, dtype = CDTYPE)
    epsa = np.asarray(epsa, dtype = FDTYPE)
    alpha, f = diffraction_alphaf(shape, ks, epsv = epsv, epsa = epsa, betamax = betamax)
    kd = ks * d
    e = E_mat(f, mode = None)
    ei = inv(e)
    pmat = phase_mat(alpha, kd[...,None,None])
    return dotmdm(f,pmat,ei,out = out) 

@cached_function
def Epn_correction_matrix(beta,phi,ks, d=1., epsv = (1,1,1), epsa = (0,0,0.), out = None):
    alpha, f = alphaf(beta,phi,epsv,epsa)  
    kd = -np.asarray(ks)*d
    pmat = phase_mat(alpha, kd[...,None,None]) 
    e = E_mat(f, mode = None)
    ei = inv(e)
    return dotmdm(e,pmat,ei, out = out)

@cached_function
def first_corrected_Epn_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
    dmat = first_Epn_diffraction_matrix(shape, ks, d, epsv, epsa, betamax = betamax)
    cmat = Epn_correction_matrix(beta, phi, ks, d, epsv, epsa)
    return dotmm(cmat,dmat, out = None)

@cached_function
def second_corrected_Epn_diffraction_matrix(shape, ks, beta,phi, d=1.,
                                 epsv = (1,1,1), epsa = (0,0,0.), betamax = BETAMAX, out = None):
    dmat = second_Epn_diffraction_matrix(shape, ks, d, epsv, epsa, betamax = betamax)
    cmat = Epn_correction_matrix(beta, phi, ks, d, epsv, epsa)
    return dotmm(dmat, cmat, out = None)


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


def _transfer_ray_4x4_2(field, wavenumbers, layer,  beta = 0, phi=0,
                    nsteps = 1, dmatpn = None,
                    out = None, tmpdata = None):
    _out = {} if tmpdata is None else tmpdata
        
    d, epsv, epsa = layer
    
    if dmatpn is None:
        kd = wavenumbers*d
    else:
        dmatp, dmatn = dmatpn
        kd = wavenumbers*d/2    

    alpha, f, fi = alphaffi(beta,phi,epsv,epsa, out = _out.get("affi"))

    p = phasem(alpha,kd[...,None,None], out = _out.get("p"))
    if tmpdata is not None:
        tmpdata["affi"] = (alpha, f, fi)
        tmpdata["p"] = p
    
    if dmatpn is not None:
        e = E_mat(f, mode = None)
        ei = inv(e, out = _out.get("ei"))
        if tmpdata is not None:
            tmpdata["ei"] = ei
          
    for j in range(nsteps):
        if dmatpn is None:
            field = dotmdmf(f,p,fi,field, out = out)  
        else:
            field = dotmdmf(e,p,fi,field, out = out) 
            diffract(field[...,0::2,:,:], dmatp, out = field[...,0::2,:,:])
            diffract(field[...,1::2,:,:], dmatn, out = field[...,1::2,:,:])
            field = dotmdmf(f,p,ei,field, out = field)              
         
    return field

def _transfer_ray_4x4_4(field, wavenumbers, layer,  beta = 0, phi=0,
                    nsteps = 1, dmat = None,
                    out = None):

    d, epsv, epsa = layer
    
    if dmat is None:
        kd = wavenumbers*d
    else:
        kd = wavenumbers*d/2    

    alpha, f, fi = alphaffi(beta,phi,epsv,epsa)
    p = phasem(alpha,kd[...,None,None])
  
    for j in range(nsteps):
        if dmat is None:
            field = dotmdmf(f,p,fi,field, out = out)  
        else:
            field = dotmdmf(f,p,fi,field, out = out) 
            field = diffract(field, dmat, out = field)
            field = dotmdmf(f,p,fi,field, out = field)              
         
    return field


def _transfer_ray_4x4_1(field, wavenumbers, layer, dmat1, dmat2, beta = 0, phi=0,
                    nsteps = 1, 
                    betamax = BETAMAX, out = None):
        
    d, epsv, epsa = layer

    kd = wavenumbers*d 

    alpha, f = alphaf(beta,phi,epsv,epsa)
    p = phasem(alpha,kd[...,None,None])
    
    e = E_mat(f, mode = None)
    ei = inv(e)

    for j in range(nsteps):
        field = dotmf(dmat1,field, out = out)
        field = ifft2(field, out = field)
        field = dotmdmf(e,p,ei,field, out = field)  
        field = fft2(field, out = field)
        field = dotmf(dmat2,field, out = out)
                  
    return field


def _transfer_ray_4x4_3(field, wavenumbers, layer, dmat1, dmat2, beta = 0, phi=0,
                    nsteps = 1, 
                    betamax = BETAMAX, out = None):
        
    d, epsv, epsa = layer

    kd = wavenumbers*d 

    alpha, f, fi = alphaffi(beta,phi,epsv,epsa)
    p = phasem(alpha,kd[...,None,None])

    for j in range(nsteps):
        field = dotmf(dmat1,field, out = out)
        field = ifft2(field, out = field)
        field = dotmdmf(f,p,fi,field, out = field)  
        field = fft2(field, out = field)
        field = dotmf(dmat2,field, out = out)
                  
    return field


def propagate_4x4_effective_2(field, wavenumbers, layer, effective_layer, beta = 0, phi=0,
                    nsteps = 1, diffraction = True, 
                    betamax = BETAMAX,out = None, tmpdata = None):
    
    d_eff, epsv_eff, epsa_eff = effective_layer
    shape = field.shape[-2:]
    if diffraction <= 1:
        
        if diffraction != 0:

            dmatp, dmatn = corrected_Epn_diffraction_matrix(shape, wavenumbers, beta,phi, d = d_eff,
                                 epsv = epsv_eff, epsa = epsa_eff, betamax = betamax)
        
        else:
            dmatp, dmatn = None, None

        return _transfer_ray_4x4_2(field, wavenumbers, layer,
                                beta = beta, phi = phi, nsteps =  nsteps, dmatpn = (dmatp,dmatn),
                                out = out, tmpdata = tmpdata)
    else:
        fout = np.zeros_like(field)
        _out = None
        field = fft2(field)

        try: 
            broadcast_shape = beta.shape
            beta = beta[...,0]
            phi = phi[...,0]
        except IndexError:
            broadcast_shape = ()
            
        windows, (betas, phis) = fft_mask(field.shape, wavenumbers, int(diffraction), 
                 betax_off = beta*np.cos(phi), betay_off = beta*np.sin(phi), betamax = betamax)    

        n = len(windows)
        betas = betas.reshape((n,) + broadcast_shape)
        phis = phis.reshape((n,) + broadcast_shape)
        
        if diffraction != 0:
            dmatps = corrected_E_diffraction_matrix(field.shape[-2:], wavenumbers, betas,phis, d = d_eff,
                                 epsv = epsv_eff, epsa = epsa_eff, mode = 1, betamax = betamax)
            dmatns = corrected_E_diffraction_matrix(field.shape[-2:], wavenumbers, betas,phis, d = d_eff,
                                 epsv = epsv_eff, epsa = epsa_eff, mode = -1, betamax = betamax)
        
        else:
            dmatps = [None]*n
            dmatns = [None]*n
      
        for window, b, p, dmatp, dmatn  in zip(windows, betas, phis, dmatps,dmatns):
            fpart = np.multiply(field, window, out = _out)
            fpart_re = ifft2(fpart, out = fpart)

            _out =  _transfer_ray_4x4_2(fpart_re, wavenumbers, layer, 
                                beta = b, phi = p, nsteps =  nsteps,
                                dmatpn = (dmatp,dmatn), tmpdata = tmpdata)                       
            fout = np.add(fout, _out, out = fout)

        
#        for window, b, p  in zip(windows, betas, phis):
#            fpart = field * window
#            fpart_re = ifft2(fpart)
#            
#            b = b.reshape(beta_shape)
#            p = p.reshape(beta_shape)
#            
#            if diffraction != 0:
#                dmat = corrected_field_diffraction_matrix(field.shape[-2:], wavenumbers, b,p, d=d_eff,
#                                     epsv = epsv_eff, epsa = epsa_eff)
#            else:
#                dmat = None
#
#            _out =  _transfer_ray_4x4_2(fpart_re, wavenumbers, layer, 
#                                beta = b, phi = p, nsteps =  nsteps,
#                                dmat = dmat,_reuse = _reuse)                       
#            fout += _out

        if out is not None:
            out[...] = fout
        else:
            out = fout
        return out

def propagate_4x4_effective_4(field, wavenumbers, layer, effective_layer, beta = 0, phi=0,
                    nsteps = 1, diffraction = True, 
                    betamax = BETAMAX,out = None):
    
    d_eff, epsv_eff, epsa_eff = effective_layer
    
    if diffraction <= 1:
        
        if diffraction != 0:
            dmat = corrected_field_diffraction_matrix(field.shape[-2:], wavenumbers, beta,phi, d=d_eff,
                                 epsv = epsv_eff, epsa = epsa_eff, betamax = betamax)

        else:
            dmat = None
        
        return _transfer_ray_4x4_4(field, wavenumbers, layer,
                                beta = beta, phi = phi, nsteps =  nsteps, dmat = dmat,
                                out = out)
    else:
        fout = np.zeros_like(field)
        _out = None
        field = fft2(field)

        try: 
            broadcast_shape = beta.shape
            beta = beta[...,0]
            phi = phi[...,0]
        except IndexError:
            broadcast_shape = ()
            
        windows, (betas, phis) = fft_mask(field.shape, wavenumbers, int(diffraction), 
                 betax_off = beta*np.cos(phi), betay_off = beta*np.sin(phi), betamax = betamax)    

        n = len(windows)
        betas = betas.reshape((n,) + broadcast_shape)
        phis = phis.reshape((n,) + broadcast_shape)
        
        if diffraction != 0:
            dmats = corrected_field_diffraction_matrix(field.shape[-2:], wavenumbers, betas,phis, d=d_eff,
                                 epsv = epsv_eff, epsa = epsa_eff)
        else:
            dmats = [None]*n
      
        for window, b, p, dmat  in zip(windows, betas, phis, dmats):
            fpart = np.multiply(field, window, out = _out)
            fpart_re = ifft2(fpart, out = fpart)

            _out =  _transfer_ray_4x4_4(fpart_re, wavenumbers, layer, 
                                beta = b, phi = p, nsteps =  nsteps,
                                dmat = dmat)                       
            fout = np.add(fout, _out, out = fout)

        
#        for window, b, p  in zip(windows, betas, phis):
#            fpart = field * window
#            fpart_re = ifft2(fpart)
#            
#            b = b.reshape(beta_shape)
#            p = p.reshape(beta_shape)
#            
#            if diffraction != 0:
#                dmat = corrected_field_diffraction_matrix(field.shape[-2:], wavenumbers, b,p, d=d_eff,
#                                     epsv = epsv_eff, epsa = epsa_eff)
#            else:
#                dmat = None
#
#            _out =  _transfer_ray_4x4_2(fpart_re, wavenumbers, layer, 
#                                beta = b, phi = p, nsteps =  nsteps,
#                                dmat = dmat,_reuse = _reuse)                       
#            fout += _out

        if out is not None:
            out[...] = fout
        else:
            out = fout
        return out

def propagate_4x4_effective_1(field, wavenumbers, layer, effective_layer, beta = 0, phi=0,
                    nsteps = 1, diffraction = True, 
                    betamax = BETAMAX,out = None,_reuse = False ):
    d_eff, epsv_eff, epsa_eff = effective_layer
    
    if diffraction == 1:
        dmat1 = first_corrected_Epn_diffraction_matrix(field.shape[-2:], wavenumbers, beta, phi,d_eff/2, epsv = epsv_eff, 
                                        epsa =  epsa_eff,betamax = betamax) 
        dmat2 = second_corrected_Epn_diffraction_matrix(field.shape[-2:], wavenumbers, beta, phi,d_eff/2, epsv = epsv_eff, 
                                        epsa =  epsa_eff,betamax = betamax) 
        return _transfer_ray_4x4_1(field, wavenumbers, layer,dmat1, dmat2, 
                                beta = beta, phi = phi, nsteps =  nsteps, 
                                betamax = betamax,  out = out)
    else:
        fout = np.zeros_like(field)
        _out = None

        try: 
            broadcast_shape = beta.shape
            beta = beta[...,0]
            phi = phi[...,0]
        except IndexError:
            broadcast_shape = ()
            
        windows, (betas, phis) = fft_mask(field.shape, wavenumbers, int(diffraction), 
                 betax_off = beta*np.cos(phi), betay_off = beta*np.sin(phi), betamax = betamax)    

        n = len(windows)
        betas = betas.reshape((n,) + broadcast_shape)
        phis = phis.reshape((n,) + broadcast_shape)

        dmats1 = first_corrected_Epn_diffraction_matrix(field.shape[-2:], wavenumbers, betas, phis,d_eff/2, epsv = epsv_eff, 
                                        epsa =  epsa_eff,betamax = betamax) 
        dmats2 = second_corrected_Epn_diffraction_matrix(field.shape[-2:], wavenumbers, betas, phis,d_eff/2, epsv = epsv_eff, 
                                        epsa =  epsa_eff,betamax = betamax) 

        for window, beta, phi, dmat1, dmat2  in zip(windows, betas, phis, dmats1, dmats2):
            fpart = np.multiply(field, window, out = _out)
            
            _out =  _transfer_ray_4x4_1(fpart, wavenumbers, layer, dmat1,dmat2,
                                beta = beta, phi = phi, nsteps =  nsteps,
                                betamax = betamax, out = _out)                       
            fout = np.add(fout, _out, out = fout)


#        for window, b, p  in zip(windows, betas, phis):
#            fpart = np.multiply(field, window, out = _out)
#            
#            beta = b.reshape(broadcast_shape)
#            phi = p.reshape(broadcast_shape)
#
#            dmat1 = second_field_diffraction_matrix(field.shape[-2:], wavenumbers, beta, phi,d_eff/2, epsv = epsv_eff, 
#                                            epsa =  epsa_eff,betamax = betamax) 
#            dmat2 = first_field_diffraction_matrix(field.shape[-2:], wavenumbers, beta, phi,d_eff/2, epsv = epsv_eff, 
#                                            epsa =  epsa_eff,betamax = betamax) 
#
#            _out =  _transfer_ray_4x4_1(fpart, wavenumbers, layer, dmat1,dmat2,
#                                beta = beta, phi = phi, nsteps =  nsteps,
#                                betamax = betamax, out = _out,_reuse = _reuse)                       
#            fout = np.add(fout, _out, out = fout)

        if out is not None:
            out[...] = fout
        else:
            out = fout
        return out


def propagate_4x4_effective_3(field, wavenumbers, layer, effective_layer, beta = 0, phi=0,
                    nsteps = 1, diffraction = True, 
                    betamax = BETAMAX,out = None):
    d_eff, epsv_eff, epsa_eff = effective_layer
    
    if diffraction == 1:
        dmat1 = second_field_diffraction_matrix(field.shape[-2:], wavenumbers, beta, phi,d_eff/2, epsv = epsv_eff, 
                                        epsa =  epsa_eff,betamax = betamax) 
        dmat2 = first_field_diffraction_matrix(field.shape[-2:], wavenumbers, beta, phi,d_eff/2, epsv = epsv_eff, 
                                        epsa =  epsa_eff,betamax = betamax) 
        return _transfer_ray_4x4_3(field, wavenumbers, layer,dmat1, dmat2, 
                                beta = beta, phi = phi, nsteps =  nsteps, 
                                betamax = betamax,  out = out)
    else:
        fout = np.zeros_like(field)
        _out = None

        try: 
            broadcast_shape = beta.shape
            beta = beta[...,0]
            phi = phi[...,0]
        except IndexError:
            broadcast_shape = ()
            
        windows, (betas, phis) = fft_mask(field.shape, wavenumbers, int(diffraction), 
                 betax_off = beta*np.cos(phi), betay_off = beta*np.sin(phi), betamax = betamax)    

        n = len(windows)
        betas = betas.reshape((n,) + broadcast_shape)
        phis = phis.reshape((n,) + broadcast_shape)

        dmats1 = second_field_diffraction_matrix(field.shape[-2:], wavenumbers, betas, phis,d_eff/2, epsv = epsv_eff, 
                                        epsa =  epsa_eff,betamax = betamax) 
        dmats2 = first_field_diffraction_matrix(field.shape[-2:], wavenumbers, betas, phis,d_eff/2, epsv = epsv_eff, 
                                        epsa =  epsa_eff,betamax = betamax) 

        for window, beta, phi, dmat1, dmat2  in zip(windows, betas, phis, dmats1, dmats2):
            fpart = np.multiply(field, window, out = _out)
            
            _out =  _transfer_ray_4x4_3(fpart, wavenumbers, layer, dmat1,dmat2,
                                beta = beta, phi = phi, nsteps =  nsteps,
                                betamax = betamax, out = _out)                       
            fout = np.add(fout, _out, out = fout)


#        for window, b, p  in zip(windows, betas, phis):
#            fpart = np.multiply(field, window, out = _out)
#            
#            beta = b.reshape(broadcast_shape)
#            phi = p.reshape(broadcast_shape)
#
#            dmat1 = second_field_diffraction_matrix(field.shape[-2:], wavenumbers, beta, phi,d_eff/2, epsv = epsv_eff, 
#                                            epsa =  epsa_eff,betamax = betamax) 
#            dmat2 = first_field_diffraction_matrix(field.shape[-2:], wavenumbers, beta, phi,d_eff/2, epsv = epsv_eff, 
#                                            epsa =  epsa_eff,betamax = betamax) 
#
#            _out =  _transfer_ray_4x4_1(fpart, wavenumbers, layer, dmat1,dmat2,
#                                beta = beta, phi = phi, nsteps =  nsteps,
#                                betamax = betamax, out = _out,_reuse = _reuse)                       
#            fout = np.add(fout, _out, out = fout)

        if out is not None:
            out[...] = fout
        else:
            out = fout
        return out

 
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
