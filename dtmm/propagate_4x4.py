"""
The 4x4 layer propagation functions
"""
from __future__ import absolute_import, print_function, division

from dtmm.conf import BETAMAX
from dtmm.wave import eigenwave, betaphi
from dtmm.tmm import alphaffi, phasem,  alphaf,  E_mat
from dtmm.linalg import dotmf, dotmdmf, inv
from dtmm.diffract import diffract
from dtmm.fft import fft2, ifft2
import numpy as np
from dtmm.mode import fft_mask
from dtmm.matrix import corrected_Epn_diffraction_matrix, corrected_field_diffraction_matrix, \
         first_corrected_Epn_diffraction_matrix, second_corrected_Epn_diffraction_matrix, \
         first_field_diffraction_matrix, second_field_diffraction_matrix

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

            dmatpn = corrected_Epn_diffraction_matrix(shape, wavenumbers, beta,phi, d = d_eff,
                                 epsv = epsv_eff, epsa = epsa_eff, betamax = betamax)
        
        else:
            dmatpn = None

        return _transfer_ray_4x4_2(field, wavenumbers, layer,
                                beta = beta, phi = phi, nsteps =  nsteps, dmatpn = dmatpn,
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
        
        dmatps, dmatns = corrected_Epn_diffraction_matrix(field.shape[-2:], wavenumbers, betas,phis, d = d_eff,
                                 epsv = epsv_eff, epsa = epsa_eff, betamax = betamax)


      
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
    elif diffraction > 1:
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
    else:
        raise ValueError("Invalid diffraction value")


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
    elif diffraction > 1:
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
