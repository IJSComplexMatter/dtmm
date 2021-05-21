"""
The 2x2 layer propagation functions
"""
from __future__ import absolute_import, print_function, division

from dtmm.conf import BETAMAX, field_has_vec_layout, field_shape
from dtmm.wave import eigenwave, betaphi
from dtmm.tmm import alphaf, E2H_mat, E_mat, Eti_mat, phase_mat, Etri_mat, tr_mat

from dtmm.linalg import  inv, dotmd, dotmv, dotmdmv, dotmv_vec
from dtmm.linalg import dotmdmf as _dotmdmf
from dtmm.linalg import dotmf as _dotmf
from dtmm.diffract import diffract, E_tr_matrix
from dtmm.fft import fft2, ifft2
import numpy as np

from dtmm.matrix import corrected_E_diffraction_matrix,second_E_diffraction_matrix,first_E_diffraction_matrix
from dtmm.mode import fft_mask


def dotmf(*args,**kwargs):
    if field_has_vec_layout():
        return dotmv_vec(*args,**kwargs)
    else:
        return _dotmf(*args,**kwargs)

def dotmdmf(*args,**kwargs):
    if field_has_vec_layout():
        return dotmdmv(*args,**kwargs)
    else:
        return _dotmdmf(*args,**kwargs)

def _transfer_ray_2x2_1(fft_field, wavenumbers, layer, effective_layer_in,effective_layer_out, dmat1, dmat2, beta = 0, phi=0,
                    nsteps = 1, mode = +1, reflection = True, betamax = BETAMAX, refl = None, bulk = None, out = None, tmpdata = None):
    _out = {} if tmpdata is None else tmpdata
    #fft_field = fft2(fft_field, out = out)
    shape = field_shape(fft_field)
    d_in, epsv_in,epsa_in = effective_layer_in     
    
    d_out, epsv_out,epsa_out = effective_layer_out    
    
    if reflection:
        tmat,rmat = E_tr_matrix(shape, wavenumbers, epsv_in = epsv_in, epsa_in = epsa_in,
                            epsv_out = epsv_out, epsa_out = epsa_out, mode = mode, betamax = betamax)
    
    d, epsv, epsa = layer
    alpha, fmat = alphaf(beta,phi, epsv, epsa, out = _out.get("alphaf"))
    
    e = E_mat(fmat, mode = mode, copy = False) #2x2 E-only view of fmat
    ei = inv(e, out = _out.get("ei"))
#    
    kd = wavenumbers * d   
    p = phase_mat(alpha,kd[...,None,None], mode = mode, out = _out.get("p"))
    
    if tmpdata is not None:
        _out["alphaf"] = alpha, fmat
        _out["ei"] = ei
        _out["p"] = p
    
    for j in range(nsteps):
        if j == 0 and reflection:
            #reflect only at the beginning
            if refl is not None:
                trans = refl.copy()
                refl = dotmf(rmat, fft_field, out = refl)
                fft_field = dotmf(tmat, fft_field, out = out)
                fft_field = np.add(fft_field,trans, out = fft_field)
                
                if mode == -1 and bulk is not None:
                    field = ifft2(fft_field)
                    e2h = E2H_mat(fmat, mode = mode)
                    bulk[...,::2,:,:] += field 
                    bulk[...,1::2,:,:] +=  dotmf(e2h, field, out = field)

                fft_field = dotmf(dmat1, fft_field, out = fft_field)
                out = fft_field
            else:
                fft_field = dotmf(tmat, fft_field, out = out)
                if dmat1 is not None:
                    fft_field = dotmf(dmat1, fft_field, out = fft_field)
                out = fft_field
        else:
            if dmat1 is not None:
                fft_field = dotmf(dmat1, fft_field, out = out)
            out = fft_field
        field = ifft2(fft_field, out = out)
        field = dotmdmf(e,p,ei,field, out = field)
        fft_field = fft2(field, out = field)
        if dmat2 is not None:
            fft_field = dotmf(dmat2, fft_field, out = fft_field)
    #return fft_field, refl  
    
    #out = ifft2(fft_field, out = out)
   
    if mode == +1 and bulk is not None:
        field = ifft2(fft_field)
        e2h = E2H_mat(fmat, mode = mode)
        if field_has_vec_layout():
            bulk[...,1::2] +=  dotmf(e2h, field)
            bulk[...,::2] += field            
        else:
            bulk[...,1::2,:,:] +=  dotmf(e2h, field)
            bulk[...,::2,:,:] += field
    
    return fft_field, refl



def _transfer_ray_2x2_2(field, wavenumbers, in_layer, out_layer, dmat = None, beta = 0, phi=0,
                    nsteps = 1, mode = +1,  reflection = True, betamax = BETAMAX, refl = None, bulk = None, out = None, tmpdata = None):
    _out = {} if tmpdata is None else tmpdata
    if in_layer is not None:
        d, epsv,epsa = in_layer    
        alpha, fmat_in = alphaf(beta,phi, epsv, epsa, out = _out.get("afin"))
        if tmpdata is not None:
            _out["afin"] = alpha, fmat_in
    d, epsv,epsa = out_layer        
    alpha, fmat = alphaf(beta,phi, epsv, epsa, out = _out.get("afout"))    
    e = E_mat(fmat, mode = mode, copy = False) #2x2 E-only view of fmat
#    if refl is not None:
#        ein = E_mat(fmat_in, mode = mode * (-1)) #reverse direction
#
#    if reflection == 0:
#        kd = wavenumbers * d
#    else:   
    kd = wavenumbers * d /2  
    p = phase_mat(alpha,kd[...,None,None], mode = mode, out = _out.get("p"))

    ei0 = inv(e, out = _out.get("ei0"))
    ei = ei0

    if tmpdata is not None:
        _out["afout"] = alpha, fmat
        _out["ei0"] = ei
        _out["p"] = p
    
    for j in range(nsteps):
        #reflect only at the beginning
        if j == 0 and reflection != 0:
            #if we need to track reflections (multipass)
            if refl is not None:
                #ei,eri = Etri_mat(fmat_in, fmat, mode = mode, out = _out.get("eieri"))
                tmat,rmat = tr_mat(fmat_in, fmat, mode = mode, out = _out.get("eieri"))
                if tmpdata is not None:
                    #_out["eieri"] = ei,eri
                    _out["eieri"] = tmat, rmat
                trans = refl.copy()
 
                #refl = dotmf(rmat, field, out = refl)
                refl = dotmf(tmat, field, out = refl)
                refl = np.subtract(field, refl, out = refl)

                field = dotmf(tmat,field, out = out)
                field = np.add(field,trans, out = field)
                
                
                if mode == -1 and bulk is not None:
                    #tmp_field = dotmf(e,field)
                    tmp_field = field
                    e2h = E2H_mat(fmat, mode = mode)
                    bulk[...,::2,:,:] += tmp_field 
                    bulk[...,1::2,:,:] +=  dotmf(e2h, tmp_field)
                
                field = dotmf(ei,field, out = field)
                
                if d != 0.:                
                    field = dotmf(dotmd(e,p),field, out = field)
                    
                else:
                    field = dotmf(e,field, out = field)
                out = field   
                
#                rmat = dotmm(ein, eri, out = eri)
#                trans = refl.copy()
#                refl = dotmf(rmat, field, out = refl)     
#                ei0 = inv(e)
#                field = dotmf(ei,field, out = out)
#                field = dotmf(e,field) + trans
#                if d != 0.:
#                    field = dotmdmf(e,p,ei0,field, out = out)
                    
                ei = ei0
            else:
                ei = Eti_mat(fmat_in, fmat, mode = mode, out = _out.get("eti"))
                field = dotmdmf(e,p,ei,field, out = out)  
                out = field
                if tmpdata is not None:
                    _out["eti"] = ei
                ei = ei0
        else:
            #no need to compute if d == 0!.. identity
            if d != 0.:
                field = dotmdmf(e,p,ei,field, out = out) 
                out = field
        if dmat is not None:   
            field = diffract(field, dmat, out = out)
            out = field
        #no need to compute if d == 0!.. identity
        if d != 0.:
            field = dotmdmf(e,p,ei,field, out = out) 
            
    if mode == +1 and bulk is not None:
        e2h = E2H_mat(fmat, mode = mode)
        if field_has_vec_layout():
            bulk[...,1::2] +=  dotmf(e2h, field)
            bulk[...,::2] += field            
        else:
            bulk[...,1::2,:,:] +=  dotmf(e2h, field)
            bulk[...,::2,:,:] += field
     

    return field, refl

        
def propagate_2x2_effective_1(field, wavenumbers, layer_in, layer_out, effective_layer_in, 
                            effective_layer_out, beta = 0, phi = 0,
                            nsteps = 1, diffraction = True, reflection = True, 
                            betamax = BETAMAX,  mode = +1,  tmpdata = None, split_diffraction = False,
                            refl = None, bulk = None, out = None):
    d_out, epsv_out,epsa_out = effective_layer_out    
    shape = field_shape(field)
    
    if diffraction <= 1:
        if diffraction == 1:
            dmat1 = first_E_diffraction_matrix(shape, wavenumbers, beta, phi,d_out/2, epsv = epsv_out, 
                                        epsa =  epsa_out, mode = mode, betamax = betamax) 
            dmat2 = second_E_diffraction_matrix(shape, wavenumbers, beta, phi,d_out/2, epsv = epsv_out, 
                                        epsa =  epsa_out, mode = mode, betamax = betamax) 
        else:
            dmat1, dmat2 = None,None
        return _transfer_ray_2x2_1(field, wavenumbers, layer_out, effective_layer_in, effective_layer_out,dmat1, dmat2,
                                beta = beta, phi = phi, nsteps =  nsteps,reflection = reflection,
                                betamax = betamax, mode = mode, refl = refl, bulk = bulk, out = out, tmpdata = tmpdata)            
    elif diffraction > 1:
        fout = np.zeros_like(field)
        reflpart = None
        fpart = None
        _out = None

        if refl is not None:
            _refl = np.zeros_like(refl)
        else:
            _refl = None

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
        
        if split_diffraction == False:

            dmats1 = first_E_diffraction_matrix(shape, wavenumbers, betas, phis,d_out/2, epsv = epsv_out, 
                                        epsa =  epsa_out, mode = mode, betamax = betamax) 
            dmats2 = second_E_diffraction_matrix(shape, wavenumbers, betas, phis,d_out/2, epsv = epsv_out, 
                                        epsa =  epsa_out, mode = mode, betamax = betamax) 
            
            idata = zip(windows, betas, phis, dmats1, dmats2)
        else:
            idata = zip(windows, betas, phis)
            
        for data  in idata:
            if split_diffraction == False:
                window, beta, phi, dmat1, dmat2 = data
            else:
                window, beta, phi = data
                dmat1 = first_E_diffraction_matrix(shape, wavenumbers, beta, phi,d_out/2, epsv = epsv_out, 
                                        epsa =  epsa_out, mode = mode, betamax = betamax) 
                dmat2 = second_E_diffraction_matrix(shape, wavenumbers, beta, phi,d_out/2, epsv = epsv_out, 
                                        epsa =  epsa_out, mode = mode, betamax = betamax) 
                        
            
            fpart = np.multiply(field, window, out = fpart)
            
            if refl is not None:
                reflpart = np.multiply(refl, window, out = reflpart)
            else:
                reflpart = None
            _out, __refl = _transfer_ray_2x2_1(fpart, wavenumbers,layer_out, 
                                                effective_layer_in, effective_layer_out,
                                                dmat1, dmat2,
                            beta = beta, phi = phi, 
                            nsteps =  nsteps,reflection = reflection,
                            betamax = betamax, mode = mode, bulk = bulk,
                            out = _out,  refl = reflpart, tmpdata = tmpdata)
             
            np.add(fout, _out, fout)
            if refl is not None and reflection != 0:
                np.add(_refl, __refl, out = _refl)


    if out is not None:
        out[...] = fout
    else:
        out = fout
    if refl is not None:
        refl[...] = _refl
    return out, refl   

def propagate_2x2_effective_2(field, wavenumbers, layer_in, layer_out, effective_layer_in, 
                            effective_layer_out, beta = 0, phi = 0,
                            nsteps = 1, diffraction = True, split_diffraction = False,
                            reflection = True, 
                            betamax = BETAMAX,  mode = +1, 
                            refl = None, bulk = None, out = None, tmpdata = None):
    
    shape = field.shape[-2:]
    d_eff, epsv_eff, epsa_eff = effective_layer_out
    

    
    if diffraction <= 1:
        if diffraction:
            dmat = corrected_E_diffraction_matrix(shape, wavenumbers, beta,phi, d = d_eff,
                                 epsv = epsv_eff, epsa = epsa_eff, mode = mode, betamax = betamax)
        else:
            dmat = None
        
        return _transfer_ray_2x2_2(field, wavenumbers, layer_in, layer_out, dmat = dmat,
                                beta = beta, phi = phi, nsteps =  nsteps,
                                reflection = reflection,
                                betamax = betamax, mode = mode, refl = refl, bulk = bulk, out = out, tmpdata = tmpdata)            
    else:
        fout = 0.
        fpart = None
        ffield = fft2(field)
        if refl is not None:
            frefl = fft2(refl)
            _refl = 0.
        else:
            _refl = None
        try: 
            broadcast_shape = beta.shape
            beta = beta[...,0]
            phi = phi[...,0]
        except IndexError:
            #beta and phi are scalar
            broadcast_shape = ()
            
        windows, (betas, phis) = fft_mask(field.shape, wavenumbers, int(diffraction), 
                 betax_off = beta*np.cos(phi), betay_off = beta*np.sin(phi), betamax = betamax)    

        n = len(windows)
        betas = betas.reshape((n,) + broadcast_shape)
        phis = phis.reshape((n,) + broadcast_shape)


        if split_diffraction == False:

            dmats = corrected_E_diffraction_matrix(shape, wavenumbers, betas,phis, d = d_eff,
                                     epsv = epsv_eff, epsa = epsa_eff, mode = mode, betamax = betamax)
            
            idata = zip(windows, betas, phis, dmats)
        else:
            idata = zip(windows, betas, phis)
            

        for data  in idata:
            if split_diffraction == False:
                window, beta, phi, dmat = data  
            else:
                window, beta, phi = data
                dmat = corrected_E_diffraction_matrix(shape, wavenumbers, beta,phi, d = d_eff,
                                     epsv = epsv_eff, epsa = epsa_eff, mode = mode, betamax = betamax)
                        
            fpart = np.multiply(ffield, window, out = fpart)
            fpart_re = ifft2(fpart, out = fpart)
            
            if refl is not None:
                reflpart = frefl* window
                reflpart_re = ifft2(reflpart)
            else:
                reflpart_re = None
            _out, __refl = _transfer_ray_2x2_2(fpart_re, wavenumbers, layer_in, layer_out, dmat = dmat,
                                beta = beta, phi = phi, nsteps =  nsteps,
                                betamax = betamax, reflection = reflection,
                                mode = mode, bulk = bulk,out = out,  refl = reflpart_re,tmpdata = tmpdata)                       
    
            fout += _out
            if refl is not None and reflection != 0:
                _refl += __refl

    if out is not None:
        out[...] = fout
    else:
        out = fout
    if refl is not None:
        refl[...] = _refl

    return out, refl
 
    
def propagate_2x2_full(field, wavenumbers, layer, input_layer = None, 
                    nsteps = 1,  mode = +1, reflection = True,
                    betamax = BETAMAX, refl = None, bulk = None, out = None):

    shape = field.shape[-2:]
   
    d, epsv, epsa = layer
    if input_layer is not None:
        d_in, epsv_in, epsa_in = input_layer

    kd = wavenumbers*d/nsteps
    
    if out is None:
        out = np.empty_like(field)
        
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
            
            if bulk is not None:
                obulk = bulk[...,i,:,:,:]
            
            if refl is not None:
                tampl = fft2(refl[...,i,:,:,:])[...,mask]
                orefl = refl[...,i,:,:,:]
                orefl[...] = 0.
              
            for bp in sorted(zip(range(len(betas)),betas,phis,iind,jind),key = lambda el : el[1], reverse = False):     
                #for j,bp in enumerate(zip(betas,phis,iind,jind)):     
                              
                j, beta, phi, ieig, jeig = bp

                out_af = alphaf(beta,phi,epsv,epsa)
                alpha,fmat_out = out_af 
                e = E_mat(fmat_out, mode = mode)
                ei0 = inv(e)
                ei = ei0                        
                pm = phase_mat(alpha,kd[i,None,None],mode = mode)
                w = eigenwave(amplitude.shape[:-1]+shape, ieig,jeig, amplitude = amplitude[...,j])
                if step == 0 and reflection != False:
                    alphain, fmat_in = alphaf(beta,phi,epsv_in,epsa_in)
                    if refl is not None:
                        ei,eri = Etri_mat(fmat_in, fmat_out, mode = mode)
                        ein =  E_mat(fmat_in, mode = -1*mode)
                        t = eigenwave(amplitude.shape[:-1]+shape, ieig,jeig, amplitude = tampl[...,j])
                        r = dotmf(eri, w)
                        r = dotmf(ein,r, out = r)
                        np.add(orefl,r,orefl)
                    
                        w = dotmf(ei, w, out = w)
                        t = dotmf(ei0,t, out = t)
                        w = np.add(t,w,out = w)
  
                    else:
                        ei = Eti_mat(fmat_in, fmat_out, mode = mode)
                        w = dotmf(ei, w, out = w)
                    w = dotmf(dotmd(e,pm),w, out = w)
                    np.add(ofield,w,ofield)                         
                    
                else:
                    w = dotmdmf(e,pm,ei,w, out = w)
                    np.add(ofield,w,ofield)

                if bulk is not None:
                    e2h = E2H_mat(fmat_out, mode = mode)
                    obulk[...,1::2,:,:] +=  dotmf(e2h, w)
                    obulk[...,::2,:,:] += w
                
            out[...,i,:,:,:] = ofield
                    
        field = out
    return out, refl

