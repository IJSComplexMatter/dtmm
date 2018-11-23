"""
Polarization manipulation functions.
"""
from __future__ import absolute_import, print_function, division

from dtmm.conf import cached_result, BETAMAX, FDTYPE, CDTYPE
from dtmm.wave import betaphi
from dtmm.tmm import polarizer4x4, alphaf
from dtmm.linalg import  dotmf
from dtmm.fft import fft2, ifft2
from dtmm.jones import jonesvec, polarizer
import numpy as np

from dtmm.diffract import diffraction_alphaf

@cached_result
def mode_polarizer(shape, ks, jones = (1,0), epsv = (1.,1.,1.), 
                            epsa = (0.,0.,0.), betamax = BETAMAX):
    """Returns a mode polarizer that should be applied in fft space. This is the 
    most general polarizer that does not introduce any reflections for any kind
    of nonhomogeneous or homogeneous field."""
    ks = np.asarray(ks, FDTYPE)
    ks = abs(ks)
    epsv = np.asarray(epsv, CDTYPE)
    epsa = np.asarray(epsa, FDTYPE)
    beta, phi = betaphi(shape,ks)
    alpha, f = diffraction_alphaf(shape, ks, epsv = epsv, 
                            epsa = epsa, betamax = betamax)

    beta, phi = betaphi(shape,ks)
    jones = jonesvec(jones, phi)
    pmat = polarizer4x4(jones, f)
    return pmat

@cached_result
def ray_polarizer(jones = (1,0), beta = 0, phi = 0, epsv = (1.,1.,1.), 
                            epsa = (0.,0.,0.),):
    """Returns a ray polarizer that should be applied in real space. Good for
    beams that can be approximated with a single wave vector and with a direction of 
    ray propagation beta and phi.
    
    See also mod_polarizer, which is for non-planewave field data."""
    epsv = np.asarray(epsv, CDTYPE)
    epsa = np.asarray(epsa, FDTYPE)
    beta = np.asarray(beta, FDTYPE)
    phi = np.asarray(phi, FDTYPE)
    
    alpha, f = alphaf(beta, phi, epsv, epsa)
    jones = jonesvec(jones, phi)
    pmat = polarizer4x4(jones, f)
    return pmat


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
    

def apply_mode_polarizer(pmat,field, out = None):
    """A convenience function. It applyies mode polarizer to field data."""
    fft = fft2(field, out = out)
    pfft = dotmf(pmat, fft ,out  = fft)
    return ifft2(fft, out = pfft)
 