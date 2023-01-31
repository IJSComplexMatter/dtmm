#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Light source creation functions and objects
"""

from dtmm import tmm
from dtmm.conf import DTMMConfig, get_default_config_option, BETAMAX
import numpy as np


def planewaves(shape, wavelength, pixelsize = 1., beta = 0., phi = 0., amplitude = 1.,eigenmodes = None)

def beam(shape, wavelengths, pixelsize = 1., beta = 0., phi = 0., amplitude = 1.,
                      n = None, focus = 0., window = None, mode = +1, 
                      jones = None, diffraction = True, eigenmodes = None, compression = 1., 
                      betamax = BETAMAX):
    verbose_level = DTMMConfig.verbose
    if verbose_level > 0:
        print("Building illumination data.") 
        
    n = get_default_config_option("n_cover",n)
    
    fmat = tmm.f_iso(beta, phi, n)
    m = tmm.fvec(fmat, jones = jones, amplitude = amplitude, mode = mode)
    
    
    f = np.moveaxis(m,-1,0)
    f = f[None,...]

    

    
    