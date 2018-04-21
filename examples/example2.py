#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hello nematic droplet example"""
import os
import numpy as np
os.environ["DTMM_PARALLEL"] = "0"
import dtmm
dtmm.conf.set_fftlib("mkl_fft")
#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, XSIZE, YSIZE =60,128,128
#: illumination wavelengths in nm
WAVELENGTHS = range(380,780,20)
#: lets make some experimental data

nout = 1.0
no = 1.5
ne = 1.6
nhost = 1.5

data = dtmm.nematic_droplet_data((NLAYERS, XSIZE, YSIZE), radius = 300,
           profile = "y", no = no, ne = ne, nhost = nhost)

data[0][0] = 1

field_waves, cmf = dtmm.illumination_data((XSIZE, YSIZE), WAVELENGTHS, refind = nout,
                                             pixelsize = PIXELSIZE, diameter = 0.8, alpha = 0.1, beta = 0.) 
#field_waves[0][...,31,31] = 0.
window = dtmm.aperture((128,128),alpha = 0.2)
window = None

out = dtmm.transmit_field(field_waves, data,diffraction = True, mode = "b", 
                          npass = 1, nsteps = 2,n_in = nout,n_out = nout, window = window)

viewer = dtmm.field_viewer(out, cmf, refind = nout, sample = 0, 
                           polarizer = 90, focus = 0, analizer = 90, 
                           mode = "t", intensity = 1)
viewer.plot()
viewer.show()