#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hello nematic droplet example"""
import os
#os.environ["DTMM_PARALLEL"] = "0"

import dtmm
import numpy as np
dtmm.conf.set_verbose(1)
dtmm.conf.set_fftlib("mkl_fft")
#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH =60,96,96
#: illumination wavelengths in nm
WAVELENGTHS = range(380,780,40)
#: lets make some experimental data

optical_data = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), radius = 30,
           profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)

#t,e,a = optical_data

#mask = a[...,2]!= 0.

#a[mask,2] = 0.

beta, phi = dtmm.illumination_betaphi(0.1,23)

window = dtmm.aperture((HEIGHT,WIDTH), 1,0.1)

field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, beta = beta, phi = phi,
                                             pixelsize = PIXELSIZE, window = window) 

window = dtmm.aperture((HEIGHT,WIDTH), 1,0.1)

field_data_out = dtmm.transfer_field(field_data_in, optical_data, beta = beta, 
                                     phi = phi, npass = 1,nstep = 1, diffraction = True)

viewer = dtmm.field_viewer(field_data_out, sample = 0, intensity = 2,
                polarizer = 0, focus = -20, analyzer = 90)
fig, ax = viewer.plot()
fig.show()