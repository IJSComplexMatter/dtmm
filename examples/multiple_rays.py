#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple ray example"""

import dtmm
import numpy as np
dtmm.conf.set_verbose(2)

#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 60,96,96
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,21)
#: lets make some experimental data

optical_data = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), radius = 30,
           profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)

#NA 0.25, diaphragm with diameter 8 pixels, around 4*4*pi rays
beta, phi, intensity = dtmm.illumination_rays(0.25,6)

window = dtmm.aperture((HEIGHT,WIDTH), 1,0.2)

field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, PIXELSIZE, 
                                       beta,phi, intensity,
                                       window = window, n = 1.5, focus = 30) 


field_data_out = dtmm.transfer_field(field_data_in, optical_data, beta = beta, phi = phi, 
                                     nin = 1.5,nout =1.5)

viewer = dtmm.field_viewer(field_data_out, sample = 0, intensity = 2,n=1.5,
                polarizer = 0, focus = -30, analyzer = 90, betamax = 0.4)
fig, ax = viewer.plot()
fig.show()