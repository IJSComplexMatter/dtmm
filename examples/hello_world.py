#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hello nematic droplet example"""

import dtmm
import numpy as np

import matplotlib

if matplotlib.__version__.split(".")[0] != "2":
    1/0


#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH =60,96,96
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,10)
#: lets make some experimental data

optical_data = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), radius = 30,
           profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)

field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS,
                                            pixelsize = PIXELSIZE) 

field_data_out = dtmm.transmit_field(field_data_in, optical_data)

viewer = dtmm.field_viewer(field_data_out, sample = 0,
                polarizer = 0, focus = -20, analizer = 90, mode ="b")
viewer.plot()
viewer.show()
