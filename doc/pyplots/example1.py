#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hello nematic droplet example
"""

import numpy as np
import dtmm
#: pixel size in nm
PIXELSIZE = 400 
#: compute box dimensions
NLAYERS, XSIZE, YSIZE = 50, 164, 164
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,700,7)
#: lets make some experimental data
stack, mask, material = dtmm.nematic_droplet_data((NLAYERS, XSIZE, YSIZE), 
                radius = 20, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)

field, wavenumbers, cmf = dtmm.illumination_data((XSIZE, YSIZE), WAVELENGTHS, refind = 1.5,
                                             pixelsize = PIXELSIZE, diameter = 0.9) 
#out = dtmm.transmit_field(field, wavenumbers, stack, mask, material)

viewer = dtmm.field_viewer(field, wavenumbers, cmf, refind = 1.5, 
                           polarizer = 0, focus = -25, analizer = 0)
viewer.plot()
viewer.show()