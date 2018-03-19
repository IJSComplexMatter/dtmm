#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hello nematic droplet example"""

import dtmm
#: pixel size in nm
PIXELSIZE = 100 
#: compute box dimensions
NLAYERS, XSIZE, YSIZE = 60, 128, 128
#: illumination wavelengths in nm
WAVELENGTHS = range(380,780,40)
#: lets make some experimental data
data = dtmm.nematic_droplet_data((NLAYERS, XSIZE, YSIZE), radius = 30,
           profile = "r", no = 1.5, ne = 1.7, nhost = 1.5)

field_waves, cmf = dtmm.illumination_data((XSIZE, YSIZE), WAVELENGTHS, refind = 1.5,
                                             pixelsize = PIXELSIZE, diameter = 0.8) 
out = dtmm.transmit_field(field_waves, data)

viewer = dtmm.field_viewer(out, cmf, refind = 1.5, sample = 0, 
                           polarizer = 0, focus = -30, analizer = 90)
viewer.plot()
viewer.show()