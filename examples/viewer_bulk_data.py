"""Nematic droplet example using bulk data calculation and bulk viewer"""

import dtmm
import numpy as np

#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 60, 96, 96
#: illumination wavelengths in nm
WAVELENGTHS = [550]
#: create some experimental data (stack)
optical_data = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), 
          radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)

#: create non-polarized input light
field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, 
                                            pixelsize = PIXELSIZE) 
#: transfer input light through stack and compute bulk EM field
field_data_out = dtmm.transfer_field(field_data_in, optical_data, ret_bulk = True)

#: visualize output field
viewer = dtmm.field_viewer(field_data_out, bulk_data = True)
viewer.set_parameters(sample = 0, intensity = 2,
                polarizer = 0, focus = 20, analyzer = 90)

fig,ax = viewer.plot()
