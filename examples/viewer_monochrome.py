"""Same as hello world, but with a monochrome camera and blue LED light"""

import dtmm
import numpy as np

#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 60, 96, 96
#: illumination wavelengths in nm
WAVELENGTHS = [410,450,490]
#: create some experimental data (stack)
optical_data = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), 
          radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)

#: create non-polarized input light
field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, 
                                            pixelsize = PIXELSIZE) 

#: transfer input light through stack
field_data_out = dtmm.transfer_field(field_data_in, optical_data)

#: here we load a custom illuminant table (it could be loaded from a file as well) 
#: a grayscale CMOS spectral response function and build a transmission color matching function
illuminant = [[410,0],[430,0.8],[450,1],[470,0.8],[490,0]]
cmf = dtmm.color.load_tcmf(WAVELENGTHS,cmf = "CMOS",illuminant = illuminant)
#: visualize output field
viewer = dtmm.pom_viewer(field_data_out, cmf = cmf)
viewer.set_parameters(sample = 0, intensity = 1, 
                polarizer = "h", focus = -18, analyzer = "v")

fig,ax = viewer.plot()
