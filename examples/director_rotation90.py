"""This example shows how to rotate director field."""

import dtmm
import numpy as np

dtmm.conf.set_verbose(1)

#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 96,96,96
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,11)
#: create some experimental data (stack)
director = dtmm.nematic_droplet_director((NLAYERS, HEIGHT, WIDTH), 
          radius = 30, profile = "r")

rotm = dtmm.rotation_matrix((0.2,0.4,0.6))
director = dtmm.rotate_director(rotm, director)

optical_data = dtmm.data.director2data(director,no = 1.5, ne = 1.6)
#: create non-polarized input light
field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS,
                                            pixelsize = PIXELSIZE) 
#: transfer input light through stack
field_data_out = dtmm.transfer_field(field_data_in, optical_data)

#: visualize output field
viewer = dtmm.field_viewer(field_data_out)
viewer.set_parameters(sample = 0, intensity = 2.,
                polarizer = 0, focus = -34, analyzer = 90)

fig,ax = viewer.plot()
fig.show()
