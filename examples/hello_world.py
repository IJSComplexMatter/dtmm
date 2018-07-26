"""Hello nematic droplet example."""

import dtmm
import numpy as np

dtmm.conf.set_verbose(2)
dtmm.conf.set_nthreads(1)
import mkl
mkl.set_num_threads(1)

#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 60,96,96
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,11)
#: create some experimental data (stack)
optical_data = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), 
          radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)
#: create non-polarized input light
field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS,
                                            pixelsize = PIXELSIZE) 
#: transfer input light through stack
field_data_out = dtmm.transfer_field(field_data_in, optical_data)

#: visualize output field
viewer = dtmm.field_viewer(field_data_out)
viewer.set_parameters(sample = 0, intensity = 2.,
                polarizer = 0, focus = -16, analyzer = 90)

fig,ax = viewer.plot()
fig.show()
