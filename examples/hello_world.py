"""Hello nematic droplet example."""

import dtmm
import numpy as np
import matplotlib.pyplot as plt
dtmm.conf.set_verbose(2)
dtmm.conf.set_numba_threads(1)
#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 60, 256, 256
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,9)
#: create some experimental data (stack)
optical_data = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), 
          radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)

#: create non-polarized input light
field_data_in = dtmm.field.illumination_data((HEIGHT, WIDTH), WAVELENGTHS,
                                            pixelsize = PIXELSIZE) 
#: transfer input light through stack
field_data_out = dtmm.transfer_field(field_data_in, optical_data)

#: visualize output field
viewer = dtmm.pom_viewer(field_data_out)
viewer.set_parameters(polarizer = "h", analyzer = "v", focus = -18)
fig,ax = viewer.plot()

plt.show()


