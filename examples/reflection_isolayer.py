"""Example on multiple reflections"""

import dtmm
import numpy as np
dtmm.conf.set_verbose(1)

THICKNESS = 100.
#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 1,128,128
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(500,600,3)
#: some experimental data

eps = dtmm.refind2eps([4,4,4]) #let us make some high reflective index medium, to increase reflections.
eps = np.broadcast_to(eps, (NLAYERS,HEIGHT, WIDTH,3))
angles = np.zeros_like(eps)
optical_data = dtmm.validate_optical_data(([THICKNESS], eps, angles))


window = dtmm.aperture((HEIGHT, WIDTH),0.2,1.)

field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, window = window,
                      beta = 0.4,      pixelsize = PIXELSIZE,focus = 25.) 

field_data_out = dtmm.transfer_field(field_data_in, optical_data, beta = 0.4, npass = 7)

viewer = dtmm.field_viewer(field_data_out, mode = "t", intensity = 200.)

fig,ax = viewer.plot(imax = 1000., fmax = 1000)
fig.show()
