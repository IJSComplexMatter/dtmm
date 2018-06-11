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
WAVELENGTHS = np.linspace(500,600,11)
#: some experimental data
BETA = 0.4

epsv = dtmm.refind2eps([4,4,4]) #let us make some high reflective index medium, to increase reflections.
epsv = np.broadcast_to(epsv, (NLAYERS,HEIGHT, WIDTH,3))
epsa = np.zeros_like(epsv)
optical_data = dtmm.validate_optical_data(([THICKNESS], epsv, epsa))


window = dtmm.aperture((HEIGHT, WIDTH),0.2,1.)

field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, window = window,
                      beta = BETA,  pixelsize = PIXELSIZE,focus = 25.) 

field_data_out = dtmm.transfer_field(field_data_in, optical_data, beta = BETA, npass = 3)

viewer = dtmm.field_viewer(field_data_out, mode = "r", intensity = 10.)

fig,ax = viewer.plot(imax = 1000., fmax = 100)
fig.show()
