"""Example on multiple reflections"""

import dtmm
import numpy as np
dtmm.conf.set_verbose(1)

THICKNESS = 100.
#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 1,96,96
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(500,600,11)
#: some experimental data
BETA = 0.4

epsv = dtmm.refind2eps([4,4,4]) #high reflective index medium, to increase reflections.
epsv = np.broadcast_to(epsv, (NLAYERS,HEIGHT, WIDTH,3))
epsa = np.zeros_like(epsv)
optical_data = dtmm.validate_optical_data(([THICKNESS], epsv, epsa))

window = dtmm.aperture((HEIGHT, WIDTH),0.2,1.)

#illumination data is focused at the top (exit) surface (actual focus = focus * ref_ind = 25*4=100)
field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, window = window,
                      beta = BETA,  pixelsize = PIXELSIZE,focus = 25.) 

field_data_out = dtmm.transfer_field(field_data_in, optical_data, beta = BETA, npass =3, norm = 1)

#: visualize output field
viewer1 = dtmm.field_viewer(field_data_out, mode = "t",intensity = 100) 
fig1,ax1 = viewer1.plot()
ax1.set_title("Transmitted field")

#: residual back propagating field is close to zero
viewer2 = dtmm.field_viewer(field_data_out, mode = "r",intensity = 100)
fig2,ax2 = viewer2.plot()
ax2.set_title("Residual field")

viewer3 = dtmm.field_viewer(field_data_in, mode = "r", intensity = 100)
fig3,ax3 = viewer3.plot()
ax3.set_title("Reflected field")