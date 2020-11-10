"""Example on multiple reflections"""

import dtmm
import numpy as np
dtmm.conf.set_verbose(1)

THICKNESS =  100
#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 1,96,96
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(400,700,9)

#: some experimental data
BETA = 0.4

epsv = dtmm.refind2eps([4,4,4]) #high reflective index medium, to increase reflections.
epsv = np.broadcast_to(epsv, (NLAYERS,HEIGHT, WIDTH,3)).copy()
#epsv[...] = 1.
#epsv[0] = 1.
#epsv[-1] = 1.
epsa = np.zeros_like(epsv)
epsa[0] = np.pi/4
#epsa[1] = np.pi/2
optical_data = dtmm.validate_optical_data(([THICKNESS]*NLAYERS, epsv, epsa))

window = dtmm.aperture((HEIGHT, WIDTH),0.2,1)

#illumination data is focused at the top (exit) surface (actual focus = focus * ref_ind = 25*4=100)
field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, window = window,
                      beta = BETA,  pixelsize = PIXELSIZE,focus = 25.) 

field_data_out = dtmm.transfer_field(field_data_in, optical_data, beta = BETA,  phi = 0.,
            reflection = 2, diffraction =1, npass = 3, method = "2x2")

viewer4 = dtmm.field_viewer(field_data_in, mode = "t", intensity = 1)
fig4,ax4 = viewer4.plot()
ax4.set_title("Input field")

#: visualize output field
viewer1 = dtmm.field_viewer(field_data_out, mode = "t",intensity = 1) 
fig1,ax1 = viewer1.plot()
ax1.set_title("Transmitted field")

#: residual back propagating field is close to zero
viewer2 = dtmm.field_viewer(field_data_out, mode = "r",intensity = 1)
fig2,ax2 = viewer2.plot()
ax2.set_title("Residual field")

viewer3 = dtmm.field_viewer(field_data_in, mode = "r", intensity = 1)
fig3,ax3 = viewer3.plot()
ax3.set_title("Reflected field")

