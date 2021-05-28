"""Example on multiple reflections"""

import dtmm
import numpy as np
dtmm.conf.set_verbose(2)

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
epsa = (0.,0.,0.)

optical_data = [(THICKNESS, epsv, epsa)]

window = dtmm.aperture((HEIGHT, WIDTH),0.2,1)

#illumination data is focused at the top (exit) surface (actual focus = focus * ref_ind = 25*4=100)
field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, window = window,
                      beta = BETA,  pixelsize = PIXELSIZE,focus = 25.,n=1) 

# illumination field is defined in vacuum, so we make input and output medium with n = 1.
field_data_out = dtmm.transfer_field(field_data_in, optical_data, npass = -1, nin = 1, nout = 1)

#we are viewing computed data in vacuum, so set n = 1.
viewer4 = dtmm.field_viewer(field_data_in, mode = +1, intensity = 1,n = 1.)
fig4,ax4 = viewer4.plot()
ax4.set_title("Input field")

#: visualize output field
viewer1 = dtmm.field_viewer(field_data_out, mode = +1,intensity = 1,n = 1.) 
fig1,ax1 = viewer1.plot()
ax1.set_title("Transmitted field")

#: residual back propagating field is close to zero
viewer2 = dtmm.field_viewer(field_data_out, mode = -1,intensity = 1,n = 1.)
fig2,ax2 = viewer2.plot()
ax2.set_title("Residual field")

viewer3 = dtmm.field_viewer(field_data_in, mode = -1, intensity = 1,n = 1.)
fig3,ax3 = viewer3.plot()
ax3.set_title("Reflected field")

