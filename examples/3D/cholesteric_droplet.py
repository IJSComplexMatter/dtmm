"""Cholester droplet example.

In this example, we use multiple passes and a 4x4 method to compute reflections 
off the cholesteric droplet. For cholesterics one should take the norm = 2 argument in the
computation of the tranfered field and have npass > 1 to compute reflections.

It is best to take odd number of passes, so that one can check the residual field
(the back propagating part) of the output field . With each iteration
the back propagating part of the output field reduces.

In this example we calculate off-axis reflection; the beta parameter is not zero.
The actual beta in the experiment differs because the illumination_data
returns the nearest eigenwave mode.
"""

import dtmm
import numpy as np

dtmm.conf.set_verbose(1)
dtmm.conf.set_betamax(0.9)

#: pixel size in nm
PIXELSIZE = 50
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 40,96,96
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,9)
#: create some experimental data (stack) left-handed cholesteric
optical_data = [dtmm.cholesteric_droplet_data((NLAYERS, HEIGHT, WIDTH), 
          radius = 20, pitch = 7, no = 1.5, ne = 1.65, nhost = 1.5)] #approx 50*7*1.5 nm bragg reflection

#: create right-handed polarized input light
beta = 0.3 #make it off-axis 
#window = dtmm.aperture((HEIGHT, WIDTH),0.9,0.)
window = None

#jones = dtmm.jonesvec((1,1j)) 
jones = None

focus= 20 #this will focus field diaphragm in the middle of the stack
field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, jones = jones, beta = beta,
                       pixelsize = PIXELSIZE, n = 1.5, focus = focus, window = window) 

#: transfer input light through stackt
field_data_out = dtmm.transfer_field(field_data_in, optical_data, beta = beta, phi = 0,
                                     diffraction = 1, method = "4x4",  smooth = 0.1,
                                     reflection = 2,nin = 1.5, nout = 1.5,npass = 5,norm = 2)

#: visualize output field
viewer1 = dtmm.field_viewer(field_data_out, mode = "t",n = 1.5, intensity = 0.5, focus = -20) 
fig1,ax1 = viewer1.plot()
ax1.set_title("Transmitted field")

#: residual back propagating field is close to zero
viewer2 = dtmm.field_viewer(field_data_out, mode = "r",n = 1.5)
fig2,ax2 = viewer2.plot()
ax2.set_title("Residual field")

viewer3 = dtmm.field_viewer(field_data_in, mode = "r", n = 1.5, polarization_mode = "mode", polarizer = "LCP", analyzer = "LCP")
fig3,ax3 = viewer3.plot()
ax3.set_title("Reflected field")

