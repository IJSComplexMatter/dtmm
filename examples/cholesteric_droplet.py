"""Cholester droplet example.

In this example, we use multiple passes to compute reflections of the cholesteric
droplet. For cholesterics one should take the norm = 2 argument in the
computation of the tranfered field and have npass > 1 to compute reflections.

It is best to take odd number of passes, so that one can check the residual field
(the back propagating part) of the output field . With each iteration
the back propagating part of the output field reduces.

In this example we use a window function to shape the input light with circular 
aperture in order to eliminate boundary effects. Also, we calculate off-axis 
reflection; the beta parameter is not zero.
"""

import dtmm
import numpy as np

dtmm.conf.set_verbose(1)

#: pixel size in nm
PIXELSIZE = 50
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 40,96,96
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,21)
#: create some experimental data (stack) left-handed cholesteric
optical_data = dtmm.cholesteric_droplet_data((NLAYERS, HEIGHT, WIDTH), 
          radius = 20, pitch = 7, no = 1.5, ne = 1.6, nhost = 1.5) #approx 50*7*1.5 nm bragg reflection

#: create right-handed polarized input light
beta = 0.2 #make it off-axis 
window = dtmm.aperture((HEIGHT, WIDTH),0.9)
jones = dtmm.jonesvec((1,-1j)) #right handed input light
focus= 20 #thi will focus aperture in the middle of the stack
field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, jones = jones, beta = beta,
                       pixelsize = PIXELSIZE, n = 1.5, window = window, focus = focus) 

#: transfer input light through stack
field_data_out = dtmm.transfer_field(field_data_in, optical_data, beta = beta, nin = 1.5, nout = 1.5,npass = 3, norm = 2)

#: visualize output field
viewer1 = dtmm.field_viewer(field_data_out, mode = "t",n = 1.5, intensity = 0.5, focus = -20) 
fig1,ax1 = viewer1.plot()
ax1.set_title("Transmitted field")

#: residual back propagating field is close to zero
viewer2 = dtmm.field_viewer(field_data_out, mode = "r",n = 1.5)
fig2,ax2 = viewer2.plot()
ax2.set_title("Residual field")

viewer3 = dtmm.field_viewer(field_data_in, mode = "r", n = 1.5)
fig3,ax3 = viewer3.plot()
ax3.set_title("Reflected field")