"""Hello nematic droplet example with dispersive material.
Compare this with the hello_world.py example.

In this example we are using Cauchy approximation with n = 2 coefficients:
    
refind = a + b/lambda**2

we take the epsv values as a reference, so first coefficient is just epv**0.5,
and then we add anisotropic dispersion. The b coefficient depends on the 
anistropy.

Units of coefficient b are microns**2
"""

import dtmm
import numpy as np
import matplotlib.pyplot as plt
from dtmm.data import EpsilonCauchy

dtmm.conf.set_verbose(2)
#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 60, 96,96
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,9)
#: create some experimental data (stack)
d, epsv, epsa = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), 
          radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)

# First we prepare empty coefficients table. The shape argument must match the problem shape.
# We are using only two terms of expansion, to simplify things. Normally, you should use at least n = 3.
epsc = EpsilonCauchy(shape = (NLAYERS,HEIGHT,WIDTH), n = 2)
epsc.coefficients[...,0] = (epsv)**0.5  # a term, just set to refractive index
# the scale term for the b coefficient. We are using very strong dispersion, to make the effect visible.
scale = 0.08
epsc.coefficients[...,0:2,1] = scale*(epsv[...,0:2])**0.5  # b term ordinary
epsc.coefficients[...,2,1] = scale*(epsv[...,2])**0.5  # b term extraordinary

#e psc is a callable object. To evaulate coefficients and calculate eps values you
# must call it with the wavelength argument, e.g.
# >>> epsc(550)

#normalize so that we have same refractive index at 550 nm
epsc.coefficients[...,0] += ((epsv)**0.5 - epsc(550)**0.5)

#redefine optical data, setting epsc as material eps values
optical_data = [(d, epsc, epsa)]

#: create non-polarized input light
field_data_in = dtmm.field.illumination_data((HEIGHT, WIDTH), WAVELENGTHS,
                                            pixelsize = PIXELSIZE) 

#: transfer input light through stack. We must split wavelengths and compute for each wavelength.
#: the algorithm computes epsv values from the cauchy coefficient for each of the wavelengths.
field_data_out = dtmm.transfer_field(field_data_in, optical_data, split_wavelengths = True)

#: visualize output field
viewer = dtmm.pom_viewer(field_data_out)
viewer.set_parameters(polarizer = "h", analyzer = "v", focus = -18)
fig,ax = viewer.plot()

plt.show()


