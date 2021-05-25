"""Hello nematic droplet example with dispersive material."""

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
          radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)[0]
"""
we will be using Cauchy approximation with n = 2 coefficients.
refind = a + b/lambda**2

#e take the epsv values as a reference, so first coefficient is just epv**0.5,
and then w add some small anisotropic dispersion the b coefficient depends on the 
anistropy.

units of coefficient b are microns**2
"""
epsc = EpsilonCauchy((NLAYERS,HEIGHT,WIDTH), n = 2)
epsc.coefficients[...,0] = (epsv)**0.5  # a term, just set to refractive index
epsc.coefficients[...,0:2,1] = 0.005*(epsv[...,0:2])**0.5   # b term ordinary
epsc.coefficients[...,2,1] = 0.005*(epsv[...,2])**0.5  # b term extraordinary

#redefien optical data, setting epsc as 
optical_data = [(d, epsc, epsa)]

#: create non-polarized input light
field_data_in = dtmm.field.illumination_data((HEIGHT, WIDTH), WAVELENGTHS,
                                            pixelsize = PIXELSIZE) 

#: transfer input light through stack. We must split wavelength and compute for each wavelength.
#: the algorithm computes epsv values from the cauchy coefficient for each of the wavelengths.
field_data_out = dtmm.transfer_field(field_data_in, optical_data, split_wavelengths = True)

#: visualize output field
viewer = dtmm.pom_viewer(field_data_out)
viewer.set_parameters(polarizer = "h", analyzer = "v", focus = -18)
fig,ax = viewer.plot()

plt.show()


