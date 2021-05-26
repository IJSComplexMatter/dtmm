"""Demonstrates dark field microscopy simulation.

Here we define illumination field data at larger beta values, and viewe the computed
filed with pom_viewer that cuts large beta values to simulate dark field microscopy.

Therefore, we set NA (0.5) of the microscope objective lower than that of the illumination
annular eperture inner width (0.6). 
"""

import dtmm
import numpy as np
dtmm.conf.set_verbose(2)

#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 60,96,96
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,9)
#: lets make some experimental data
optical_data = [dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), radius = 30,
           profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)]

#using annular aperture, approximate ratio of aperture dimaters is inner/outer = 6/7
#: max NA is set to 0.7., this means input rays from NA of 0.6 to 0.7 
beta, phi, intensity = dtmm.illumination_rays(0.7,(6,7))

#: care must be taken when using high NA light source, we must usee a high betamax parameter
field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, PIXELSIZE, 
                                       beta,phi, intensity,
                                       focus = 30, betamax = 1)
 
#: also in the calculation, we must perform it at high betamax
field_data_out = dtmm.transfer_field(field_data_in, optical_data, multiray = True, betamax = 1)

# we must set the numarical aperture of the objective lower than the minimum NA of the illumination.
viewer = dtmm.pom_viewer(field_data_out, intensity = 10, NA = 0.5, focus = -20,
                           beta = beta)
fig, ax = viewer.plot()
fig.show()

