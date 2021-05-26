"""Two spherical particles at differnet focal planes"""

import dtmm
import numpy as np

dtmm.conf.set_verbose(2)

#: pixel size in nm
PIXELSIZE = 80
#: compute box dimensionss
NLAYERS, HEIGHT, WIDTH = 30, 256, 256
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,11)
#: create some experimental data (stack)

m1 = dtmm.sphere_mask((NLAYERS, HEIGHT, WIDTH),15, (20,0,0))
m2 = dtmm.sphere_mask((NLAYERS, HEIGHT, WIDTH), 15, (-20,0,0))

eps1 = np.ones((NLAYERS, HEIGHT, WIDTH,3)) * 1.5**2
angles1 = np.zeros((NLAYERS, HEIGHT, WIDTH,3))
eps2 = np.ones((NLAYERS, HEIGHT, WIDTH,3)) * 1.5**2
angles2 = np.zeros((NLAYERS, HEIGHT, WIDTH,3))

eps1[m1,0] = 1.6**2
eps1[m1,1] = 1.6**2
eps1[m1,2] = 1.6**2

eps2[m2,0] = 1.6**2
eps2[m2,1] = 1.6**2
eps2[m2,2] = 1.6**2

d = np.ones((NLAYERS,))

optical_data = [(d, eps1, angles1),(d, eps2, angles2)]


#NA 0.15, diaphragm with diameter 5 pixels, around 2*2*pi rays
beta, phi, intensity = dtmm.illumination_rays(0.35,5)

window = dtmm.aperture((HEIGHT,WIDTH), 1,0.1)

field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, PIXELSIZE, 
                                       beta,phi, intensity,
                                       window = window, focus = 30) 

field_data_out = dtmm.transfer_field(field_data_in, optical_data, multiray = True, split_rays = True,
                                     betamax = 1., diffraction = 1, reflection = 2)


#: visualize output field
viewer = dtmm.pom_viewer(field_data_out)
viewer.set_parameters(sample = 0, intensity = 0.8, 
                polarizer = None, focus = -15, analyzer = None)
fig,ax = viewer.plot()

