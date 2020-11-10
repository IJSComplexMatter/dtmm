"""Hello nematic droplet example."""

import dtmm
import numpy as np

dtmm.conf.set_verbose(2)

#: pixel size in nm
PIXELSIZE = 80
#: compute box dimensionss
NLAYERS, HEIGHT, WIDTH = 50, 512, 512
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,11)
#: create some experimental data (stack)

m1 = dtmm.sphere_mask((NLAYERS, HEIGHT, WIDTH),25, (26,0,0))
m2 = dtmm.sphere_mask((NLAYERS, HEIGHT, WIDTH), 25, (-26,0,0))
m = m1 | m2

refind = np.ones((NLAYERS, HEIGHT, WIDTH,3)) * 1.3
angles = np.zeros((NLAYERS, HEIGHT, WIDTH,3))

refind[m,0] = 1.6
refind[m,1] = 1.6
refind[m,2] = 1.6

d = np.ones((NLAYERS,))

optical_data = d, refind, angles


#NA 0.15, diaphragm with diameter 5 pixels, around 2*2*pi rays
beta, phi, intensity = dtmm.illumination_rays(0.15,5)

beta = 0.
phi = 0.
intensity = 1

window = dtmm.aperture((HEIGHT,WIDTH), 1,0.1)

field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, PIXELSIZE, 
                                       beta,phi, intensity,
                                       window = window, n = 1.3, focus = 30) 

field_data_out = dtmm.transfer_field(field_data_in, optical_data, multiray = True, split_rays = False,
                                     nin = 1.3,nout =1.3, betamax = 1., diffraction = 1, reflection = 1)


#: visualize output field
viewer = dtmm.field_viewer(field_data_out, betamax = 1., n = 1.3)
viewer.set_parameters(sample = 0, intensity = 1, 
                polarizer = None, focus = -14, analyzer = None)
fig,ax = viewer.plot()

