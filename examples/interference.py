"""An example how to compute reflections"""

import dtmm
import numpy as np
dtmm.conf.set_verbose(1)
import matplotlib.pyplot as plt

#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 60,96,96
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,81)
#: some experimental data

thickness = [2.]
epsv = np.ones((1,HEIGHT,WIDTH,3))*2.25
epsv = epsv + np.linspace(0,2,HEIGHT)[None,:,None,None]
epsa = np.zeros_like(epsv)

optical_data = (thickness, epsv, epsa)


#optical_data = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), 
#          radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)

window = dtmm.aperture((HEIGHT,WIDTH),alpha = 0.2)

field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS,window = window,
                                            pixelsize = PIXELSIZE, beta = 0.4,betamax = 0.9) 

field_data_out = dtmm.transfer_field(field_data_in, optical_data, beta = 0.4,npass = 3,diffraction = True, betamax = 0.9,nstep = 4)

#fig,axes = plt.subplots(1,1)

viewer = dtmm.field_viewer(field_data_in, mode = "r")
viewer.set_parameters(sample = None, intensity = 1.,
                polarizer = None, focus = 0, analyzer = None)

fig,ax = viewer.plot()


fig.show()
