"""This example shows that calculating 4x4 reflection is less stable and noisier.
"""

import dtmm
import numpy as np

#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 60, 96, 96
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,9)
#: create some experimental data (stack)
block_data = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), 
          radius = 30, profile = "r", no = 3.5, ne = 3.6, nhost = 3.5)

betamax = np.inf #no need to cutoff in 2x2 mode, in the 4x4 mode, filtering is done automatically
NA = 0.9 #NA of the microscope objective... also this filters out high frequency modes

window = dtmm.aperture((96,96),0.95,0.)
window = None
#: create non-polarized input light
(fin2,w,p) = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, beta = 0.,
                                            pixelsize = PIXELSIZE, window = window) 
(fin4,w,p) = (fin2.copy(),w,p)


(f2,w,p)  = dtmm.transfer_field((fin2,w,p), [block_data],  beta = 0., phi = 0., betamax = betamax,
             method = "2x2", ret_bulk = False, npass = 9, reflection = 2) 

(f4,w,p)  = dtmm.transfer_field((fin4,w,p), [block_data],  beta = 0., phi = 0., betamax = betamax,
             method = "4x4", ret_bulk = False, npass = 9, reflection = 2) 


viewer2in = dtmm.field_viewer((fin2,w,p), bulk_data = False, n = 1, mode = "r", betamax = NA)
viewer2in.set_parameters(sample = 0, intensity = 1,
                polarizer = 0,  analyzer = 0)

fig,ax = viewer2in.plot()

viewer4in = dtmm.field_viewer((fin4,w,p), bulk_data = False, n = 1, mode = "r", betamax = NA)
viewer4in.set_parameters(sample = 0, intensity = 1,
                polarizer = 0,  analyzer = 0)

fig,ax = viewer4in.plot()

viewer2 = dtmm.field_viewer((f2,w,p), bulk_data = False, n = 1, mode = "t", betamax = NA)
viewer2.set_parameters(sample = 0, intensity = 1,
                polarizer = 0,  analyzer = 0)

fig,ax = viewer2.plot()

viewer4 = dtmm.field_viewer((f4,w,p), bulk_data = False, n = 1, mode = "t", betamax = NA)
viewer4.set_parameters(sample = 0, intensity = 1,
                polarizer = 0,  analyzer = 0)

fig,ax = viewer4.plot()


