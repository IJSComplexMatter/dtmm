"""This example shows how to work with bulk data.

If tranfer_field is called with ret_bulk = True, it calculates and returns 
bulk field data. The first element is field just in front of the stack (at the input surface), and 
the last element is for the field just after the exit surface.

The rest of the elements are the computed field after 
passing each of the layers (depending o the direction of propagation - number of pass) 

With 4x4 method to view the bulk field you need to use even number of passes 
(or npass = 1) with odd number of passes you will se the residual field propagation.

"""

import dtmm
import numpy as np

#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 58, 96, 96
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,9)
#: create some experimental data (stack)
optical_data = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), 
          radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)
#optical_data[0][...]=1
#optical_data[1][0,...]=1
#optical_data[1][-1,...]=1

window = dtmm.aperture((96,96),0.95,0.)
window = None
#: create non-polarized input light
field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, beta = 0.,
                                            pixelsize = PIXELSIZE, window = window) 

f,w,p  = dtmm.transfer_field(field_data_in, optical_data,  beta = 0., phi = 0., betamax = 0.8,
             method = "4x4", ret_bulk = True, npass = 4, reflection = 2, smooth = 0.1) #must be even for 4x4 method


viewer = dtmm.field_viewer(field_data_in, bulk_data = False, n = 1, mode = "r", betamax = 0.9)
viewer.set_parameters(sample = 0, intensity = 1,
                polarizer = 0, focus = 0, analyzer = 0)

fig,ax = viewer.plot()


viewer_bulk = dtmm.field_viewer((f,w,p), bulk_data = True)
viewer_bulk.set_parameters(sample = 0, intensity = 1,
                polarizer = 0, focus = 0, analyzer = 0)

fig,ax = viewer_bulk.plot()