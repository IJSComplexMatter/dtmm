"""This example uses a standard (non-diffractive) 4x4 
method to calculate transmitance and reflections of a nematic droplet.
"""

import dtmm
import numpy as np

dtmm.conf.set_verbose(2)

#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 60,96,96
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,11)
#: create some experimental data (stack)
d, epsv, epsa = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), 
          radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)
#: create non-polarized input light

f,w,p = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS,
                                            pixelsize = PIXELSIZE, beta = 0., phi = 0.) 

field_data_in = f,w,p

#transpose to field vector
ft = dtmm.field.transpose(f)

# build kd phase values
kd = d[:,None]*(dtmm.k0(WAVELENGTHS, PIXELSIZE))

#build stack matrix... 
cmat = dtmm.tmm.stack_mat(kd[...,None,None], epsv, epsa)
fout = dtmm.tmm.transmit(ft,cmat)

# invere transpose to build field data for visualization
field_data_out = dtmm.field.itranspose(fout),w,p

#: visualize output field
viewer = dtmm.field_viewer(field_data_out, diffraction = False)
viewer.set_parameters(sample = 0, intensity = 2,
                polarizer = 0, analyzer = 90)

fig,ax = viewer.plot()
fig.show()

viewer2 = dtmm.field_viewer(field_data_in, mode = "r")
viewer2.set_parameters(sample = 0, intensity = 2,
                polarizer = 0,  analyzer = 90)

fig,ax = viewer2.plot()
fig.show()


