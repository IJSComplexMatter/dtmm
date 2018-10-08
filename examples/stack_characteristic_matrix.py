"""This example uses a standard (non-diffractive) 4x4 
method to calculate transmitance and reflections of a nematic droplet"""

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
optical_data = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), 
          radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)
#: create non-polarized input light

f,w,p = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS,
                                            pixelsize = PIXELSIZE, beta = 0., phi = 0.) 

field_data_in = f,w,p

ft = f.transpose((0,1,3,4,2)).copy()

cmat = dtmm.tmm.stack_mat(dtmm.k0(WAVELENGTHS, PIXELSIZE)[...,None,None],optical_data)
fout = dtmm.tmm.transmit(ft,cmat, nin = 1.,nout = 1.)


#f = fout.transpose((0,1,4,2,3)).copy()

field_data_out = fout.transpose((0,1,4,2,3)).copy(),w,p
#field_data_in = ft.transpose((0,1,4,2,3)).copy(),w,p

#: transfer input light through stack
field_data_out = dtmm.transfer_field(field_data_in, optical_data,interference =True,  npass = 3, norm = 0)

#: visualize output field
viewer = dtmm.field_viewer(field_data_in, mode = "r")
viewer.set_parameters(sample = 0, intensity = 2,
                polarizer = 0, focus = 0, analyzer = 90)

fig,ax = viewer.plot()
fig.show()

