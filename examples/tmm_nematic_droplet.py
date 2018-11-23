"""This example uses a standard (non-diffractive) 4x4 and 2x2 methods
to calculate transmitance and reflections of a nematic droplet.
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
#: create some experimental data (stack)
d, epsv, epsa = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), 
          radius = 30, profile = "x", no = 1.5, ne = 1.6, nhost = 1.5)
#: create non-polarized input light

f,w,p = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS,
                                            pixelsize = PIXELSIZE, beta = 0., phi = 0.) 

field_data_in = f,w,p

#transpose to field vector
ft = dtmm.field.transpose(f)

# build kd phase values
kd = d[:,None]*(dtmm.k0(WAVELENGTHS, PIXELSIZE))

#build stack matrix... 
cmat = dtmm.tmm.stack_mat(kd[...,None,None], epsv, epsa, method = "2x2")
fout_2x2 = dtmm.tmm.transmit2x2(ft,cmat)

cmat = dtmm.tmm.stack_mat(kd[...,None,None], epsv, epsa, method = "4x2")
fout_4x2 = dtmm.tmm.transmit(ft,cmat)

cmat = dtmm.tmm.stack_mat(kd[...,None,None], epsv, epsa, method = "4x4")
fout_4x4 = dtmm.tmm.transmit(ft,cmat)

# inverse transpose to build field data for visualization
field_data_out_2x2 = dtmm.field.itranspose(fout_2x2),w,p
field_data_out_4x2 = dtmm.field.itranspose(fout_4x2),w,p
field_data_out_4x4 = dtmm.field.itranspose(fout_4x4),w,p

#: visualize output field
viewer1 = dtmm.field_viewer(field_data_out_2x2, diffraction = False)
viewer1.set_parameters(sample = 45, intensity = 2,
                polarizer = 0, analyzer = 90)

fig,ax = viewer1.plot()
ax.set_title("2x2 method (jones method)")
fig.show()

viewer2 = dtmm.field_viewer(field_data_out_4x2, diffraction = False )
viewer2.set_parameters(sample = 45, intensity = 2,
                polarizer = 0,  analyzer = 90)

fig,ax = viewer2.plot()
ax.set_title("4x2 method (4x4 with single reflection)")
fig.show()

viewer3 = dtmm.field_viewer(field_data_out_4x4, diffraction = False )
viewer3.set_parameters(sample = 45, intensity = 2,
                polarizer = 0,  analyzer = 90)

fig,ax = viewer3.plot()
ax.set_title("4x4 method (4x4 with interference)")
fig.show()

viewer4 = dtmm.field_viewer(field_data_in, mode = "r" )
viewer4.set_parameters(sample = 45, intensity = 2,
                polarizer = 0,  analyzer = 90)

fig,ax = viewer4.plot()
ax.set_title("4x4 method - reflected field")
fig.show()


