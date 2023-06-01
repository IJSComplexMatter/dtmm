"""Hello nematic droplet example."""

import dtmm
import numpy as np
import matplotlib.pyplot as plt

from dtmm import wave, field, tmm

dtmm.conf.set_verbose(2)
#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 60, 128,128
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,9)
WAVELENGTHS = [550]
#: create some experimental data (stack)
#optical_block = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), 
#          radius = 30, profile = "r", no = 1.5, ne = 1.7, nhost = 1.5)


#planewave = wave.eigenwave((128,128),30,0)

jones = np.zeros((128,128,2),"float")
jones[0::4,:,0] = 1
jones[1::4,:,0] = 1
jones[2::4,:,1] = 1
jones[3::4,:,1] = 1
#jones[2::8,:,0] = 1
#jones[3::8,:,0] = 1

#jones[4::8,:,1] = 1
#jones[5::8,:,1] = 1
#jones[6::8,:,1] = 1
#jones[7::8,:,1] = 1

#jones[...,0] = planewave.real
#jones[...,1] = planewave.imag

fmat = tmm.f_iso()
m = tmm.fvec(fmat, jones = jones)
f = np.moveaxis(m,-1,0)
f = f[None,...]


#: create non-polarized input light
#field_data_in = dtmm.field.illumination_data((HEIGHT, WIDTH), WAVELENGTHS,
#                                            pixelsize = PIXELSIZE, jones = jones) 
#_,w,p = field_data_in
field_data_in = f,WAVELENGTHS,PIXELSIZE

#: transfer input light through stack
# solver = dtmm.solver.matrix_data_solver((HEIGHT,WIDTH), WAVELENGTHS, PIXELSIZE, method= "2x2", betamax = 0.5,resolution = 1)
# solver.set_optical_data([optical_block], resize = 2)
# solver.calculate_field_matrix(nin = 1.5, nout = 1.5)
# solver.calculate_stack_matrix()
# solver.calculate_transmittance_matrix()
# field_out = solver.transfer_field(field_data_in[0])
# field_data_out = field_out, solver.wavelengths, solver.pixelsize
#: transfer input light through stack
#field_data_out = dtmm.transfer_field(field_data_in, [optical_block], betamax = 1)
#: visualize output field
viewer1 = dtmm.pom_viewer(field_data_in, magnification = 1, d_cover = 0, NA = 0.75)
viewer1.set_parameters( analyzer = None, focus = -18)
fig,ax = viewer1.plot()


viewer2 = dtmm.pom_viewer(field_data_in, magnification = 50, d_cover = 0, NA = 0.75)
viewer2.set_parameters( analyzer = None, focus = -18)
fig,ax = viewer2.plot()

#plt.show()

plt.figure(figsize = (9,4))

plt.subplot(121)
plt.imshow(stokes2psi(jones2stokes(np.moveaxis(viewer1.calculate_field()[0],0,-1))),cmap = "twilight",vmin = -np.pi/2, vmax = np.pi/2)
plt.title("SLM source")
plt.colorbar()
plt.subplot(122)
plt.title("SLM image")
plt.imshow(stokes2psi(jones2stokes(np.moveaxis(viewer2.calculate_field()[0],0,-1))),cmap = "twilight",vmin = -np.pi/2, vmax = np.pi/2)
plt.colorbar()