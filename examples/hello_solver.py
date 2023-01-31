"""Hello nematic droplet example using matrix solver."""

import dtmm.solver
import dtmm
import numpy as np
import matplotlib.pyplot as plt
dtmm.conf.set_verbose(2)
#: pixel size in nm
PIXELSIZE = 50
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 60, 1, 196
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,9)

no = 1.5
ne = 1.6

epsv1 = np.zeros((60,1,WIDTH,3))
epsv1[...,0] = no**2 + 0.5*np.cos(np.arange(0,4*np.pi, 4*np.pi/WIDTH)) 
epsv1[...,1] = no**2 + 0.5*np.cos(np.arange(0,4*np.pi, 4*np.pi/WIDTH)) 
epsv1[...,2] = no**2 + 0.5*np.cos(np.arange(0,4*np.pi, 4*np.pi/WIDTH)) 

epsa1 = np.zeros((60,1,WIDTH,3))
epsa1[...,1] = np.pi/2
epsa1[...,2] = np.arange(0,np.pi, np.pi/WIDTH)

# epsv2 = np.zeros((10,HEIGHT,1,3))
# epsv2[:,:,0,0] = no**2 + 0.1*np.cos(np.arange(0,4*np.pi, 4*np.pi/WIDTH)) 
# epsv2[:,:,0,1] = no**2 + 0.1*np.cos(np.arange(0,4*np.pi, 4*np.pi/WIDTH)) 
# epsv2[:,:,0,2] = no**2 + 0.1*np.cos(np.arange(0,4*np.pi, 4*np.pi/WIDTH)) 

# epsa2 = np.zeros((10,HEIGHT,1,3))
# epsa2[:,:,0,1] = np.pi/2
# epsa2[:,:,0,2] = np.arange(0,np.pi, np.pi/WIDTH)

optical_block1 = ((1,)*60, epsv1 ,epsa1)
#optical_block2 = ((6,)*10, epsv2, epsa2)

optical_data = [optical_block1]
optical_data = dtmm.data.validate_optical_data(optical_data, shape = (HEIGHT, WIDTH), dim = 3)

w = dtmm.window.aperture((HEIGHT,WIDTH),diameter = 0.5)
#: create non-polarized input light
field_data_in = dtmm.field.illumination_data((HEIGHT, WIDTH), WAVELENGTHS,
                                            pixelsize = PIXELSIZE, window = w) 
#: transfer input light through stack
solver = dtmm.solver.matrix_data_solver((HEIGHT,WIDTH), WAVELENGTHS, PIXELSIZE, method= "4x4")
solver.set_optical_data(optical_data, resize =2)
solver.calculate_field_matrix(nin = 1.5, nout = 1.5)
solver.calculate_stack_matrix()
solver.calculate_reflectance_matrix()
#solver.calculate_transmittance_matrix()
field_out = solver.transfer_field(field_data_in[0])

field_data_out = field_out, solver.wavelengths, solver.pixelsize


#: visualize output field
viewer3 = dtmm.pom_viewer(field_data_out)
viewer3.set_parameters(polarizer = "h", analyzer = "h", rows = 196)
fig,ax = viewer3.plot()

plt.show()


