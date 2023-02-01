"""Hello nematic droplet example."""

import dtmm
import numpy as np
import matplotlib.pyplot as plt
import dtmm.conf
dtmm.conf.set_verbose(2)
#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 60, 96, 96
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,9)

#: create some experimental data (stack)
optical_block = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), 
          radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)

#: create non-polarized input light
field_data_in = dtmm.field.illumination_data((HEIGHT, WIDTH), WAVELENGTHS,
                                            pixelsize = PIXELSIZE, beta = 0.4, phi = 0.1) 

#: transfer input light through stack
# solver = dtmm.solver.matrix_data_solver((HEIGHT,WIDTH), WAVELENGTHS, PIXELSIZE, method= "2x2", betamax = 0.5,resolution = 1)
# solver.set_optical_data([optical_block], resize = 2)
# solver.calculate_field_matrix(nin = 1.5, nout = 1.5)
# solver.calculate_stack_matrix()
# solver.calculate_transmittance_matrix()
# field_out = solver.transfer_field(field_data_in[0])
# field_data_out = field_out, solver.wavelengths, solver.pixelsize
#: transfer input light through stack
field_data_out = dtmm.transfer_field(field_data_in, [optical_block], betamax = 1, diffraction = 3)
#: visualize output field
viewer1 = dtmm.pom_viewer(field_data_out, magnification = 100, d_cover = 0, NA = 0.9)
viewer1.set_parameters(polarizer = "v", analyzer = "h", focus = -18)
fig,ax = viewer1.plot()

field_data_out = dtmm.transfer_field(field_data_in, [optical_block],beta = 0, phi = 0, betamax = 1, diffraction = 3)

viewer2 = dtmm.pom_viewer(field_data_out, magnification = 100., d_cover = 0, NA = 0.9)
viewer2.set_parameters(polarizer = "v", analyzer = "h", focus = -18)
fig,ax = viewer2.plot()

plt.show()


