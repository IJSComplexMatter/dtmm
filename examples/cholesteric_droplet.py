"""Hello nematic droplet example."""
import os
os.environ["OMP_NUM_THREADS"] = "4"



import dtmm
import numpy as np
import matplotlib.pyplot as plt
import dtmm.conf
dtmm.conf.set_verbose(2)
dtmm.conf.set_fft_planner(1)

#: pixel size in nm

SCALE = 1

RESOLUTION = 2

RADIUS = 20
CORE = 0
PITCH = 0.375

HEIGHT = RADIUS*2.1
WIDTH = RADIUS*2.1




PIXELSIZE = 100//SCALE

RADIUS = int(RADIUS*1000/PIXELSIZE)
CORE = int(CORE*1000/PIXELSIZE)
HEIGHT = int(HEIGHT*1000/PIXELSIZE)
WIDTH = int(WIDTH*1000/PIXELSIZE)


HEIGHT = 400
WIDTH = 400

PITCH = PITCH * 1000 / PIXELSIZE

D = 1/RESOLUTION

#: compute box dimensions
NLAYERS = int(2*RADIUS *RESOLUTION)
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,11)
#WAVELENGTHS = np.linspace(420,540,4)
#WAVELENGTHS = [500]

NO= 1.525
NE = NO + 0.236

NO = 1.5
NE = 1.6


#: create some experimental data (stack)
optical_block = dtmm.cholesteric_droplet_data((NLAYERS, HEIGHT, WIDTH), 
          radius = RADIUS, pitch = PITCH, no = NO, ne = NE, nhost = 1.5, core = CORE, d = D, view = 'z')


#: create non-polarized input light
field_data_in = dtmm.field.illumination_data((HEIGHT, WIDTH), WAVELENGTHS,
                                            pixelsize = PIXELSIZE, beta = 0., phi = 0.1) 



#: transfer input light through stack
#solver = dtmm.solver.matrix_data_solver((HEIGHT,WIDTH), WAVELENGTHS, PIXELSIZE, method= "4x4", betamax = 1.,resolution = 1)
#solver.set_optical_data([optical_block], resize = 1)
#solver.calculate_field_matrix(nin = 1.5, nout = 1.5)
#solver.calculate_stack_matrix()



#solver.calculate_reflectance_matrix()
#field_out = solver.transfer_field(field_data_in[0])
#field_data_out = field_out, solver.wavelengths, solver.pixelsize
#: transfer input light through stack

for nlayers in np.linspace(0,NLAYERS,2)[1:]:
    nlayers = int(nlayers)
    print('Thickness', nlayers)
    data = [(optical_block[0][0:nlayers], optical_block[1][0:nlayers], optical_block[2][0:nlayers])]

    field_data_out = dtmm.transfer_field(field_data_in, data, 
                                     betamax = 0.9, reflection = 2, 
                                     diffraction = 1, npass = 3, 
                                     method = '4x4',
                                     norm = 2,
                                     split_wavelengths = False)
#: visualize output field
viewer1 = dtmm.field_viewer(field_data_in, mode = "r", n = 1.5)
viewer1.set_parameters(polarizer = "v", analyzer = "h", focus = 100)
fig,ax = viewer1.plot()

viewer2 = dtmm.field_viewer(field_data_out, mode = "t", n = 1.5)
viewer2.set_parameters(polarizer = "v", analyzer = "h", focus = -100)
fig,ax = viewer2.plot()

viewer3 = dtmm.field_viewer(field_data_out, mode = "r", n = 1.5)
viewer3.set_parameters(polarizer = "v", analyzer = "h", focus = 00)
fig,ax = viewer3.plot()

plt.show()


