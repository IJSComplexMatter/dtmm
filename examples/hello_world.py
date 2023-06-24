"""Hello nematic droplet example."""

import dtmm
import numpy as np
import matplotlib.pyplot as plt
import dtmm.conf
from dtmm.sim import ScatteringBlockSolver3D, MatrixBlockSolver3D

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




SCALE = 1

RESOLUTION = 4

RADIUS = 4
CORE = 0
PITCH = 0.375

HEIGHT = RADIUS*2.1
WIDTH = RADIUS*2.1




PIXELSIZE = 100//SCALE

RADIUS = int(RADIUS*1000/PIXELSIZE)
CORE = int(CORE*1000/PIXELSIZE)
HEIGHT = int(HEIGHT*1000/PIXELSIZE)
WIDTH = int(WIDTH*1000/PIXELSIZE)



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
                                            pixelsize = PIXELSIZE, beta = 0., phi = 0., n = 1.5) 

field_data_back = dtmm.field.illumination_data((HEIGHT, WIDTH), WAVELENGTHS,
                                            pixelsize = PIXELSIZE, beta = 0., phi = 0., n = 1.5, backdir = True) 


field_data_in[0][...] = (field_data_back[0]*0 +field_data_in[0])

#: transfer input light through stack
# solver = dtmm.solver.matrix_data_solver((HEIGHT,WIDTH), WAVELENGTHS, PIXELSIZE, method= "2x2", betamax = 0.5,resolution = 1)

solver = ScatteringBlockSolver3D((HEIGHT,WIDTH), WAVELENGTHS, PIXELSIZE, method= "4x4", betamax = 0.9,resolution = 1)

solver.set_optical_block(optical_block, resize = 0)

deltaeps = solver.epsv - solver.epsv_eff
epsa = solver.epsa
epsv_eff = solver.epsv_eff

# solver.set_optical_block((optical_block[0], solver.epsv_eff + deltaeps * 0.5, solver.epsv), resize = 0)



# solver.calculate_field_matrix(nin = 1.5, nout = 1.5)
# solver.calculate_stack_matrix()
# solver.calculate_reflectance_matrix()

# solver.transfer_field(field_data_in[0])
# solver.scatter_field()
# solver.transfer_field()

# solver.clear_matrices()
# solver.set_optical_block((optical_block[0], solver.epsv_eff + deltaeps * 0.8, solver.epsv), resize = 0)
# solver.calculate_stack_matrix()
# solver.calculate_field_matrix(nin = 1.5, nout = 1.5)
# solver.calculate_reflectance_matrix()

# solver.scatter_field()
# solver.transfer_field()

solver = ScatteringBlockSolver3D((HEIGHT,WIDTH), WAVELENGTHS, PIXELSIZE, method= "4x4", betamax = 0.9,resolution = 1)


field_in = field_data_in[0]
field_out = np.zeros_like(field_in)

for i in [1]:
    
    solver = ScatteringBlockSolver3D((HEIGHT,WIDTH), WAVELENGTHS, PIXELSIZE, method= "4x4", betamax = 0.9,resolution = 1)
    
    solver.set_optical_block((optical_block[0], epsv_eff + deltaeps*i , epsa), resize = 0)
    solver.calculate_stack_matrix()
    solver.calculate_field_matrix(nin = 1.5, nout = 1.5)
    solver.calculate_reflectance_matrix()
    solver.field_in = field_in
    print ('sds',field_in.mean())
    
    solver.transfer_field()
    solver.scatter_field()
    solver.field_out = field_out
    solver.field_in = field_in

    solver.transfer_field()

    #solver.scatter_field()
    #solver.transfer_field()     
    field_in = solver.field_in.copy() 




# solver.scatter_field()
# solver.transfer_field()
# solver.scatter_field()
# solver.transfer_field()
# solver.scatter_field()
# solver.transfer_field()
# solver.scatter_field()
# solver.transfer_field()
field_out = solver.field_out.copy()
field_in = solver.field_in.copy()

# for i in range(10):

#     solver.scatter_field()
#     scattered0 = solver.scattered_modes

#     zeros = np.zeros_like(field_out)

#     solver = ScatteringBlockSolver3D((HEIGHT,WIDTH), WAVELENGTHS, PIXELSIZE, method= "4x4", betamax = 0.9,resolution = 1)

#     solver.set_optical_block(optical_block, resize = 0)
#     solver.calculate_field_matrix(nin = 1.5, nout = 1.5)
#     solver.calculate_stack_matrix()
#     solver.calculate_reflectance_matrix()



#     field1_out = solver.transfer_field(zeros).copy()
#     solver.scattered_modes = scattered0 
#     field1_out = solver.transfer_field(zeros).copy()
#     field1_in = solver.field_in.copy()
    
#     field_out += field1_out
#     field_in += field1_in
    

field_data_in = field_in , WAVELENGTHS, PIXELSIZE
field_data_out = field_out, WAVELENGTHS, PIXELSIZE

scattered_field = solver.scattered_field, field_data_in[1], field_data_in[2]

#: transfer input light through stack
#field_data_out = dtmm.transfer_field(field_data_in, [optical_block], betamax = 1, diffraction = 1)
#: visualize output field
#viewer1 = dtmm.pom_viewer(field_data_out, magnification = 100, d_cover = 0, NA = 0.9)

viewer3 = dtmm.field_viewer(field_data_in, mode = 't', n = 1.5)
viewer3.set_parameters(polarizer = "v", analyzer = "h", focus = -18)
fig,ax = viewer3.plot()

viewer1 = dtmm.field_viewer(field_data_in, mode = 'r', n = 1.5)
viewer1.set_parameters(polarizer = "v", analyzer = "h", focus = -18)
fig,ax = viewer1.plot()

viewer2 = dtmm.field_viewer(field_data_out, mode = 't', n = 1.5)
viewer2.set_parameters(polarizer = "v", analyzer = "h", focus = -18)
fig,ax = viewer2.plot()



