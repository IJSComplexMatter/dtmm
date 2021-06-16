"""
This script calculates reflected and transmitted waves from an input white-light
plane wave and calculates the microscope images in transmission mode and in
reflection mode. We also plot the reflection efficiency for one of the 
modes and transmission efficiency for the central mode.

"""
import dtmm
import numpy as np
import matplotlib.pyplot as plt

from dtmm import tmm2d, rotation, data, wave, field, tmm, solver

from cholesteric_grating import epsv, WAVELENGTHS, WIDTH, PIXELSIZE, field_data_in, NLAYERS, d,epsa, nin, nout

#add axis. We will work in 3D
epsv = epsv[...,None,:,:]#.real

# from dtmm.data import EpsilonCauchy
# epsc = EpsilonCauchy(shape = (NLAYERS,1,WIDTH), n = 2)
# epsc.coefficients[...,0] = (epsv)**0.5  # a term, just set to refractive index
# # very strong dispersion.
# b = 0.01
# epsc.coefficients[...,0:2,1] = b*(epsv[...,0:2])**0.5   # b term ordinary
# epsc.coefficients[...,2,1] = b*(epsv[...,2])**0.5  # b term extraordinary

# #normalize so that we have same refractive index at 550 nm
# epsc.coefficients[...,0] += ((epsv)**0.5 - epsc(550)**0.5)


beta, phi, intensity = 0,0,1


optical_data = [(d,epsv,epsa[...,None,:,:])] 

sim = solver.MatrixDataSolver3D((1,WIDTH), wavelengths = WAVELENGTHS, pixelsize = PIXELSIZE)
sim.set_optical_data(optical_data)
sim.calculate_stack_matrix(keep_layer_matrices = False, keep_stack_matrices = False)
sim.calculate_field_matrix(nin, nout)
sim.calculate_reflectance_matrix()
sim.transfer_field(field_data_in[0])

field_data_out = sim.get_field_data_out()

fmode_in  = sim.modes_in
fmode_out  = sim.modes_out
fmatin = sim.field_matrix_in
fmatout = sim.field_matrix_out

cols = 1
rows = WIDTH


ax1 = plt.subplot(121)
ax2 = plt.subplot(122)


t_rcp = []
r_rcp = []

t_lcp = []
r_lcp = []
ws = []

mode = 3

for fin,fout, w, f0in,f0out in zip(fmode_in, fmode_out, WAVELENGTHS, fmatin, fmatout):
            
    fr_rcp = fin[0] - 1j * fin[1] #right handed
    ft_rcp  = fout[0] - 1j * fout[1]
    fr_lcp = fin[0] + 1j * fin[1] 
    ft_lcp  = fout[0] + 1j * fout[1]    
    i_rcp = tmm.intensity(fr_rcp[0])
    i_lcp = tmm.intensity(fr_lcp[0])
    try:
        r_rcp.append(tmm.intensity(fr_rcp[mode])/i_rcp)
        t_rcp.append(tmm.intensity(ft_rcp[0])/i_rcp)
        
        r_lcp.append(tmm.intensity(fr_lcp[mode])/i_lcp)
        t_lcp.append(tmm.intensity(ft_lcp[0])/i_lcp)
        ws.append(w)
    except IndexError:
        #nonexistent mode.. break
        break
    
ax1.plot(ws,r_rcp, label = "rcp mode {} ".format(mode))
ax1.plot(ws,r_lcp, label = "lcp mode {} ".format(mode))
ax2.plot(ws,t_rcp, label = "rcp mode 0")
ax2.plot(ws,t_lcp, label = "lcp mode 0")

ax1.set_title("reflection")
ax1.set_xlabel("wavelength")
ax2.set_title("transmission")
ax2.set_xlabel("wavelength")

ax1.legend()
ax2.legend()

viewer1 = dtmm.field_viewer(field_data_in, mode = "r", n = 1.5, focus = 0,intensity = 1, cols = cols,rows = rows, analyzer = "h")
viewer2 = dtmm.field_viewer(field_data_out, mode = "t", n = 1.5, focus = 0,intensity = 1, cols = cols,rows = rows, analyzer = "h")
viewer3 = dtmm.field_viewer(field_data_in, mode = "t", n = 1.5, focus = 0,intensity = 1, cols = cols,rows = rows, analyzer = "h")

fig,ax = viewer1.plot()
ax.set_title("Reflected field")

fig,ax = viewer2.plot()
ax.set_title("Transmitted field")

fig,ax = viewer3.plot()
ax.set_title("Input field")

plt.show()