"""
Same as 
"""

import dtmm
import numpy as np
dot = dtmm.linalg.dotmm
dotd = dtmm.linalg.dotmd
dotmdm = dtmm.linalg.dotmdm
dotmv = dtmm.linalg.dotmv

#:thickness of cholesteric layer in microns
thickness = 4
#: number of layers (should be high enough...) 
nlayers = 100

PIXELSIZE = 1000*thickness/nlayers


nwavelengths = 100
#which wavelengths to compute
wavelengths = np.linspace(400,700, nwavelengths)
#wavelengths = [400,450]
# input layer ref. index 
nin = [1.]*3
# output refractive index
nout = [1.]*3
#:ordinary refractive index of cholesteric
no = 1.5
#:extraordinary
ne = 1.62

step = thickness*1000/nlayers #in nanometers

phi =  0.+np.linspace(0, np.pi/2, nlayers)
eps_angles = np.zeros(shape = (nlayers,3), dtype = "float32") #in case we compiled for float32, this has to be float not duouble
eps_angles[:,1] = np.pi/4 #theta angle - in plane director
eps_angles[:,2] = phi

d = np.ones((nlayers,),"float32")#*step #* (thickness/nlayers)

n = [no,no,ne] 

#: layer thickness time wavenumber
#kd = 2*np.pi/wavelengths * step
#: input epsilon -air
eps_in = dtmm.refind2eps(nin)
#: layer epsilon
eps_layer = dtmm.refind2eps(n)
eps_layers = np.array([eps_layer]*(nlayers), dtype = "complex64")




#: ray beta parameters; beta is nin*np.sin(theta)
beta = 0.5
phi = 0.2

s = 1

window = dtmm.aperture((s,s))

eps_v = np.zeros(shape = (nlayers, s,s,3), dtype = eps_layer.dtype)
eps_a = np.zeros(shape = (nlayers, s,s,3), dtype = eps_angles.dtype)
eps_v[...] = eps_layers[:,None,None,:]
eps_a[...] = eps_angles[:,None,None,:]

optical_data = (d, eps_v, eps_a)

#: create non-polarized input light
field_data_in = dtmm.illumination_data((s,s), wavelengths, beta = beta, phi = phi, window = window,
                                            pixelsize = PIXELSIZE) 
#: transfer input light through stack
field_data_out = dtmm.transfer_field(field_data_in, optical_data, beta = beta, phi = phi, method = "2x2", reflection = 2, diffraction = False, npass =5)

#: visualize output field
viewer = dtmm.field_viewer(field_data_out)
viewer.set_parameters(sample = 0, intensity = 1,
                polarizer = 90, focus = -18, analyzer = 90)


#: create non-polarized input light
field_data_in = dtmm.illumination_data((s,s), wavelengths, beta = beta, phi = phi, window = window,
                                            pixelsize = PIXELSIZE) 
field_data_out = dtmm.transfer_field(field_data_in, optical_data, beta = beta, phi = phi, method = "4x4", reflection = 2, diffraction = False, npass =5)

#: visualize output field
viewer2 = dtmm.field_viewer(field_data_out)
viewer2.set_parameters(sample = 0, intensity = 1,
                polarizer = 90, focus = -18, analyzer = 90)

#fig,ax = viewer.plot()

s1 = viewer.calculate_specter()[s//2,s//2]
s2 = viewer2.calculate_specter()[s//2,s//2]

plt.plot(wavelengths, 2*s1,"--")
plt.plot(wavelengths, 2*s2,"-.")
