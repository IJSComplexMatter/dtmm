"""Example on multiple reflections"""

import dtmm
import numpy as np
dtmm.conf.set_verbose(1)
import matplotlib.pyplot as plt

THICKNESS = 0.6
#: pixel size in nm
PIXELSIZE = 100
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 50,96,96
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(450,650,21)
#: some experimental data

eps = dtmm.refind2eps([1.5,1.5,1.6])
eps = np.broadcast_to(eps, (NLAYERS,HEIGHT, WIDTH,3))
angles = np.zeros_like(eps)
angles[...,1] = np.pi/2 #theta angle

for i in range(HEIGHT):
    angles[:,i,:,2] = np.linspace(0,(14.+6*i/HEIGHT)*np.pi,NLAYERS)[:,None] #theta angle
optical_data = dtmm.validate_optical_data(([THICKNESS]*NLAYERS, eps, angles))

eps = dtmm.refind2eps([1.5,1.5,1.6])
eps = np.broadcast_to(eps, (NLAYERS,3))
angles = np.zeros_like(eps)
angles[...,1] = np.pi/2 #theta angle
angles[...,2] = np.linspace(0,20*np.pi,NLAYERS) #theta angle
eff_data = dtmm.validate_optical_data(([THICKNESS]*NLAYERS, eps, angles), homogeneous = True)



beta =0.5#.5#[0.,0.5]
phi = 0#[0.,0.]


window = dtmm.aperture((HEIGHT, WIDTH),1,0.4)
#window = None
jones = dtmm.jonesvec((1,-1j))
#jones = dtmm.jonesvec((1,1))

field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, jones = jones, focus = 0*0.3,
                      beta = beta,phi = phi, pixelsize = PIXELSIZE, n = 1.5, window = window) 


f0 = field_data_in[0].copy()
#f[...,:] = f[...,:,:,:] * np.linspace(0,1,WIDTH)[None,None,:]

window = None#dtmm.aperture((HEIGHT, WIDTH),1,0.3)

field_data_out = dtmm.transfer.transfer_field(field_data_in, optical_data,diffraction = True,interference = True,
                                     beta = beta, phi = phi, nstep = 1,npass =2, nin = 1.5, nout = 1.5)


viewer = dtmm.field_viewer(field_data_out, mode = "t",intensity = 1, n = 1.5)
fig,ax = viewer.plot(imax = 100, fmax =100,fmin = -100)

#plt.figure()


