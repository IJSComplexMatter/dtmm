"""
A jones calculus example. 

You can use the package for simple normal incidence Jones calculus. In the 
:mod:`dtmm.jones` you will find all the functionality to work with jones calculus. 
For example, we can compute the transmittance properties of a simple Twisted 
Nematic director profile. We compute wavelength-dependent transmittance of 
normally white and normally black TN modes. We build a left-handed twisted 
nematic in a 4 microns cell and in the first minimum condition - max transmission 
at 550 nm. 

The matrix creation functions in the jones module obey numpy broadcasting rules,
so we can build and multiply matrices at different wavelengths simultaneously. 
See the source code of the example below.
"""

import numpy as np
from dtmm import jones
import matplotlib.pyplot as plt

#---------------------------
# user options
#---------------------------

#:thickness of LC cell in microns
thickness = 4
#: number of layers (should be high enough...) 
nlayers = 100
#: which wavelengths to compute (in nanometers)
k = np.linspace(2*np.pi/700 ,2*np.pi/400, 200)
wavelengths = 2*np.pi/k
#:ordinary refractive index of LC
no = 1.5
#:extraordinary
ne = 1.62

#---------------
# implementation
#---------------

step = thickness*1000/nlayers #in nanometers

x_jvec = jones.jonesvec((1,0))
y_jvec = jones.jonesvec((0,1))

phis =  np.linspace(0, np.pi/2, nlayers) #twist rotation angle
phase = (ne - no) * k * step #phase retardation in each of the layers

matrices = [jones.polarizer(x_jvec)] #x polarizer

#add retarders... left handed TN
for phi in phis:
    matrices.append(jones.retarder(phase, phi))
 
#next, we multiply matrices together in reverse order ...tn2.tn1.tn0.x      
jmat = jones.multi_dot(matrices, reverse = True)   
 
normally_white_jmat = jones.dotmm(jones.polarizer(y_jvec), jmat) #crossed polarizers
normally_black_jmat = jones.dotmm(jones.polarizer(x_jvec), jmat) #parallel polarizers

nw_jvec = jones.dotmv(normally_white_jmat, x_jvec)
nb_jvec = jones.dotmv(normally_black_jmat, x_jvec)

plt.title("First minimum TN transmittance")
plt.plot(wavelengths, jones.jones_intensity(nw_jvec), label = "Normally white mode")
plt.plot(wavelengths, jones.jones_intensity(nb_jvec), label = "Normally black mode")
plt.legend()
plt.xlabel("wavelength")
plt.ylabel("transmittance")


