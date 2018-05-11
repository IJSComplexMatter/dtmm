"""EM field example"""

import dtmm
import numpy as np
import matplotlib.pyplot as plt


WAVELENGTHS = [500,600]
SIZE = (128,128)

window = dtmm.aperture((128,128))

field_data = dtmm.illumination_data(SIZE, WAVELENGTHS,window = window,jones = (1,0),
         pixelsize = 200, beta = (0,0.1,0.2), phi = (0.,0.,np.pi/6))

field = field_data[0]

#500nm

Ex = field[:,0,0] #Ex of the x-polarized light

subplots = (231,232,233)

for i,subplot in enumerate(subplots):
    plt.subplot(subplot)
    plt.imshow(Ex[i].real, origin = "lower")

#600nm

Ex = field[:,1,0] 

subplots = (234,235,236)

for i,subplot in enumerate(subplots):
    plt.subplot(subplot)
    plt.imshow(Ex[i].real, origin = "lower")
    
plt.show()
