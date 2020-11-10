"""
Same as tmm_twisted_nematic.py, but computed with transfer_field function. Instead
of computing the characteristic 4x4 and 2x2 matrices for a stack of layers we 
build a single pixel optical data and compute the transmission with 2x2 method
and diffraction disabled.

It is important to have diffraction disabled, as otherwise
the method perpforms mode filtering thorugh fft. FFT on single pixel data has only
one mode (beta = 0) and therefore to compute reflections correctly it is important
that diffraction is disabled both in illuminnation_data and transfer_field functions.

It has to be stressed that both results (before applying output polarizer) should
be equal. However, since the field_viewer uses a fft mode polarizer to calculate
specter and since there is only beta = 0 mode in single pixel data
it is not 100% exact for off-axis input filed with beta > 0. In this example we
compute power from the computed specter without the field_viewer.
"""

import dtmm
import numpy as np

from tmm_twisted_nematic import step, d ,eps_values, eps_angles, wavelengths,\
    beta, phi, x_polarizer, y_polarizer
from dtmm.field import field2poynting
from dtmm.linalg import dotmf

nlayers = len(d)
#add axes to eps arrays.. to make a single pixel optical data
optical_data = (d[1:-1], eps_values[1:-1][:,None,None,:], eps_angles[1:-1][:,None,None,:])

optical_data = (d, eps_values[:,None,None,:], eps_angles[:,None,None,:])

#: create non-polarized input light
field_data_in = dtmm.field.illumination_data((1,1), wavelengths, beta = beta, phi = phi, diffraction = False,
                                            pixelsize = step) 
#: transfer input light through stack
field_data_out = dtmm.transfer_field(field_data_in, optical_data, beta = beta, phi = phi, method = "2x2", reflection = 2, diffraction = False)
f,w,p = field_data_out


Txx2 = field2poynting(dotmf(x_polarizer,f[0]))[...,0,0]*2 #times two, because field_data_in[0][0] has intensity of 0.5
Tyx2 = field2poynting(dotmf(y_polarizer,f[0]))[...,0,0]*2
Txy2 = field2poynting(dotmf(x_polarizer,f[1]))[...,0,0]*2#times two, because field_data_in[0][1] has intensity of 0.5
Tyy2 = field2poynting(dotmf(y_polarizer,f[1]))[...,0,0]*2

# uncomment below to see how the viewer calculates.. it is same for beta = 0, 
# but slightly different for beta > 0

#viewer = dtmm.field_viewer(field_data_out, diffraction = False)
#Txx2 = viewer.calculate_specter(polarizer = 0, analyzer = 0)[0,0]*2 
#Tyy2 = viewer.calculate_specter(polarizer = 90, analyzer = 90)[0,0]*2
#Txy2 = viewer.calculate_specter(polarizer = 90, analyzer = 0)[0,0]*2
#Tyx2 = viewer.calculate_specter(polarizer = 0, analyzer = 90)[0,0]*2


field_data_out = dtmm.transfer_field(field_data_in, optical_data, beta = beta, phi = phi, method = "2x2", reflection = 2, diffraction = False, npass =5)
f,w,p = field_data_out

Txx2r = field2poynting(dotmf(x_polarizer,f[0]))[...,0,0]*2
Tyx2r = field2poynting(dotmf(y_polarizer,f[0]))[...,0,0]*2
Txy2r = field2poynting(dotmf(x_polarizer,f[1]))[...,0,0]*2
Tyy2r = field2poynting(dotmf(y_polarizer,f[1]))[...,0,0]*2

# uncomment below to see how the viewer calculates.. it is same for beta = 0, 
# but slightly different for beta > 0

#viewer = dtmm.field_viewer(field_data_out, diffraction = False)
#Txx2r = viewer.calculate_specter(polarizer = 0, analyzer = 0)[0,0]*2
#Tyy2r = viewer.calculate_specter(polarizer = 90, analyzer = 90)[0,0]*2
#Txy2r = viewer.calculate_specter(polarizer = 90, analyzer = 0)[0,0]*2
#Tyx2r = viewer.calculate_specter(polarizer = 0, analyzer = 90)[0,0]*2


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.subplot(121)

    plt.plot(wavelengths, Txx2, label = "Txx2")

    plt.plot(wavelengths, Tyx2, label = "Tyx2")

    plt.plot(wavelengths, Txx2r, label = "Txx2r")   
    plt.plot(wavelengths, Tyx2r, label = "Tyx2r")
    plt.legend()
    
    plt.subplot(122)

    plt.plot(wavelengths, Txy2, label = "Txy2")  
    plt.plot(wavelengths, Tyy2, label = "Tyy2")

    plt.plot(wavelengths, Txy2r, label = "Txy2r")

    plt.plot(wavelengths, Tyy2r, label = "Tyy2r")

    
    plt.legend()