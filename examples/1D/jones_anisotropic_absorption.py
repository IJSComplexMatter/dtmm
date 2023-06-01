"""
"""

import numpy as np
from dtmm import jones, tmm
import matplotlib.pyplot as plt
from dtmm.linalg import dotmv
from dtmm.data import director2Q, Q2eps, eps2epsva

#---------------------------
# user options
#---------------------------

#: total thickness of the cell in microns
thickness = 100
#: in nanometers
wavelength = 633
#: k value in inverse microns
k = 2*np.pi/ wavelength *1000
#: the kd term of layers. We have a single layer.
kd = [k * thickness]


# absorption coefficients, the complex part of the dielectric constant
sigma_o = 0.0005
sigma_e = 0.0000001

# refractive index of the material with no absorption
refractive_index = 1.5
# birefringence of the material with no absorption
anisotropy = 0.011

epsv_o = refractive_index**2
epsv_e = (refractive_index + anisotropy)**2

# complex refractive indices of fully ordered and absorbing material
no = np.sqrt(epsv_o + 1j * sigma_o)
ne = np.sqrt(epsv_e + 1j * sigma_e)

def order_parameter(b):
    """Returns the order parameter from the normalized field strength"""
    b = np.asarray(b)
    out = (3 + b**2 - 3*b/np.tanh(b))/(b**2)
    if b.ndim == 0:
        if b == 0:
            return 0.
        else:
            return out
    else:
        mask = (b == 0)
        out[mask] = 0
        return out


def material(director, s = 1., no = 1.5, ne = 1.6):
    """Returns material epsv and epsa parameters"""
    # now convert the director to Q tensor using a given order parameter ranging from 0 to 1
    # force the Q tensor to be complex because we use complex no and ne later
    Q = director2Q(director, order = s) + 0j
    #convert the Q tensor to a uniaxial eps tensor
    eps = Q2eps(Q, no = no, ne = ne)
    # convert the eps tensor to eigenframe and euler angles
    epsv,epsa = eps2epsva(eps)
    return epsv, epsa
    

def transmittance(epsv, epsa, polarizer = 0, analyzer = 0):
       
    # input and output material (glass)
    nin = 1.5
    nout = 1.5
    #convert angle to jones vector
    analyzer = (np.cos(analyzer),np.sin(analyzer))
    jin = (np.cos(polarizer),np.sin(polarizer))
    
    # converts the analyzer jones vector to a jones polarizer matrix
    analyzer = jones.polarizer(analyzer) 
    
    # transmit the joines vector through the stack
    jout = tmm.transfer2x2(jin, kd, epsv, epsa, nin = nin, nout = nout)
    
    #multiply with polarizer matrix 
    jout = dotmv(analyzer,jout)
    #compute and return the intensity
    return (np.abs(jout)**2).sum(axis = -1)



import matplotlib.pyplot as plt

P = np.pi/4
A = -np.pi/4

P = -np.pi/4
A = np.pi/4
P=0
A=0

x = np.linspace(0,40,100)
s = order_parameter(x)
#no = np.asarray(no)#[None]
#ne = np.asarray(ne)#[None]

#define the director field. Must be a of shape (n,...,3) where n is number of layers
director = np.asarray([[(1,0,0)] * len(x)])

epsv, epsa = material(director, s = s, no = no, ne = ne)

y = transmittance(epsv,epsa, polarizer = P, analyzer = A)

plt.plot(x,y)


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
    
#     plt.subplot(121)

#     plt.plot(wavelengths,Rxx4, label = "Rxx")
#     plt.plot(wavelengths,Ryx4, label = "Ryx")
#     plt.plot(wavelengths,Txx4, label = "Txx")
#     plt.plot(wavelengths,Tyx4, label = "Tyx")
#     plt.plot(wavelengths,Rxx4+Txx4+Tyx4+Ryx4, "--", label = "T+R")
#     plt.xlabel("wavelength")
#     plt.legend(loc = 5)

#     plt.subplot(122)
#     plt.plot(wavelengths,Rxy4, label = "Rxy")
#     plt.plot(wavelengths,Ryy4, label = "Ryy")
#     plt.plot(wavelengths,Txy4, label = "Txy")
#     plt.plot(wavelengths,Tyy4, label = "Tyy")    
#     plt.plot(wavelengths,Rxy4+Txy4+Tyy4+Ryy4, "--", label = "T+R")    
#     plt.xlabel("wavelength")
    
#     plt.legend(loc = 5)