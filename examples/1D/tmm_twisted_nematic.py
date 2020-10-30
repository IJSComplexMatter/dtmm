"""
An example on standard 4x4 berreman and 2x2 jones method (with and without
reflections) for calculation of reflection and transmission coefficient 
for x and y polarizations off a left-handed twisted nematic in a 4 microns cell
and in first minimum condition - max transmission at 550 nm.

A a typical LCD, the LC material is covered by a thick glass. Thick glass 
cannot be simulated by the 4x4 method, so here we show results of transmission
through a LC film in air (input and output refractive index is 1). The 4x4
gives strong interference, while the 2x2 method with reflections inlcudes single
reflection from both of the interfaces. The 2x2 method wothoud reflections
works poorly at high beta values, because depolarization of the field coming from
air to LC is neglected, works good for beta close to zero though.
"""

import dtmm
import numpy as np
from dtmm import tmm, linalg, jones4

CDTYPE = dtmm.conf.CDTYPE
FDTYPE = dtmm.conf.FDTYPE

dtmm.conf.set_verbose(2)

dot = linalg.dotmm
dotd = linalg.dotmd
dotmdm = linalg.dotmdm
dotmv = linalg.dotmv

#---------------------------
# user options
#---------------------------

#which method to use. 
method = "4x4" # 4x4 full interference 
#method = "4x4_1" #4x4 with single reflections
#method = "2x2" #2x2 no reflections
#method = "2x2_1" #2x2 single reflections

#:thickness of LC cell in microns
thickness = 4
#: number of layers (should be high enough...) 
nlayers = 100
#: which wavelengths to compute

k = np.linspace(2*np.pi/700 ,2*np.pi/400, 200)
wavelengths = 2*np.pi/k
# input layer ref. index 
nin = 1.
# output refractive index
nout = 1.
#:ordinary refractive index of cholesteric
no = 1.5
#:extraordinary
ne = 1.62
#: ray beta parameters; beta is nin*np.sin(theta)
beta = 0.
phi = 0

nglass = 1.
dglass = 100


nwavelengths = len(wavelengths)

step = thickness*1000/nlayers #in nanometers


_phi =  np.linspace(0, np.pi/2, nlayers)
eps_angles = np.zeros(shape = (nlayers+2,3), dtype = FDTYPE) #in case we compiled for float32, this has to be float not duouble
eps_angles[1:-1,1] = np.pi/2 #theta angle - in plane director
eps_angles[1:-1,2] = _phi

d = np.ones((nlayers+2,),FDTYPE) 
d[0] = dglass #first layer is glass thickness
d[-1] = dglass #last too...

n = [no,no,ne] 

#: layer thickness times wavenumber
kd = 2*np.pi/wavelengths[None,:]* step * d[:,None]
#: input epsilon -air
eps_in = dtmm.refind2eps([nin]*3)
#: layer epsilon
eps_layer = dtmm.refind2eps(n)
eps_values = np.array([eps_layer]*(nlayers+2), dtype = CDTYPE)
eps_values[0] =  dtmm.refind2eps([nglass]*3)
eps_values[-1] =  dtmm.refind2eps([nglass]*3)


jx = jones4.jonesvec((1,0), phi)
jy = jones4.jonesvec((0,1), phi)

fin = tmm.f_iso(n = nin)
fout = tmm.f_iso(n = nout)

pmat = tmm.projection_mat(fout,mode = +1)
mmat = tmm.projection_mat(fin, mode = -1)

fvec = tmm.fvec(fin,jones = jx)
rfvec4 = np.array([fvec]*nwavelengths)
tfvec4 = tmm.transfer(rfvec4, kd, eps_values, eps_angles, nin = nin, nout = nout,method = method)
rfvec4 = dotmv(mmat,rfvec4)

x_polarizerin = jones4.polarizer4x4(jx, fin) 
y_polarizerin = jones4.polarizer4x4(jy, fin) 
x_polarizer = jones4.polarizer4x4(jx, fout) 
y_polarizer = jones4.polarizer4x4(jy, fout) 

intensity = tmm.intensity

Rxx4 = intensity(dotmv(x_polarizerin,rfvec4))
Ryx4 = intensity(dotmv(y_polarizerin,rfvec4))
Txx4 = intensity(dotmv(x_polarizer,tfvec4))
Tyx4 = intensity(dotmv(y_polarizer,tfvec4))

fvec = tmm.fvec(fin,jones = jy)
rfvec4 = np.array([fvec]*nwavelengths)
tfvec4 = tmm.transfer(rfvec4, kd, eps_values, eps_angles,nin = nin, nout = nout, method = method)
rfvec4 = dotmv(mmat,rfvec4)

Rxy4 = intensity(dotmv(x_polarizerin,rfvec4))
Ryy4 = intensity(dotmv(y_polarizerin,rfvec4))
Txy4 = intensity(dotmv(x_polarizer,tfvec4))
Tyy4 = intensity(dotmv(y_polarizer,tfvec4))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    plt.subplot(121)

    plt.plot(wavelengths,Rxx4, label = "Rxx")
    plt.plot(wavelengths,Ryx4, label = "Ryx")
    plt.plot(wavelengths,Txx4, label = "Txx")
    plt.plot(wavelengths,Tyx4, label = "Tyx")
    plt.plot(wavelengths,Rxx4+Txx4+Tyx4+Ryx4, "--", label = "T+R")
    plt.xlabel("wavelength")
    plt.legend(loc = 5)

    plt.subplot(122)
    plt.plot(wavelengths,Rxy4, label = "Rxy")
    plt.plot(wavelengths,Ryy4, label = "Ryy")
    plt.plot(wavelengths,Txy4, label = "Txy")
    plt.plot(wavelengths,Tyy4, label = "Tyy")    
    plt.plot(wavelengths,Rxy4+Txy4+Tyy4+Ryy4, "--", label = "T+R")    
    plt.xlabel("wavelength")
    
    plt.legend(loc = 5)


