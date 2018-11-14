"""
A low level example on standard 4x4 berreman for calculation of reflection
and transmission coefficient for p and s polarizations off a single layer deposited on top of a
substrate with refractive index nout. Calculation is done for multiple input angles
(beta parameters).
"""

import dtmm
import numpy as np

#: wavelength of light in vacuum in nanometers
wavelength = 550 #in nanometers
#: thickness of layer in microns
thickness = 2.
#: refractive indices of the layer
n = [1.5,1.5,1.6] #anisotropic 
#n = [np.sqrt(-10+0.32j)]*3#silver
#: euler angles of the optical axes (will make effect for anisotropic)
eps_angles = np.array([0,np.pi/2,np.pi/4], dtype = "float32") #in case we compiled for float32, this has to be float not duouble
#: refractive indices if the substrate
nout = [4.]*3
#: layer thickness time wavenumber
kd = 2*np.pi/wavelength * thickness * 1000.
#: input epsilon -air
eps_in = dtmm.refind2eps([1,1,1])
#: layer epsilon
eps_layer = dtmm.refind2eps(n)
#: substrate epsilon
eps_out = dtmm.refind2eps(nout)
#: ray beta parameters; beta is nin*np.sin(theta)
betas = np.array(np.linspace(0.0,0.99999,10000),dtype = "float32") #in case we compiled for float32, this has to be float not duouble
#: phi angle - will make a diffference for anisotropic layer
phi = 0

#build field matrices
a,f,fi = dtmm.tmm.alphaffi(betas,phi,eps_layer, eps_angles)
aout,fout,fiout = dtmm.tmm.alphaffi(betas,phi,eps_out, eps_angles) #eps_angles does nothing because eps_out is isotropic
ain,fin,fiin = dtmm.tmm.alphaffi(betas,phi,eps_in, eps_angles)#eps_angles does nothing because eps_in is isotropic


dot = dtmm.linalg.dotmm
dotd = dtmm.linalg.dotmd
dotmdm = dtmm.linalg.dotmdm
#p = dtmm.tmm.phasem(a,-kd)
p = np.exp(-1j*a*kd)

#characteristic matrix
cmat = dotmdm(f,p,fi)

m = dot(fiin,dot(cmat,fout))

#for isotropic input and output medium case
# this are the p- and s- amplitude reflection coefficients
det = m[...,0,0]*m[...,2,2]-m[...,0,2]*m[...,2,0]
rpp = (m[...,1,0]*m[...,2,2]-m[...,1,2]*m[...,2,0])/det
rss = (m[...,0,0]*m[...,3,2]-m[...,0,2]*m[...,3,0])/det
rps = (m[...,0,0]*m[...,1,2]-m[...,0,2]*m[...,1,0])/det
rsp = (m[...,2,2]*m[...,3,0]-m[...,2,0]*m[...,3,2])/det


tps = -m[...,0,2]/det
tsp = -m[...,2,0]/det
tpp = m[...,2,2]/det
tss = m[...,0,0]/det

#you need poynting vector to calculate total reflectance and transmittance
pin = dtmm.tmm.poynting(fin)
pout = dtmm.tmm.poynting(fout)


Rpp = np.abs(rpp)**2 *np.abs(pin[...,1]/pin[...,0])
Rss = np.abs(rss)**2 *np.abs(pin[...,3]/pin[...,2])
Rps = np.abs(rps)**2 *np.abs(pin[...,1]/pin[...,2])
Rsp = np.abs(rsp)**2 *np.abs(pin[...,3]/pin[...,0])

Tpp = np.abs(tpp)**2 *np.abs(pout[...,0]/pin[...,0])
Tss = np.abs(tss)**2 *np.abs(pout[...,2]/pin[...,2])
Tsp = np.abs(tsp)**2 *np.abs(pout[...,2]/pin[...,0])
Tps = np.abs(tps)**2 *np.abs(pout[...,0]/pin[...,2])

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.plot(betas,Rss, label = "Rss")
    plt.plot(betas,Rpp, label = "Rpp")
    plt.plot(betas,Tss, label = "Tss")
    plt.plot(betas,Tpp, label = "Tpp")
    plt.plot(betas,Rsp, label = "Rsp")
    plt.plot(betas,Rps, label = "Rps")
    plt.plot(betas,Tsp, label = "Tsp")
    plt.plot(betas,Tps, label = "Tps")
    
    plt.plot(betas,Tss+Rss+Tps+Rps, "--", label = "Tss+Rss+Tps+Rps")
    plt.plot(betas,Tpp+Rpp + Rsp + Tsp, "-.",label = "Tpp+Rpp+Tsp+Rsp")
    
    plt.xlabel("beta")
    
    plt.legend()
