"""
A low level example on standard 4x4 berreman for calculation of reflection
and transmission coefficient for p and s polarizations off a single layer deposited on top of a
substrate with refractive index nout. 

This is a reference implementation... for testing. Reflection coefficients
are computed with eigenfield amplitudes instead of total field as done
in berreman_reflection_simple. Results of both calculation must be identical
for any type of material parameters.
"""

import dtmm
import numpy as np

from tmm_reflection import betas, phi, eps_layer, eps_angles, eps_in, eps_out, kd


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
pin = dtmm.tmm.fmat2poynting(fin)
pout = dtmm.tmm.fmat2poynting(fout)

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

    plt.subplot(121)

    plt.plot(betas,Rss, label = "Rss")
    plt.plot(betas,Tss, label = "Tss")
    plt.plot(betas,Rps, label = "Rps")
    plt.plot(betas,Tps, label = "Tps")
    
    plt.plot(betas,Tss+Rss+Tps+Rps, "--", label = "T+R")
    plt.legend()
    plt.xlabel("beta")
    
    plt.subplot(122)
    
    plt.plot(betas,Rpp, label = "Rpp")    
    plt.plot(betas,Tpp, label = "Tpp")
    plt.plot(betas,Rsp, label = "Rsp")
    plt.plot(betas,Tsp, label = "Tsp")
    
    plt.plot(betas,Tpp+Rpp + Rsp + Tsp, "-.",label = "T+R")
    
    #plt.plot(betas,r+t, "--", label = "T+R")
    
    plt.xlabel("beta")
    
    plt.legend()
