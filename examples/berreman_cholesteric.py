"""
A low level example on standard 4x4 berreman for calculation of reflection
and transmission coefficient for p and s polarizations off a 
double left- and right-hand cholesteric structure - a 100% reflection mirror
with pitch that correspond to 540nm 
"""

import dtmm
import numpy as np

#: cholesteric pitch in nm
pitch = 350 #350*eff_refind=350*1.55 = 540
#:thickness of cholesteric layer in microns
thickness = 20
#: number of layers (should be high enough...) 
nlayers = 1000
#which wavelengths to compute
wavelengths = np.linspace(400,700, 1000)
# input layer ref. index - we match it to cholesteric to avoid additional reflections
nin = [1.5]*3
# output refractive index
nout = [1.5]*3
#:ordinary refractive index of cholesteric
no = 1.5
#:extraordinary
ne = 1.6 

#build cholesteric
step = thickness*1000/nlayers
phi =  2*np.pi/pitch*np.arange(nlayers)*step
eps_angles = np.zeros(shape = (nlayers,3), dtype = "float32") #in case we compiled for float32, this has to be float not duouble
eps_angles[...,1] = np.pi/2 #theta angle - in plane director
eps_angles[:nlayers//2,2] = phi[:nlayers//2]
eps_angles[nlayers//2:,2] = -phi[nlayers//2:]

n = [no,no,ne] 

#: layer thickness time wavenumber
kd = 2*np.pi/wavelengths * step
#: input epsilon -air
eps_in = dtmm.refind2eps(nin)
#: layer epsilon
eps_layer = dtmm.refind2eps(n)
eps_layers = np.array([e for e in eps_layer])
#: substrate epsilon
eps_out = dtmm.refind2eps(nout)
#: ray beta parameters; beta is nin*np.sin(theta)
beta = 0.
phi = 0.

#build field matrices
a,f,fi = dtmm.tmm.alphaffi(beta,phi,eps_layers, eps_angles)
aout,fout,fiout = dtmm.tmm.alphaffi(beta,phi,eps_out, eps_angles[0]) #eps_angles does nothing because eps_out is isotropic
ain,fin,fiin = dtmm.tmm.alphaffi(beta,phi,eps_in, eps_angles[0])#eps_angles does nothing because eps_in is isotropic

dot = dtmm.linalg.dotmm
dotd = dtmm.linalg.dotmd
dotmdm = dtmm.linalg.dotmdm
p = dtmm.tmm.phasem(a,-kd[...,None])
#p = np.exp(-1j*a*kd)

#characteristic matrix
mmat = dotmdm(f,p,fi)

cmat = mmat[:,0].copy()
for i in range(nlayers-1):
    cmat = dot(cmat, mmat[:,i+1])

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

    plt.plot(wavelengths,Rss+Rps, label = "Rss+Rps")
    plt.plot(wavelengths,Rpp+Rsp, label = "Rpp+Rsp")
    plt.plot(wavelengths,Tpp+Tsp, label = "Tpp+Tsp")
    plt.plot(wavelengths,Tpp+Tsp, label = "Tpp+Tsp")
    
    plt.plot(wavelengths,Tss+Rss+Tps+Rps, "--", label = "Tss+Rss+Tps+Rps")
    plt.plot(wavelengths,Tpp+Rpp + Rsp + Tsp, "-.",label = "Tpp+Rpp+Tsp+Rsp")
    
    plt.xlabel("beta")
    
    plt.legend()
