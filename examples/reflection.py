
"""
A low level example on use of standard 4x4 berreman for calculation of reflextion
coefficient for p and s polarizations
"""

import dtmm
import numpy as np
import matplotlib.pyplot as plt

betas = np.array(np.linspace(0.00,0.9999,100),dtype = "float32")
phi = np.pi/4

n0 = 1.
n1 = np.sqrt(-10+0.32j) #silver

angles = np.array([0,0.2,0.0], dtype = "float32")

a,f,fi = dtmm.alphaffi_xy(betas,phi,angles,dtmm.refind2eps([n1,n1,n1]))
a0,f0,fi0 = dtmm.alphaffi_xy(betas,phi,angles,dtmm.refind2eps([n0,n0,n0]))

aiso,fiso,fiiso = dtmm.alphaffi_xy(betas,phi,angles,dtmm.refind2eps([n1,n1,n1]))
a0iso,f0iso,fi0iso = dtmm.alphaffi_xy(betas,phi,angles,dtmm.refind2eps([n0,n0,n0]))

dot = dtmm.linalg.dotmm
dotd = dtmm.linalg.dotmd

ad = dtmm.phasem(a,1)

m = dot(fi0iso,dot(dotd(f,ad),dot(fi,fiso)))

det = m[...,0,0]*m[...,2,2]-m[...,0,2]*m[...,2,0]

rpp = (m[...,1,0]*m[...,2,2]-m[...,1,2]*m[...,2,0])/det
rss = (m[...,0,0]*m[...,3,2]-m[...,0,2]*m[...,3,1])/det
tpp = m[...,2,2]/det
tss = m[...,0,0]/det

plt.plot(betas,np.abs(rss)**2)
plt.plot(betas,np.abs(rpp)**2)

