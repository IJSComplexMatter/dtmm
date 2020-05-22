"""
An example on standard 4x4 berreman for calculation of reflection
and transmission coefficient for p and s polarizations off a single layer deposited on top of a
substrate with refractive index nout. Calculation is done for multiple input angles
(beta parameters).

See also berreman_reflection_reference for alternative implementation. 
"""

import dtmm
import numpy as np
from dtmm import tmm, jones

#: wavelength of light in vacuum in nanometers
wavelength = 550 
#: thickness of layer in microns
thickness = 2.
#: refractive indices of the anisotropic layer
n = [ 1.5,1.5,1.5] 
#n = [np.sqrt(-10+0.32j)]*3#silver
#: euler angles of the optical axes (will make effect for anisotropic)
eps_angles = np.array([0,0.2,0.4], dtype = "float32") #in case we compiled for float32, this has to be float not duouble
#: refractive indices if the substrate
nout = [   1.]*3
#: phase retardation - layer thickness times wavenumber
kd = 2*np.pi/wavelength * thickness * 1000. 
#: input epsilon -air
eps_in = dtmm.refind2eps([1.,1.,1.])
#: layer epsilon
eps_layer = dtmm.refind2eps(n)
#: substrate epsilon
eps_out = dtmm.refind2eps(nout)
#: ray beta parameters; beta is nin*np.sin(theta)
betas = np.array(np.linspace(0.0,0.9999999,1000),dtype = "float32") #in case we compiled for float32, this has to be float not duouble
#: phi angle of the input light - will make a diffference for anisotropic layer
phi = 0

_phi = 0

pol1 = (-np.sin(_phi),np.cos(_phi))
pol2 = (np.cos(_phi),np.sin(_phi))


#: we must view them in rotated frame, with respect to the ray phi value.
pol1 = jones.jonesvec(pol1, phi) 
pol2 = jones.jonesvec(pol2, phi) 


#build field matrices
a,f,fi = tmm.alphaffi(betas,phi,eps_layer, eps_angles)
aout,fout,g = tmm.alphaffi(betas,phi,eps_out, eps_angles) #eps_angles does nothing because eps_out is isotropic
ain,fin,g = tmm.alphaffi(betas,phi,eps_in, eps_angles)#eps_angles does nothing because eps_in is isotropic

dot = dtmm.linalg.dotmm
dotd = dtmm.linalg.dotmd
dotmdm = dtmm.linalg.dotmdm
dotmv = dtmm.linalg.dotmv
intensity = tmm.intensity
intensity = tmm.poynting
p = tmm.phase_mat(a,-kd)
#p = np.exp(-1j*a*kd)

#characteristic matrix
cmat = dotmdm(f,p,fi)

#field projection matrices - used to take the forward propagating or backward propagating waves
pmat = tmm.projection_mat(fout,mode = +1)
mmat = tmm.projection_mat(fin, mode = -1)
pmatin = tmm.projection_mat(fin,mode = +1)
#: build EM field 4-vector (Ex, Hy, Ey, Hx) of a given polarization
fvec = tmm.fvec(fin,jones = pol1)
fvec = dotmv(pmatin,fvec)

#: transmit the field and update fvec with reflected waves
tfvec = tmm.transmit(fvec, cmat, fmatin = fin, fmatout = fout)
#: take the backward propagating waves in input field
rfvec = dotmv(mmat,fvec)
tfvec = dotmv(pmat,tfvec) #no need to do this.. there is no backpropagating waves in the output
tfvecin = dotmv(pmatin,fvec)

y_polarizer = tmm.polarizer4x4(pol1, fout) #y == s polarization
x_polarizer = tmm.polarizer4x4(pol2, fout) #x == p polarization

y_polarizerin = tmm.polarizer4x4(pol1, fin) #y == s polarization
x_polarizerin = tmm.polarizer4x4(pol2, fin) #x == p polarization


Rss = -intensity(dotmv(y_polarizerin,rfvec))
Rps = -intensity(dotmv(x_polarizerin,rfvec))
Tss = intensity(dotmv(y_polarizer,tfvec))
Tps = intensity(dotmv(x_polarizer,tfvec))
Tin1 = intensity(dotmv(x_polarizer,tfvecin))
Tin2 = intensity(dotmv(y_polarizer,tfvecin))

#now do the same for x polarization
fvec = tmm.fvec(fin,jones = pol2)
tfvec = tmm.transmit(fvec, cmat, fmatin = fin, fmatout = fout)
rfvec = dotmv(mmat,fvec)
#tfvec = dotmv(pmat,tfvec)
Tsp = intensity(dotmv(y_polarizer,tfvec))
Tpp = intensity(dotmv(x_polarizer,tfvec))
Rsp = -intensity(dotmv(y_polarizerin,rfvec))
Rpp = -intensity(dotmv(x_polarizerin,rfvec))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    plt.subplot(121)

    plt.plot(betas,Rss, label = "Rss")
    plt.plot(betas,Tss, label = "Tss")
    plt.plot(betas,Rps, label = "Rps")
    plt.plot(betas,Tps, label = "Tps")
    #plt.plot(betas,Tin1, label = "T1")
    #plt.plot(betas,Tin2, label = "T2")
    
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
