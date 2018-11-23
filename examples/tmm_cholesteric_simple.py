"""
A low level example on standard 4x4 berreman for calculation of reflection
and transmission coefficient for p and s polarizations off a 
left-handed cholesteric structure - a reflection mirror for right-handed light
"""

import dtmm
import numpy as np
dot = dtmm.linalg.dotmm
dotd = dtmm.linalg.dotmd
dotmdm = dtmm.linalg.dotmdm
dotmv = dtmm.linalg.dotmv


#: cholesteric pitch in nm
pitch = 350 #350*eff_refind=350*1.55 = 540
#:thickness of cholesteric layer in microns
thickness = 20
#: number of layers (should be high enough...) 
nlayers = 500

nwavelengths = 800
#which wavelengths to compute
wavelengths = np.linspace(400,700, nwavelengths)
# input layer ref. index 
nin = [1.5]*3
# output refractive index
nout = [1.5]*3
#:ordinary refractive index of cholesteric
no = 1.5
#:extraordinary
ne = 1.6 

step = thickness*1000/nlayers #in nanometers

ii = np.arange(nlayers)
phi =  2*np.pi/pitch*ii*step
eps_angles = np.zeros(shape = (nlayers,3), dtype = "float32") #in case we compiled for float32, this has to be float not duouble
eps_angles[...,1] = np.pi/2 #theta angle - in plane director
eps_angles[...,2] = phi

d = np.ones((nlayers,),"float32") * (thickness/nlayers)

n = [no,no,ne] 

#: layer thickness time wavenumber
kd = 2*np.pi/wavelengths * step
#: input epsilon -air
eps_in = dtmm.refind2eps(nin)
#: layer epsilon
eps_layer = dtmm.refind2eps(n)
eps_layers = np.array([eps_layer]*nlayers, dtype = "complex64")
#: substrate epsilon
eps_out = dtmm.refind2eps(nout)
#: ray beta parameters; beta is nin*np.sin(theta)
beta = 0.3
phi = 0.4

#stack = d,eps_layers, eps_angles

#cmat = dtmm.tmm.stack_mat(stack,kd, beta = beta, phi = phi)

#build field matrices
a,f,fi = dtmm.tmm.alphaffi(beta,phi,eps_layers, eps_angles)
aout,fout,fiout = dtmm.tmm.alphaffi(beta,phi,eps_out, eps_angles[0]) #eps_angles does nothing because eps_out is isotropic
ain,fin,fiin = dtmm.tmm.alphaffi(beta,phi,eps_in, eps_angles[0])#eps_angles does nothing because eps_in is isotropic

#now build layer matrices

#: we are propagating backward--- minus sign must be taken in the phase
p = dtmm.tmm.phase_mat(a,-kd[...,None])
#characteristic matrix
m = dotmdm(f,p,fi)

#we could have built layer matrices directly, not the ommision of negative value in front of kd:
#m = dtmm.tmm.layer_mat(kd[...,None],eps_layers, eps_angles)

#cmat = m[:,0].copy()
#for i in range(nlayers-1):
#    cmat = dot(cmat, m[:,i+1])

# multiply over second axis (first axis is wavelenght)
cmat = dtmm.linalg.multi_dot(m, axis = 1) 


jleft = dtmm.jonesvec((1,1j))
jright = dtmm.jonesvec((1,-1j))


#field projection matrices - used to take the forward propagating or backward propagating waves
pmat = dtmm.tmm.projection_mat(fout,mode = +1)
mmat = dtmm.tmm.projection_mat(fin, mode = -1)

fvec = dtmm.tmm.field4(fin,jones = jleft)
fvec = np.array([fvec]*nwavelengths)

tfvec = dtmm.tmm.transmit(fvec, cmat, fmatin = fin[None,...], fmatout = fout[None,...])
rfvec = dotmv(mmat,fvec)
#tfvec = dotmv(pmat,tfvec) #no need to do this.. there is no backpropagating waves in the output

l_polarizerin = dtmm.tmm.polarizer4x4(fin,jleft, phi) #x == s polarization
r_polarizerin = dtmm.tmm.polarizer4x4(fin,jright, phi) #y == p polarization
l_polarizer = dtmm.tmm.polarizer4x4(fout,jleft, phi) #x == s polarization
r_polarizer = dtmm.tmm.polarizer4x4(fout,jright, phi) #y == p polarization

Rll = dtmm.tmm.intensity(dotmv(l_polarizerin,rfvec))
Rrl = dtmm.tmm.intensity(dotmv(r_polarizerin,rfvec))
Tll = dtmm.tmm.intensity(dotmv(l_polarizer,tfvec))
Trl = dtmm.tmm.intensity(dotmv(r_polarizer,tfvec))


fvec = dtmm.tmm.field4(fin,jones = jright)
fvec = np.array([fvec]*nwavelengths)


tfvec = dtmm.tmm.transmit(fvec, cmat, fmatin = fin[None,...], fmatout = fout[None,...])
rfvec = dotmv(mmat,fvec)
#tfvec = dotmv(pmat,tfvec) #no need to do this.. there is no backpropagating waves in the output

Rlr = dtmm.tmm.intensity(dotmv(l_polarizerin,rfvec))
Rrr = dtmm.tmm.intensity(dotmv(r_polarizerin,rfvec))
Tlr = dtmm.tmm.intensity(dotmv(l_polarizer,tfvec))
Trr = dtmm.tmm.intensity(dotmv(r_polarizer,tfvec))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    plt.subplot(211)

    plt.plot(wavelengths,Rll, label = "Rll")
    plt.plot(wavelengths,Rrl, label = "Rrl")
    plt.plot(wavelengths,Tll, label = "Tll")
    plt.plot(wavelengths,Trl, label = "Trl")
    
    plt.plot(wavelengths,Rll+Tll+Trl+Rrl, "--", label = "T+R")
    
    plt.legend(loc = 5)

    plt.subplot(212)
    plt.plot(wavelengths,Rlr, label = "Rlr")
    plt.plot(wavelengths,Rrr, label = "Rrr")
    plt.plot(wavelengths,Tlr, label = "Tlr")
    plt.plot(wavelengths,Trr, label = "Trr")
       
    plt.plot(wavelengths,Rlr+Tlr+Trr+Rrr, "--", label = "T+R")
    
    plt.xlabel("wavelength")
    
    plt.legend(loc = 5)
