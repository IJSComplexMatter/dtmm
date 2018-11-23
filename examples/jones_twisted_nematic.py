"""
An example on standard 4x4 berreman and 2x2 jones method (with and without
reflections) for calculation of reflection and transmission coefficient 
for x and y polarizations off a left-handed twisted nematic in a 4 microns cell
and in first minimum condition - max transmission at 550 nm.

"""

import dtmm
import numpy as np
dot = dtmm.linalg.dotmm
dotd = dtmm.linalg.dotmd
dotmdm = dtmm.linalg.dotmdm
dotmv = dtmm.linalg.dotmv

#---------------------------
# user options
#---------------------------

#:thickness of cholesteric layer in microns
thickness = 4
#: number of layers (should be high enough...) 
nlayers = 100
#: which wavelengths to compute
wavelengths = np.linspace(400,700, 100)
# input layer ref. index 
nin = 1
# output refractive index
nout = 1
#:ordinary refractive index of cholesteric
no = 1.5
#:extraordinary
ne = 1.62
#: ray beta parameters; beta is nin*np.sin(theta)
beta = 0.4
phi = 0



#------------------------------------
# implementation
#------------------------------------

nwavelengths = len(wavelengths)

step = thickness*1000/nlayers #in nanometers

_phi =  0.+np.linspace(0, np.pi/2, nlayers)
eps_angles = np.zeros(shape = (nlayers+2,3), dtype = "float32") #in case we compiled for float32, this has to be float not duouble
eps_angles[1:-1,1] = np.pi/2 #theta angle - in plane director
eps_angles[1:-1,2] = _phi

d = np.ones((nlayers+2,),"float32") 
d[0] = 0 #first layer is zero thickness
d[-1] = 0 #last too...

n = [no,no,ne] 

#: layer thickness times wavenumber
kd = 2*np.pi/wavelengths[:,None]* step * d
#: input epsilon -air
eps_in = dtmm.refind2eps([nin]*3)
#: layer epsilon
eps_layer = dtmm.refind2eps(n)
eps_layers = np.array([eps_layer]*(nlayers+2), dtype = "complex64")
eps_layers[0] =  dtmm.refind2eps([nin]*3)
eps_layers[-1] =  dtmm.refind2eps([nout]*3)


#stack = d,eps_layers, eps_angles

#cmat = dtmm.tmm.stack_mat(stack,kd, beta = beta, phi = phi)

#build field matrices
a,f,fi = dtmm.tmm.alphaffi(beta,phi,eps_layers, eps_angles)

fout = f[-1]
fin = f[0]
#now build layer matrices

#: we are propagating backward--- minus sign must be taken in the phase
p = dtmm.tmm.phase_mat(a,-kd)
pr = dtmm.tmm.phase_mat(a,kd)
#characteristic matrix

e = dtmm.tmm.E_mat(f[1:], mode = +1)
ei = dtmm.linalg.inv(e)
eti = dtmm.tmm.Eti_mat(f[:-1], f[1:], mode = +1)
p2 = dtmm.tmm.phase_mat(a,kd, mode = +1)[:,1:]
m2 = dotmdm(e,p2,ei)
m2t = dotmdm(e,p2,eti)
m = dotmdm(f,p,fi)
mr = dotmdm(f,pr,fi)


#we could have built layer matrices directly, not the ommision of negative value in front of kd:
#m = dtmm.tmm.layer_mat(kd,eps_layers, eps_angles)

# multiply matrices together over second axis (first axis is wavelenght, second are layers)
cmat = dtmm.linalg.multi_dot(m, axis = 1) 
cmat2 = dtmm.linalg.multi_dot(m2, axis = 1, reverse =True) 
cmat2t = dtmm.linalg.multi_dot(m2t, axis = 1, reverse =True) 

#cmatr = dtmm.linalg.inv(cmat)

jx = dtmm.jonesvec((1,0))
jy = dtmm.jonesvec((0,1))

#field projection matrices - used to take the forward propagating or backward propagating waves
pmat = dtmm.tmm.projection_mat(fout,mode = +1)
mmat = dtmm.tmm.projection_mat(fin, mode = -1)

fvec = dtmm.tmm.field4(fin,jones = jx)
fvec = np.array([fvec]*nwavelengths)

tfvec2 = dtmm.tmm.transmit2x2(fvec, cmat2,fmatout = fout[None,...])
tfvec2r = dtmm.tmm.transmit2x2(fvec, cmat2t,fmatout = fout[None,...])

tfvec4 = dtmm.tmm.transmit(fvec, cmat, fmatin = fin[None,...], fmatout = fout[None,...])
rfvec4 = dotmv(mmat,fvec)

x_polarizer = dtmm.tmm.polarizer4x4(jx) 
y_polarizer = dtmm.tmm.polarizer4x4(jy) 

Rxx4 = dtmm.tmm.intensity(dotmv(x_polarizer,rfvec4))
Ryx4 = dtmm.tmm.intensity(dotmv(y_polarizer,rfvec4))
Txx4 = dtmm.tmm.intensity(dotmv(x_polarizer,tfvec4))
Tyx4 = dtmm.tmm.intensity(dotmv(y_polarizer,tfvec4))
Txx2 = dtmm.tmm.intensity(dotmv(x_polarizer,tfvec2))
Tyx2 = dtmm.tmm.intensity(dotmv(y_polarizer,tfvec2))
Txx2r = dtmm.tmm.intensity(dotmv(x_polarizer,tfvec2r))
Tyx2r = dtmm.tmm.intensity(dotmv(y_polarizer,tfvec2r))

fvec = dtmm.tmm.field4(fin,jones = jy)
fvec = np.array([fvec]*nwavelengths)

tfvec2 = dtmm.tmm.transmit2x2(fvec, cmat2,fmatout = fout[None,...])
tfvec2r = dtmm.tmm.transmit2x2(fvec, cmat2t,fmatout = fout[None,...])

tfvec4 = dtmm.tmm.transmit(fvec, cmat, fmatin = fin[None,...], fmatout = fout[None,...])
rfvec4 = dotmv(mmat,fvec)

Rxy4 = dtmm.tmm.intensity(dotmv(x_polarizer,rfvec4))
Ryy4 = dtmm.tmm.intensity(dotmv(y_polarizer,rfvec4))
Txy4 = dtmm.tmm.intensity(dotmv(x_polarizer,tfvec4))
Tyy4 = dtmm.tmm.intensity(dotmv(y_polarizer,tfvec4))
Txy2 = dtmm.tmm.intensity(dotmv(x_polarizer,tfvec2))
Tyy2 = dtmm.tmm.intensity(dotmv(y_polarizer,tfvec2))
Txy2r = dtmm.tmm.intensity(dotmv(x_polarizer,tfvec2r))
Tyy2r = dtmm.tmm.intensity(dotmv(y_polarizer,tfvec2r))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    plt.subplot(121)

    plt.plot(wavelengths,Rxx4, label = "Rxx4")
    plt.plot(wavelengths,Ryx4, label = "Ryx4")
    plt.plot(wavelengths,Txx4, label = "Txx4")
    plt.plot(wavelengths,Tyx4, label = "Tyx4")
    plt.plot(wavelengths,Txx2, label = "Txx2")
    plt.plot(wavelengths,Tyx2, label = "Tyx2")  
    plt.plot(wavelengths,Txx2r, "--", label = "Txx2r")
    plt.plot(wavelengths,Tyx2r, "--", label = "Tyx2r")    
    plt.plot(wavelengths,Rxx4+Txx4+Tyx4+Ryx4, "--", label = "T4+R4")
    plt.plot(wavelengths,Tyx2+Txx2, "--", label = "T2")
    plt.plot(wavelengths,Tyx2r+Txx2r, "--", label = "T2r")
    plt.xlabel("wavelength")
    plt.legend(loc = 5)

    plt.subplot(122)
    plt.plot(wavelengths,Rxy4, label = "Rxy4")
    plt.plot(wavelengths,Ryy4, label = "Ryy4")
    plt.plot(wavelengths,Txy4, label = "Txy4")
    plt.plot(wavelengths,Tyy4, label = "Tyy4")
    plt.plot(wavelengths,Txy2, label = "Txy2")
    plt.plot(wavelengths,Tyy2, label = "Tyy2")   
    plt.plot(wavelengths,Txy2r, "--", label = "Txy2r")
    plt.plot(wavelengths,Tyy2r, "--", label = "Tyy2r")        
    plt.plot(wavelengths,Rxy4+Txy4+Tyy4+Ryy4, "--", label = "T4+R4")
    plt.plot(wavelengths,Txy2+Tyy2, "--", label = "T2")
    plt.plot(wavelengths,Txy2r+Tyy2r, "--", label = "T2r")    
    plt.xlabel("wavelength")
    
    plt.legend(loc = 5)


