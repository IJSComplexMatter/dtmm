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
from dtmm import tmm, linalg, jones

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

#:thickness of LC cell in microns
thickness = 4
#: number of layers (should be high enough...) 
nlayers = 100
#: which wavelengths to compute

k = np.linspace(2*np.pi/700 ,2*np.pi/400, 1000)
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



#------------------------------------
# implementation
# we build layers and add the two additional layers for input layer -air and output layer -air
# but we scpedifie the thikcness of these layers to 0. We need this to calculate
# fresnell coefficients for the 2x2 method with reflections.
#------------------------------------

#here

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


jx = jones.jonesvec((1,0), phi)
jy = jones.jonesvec((0,1), phi)

fin = tmm.f_iso(n = nin)
fout = tmm.f_iso(n = nout)

pmat = tmm.projection_mat(fout,mode = +1)
mmat = tmm.projection_mat(fin, mode = -1)

fvec = tmm.fvec(fin,jones = jx)
rfvec4 = np.array([fvec]*nwavelengths)
tfvec4 = tmm.transfer(rfvec4, kd, eps_values, eps_angles, nin = nin, nout = nout,method = "4x4")
rfvec4 = dotmv(mmat,rfvec4)

#tfvec2r = (dtmm.tmm.transfer(fvec[i], kd[i], eps_values, eps_angles, method = "4x4") for i in range(100))

#tfvec2r = np.vstack(tuple(tfvec2r))



#
##stack = d,eps_values, eps_angles
#
##cmat = tmm.stack_mat(stack,kd, beta = beta, phi = phi)
#
##build field matrices
#a,f,fi = tmm.alphaffi(beta,phi,eps_values, eps_angles)
#
#fout = f[-1]
#fin = f[0]
#
##now build layer matrices
#
##: in 4x4 we are propagating backward--- minus sign must be taken in the phase
#p = tmm.phase_mat(a,-kd)
#
## uncoment this to test that setting backward propagating waves to zero, 
## you have single reflections only... -same result as 2x2 with reflection.
##p[...,1::2] = 0.
#
##4x4 characteristic matrix
#m = dotmdm(f,p,fi)
##we could have built layer matrices directly, note the there is no negative value in front of kd:
##m = tmm.layer_mat(kd,eps_values, eps_angles, beta = beta, phi = phi)
#
##e field 2x22 matrix.. skip the first layer (air)
#e = tmm.E_mat(f[1:], mode = +1)
## the inverse, no reflections
#ei = linalg.inv(e)
## the inverse, with reflections
#eti = tmm.Eti_mat(f[:-1], f[1:], mode = +1)
#p2 = tmm.phase_mat(a,kd, mode = +1)[:,1:]
#
##2x2 characteristic matrix
#m2 = dotmdm(e,p2,ei) #no reflections
#m2t = dotmdm(e,p2,eti) #with reflections
#
## multiply matrices together over second axis (first axis is wavelenght, second are layers)
#cmat = linalg.multi_dot(m, axis = 1) 
#
##the 2x2 matrices must be multiplied in reverse order... because we propagate forward
#cmat2 = linalg.multi_dot(m2, axis = 1, reverse = True) 
#cmat2t = linalg.multi_dot(m2t, axis = 1, reverse = True) 
#
#jx = jones.jonesvec((1,0), phi)
#jy = jones.jonesvec((0,1), phi)
#
##field projection matrices - used to take the forward propagating or backward propagating waves

#
#fvec = tmm.fvec(fin,jones = jx)
##fvec = tmm.field4old(fin,jones = jx)
#fvec = np.array([fvec]*nwavelengths)
#
#tfvec2 = tmm.transmit2x2(fvec, cmat2,fmatout = fout[None,...])
#tfvec2r = tmm.transmit2x2(fvec, cmat2t,fmatout = fout[None,...])
#
#tfvec4 = tmm.transmit(fvec, cmat, fmatin = fin[None,...], fmatout = fout[None,...])
#rfvec4 = dotmv(mmat,fvec)
#
x_polarizerin = tmm.polarizer4x4(jx, fin) 
y_polarizerin = tmm.polarizer4x4(jy, fin) 
x_polarizer = tmm.polarizer4x4(jx, fout) 
y_polarizer = tmm.polarizer4x4(jy, fout) 
#
intensity = tmm.intensity
#
Rxx4 = intensity(dotmv(x_polarizerin,rfvec4))
Ryx4 = intensity(dotmv(y_polarizerin,rfvec4))
Txx4 = intensity(dotmv(x_polarizer,tfvec4))
Tyx4 = intensity(dotmv(y_polarizer,tfvec4))

n = 60

Tyx4 = np.convolve(Tyx4,[1./n]*n, "same")
Txx4 = np.convolve(Txx4,[1./n]*n, "same")
Ryx4 = np.convolve(Ryx4,[1./n]*n, "same")
Rxx4 = np.convolve(Rxx4,[1./n]*n, "same")

#Txx2 = intensity(dotmv(x_polarizer,tfvec2))
#Tyx2 = intensity(dotmv(y_polarizer,tfvec2))
#Txx2r = intensity(dotmv(x_polarizer,tfvec2r))
#Tyx2r = intensity(dotmv(y_polarizer,tfvec2r))

#fvec = tmm.fvec(fin,jones = jy)
#fvec = np.array([fvec]*nwavelengths)

#tfvec2 = tmm.transmit2x2(fvec, cmat2,fmatout = fout[None,...])
#tfvec2r = tmm.transmit2x2(fvec, cmat2t,fmatout = fout[None,...])

#tfvec4 = tmm.transmit(fvec, cmat, fmatin = fin[None,...], fmatout = fout[None,...])
#rfvec4 = dotmv(mmat,fvec)

#Rxy4 = intensity(dotmv(x_polarizerin,rfvec4))
#Ryy4 = intensity(dotmv(y_polarizerin,rfvec4))
#Txy4 = intensity(dotmv(x_polarizer,tfvec4))
#Tyy4 = intensity(dotmv(y_polarizer,tfvec4))
#Txy2 = intensity(dotmv(x_polarizer,tfvec2))
#Tyy2 = intensity(dotmv(y_polarizer,tfvec2))



fvec = tmm.fvec(fin,jones = jx)
rfvec4 = np.array([fvec]*nwavelengths)
tfvec4 = tmm.transfer(rfvec4, kd, eps_values, eps_angles,nin = nin, nout = nout, method = "4x4_t")
fvec = tmm.fvec(fin,jones = jx)
#rfvec4 = np.array([fvec]*nwavelengths)

cmat = tmm.stack_mat(kd, eps_values, eps_angles, method = "4x4_r")
smat = tmm.system_mat(cmat, fmatin = fin, fmatout = fout)
a = dotmv(dtmm.linalg.inv(fout),tfvec4)
a = dotmv(smat,a)
rfvec4 = dotmv(fin,a)

#tmm.transfer(rfvec4, kd, eps_values, eps_angles, method = "4x4_r")
rfvec4 = dotmv(mmat,rfvec4)

Rxy4 = intensity(dotmv(x_polarizerin,rfvec4))
Ryy4 = intensity(dotmv(y_polarizerin,rfvec4))
Txy4 = intensity(dotmv(x_polarizer,tfvec4))
Tyy4 = intensity(dotmv(y_polarizer,tfvec4))


#Txy2r = intensity(dotmv(x_polarizer,tfvec2r))
#Tyy2r = intensity(dotmv(y_polarizer,tfvec2r))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    plt.subplot(121)

    plt.plot(wavelengths,Rxx4, label = "Rxx4")
    plt.plot(wavelengths,Ryx4, label = "Ryx4")
    plt.plot(wavelengths,Txx4, label = "Txx4")
    plt.plot(wavelengths,Tyx4, label = "Tyx4")
#    plt.plot(wavelengths,Txx2, label = "Txx2")
#    plt.plot(wavelengths,Tyx2, label = "Tyx2")  
#    plt.plot(wavelengths,Txx2r,label = "Txx2r")
#    plt.plot(wavelengths,Tyx2r, label = "Tyx2r")    
    plt.plot(wavelengths,Rxx4+Txx4+Tyx4+Ryx4, "--", label = "T4+R4")
#    plt.plot(wavelengths,Tyx2+Txx2, "--", label = "T2")
#    plt.plot(wavelengths,Tyx2r+Txx2r, "--", label = "T2r")
    plt.xlabel("wavelength")
    plt.legend(loc = 5)

    plt.subplot(122)
    plt.plot(wavelengths,Rxy4, label = "Rxy4")
    plt.plot(wavelengths,Ryy4, label = "Ryy4")
    plt.plot(wavelengths,Txy4, label = "Txy4")
    plt.plot(wavelengths,Tyy4, label = "Tyy4")
#    plt.plot(wavelengths,Txy2, label = "Txy2")
#    plt.plot(wavelengths,Tyy2, label = "Tyy2")   
#    plt.plot(wavelengths,Txy2r, label = "Txy2r")
#    plt.plot(wavelengths,Tyy2r, label = "Tyy2r")        
    plt.plot(wavelengths,Rxy4+Txy4+Tyy4+Ryy4, "--", label = "T4+R4")
#    plt.plot(wavelengths,Txy2+Tyy2, "--", label = "T2")
#    plt.plot(wavelengths,Txy2r+Tyy2r, "--", label = "T2r")    
    plt.xlabel("wavelength")
    
    plt.legend(loc = 5)


