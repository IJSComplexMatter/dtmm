"""
this script calculates reflected and trasnmitted waves from an input white-light
plane wave and calculates the microscope images in transmission mode and in
reflection mode.


"""
import dtmm
import numpy as np
import matplotlib.pyplot as plt

from dtmm import tmm2d, rotation, data, wave, field, jones4, tmm
dtmm.conf.set_verbose(2)

#scaling factor.. set this to 2,3,4 or any integer to decrease resolution by this factor (to increase computation speed)
SCALING = 1
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,19)

PIXELSIZE = 10

KOEHLER = False

NA = 0.4

nin = 1.5# refractive index of the  input material 
nout = 1.5# refractive index of the oputput material
no = 1.5
ne = 1.7


pitch_z = 36
pitch_x = 180

pitch_true = 1/(1/pitch_z **2 + 1/pitch_x**2)**0.5
print("pitch : {} nm".format(pitch_true * PIXELSIZE))
print("pitch * n : {} nm".format(pitch_true * PIXELSIZE * no))

size_x = pitch_x * 8
size_z = pitch_z * 2

tilt = np.arctan(pitch_z/pitch_x)

twist = np.arange(0,2*np.pi*size_z/pitch_z, 2*np.pi/pitch_z)[:,None] + np.arange(0,2*np.pi*size_x/pitch_x, 2*np.pi/pitch_x)[None,:]

director = np.empty(shape = twist.shape + (3,))
director[...,0] = np.cos(twist) #x component
director[...,1] = np.sin(twist) #y component
director[...,2] = 0 # z component

r = rotation.rotation_matrix_y(tilt)
director = rotation.rotate_vector(r, director)
epsa = data.director2angles(director)


#: pixel size in nm
PIXELSIZE = PIXELSIZE*SCALING

#: box dimensions
NLAYERS, HEIGHT, WIDTH = twist.shape[0], twist.shape[1], 1



d = np.ones(shape = (NLAYERS,))

epsv = np.empty( shape = (NLAYERS, HEIGHT, 3), dtype = dtmm.conf.CDTYPE)

epsv[...,0] = no**2
epsv[...,1] = no**2
epsv[...,2] = ne**2

d =d/SCALING
epsv = epsv[:,::SCALING]
epsa = epsa[:,::SCALING]


if KOEHLER:
    beta, phi, intensity = dtmm.illumination_rays(NA,7)
else:
    beta,phi, intensity = 0,0,1

jones = None#dtmm.jonesvec((1,1j)) #left handed input light

field_data_in = dtmm.illumination_data((HEIGHT, 1), WAVELENGTHS, jones = jones, 
                      beta= beta, phi = phi, intensity = intensity, pixelsize = PIXELSIZE, n = nin, betamax = 0.8) 


# if KOEHLER == False:
#     f,w,p = field_data_in
#     k = dtmm.k0(w,PIXELSIZE)
#     mask, modes0 = dtmm.field.field2modes(f,k)

optical_data2d = (d,epsv,epsa)

field_data_in2d = field_data_in[0][...,0],field_data_in[1],field_data_in[2]

f,w,p = field_data_in2d 
shape = f.shape[-1]
d,epsv,epsa = optical_data2d
k0 = wave.k0(w, p)

mask, fmode_in = field.field2modes1(f,k0)

fmatin = tmm2d.f_iso2d(shape = shape, betay = 0, k0 = k0, n=nin)
fmatout = tmm2d.f_iso2d(shape = shape, betay = 0, k0 = k0, n=nout)

cmat = tmm2d.stack_mat2d(k0,d, epsv, epsa, betay = 0, mask = mask)
smat = tmm2d.system_mat2d(fmatin = fmatin, cmat = cmat, fmatout = fmatout)
rmat = tmm2d.reflection_mat2d(smat)

fmode_out = tmm2d.reflect2d(fmode_in, rmat = rmat, fmatin = fmatin, fmatout = fmatout)

field_out = field.modes2field1(mask, fmode_out)
f[...] = field.modes2field1(mask, fmode_in)

field_data_out2d = field_out ,w, p

#field_data_out2d = tmm2d.transfer2d(field_data_in2d, optical_data2d,  nin = 1.5, nout = 1.5)
field_data_out = field_data_out2d[0][...,None],field_data_out2d[1],field_data_out2d[2]


cols = HEIGHT
rows = 1

for fin,fout, w in zip(fmode_in, fmode_out, WAVELENGTHS):
    i = tmm.intensity(fin[:,0]).sum()
    plt.plot((tmm.intensity(fin)/i).sum(0), label = w )
plt.legend()
    


viewer1 = dtmm.field_viewer(field_data_in, mode = "r", n = 1.5, focus = 0,intensity = 1, cols = cols,rows = rows)
viewer2 = dtmm.field_viewer(field_data_out, mode = "t", n = 1.5, focus = 0,intensity = 1, cols = cols,rows = rows)
#viewer3 = dtmm.field_viewer(field_data_out, mode = "r", n = 1.5, focus = 0,intensity = 1, cols = cols,rows = rows)

fig,ax = viewer1.plot()
ax.set_title("Reflected field")

fig,ax = viewer2.plot()
ax.set_title("Transmitted field")

#fig,ax = viewer3.plot()
#ax.set_title("Residual field")


