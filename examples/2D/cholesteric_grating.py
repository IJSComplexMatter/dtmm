"""
This script calculates reflected and transmitted waves from an input white-light
plane wave and calculates the microscope images in transmission mode and in
reflection mode. We also plot the reflection efficiency for one of the 
modes and transmission efficiency for the central mode.

"""
import dtmm
import numpy as np
import matplotlib.pyplot as plt

from dtmm import tmm2d, rotation, data, wave, field, tmm

dtmm.conf.set_verbose(2)

#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,19)
PIXELSIZE = 10

nin = 1.5# refractive index of the  input material 
nout = 1.5# refractive index of the oputput material
no = 1.5
ne = 1.7

pitch_z = 36
pitch_x = 180

pitch_true = 1/(1/pitch_z **2 + 1/pitch_x**2)**0.5
print("pitch : {} nm".format(pitch_true * PIXELSIZE))
print("pitch * n : {} nm".format(pitch_true * PIXELSIZE * no))

size_x = pitch_x * 1
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

#: box dimensions
NLAYERS, HEIGHT, WIDTH = twist.shape[0], twist.shape[1], 1

d = np.ones(shape = (NLAYERS,))

epsv = np.empty( shape = (NLAYERS, HEIGHT, 3), dtype = dtmm.conf.CDTYPE)

epsv[...,0] = no**2
epsv[...,1] = no**2
epsv[...,2] = ne**2

beta, phi, intensity = 0,0,1

#we use 3D non-polarized data and convert it to 2D
field_data_in = dtmm.illumination_data((HEIGHT, 1), WAVELENGTHS, jones = None, 
                      beta= beta, phi = phi, intensity = intensity, pixelsize = PIXELSIZE, n = nin, betamax = 0.8) 
field_data_in2d = field_data_in[0][...,0],field_data_in[1],field_data_in[2]

optical_data2d = (d,epsv,epsa)

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

# we could have used this function to transfer the field directly.
#field_data_out2d = tmm2d.transfer2d(field_data_in2d, optical_data2d,  nin = 1.5, nout = 1.5)

#convert 2D data to 3D, so that we can use pom_viewer to view the microscope image
field_data_out = field_data_out2d[0][...,None],field_data_out2d[1],field_data_out2d[2]

cols = HEIGHT
rows = 1

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)


t_rcp = []
r_rcp = []

t_lcp = []
r_lcp = []
ws = []

mode = 3

for fin,fout, w, f0in,f0out in zip(fmode_in, fmode_out, WAVELENGTHS, fmatin, fmatout):
            
    fr_rcp = fin[0] - 1j * fin[1] #right handed
    ft_rcp  = fout[0] - 1j * fout[1]
    fr_lcp = fin[0] + 1j * fin[1] #right handed
    ft_lcp  = fout[0] + 1j * fout[1]    
    i_rcp = tmm.intensity(fr_rcp[0])
    i_lcp = tmm.intensity(fr_lcp[0])
    try:
        r_rcp.append(tmm.intensity(fr_rcp[mode])/i_rcp)
        t_rcp.append(tmm.intensity(ft_rcp[0])/i_rcp)
        
        r_lcp.append(tmm.intensity(fr_lcp[mode])/i_lcp)
        t_lcp.append(tmm.intensity(ft_lcp[0])/i_lcp)
        ws.append(w)
    except IndexError:
        #nonexistent mode.. break
        break
    
ax1.plot(ws,r_rcp, label = "rcp mode {} ".format(mode))
ax1.plot(ws,r_lcp, label = "lcp mode {} ".format(mode))
ax2.plot(ws,t_rcp, label = "rcp mode 0")
ax2.plot(ws,t_lcp, label = "lcp mode 0")

ax1.set_title("reflection")
ax1.set_xlabel("wavelength")
ax2.set_title("transmission")
ax2.set_xlabel("wavelength")

ax1.legend()
ax2.legend()

viewer1 = dtmm.field_viewer(field_data_in, mode = "r", n = 1.5, focus = 0,intensity = 1, cols = cols,rows = rows, analyzer = "h")
viewer2 = dtmm.field_viewer(field_data_out, mode = "t", n = 1.5, focus = 0,intensity = 1, cols = cols,rows = rows, analyzer = "h")

fig,ax = viewer1.plot()
ax.set_title("Reflected field")

fig,ax = viewer2.plot()
ax.set_title("Transmitted field")

plt.show()


