"""
This script calculates reflected and transmitted waves from an input white-light
plane wave and calculates the microscope images in transmission mode and in
reflection mode. We also plot the reflection efficiency for one of the 
modes and transmission efficiency for the central mode.

"""
import dtmm
import numpy as np
import matplotlib.pyplot as plt

from dtmm import tmm2d, rotation, data, wave, field, tmm, solver


dtmm.conf.set_verbose(2)

#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,21)
#WAVELENGTHS = np.linspace(450,490,21)

SCALE = 1

PIXELSIZE = 10/SCALE

nin = 1.5# refractive index of the  input material 
nout = 1.5# refractive index of the oputput material
no = 1.5
ne = 1.7

pitch_z = 36*SCALE
pitch_x = 180*SCALE

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
NLAYERS, WIDTH = twist.shape[0], twist.shape[1]

d = np.ones(shape = (NLAYERS,))

epsv = np.empty( shape = (NLAYERS, WIDTH, 3), dtype = dtmm.conf.CDTYPE)

epsv[...,0] = no**2
epsv[...,1] = no**2
epsv[...,2] = ne**2

beta, phi, intensity = 0,0,1

betamax = 1.4

#we use 3D non-polarized data and convert it to 2D
field_data_in = dtmm.illumination_data((1,WIDTH), WAVELENGTHS, jones = None, 
                      beta= beta, phi = phi, intensity = intensity, pixelsize = PIXELSIZE, n = nin) 

if __name__ == "__main__":

    field_data_in2d = field_data_in[0][...,0,:],field_data_in[1],field_data_in[2]
    
    
    
    optical_data2d = [(d,epsv,epsa)]
    
    f,w,p = field_data_in2d 
    shape = f.shape[-1]
    
    k0 = wave.k0(w, p)
    
    mask, fmode_in = field.field2modes1(f,k0, betamax = betamax)
    

    fmatin = tmm2d.f_iso2d(mask = mask,  k0 = k0, n=nin, betay = 0)
    fmatout = tmm2d.f_iso2d(mask = mask,  k0 = k0, n=nout, betay = 0)
    
    cmat = tmm2d.stack_mat2d(k0,d, epsv, epsa, betay = 0, mask = mask)
    smat = tmm2d.system_mat2d(fmatin = fmatin, cmat = cmat, fmatout = fmatout)
    rmat = tmm2d.reflection_mat2d(smat)
    
    fmode_in_listed = tmm2d.list_modes(fmode_in)
    
    fmode_out_listed = tmm2d.reflect2d(fmode_in_listed , rmat = rmat, fmatin = fmatin, fmatout = fmatout)
    fmode_out = tmm2d.unlist_modes(fmode_out_listed)
    
    field_out = field.modes2field1(mask, fmode_out)
    f[...] = field.modes2field1(mask, fmode_in)
    
    
    #field_out = dtmm.tmm2d.transfer2d(field_data_in2d, (d,epsv,epsa), betay = 0., nin = nin, nout = nout, method = "4x4")[0]
    
    
    field_data_out2d = field_out ,w, p
    
    #we could have used this function to transfer the field directly.
    #field_data_out2d = tmm2d.transfer2d(field_data_in2d, optical_data2d[0],  nin = 1.5, nout = 1.5)
    
    #convert 2D data to 3D, so that we can use pom_viewer to view the microscope image
    field_data_out = field_data_out2d[0][...,None,:],field_data_out2d[1],field_data_out2d[2]
    field_data_in = field_data_in2d[0][...,None,:],field_data_in2d[1],field_data_in2d[2]
    
    cols = 1
    rows = WIDTH
    
    
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    
    
    t_rcp = []
    r_rcp = []
    
    t_lcp = []
    r_lcp = []
    ws = []
    
    
    ffin = field.modes2ffield1(mask,fmode_in)
    ffout = field.modes2ffield1(mask,fmode_out)

    fr_rcp = ffin[0] - 1j * ffin[1] #right handed
    ft_rcp  = ffout[0] - 1j * ffout[1]
    
    fr_lcp = ffin[0] + 1j * ffin[1] 
    ft_lcp  = ffout[0] + 1j * ffout[1]   
    
    fmode_in_ref = tmm2d.unlist_modes(tmm2d.project2d(tmm2d.list_modes(fmode_in), fmatin))
    ffin_ref = field.modes2ffield1(mask,fmode_in_ref)
    
    fr_rcp_ref = ffin_ref[0] - 1j * ffin_ref[1] #right handed
    fr_lcp_ref = ffin_ref[0] + 1j * ffin_ref[1] 
    i_rcp = tmm.intensity(fr_rcp_ref[...,0])
    i_lcp = tmm.intensity(fr_lcp_ref[...,0])   
    
    ws = WAVELENGTHS
    
    for mode in (-2,):
        
        y_rcp = -tmm.poynting(fr_rcp[...,mode])/i_rcp
        y_lcp = -tmm.poynting(fr_lcp[...,mode])/i_lcp
    
        ax1.plot(ws,y_rcp, label = "rcp mode {} ".format(mode))
        ax1.plot(ws,y_lcp, label = "lcp mode {} ".format(mode))

    r_rcp = (i_rcp-tmm.poynting(fr_rcp.swapaxes(-1,-2)).sum(axis = -1)) / i_rcp 
    r_lcp = (i_lcp-tmm.poynting(fr_lcp.swapaxes(-1,-2)).sum(axis = -1)) / i_lcp  

    ax1.plot(ws,r_rcp, "--", label = "rcp")
    ax1.plot(ws,r_lcp, "--", label = "lcp")
        
    ax2.plot(ws,tmm.poynting(ft_rcp[...,0])/i_rcp, label = "rcp mode 0")
    ax2.plot(ws,tmm.poynting(ft_lcp[...,0])/i_lcp, label = "lcp mode 0")
    
    t_rcp = tmm.poynting(ft_rcp.swapaxes(-1,-2)).sum(axis = -1) / i_rcp
    t_lcp = tmm.poynting(ft_lcp.swapaxes(-1,-2)).sum(axis = -1) / i_lcp

    ax2.plot(ws, t_rcp + r_rcp, "--", label = "rcp (refl + trans)")
    ax2.plot(ws, t_lcp + r_lcp, "--", label = "lcp (refl + trans)")
    
    ax1.set_title("reflection")
    ax1.set_xlabel("wavelength")
    ax2.set_title("transmission")
    ax2.set_xlabel("wavelength")
    
    ax1.legend()
    ax2.legend()
    
        
    optical_data = [(d,epsv[...,None,:,:],epsa[...,None,:,:])] 
    #we could have computed like so:
    #field_data_out = solver.transfer3d(field_data_in, optical_data, nin = 1.5, nout = 1.5,)
    
    viewer1 = dtmm.field_viewer(field_data_in, mode = "r", n = 1.5, focus = 0,intensity = 1, cols = cols,rows = rows, analyzer = "h")
    viewer2 = dtmm.field_viewer(field_data_out, mode = "t", n = 1.5, focus = 0,intensity = 1, cols = cols,rows = rows, analyzer = "h")
    viewer3 = dtmm.field_viewer(field_data_in, mode = "t", n = 1.5, focus = 0,intensity = 1, cols = cols,rows = rows, analyzer = "h")
    
    fig,ax = viewer1.plot()
    ax.set_title("Reflected field")
    
    fig,ax = viewer2.plot()
    ax.set_title("Transmitted field")
    
    fig,ax = viewer3.plot()
    ax.set_title("Input field")
    
    plt.show()