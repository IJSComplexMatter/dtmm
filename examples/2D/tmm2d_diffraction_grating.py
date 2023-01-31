"""
"""
import dtmm
import numpy as np
import matplotlib.pyplot as plt

from dtmm import tmm2d, rotation, data, wave, field, tmm

dtmm.conf.set_verbose(2)

WAVELENGTH = [540]
TWIST = 70
PIXELSIZE = 10 #equivalent to meep resolution of 100 pixels per microns
#PIXELSIZE = 20 #equivalent to meep resolution of 50 pixels per microns
PIXELSIZE = 40 #equivalent to meep resolution of 25 pixels per microns
#PIXELSIZE = 100
#PIXELSIZE = 100

no = 1.55
deltan = 0.159
ne = no + deltan

nin, nout = no,ne

gp = 1 #meep grating period in microns.

REPEAT = 1

WIDTH = REPEAT*1000/PIXELSIZE*gp #equivalent to meep gp = 1 grating period in microns.
if WIDTH != int(WIDTH):
    print("Rounding width")
WIDTH = int(WIDTH)

BETA = 0.
PHI = 0.
INTENSITY = 1


betamax = 1.

NSTEPS = 2**4

MODES = (-1*REPEAT,0,1*REPEAT) 


def simulate(thickness, grating = "uniaxial"):
    NLAYERS = max(int(1000/PIXELSIZE * thickness),1)
    
    print("nlayers", NLAYERS)
    
    phi = np.arange(0,REPEAT*np.pi, REPEAT*np.pi/WIDTH)
    assert len(phi) == WIDTH
    
    twist = TWIST if grating == "twisted" else 0
    twist1 = np.linspace(0,np.pi/180*twist, NLAYERS)
    twist2 = np.linspace(np.pi/180*twist,0, NLAYERS)
    
    d = np.ones(NLAYERS) * 1000/ PIXELSIZE / NLAYERS 
    
    epsv = np.empty(shape = (NLAYERS, WIDTH, 3), dtype = dtmm.conf.CDTYPE)
    epsv[...] = data.refind2eps([no,no,ne])
    
    twist1 = twist1[:,None] + phi[None,:]
    twist2 = twist2[:,None] + phi[None,:]
    
    director1 = np.empty(shape = (NLAYERS,WIDTH) + (3,))
    director1[...,0] = np.cos(twist1) #x component
    director1[...,1] = np.sin(twist1) #y component
    director1[...,2] = 0 # z component
    
    director2 = np.empty(shape = (NLAYERS,WIDTH) + (3,))
    director2[...,0] = np.cos(twist2) #x component
    director2[...,1] = np.sin(twist2) #y component
    director2[...,2] = 0 # z component
    
    epsa1 = data.director2angles(director1)
    epsa2 = data.director2angles(director2)
    
    if grating == "twisted":
    
        dd = np.hstack((d,d))
        epsvd = np.vstack((epsv,epsv))
        epsad = np.vstack((epsa1,epsa2))
    elif grating == "uniaxial":
        dd = d
        epsvd = epsv
        epsad = epsa1
    else:
        raise ValueError("Unknown grating type.")


    
    #we use 3D non-polarized data and convert it to 2D
    field_data_in = dtmm.illumination_data((1,WIDTH), WAVELENGTH, jones = (1,1j), 
                      beta= BETA, phi = PHI, intensity = INTENSITY, pixelsize = PIXELSIZE, n = nin) 


    #field_data_in2d = field_data_in[0][...,0,:],field_data_in[1],field_data_in[2]

    
    f,w,p = field_data_in
    
    f = f.copy()
    
    optical_data = [(dd*thickness,epsvd[...,None,:,:],epsad[...,None,:,:])]
     
    fr_lcp_ref = np.fft.fft(f[...,0,:], axis = -1)
    fr_rcp_ref = np.fft.fft(f[...,0,:], axis = -1)
    
    field_data_out = dtmm.transfer_field(field_data_in, optical_data,diffraction =1, reflection = 2, nstep = 1,  npass = 1, betamax = betamax,nin =nin,nout = nout, method = "2x2") 
    
    k0 = wave.k0(w, p)
    
    
    
    t_rcp = []
    r_rcp = []
    
    t_lcp = []
    r_lcp = []
    
    
    ffin = np.fft.fft(field_data_in[0][...,0,:], axis = -1)
    
    ffout = np.fft.fft(field_data_out[0][...,0,:],axis = -1)
    
    fr_rcp = ffin
    fr_lcp = ffin
    ft_lcp = ffout
    ft_rcp = ffout


    i1 = (tmm.poynting(field_data_out[0][...,0,:].swapaxes(-2,-1)).sum())
    
    i0 = (tmm.poynting(f[...,0,:].swapaxes(-2,-1)).sum())

    i_rcp = tmm.poynting(fr_rcp_ref.swapaxes(-2,-1)).sum(-1)
    i_lcp = tmm.poynting(fr_lcp_ref.swapaxes(-2,-1)).sum(-1)
    
    y_rcp = {}
    y_lcp = {}
    
    for mode in MODES:
        
        y_rcp[mode] = ( tmm.poynting(ft_rcp[...,mode])/i_rcp)
        y_lcp[mode] = tmm.poynting(ft_lcp[...,mode])/i_lcp
    
        # ax1.plot(ws,y_rcp, label = "rcp mode {} ".format(mode))
        # ax1.plot(ws,y_lcp, label = "lcp mode {} ".format(mode))
        

    r_rcp = (i_rcp-tmm.poynting(fr_rcp.swapaxes(-1,-2)).sum(axis = -1)) / i_rcp 
    r_lcp = (i_lcp-tmm.poynting(fr_lcp.swapaxes(-1,-2)).sum(axis = -1)) / i_lcp  

    # ax1.plot(ws,r_rcp, "--", label = "rcp")
    # ax1.plot(ws,r_lcp, "--", label = "lcp")
        
    # ax2.plot(ws,tmm.poynting(ft_rcp[...,0])/i_rcp, label = "rcp mode 0")
    # ax2.plot(ws,tmm.poynting(ft_lcp[...,0])/i_lcp, label = "lcp mode 0")
    
    t = tmm.poynting(ft_rcp.swapaxes(-1,-2))
    t_rcp = (t[:,0] + t[:,1]) / i_rcp
    t_rcp = i1/i0
    
    #t_rcp = tmm.poynting(ft_rcp.swapaxes(-1,-2)).sum(axis = -1) / i_rcp
    t_lcp = tmm.poynting(ft_lcp.swapaxes(-1,-2)).sum(axis = -1) / i_lcp
    return {"t" : (t_rcp, t_lcp), "r" : (r_rcp, r_lcp), "d" : (y_rcp, y_lcp)}

if __name__ == "__main__":
    
    
    subplots = [121,122]
    
    for grating, subplot in zip(("uniaxial", "twisted"), subplots):

    
        thickness = np.arange(0.1,3.5,0.1)
        
        t_rcp = np.empty(len(thickness))
        t_lcp = np.empty(len(thickness))
        
        r_rcp = np.empty(len(thickness))
        r_lcp = np.empty(len(thickness))
        
        d_rcp = []
        d_lcp = []
        
        for i,d in enumerate(thickness):
            print("Simulating thickness {} um".format(d))
            out = simulate(d, grating)
            
            t_rcp[i] = out["t"][0]
            t_lcp[i] = out["t"][1]
            
            r_rcp[i] = out["r"][0]
            r_lcp[i] = out["r"][1] 
            
            d_rcp.append(out["d"][0])
            d_lcp.append(out["d"][1])
            
    
        ax1 = plt.subplot(subplot)
        
        for mode in MODES:    
        
            ax1.plot(thickness, [y[mode] for y in d_rcp], label = "mode {}".format(mode))
            
        ax1.plot(thickness, t_rcp, label = "transmittance ")
        ax1.plot(thickness, r_rcp, label = "reflectance ")
        ax1.plot(thickness, t_rcp + r_rcp, "--", label = "reflectance + transmittance")
        
        # np.asarray([y[1] for y in d_lcp]) 
        
        np.save("simdata/bpm2d_{}_{}_{}_eff_m0.npy".format(PIXELSIZE,grating,gp),np.asarray([y[0] for y in d_rcp]))
        np.save("simdata/bpm2d_{}_{}_{}_eff_m1.npy".format(PIXELSIZE,grating,gp),np.asarray([y[1] for y in d_rcp]))
        np.save("simdata/bpm2d_{}_{}_{}_eff_m-1.npy".format(PIXELSIZE,grating,gp),np.asarray([y[-1] for y in d_rcp]))
        
        np.save("simdata/bpm2d_{}_{}_{}_t.npy".format(PIXELSIZE,grating,gp), t_rcp)
        np.save("simdata/bpm2d_{}_{}_{}_r.npy".format(PIXELSIZE,grating,gp), r_rcp)
        
        plt.legend()
        plt.show()