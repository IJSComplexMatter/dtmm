"""
"""
import dtmm
import numpy as np
import matplotlib.pyplot as plt

from dtmm import data, wave, field, tmm
from dtmm.solver import matrix_block_solver

dtmm.conf.set_verbose(2)

WAVELENGTH = [540]
TWIST = 70
PIXELSIZE = 10 #equivalent to meep resolution of 100 pixels per microns
#PIXELSIZE = 20 #equivalent to meep resolution of 50 pixels per microns
#PIXELSIZE = 40 #equivalent to meep resolution of 25 pixels per microns
#PIXELSIZE = 100
#PIXELSIZE = 200

no = 1.55
deltan = 0.159
ne = no + deltan

nin, nout = no,no

gp = 1 #meep grating period in microns.

WIDTH = 1000/PIXELSIZE*gp #equivalent to meep gp = 1 grating period in microns.
if WIDTH != int(WIDTH):
    print("Rounding width")
WIDTH = int(WIDTH)

BETA = 0
INTENSITY = 1
PHI = 0

betamax = 1

NSTEPS = 1#2**4

MODES = (-1,0,1)



def simulate(thickness, grating = "uniaxial"):
    NLAYERS = max(int(1000/PIXELSIZE * thickness),1)
    
    phi = np.arange(0,np.pi, np.pi/WIDTH)
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

    field_data_in2d = field_data_in[0][...,0,:],field_data_in[1],field_data_in[2]

    f,w,p = field_data_in2d 
    
    k0 = wave.k0(w, p)
    
    mask, fmode_in = field.field2modes1(f,k0, betamax = betamax)
    
    solver = matrix_block_solver(WIDTH, WAVELENGTH, PIXELSIZE, resolution = 10, mask = mask, method = "4x4")
    solver.set_optical_block((dd*thickness,epsvd,epsad))
    solver.calculate_field_matrix(nin = nin, nout = nout)
    solver.calculate_stack_matrix()
    #solver.calculate_transmittance_matrix()
    solver.calculate_reflectance_matrix()
    
    
    fmode_in_ref = tuple(f.copy() for f in fmode_in)
    
    solver.transfer_modes(fmode_in)

    fmode_out = solver.modes_out
        
    
    t_rcp = []
    r_rcp = []
    
    t_lcp = []
    r_lcp = []
    
    
    ffin = field.modes2ffield1(mask,fmode_in)
    ffout = field.modes2ffield1(mask,fmode_out)
    fr_rcp = ffin
    fr_lcp = ffin
    ft_lcp = ffout
    ft_rcp = ffout

    # fr_rcp = ffin[0] - 1j * ffin[1] #right handed
    # ft_rcp  = ffout[0] - 1j * ffout[1]
    
    # fr_lcp = ffin[0] + 1j * ffin[1] 
    # ft_lcp  = ffout[0] + 1j * ffout[1]   

    
    #fmode_in_ref = tmm2d.unlist_modes(tmm2d.project2d(tmm2d.list_modes(fmode_in), fmatin))
    ffin_ref = field.modes2ffield1(mask,fmode_in_ref)
    
    #fr_rcp_ref = ffin_ref[0] - 1j * ffin_ref[1] #right handed
    #fr_lcp_ref = ffin_ref[0] + 1j * ffin_ref[1] 
    fr_lcp_ref = ffin
    fr_rcp_ref = ffin
    i_rcp = tmm.poynting(fr_rcp_ref.swapaxes(-2,-1)).sum(-1)
    i_lcp = tmm.poynting(fr_lcp_ref.swapaxes(-2,-1)).sum(-1)
    
    y_rcp = {}
    y_lcp = {}
    
    for mode in MODES:
        
        y_rcp[mode] = (tmm.poynting(ft_rcp[...,mode])/i_rcp)
        y_lcp[mode] = tmm.poynting(ft_lcp[...,mode])/i_lcp
    
        # ax1.plot(ws,y_rcp, label = "rcp mode {} ".format(mode))
        # ax1.plot(ws,y_lcp, label = "lcp mode {} ".format(mode))

    r_rcp = (i_rcp-tmm.poynting(fr_rcp.swapaxes(-1,-2)).sum(axis = -1)) / i_rcp 
    r_lcp = (i_lcp-tmm.poynting(fr_lcp.swapaxes(-1,-2)).sum(axis = -1)) / i_lcp  

    # ax1.plot(ws,r_rcp, "--", label = "rcp")
    # ax1.plot(ws,r_lcp, "--", label = "lcp")
        
    # ax2.plot(ws,tmm.poynting(ft_rcp[...,0])/i_rcp, label = "rcp mode 0")
    # ax2.plot(ws,tmm.poynting(ft_lcp[...,0])/i_lcp, label = "lcp mode 0")
    
    t_rcp = tmm.poynting(ft_rcp.swapaxes(-1,-2)).sum(axis = -1) / i_rcp
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
        
        # for mode in MODES:    
        
        #     ax2.plot(thickness, [y[mode] for y in d_lcp], label = "mode {}".format(mode))
            
        # ax2.plot(thickness, t_lcp, label = "transmittance ")
        # ax2.plot(thickness, r_lcp, label = "reflectance ")
        # ax2.plot(thickness, t_lcp + r_lcp, "--", label = "reflectance + transmittance")
        
        np.asarray([y[1] for y in d_lcp]) 
        
        np.save("simdata/tmm2d_{}_{}_{}_eff_m0.npy".format(PIXELSIZE,grating,gp),np.asarray([y[0] for y in d_rcp]))
        np.save("simdata/tmm2d_{}_{}_{}_eff_m1.npy".format(PIXELSIZE,grating,gp),np.asarray([y[1] for y in d_rcp]))
        np.save("simdata/tmm2d_{}_{}_{}_eff_m-1.npy".format(PIXELSIZE,grating,gp),np.asarray([y[-1] for y in d_rcp]))
        
        np.save("simdata/tmm2d_{}_{}_{}_t.npy".format(PIXELSIZE,grating,gp), t_rcp)
        np.save("simdata/tmm2d_{}_{}_{}_r.npy".format(PIXELSIZE,grating,gp), r_rcp)
        plt.legend()
        plt.show()