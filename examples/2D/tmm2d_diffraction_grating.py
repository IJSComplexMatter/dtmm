"""
"""
import dtmm
import numpy as np
import matplotlib.pyplot as plt

from dtmm import tmm2d, rotation, data, wave, field, tmm, solver

dtmm.conf.set_verbose(2)

WAVELENGTH = [540]
TWIST = 70
PIXELSIZE = 20 #equivalent to meep resolution of 50 pixels per microns
no = 1.55
deltan = 0.159
ne = no + deltan
#thickness = 1.5 # in microns

nin, nout = no,no


WIDTH = 162 #almost equivalent to meep gp = 6.5 grating period in microns.


BETA = 0
INTENSITY = 1
PHI = 0

betamax = 1.4

POWER = 5

MODES = (-1,0,1)


def simulate(thickness, grating = "uniaxial"):
    NLAYERS = int(1000/PIXELSIZE * thickness)
    
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
    field_data_in = dtmm.illumination_data((1,WIDTH), WAVELENGTH, jones = None, 
                      beta= BETA, phi = PHI, intensity = INTENSITY, pixelsize = PIXELSIZE, n = nin) 


    field_data_in2d = field_data_in[0][...,0,:],field_data_in[1],field_data_in[2]

    
    f,w,p = field_data_in2d 
    
    k0 = wave.k0(w, p)
    
    mask, fmode_in = field.field2modes1(f,k0, betamax = betamax)
    

    fmatin = tmm2d.f_iso2d(mask = mask,  k0 = k0, n=nin, betay = 0)
    fmatout = tmm2d.f_iso2d(mask = mask,  k0 = k0, n=nout, betay = 0)
    
    cmat = tmm2d.stack_mat2d(k0,dd*thickness, epsvd, epsad, betay = 0, mask = mask, resolution_power = POWER)
    smat = tmm2d.system_mat2d(fmatin = fmatin, cmat = cmat, fmatout = fmatout)
    rmat = tmm2d.reflection_mat2d(smat)
    
    
    fmode_in_ref = tuple(f.copy() for f in fmode_in)
    
    fmode_out = tmm2d.reflect2d(fmode_in , rmat = rmat, fmatin = fmatin, fmatout = fmatout)

    
    f[...] = field.modes2field1(mask, fmode_in)
    
    
    t_rcp = []
    r_rcp = []
    
    t_lcp = []
    r_lcp = []
    
    
    ffin = field.modes2ffield1(mask,fmode_in)
    ffout = field.modes2ffield1(mask,fmode_out)

    fr_rcp = ffin[0] - 1j * ffin[1] #right handed
    ft_rcp  = ffout[0] - 1j * ffout[1]
    
    fr_lcp = ffin[0] + 1j * ffin[1] 
    ft_lcp  = ffout[0] + 1j * ffout[1]   
    
    #fmode_in_ref = tmm2d.unlist_modes(tmm2d.project2d(tmm2d.list_modes(fmode_in), fmatin))
    ffin_ref = field.modes2ffield1(mask,fmode_in_ref)
    
    fr_rcp_ref = ffin_ref[0] - 1j * ffin_ref[1] #right handed
    fr_lcp_ref = ffin_ref[0] + 1j * ffin_ref[1] 
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
    
    t_rcp = tmm.poynting(ft_rcp.swapaxes(-1,-2)).sum(axis = -1) / i_rcp
    t_lcp = tmm.poynting(ft_lcp.swapaxes(-1,-2)).sum(axis = -1) / i_lcp
    

    return {"t" : (t_rcp, t_lcp), "r" : (r_rcp, r_lcp), "d" : (y_rcp, y_lcp)}

if __name__ == "__main__":
    
    grating = "uniaxial"
    
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
        

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    
    for mode in MODES:    
    
        ax1.plot(thickness, [y[mode] for y in d_rcp], label = "mode {}".format(mode))
        
    ax1.plot(thickness, t_rcp, label = "transmittance ")
    ax1.plot(thickness, r_rcp, label = "reflectance ")
    ax1.plot(thickness, t_rcp + r_rcp, "--", label = "reflectance + transmittance")
    
    for mode in MODES:    
    
        ax2.plot(thickness, [y[mode] for y in d_lcp], label = "mode {}".format(mode))
        
    ax2.plot(thickness, t_lcp, label = "transmittance ")
    ax2.plot(thickness, r_lcp, label = "reflectance ")
    ax2.plot(thickness, t_lcp + r_lcp, "--", label = "reflectance + transmittance")
    
    np.asarray([y[1] for y in d_lcp]) 
    
    np.save("simdata/tmm2d_{}_eff_m0.npy".format(grating),np.asarray([y[0] for y in d_rcp]))
    np.save("simdata/tmm2d_{}_eff_m1.npy".format(grating),np.asarray([y[1] for y in d_rcp]))
    np.save("simdata/tmm2d_{}_eff_m-1.npy".format(grating),np.asarray([y[-1] for y in d_rcp]))
    
    np.save("simdata/tmm2d_{}_t.npy".format(grating), t_rcp)
    np.save("simdata/tmm2d_{}_r.npy".format(grating), r_rcp)
    
    plt.show()