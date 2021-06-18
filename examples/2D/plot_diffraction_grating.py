import numpy as np
import matplotlib.pyplot as plt

phase = np.load("simdata/meep_phase.npy")

axes = plt.subplot(121), plt.subplot(122)
gratings = ("uniaxial", "twisted")
resolution = 50

for ax, grating in zip(axes, gratings):

    if grating == "uniaxial":
        x = np.linspace(phase[0],phase[-1],200)
        y = np.cos(np.pi*x)**2
        ax.plot(x,y, "k-",label = "0th order (analytic)")
        y = np.sin(np.pi*x)**2
        ax.plot(x,y, "k--",label = "1th order (analytic)")        
    gp = 1
    
    t0_meep = np.load("simdata/meep_{}_{}_{}_eff_m0.npy".format(resolution,grating, gp))
    t0_tmm = np.load("simdata/tmm2d_{}_eff_m0.npy".format(grating))
    t1_meep = np.load("simdata/meep_{}_{}_{}_eff_m1.npy".format(resolution,grating, gp))
    t1_tmm = np.load("simdata/tmm2d_{}_eff_m1.npy".format(grating)) + np.load("simdata/tmm2d_{}_eff_m-1.npy".format(grating))
    
    ax.plot(phase, t0_meep, label = "0th order (meep)")
    ax.plot(phase, t0_tmm, label = "0th order (tmm)")
    ax.plot(phase, t1_meep, label = "1th order (meep)")
    ax.plot(phase, t1_tmm, label = "1th order (tmm)")   
    
    ax.plot(phase, t0_meep + t1_meep, "--", label = "meep trasnmission")
    ax.plot(phase, t0_tmm + t1_tmm, "--", label = "tmm trasnmission")


    ax.legend()
    
    # for mode in MODES:    
    
    #     ax1.plot(thickness, [y[mode] for y in d_rcp], label = "mode {}".format(mode))
        
    # ax1.plot(thickness, t_rcp, label = "transmittance ")
    # ax1.plot(thickness, r_rcp, label = "reflectance ")
    # ax1.plot(thickness, t_rcp + r_rcp, "--", label = "reflectance + transmittance")
    
    # for mode in MODES:    
    
    #     ax2.plot(thickness, [y[mode] for y in d_lcp], label = "mode {}".format(mode))
        
    # ax2.plot(thickness, t_lcp, label = "transmittance ")
    # ax2.plot(thickness, r_lcp, label = "reflectance ")
    # ax2.plot(thickness, t_lcp + r_lcp, "--", label = "reflectance + transmittance")
    
    # np.asarray([y[1] for y in d_lcp]) 
    
    # np.save("simdata/tmm2d_{}_eff_m0.npy".format(grating),np.asarray([y[0] for y in d_lcp]))
    # np.save("simdata/tmm2d_{}_eff_m1.npy".format(grating),np.asarray([y[1] for y in d_lcp]))
    # np.save("simdata/tmm2d_{}_eff_m-1.npy".format(grating),np.asarray([y[-1] for y in d_lcp]))
    
    # np.save("simdata/tmm2d_{}_t.npy".format(grating), t_lcp)
    # np.save("simdata/tmm2d_{}_r.npy".format(grating), r_lcp)
    
    # plt.show()