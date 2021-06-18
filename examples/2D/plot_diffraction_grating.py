import numpy as np
import matplotlib.pyplot as plt

phase = np.load("simdata/meep_phase.npy")

axes = plt.subplot(121), plt.subplot(122)
gratings = ("uniaxial", "twisted")

for ax, grating in zip(axes, gratings):

    
    t0_meep = np.load("simdata/meep_{}_eff_m0.npy".format(grating))
    t0_tmm = np.load("simdata/tmm2d_{}_eff_m0.npy".format(grating))
    t1_meep = np.load("simdata/meep_{}_eff_m1.npy".format(grating))
    t1_tmm = np.load("simdata/tmm2d_{}_eff_m1.npy".format(grating)) + np.load("simdata/tmm2d_{}_eff_m-1.npy".format(grating))
    
    ax.plot(phase, t0_meep, label = "meep (0th order)")
    ax.plot(phase, t0_tmm, label = "tmm (0th order)")
    ax.plot(phase, t1_meep, label = "meep (+-1th order)")
    ax.plot(phase, t1_tmm, label = "tmm (+-1th order)")   
    
    ax.plot(phase, t0_meep + t1_meep, "--", label = "meep trasnmission")
    ax.plot(phase, t0_tmm + t1_tmm, "--", label = "tmm trasnmission")

#    if grating == "uniaxial":
#        [math.cos(math.pi*p)**2 for p in phase]
#        ax.plot(phase, [math.cos(math.pi*p)**2 for p in phase], "--", label = "analyitical")
    
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