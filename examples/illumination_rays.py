"""This examples shows how illumination_rays function works:

It builds beta and phi and intensity parameters for Koehler illumination.
"""

import dtmm
import matplotlib.pyplot as plt
import numpy as np

#NA 0.1, 11x11 selection area, smoothness 0.1 
beta,phi,intensity = dtmm.illumination_rays(0.25, 7, smooth = 0.2)

plt.subplot(aspect = "equal")
plt.scatter(beta*np.cos(phi), beta*np.sin(phi), c = intensity, s = 50, edgecolors= "k")
plt.xlabel(r"$\beta_x$")
plt.ylabel(r"$\beta_y$")
plt.show()