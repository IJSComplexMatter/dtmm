"""This examples shows how illumination_rays function works:

It builds beta and phi and intensity parameters for Koehler illumination.
"""

import dtmm
import matplotlib.pyplot as plt
import numpy as np

#NA 0.1, 11x11 selection area, sharpness 0.7 
beta,phi,intensity = dtmm.illumination_rays(0.1,11, 0.7)

plt.scatter(beta*np.cos(phi), beta*np.sin(phi), c = intensity, s = 50, edgecolors= "k")
plt.xlabel(r"$\beta_x$")
plt.ylabel(r"$\beta_y$")
plt.gray()
plt.show()