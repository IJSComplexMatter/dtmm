#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot material angles example"""

import dtmm
data = dtmm.nematic_droplet_data((60, 128, 128), 
                radius = 30, profile = "r", no = 1.5, ne = 1.7, nhost = 1.5)
thickness, material, angles = data[0]
#matplotlib cannot handle large datasets, so crop to region of interest (center of the sphere)
fig,ax = dtmm.plot_angles(angles, center = True, xlim = (-5,5), ylim = (-5,5), zlim = (-5,5))
fig.show()
