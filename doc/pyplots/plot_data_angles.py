#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot material angles example"""

import dtmm
data = dtmm.nematic_droplet_data((60, 128, 128), 
                radius = 30, profile = "r", no = 1.5, ne = 1.7, nhost = 1.5)
thickness, id, material, angles = data
#matplotlib cannot handle large datasets, so crop to region of interest (center of the sphere)
fig = dtmm.plot_angles(angles[26:-26,58:-58,58:-58])
fig.show()
