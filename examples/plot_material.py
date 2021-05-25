#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot material example"""

import dtmm
data = dtmm.nematic_droplet_data((60,128, 128), 
                radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)
thickness, material_eps, angles = data[0]

fig, ax = dtmm.plot_material(material_eps, eps = dtmm.refind2eps([1.5,1.5,1.6]))
fig.show() #you may call  matplotlib.pyplot.show() as well
