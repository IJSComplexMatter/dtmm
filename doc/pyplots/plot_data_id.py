#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot material ID example"""

import dtmm
data = dtmm.nematic_droplet_data((60, 128, 128), 
                radius = 30, profile = "r", no = 1.5, ne = 1.7, nhost = 1.5)
thickness, material_id, material_eps, angles = data

fig = dtmm.plot_id(material_id, id = 1)
fig.show()
