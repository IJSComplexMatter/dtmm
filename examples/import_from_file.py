"""
Example showing how to import data from a file.

It generates the director file as a text file, rather than binary, to allow
the user to open it up and get a good understanding of what's going on.

This file was create to have and example that could be run straight. Based
heavily on the example in the quickstart.rst.
"""


import matplotlib
matplotlib.use("TkAgg")

import dtmm
import numpy as np

# ---- Physical parameters
no = 1.5
ne = 1.6
nhost = 1.5

# ---- Generate data
n_layers = 60
height = 96
width = 96
shape = (n_layers, height, width)
# Radius of the droplet
radius = 30
temp_director = dtmm.nematic_droplet_director(shape, radius, "r")
# Export the data to a text file
temp_director.tofile("director.txt", sep=",")

# --- Read in director and convert optical data
# Read director from text file
director = dtmm.read_director("director.txt", (n_layers, height, width, 3), sep=",")
# Create the mask for a spherical
mask = dtmm.sphere_mask(shape, radius)
# Covert director, mask, and physical parameters into optical data
optical_data = dtmm.director2data(director, mask, no, ne, nhost)

# ----
# Generate wavelengths
wavelengths = np.linspace(380, 780, 9)
# Size of each pixel in nm
pixel_size = 200
# Generate input light field
input_light_field = dtmm.field.illumination_data((height, width), wavelengths, pixel_size)

# Generate output light field using 4x4 method
output_light_field = dtmm.transfer_field(input_light_field, optical_data, method="4x4")

#: visualize output field
viewer = dtmm.field_viewer(output_light_field)
viewer.set_parameters(focus=-14, intensity=2,
                      sample=0, analyzer=90, polarizer=0)
fig, _ = viewer.plot()
# Show figure
fig.show()
