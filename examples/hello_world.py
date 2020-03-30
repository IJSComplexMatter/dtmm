"""
Hello nematic droplet example.
"""
# Change matplotlib backend to an interactive one
import matplotlib
matplotlib.use("TkAgg")

import dtmm
import numpy as np

#: Pixel size in nm
PIXELSIZE = 200
#: Set box dimensions
NLAYERS, HEIGHT, WIDTH = 60, 96, 96
#: Illumination wavelengths in nm
WAVELENGTHS = np.linspace(380, 780, 9)
#: create some experimental data (stack)
optical_data = dtmm.nematic_droplet_data(shape=(NLAYERS, HEIGHT, WIDTH),
                                         radius=30, profile="r",
                                         no=1.5, ne=1.6, nhost=1.5)
# Create aperture for illumination light
window = dtmm.aperture((96, 96))

#: create non-polarized input light
field_data_in = dtmm.field.illumination_data((HEIGHT, WIDTH), WAVELENGTHS,
                                             pixelsize=PIXELSIZE, window=window)
#: transfer input light through stack
field_data_out = dtmm.transfer_field(field_data_in, optical_data)

#: visualize output field
viewer = dtmm.field_viewer(field_data_out)
viewer.set_parameters(sample=0, intensity=2,
                      polarizer=0, focus=-14, analyzer=90)
fig, ax = viewer.plot()
# Show figure
fig.show()
