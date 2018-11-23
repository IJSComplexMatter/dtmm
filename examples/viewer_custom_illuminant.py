"""Same as hello world, but with a different illuminant"""

import dtmm
import numpy as np

#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 60, 96, 96
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,9)
#: create some experimental data (stack)
optical_data = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), 
          radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)

#: create non-polarized input light
field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, 
                                            pixelsize = PIXELSIZE) 
#: transfer input light through stack
field_data_out = dtmm.transfer_field(field_data_in, optical_data)

#: here we load illuminant A tcmf table, but you can provide your own light source.
# the illuminant argument defines the name or path to the illuminant data.
# Illuminant data format is a two column txt data file. The first column defines
# wavelength in nm, the second column the intensity  (normalization is 
# performed automatically when loading).

cmf = dtmm.load_tcmf(WAVELENGTHS, illuminant = "A")

#: visualize output field
viewer = dtmm.field_viewer(field_data_out, cmf = cmf)
viewer.set_parameters(sample = 0, intensity = 2,
                polarizer = 0, focus = -18, analyzer = 90)

fig,ax = viewer.plot()
