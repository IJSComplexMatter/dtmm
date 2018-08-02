Library Reference
=================

.. testsetup:: *

   from dtmm.color import *
   import numpy as np
   from dtmm.linalg import *

.. currentmodule:: dtmm.conf

Configuration (``dtmm.conf``)
-----------------------------

.. automodule:: dtmm.conf
   :members:

.. currentmodule:: dtmm.color

Color Management (``dtmm.color``)
---------------------------------

.. automodule:: dtmm.color

Load functions
++++++++++++++

.. autofunction:: dtmm.color.load_cmf

.. autofunction:: dtmm.color.load_tcmf

.. autofunction:: dtmm.color.load_specter

Conversion functions
++++++++++++++++++++


.. autofunction:: dtmm.color.specter2color

.. autofunction:: dtmm.color.apply_srgb_gamma

.. autofunction:: dtmm.color.apply_gamma

.. autofunction:: dtmm.color.spec2xyz

.. autofunction:: dtmm.color.xyz2srgb

.. autofunction:: dtmm.color.xyz2gray

.. currentmodule:: dtmm.data

Optical data (``dtmm.data``)
----------------------------

.. automodule:: dtmm.data

Creation functions
++++++++++++++++++

.. autofunction:: dtmm.data.nematic_droplet_data

.. autofunction:: dtmm.data.nematic_droplet_director

.. autofunction:: dtmm.data.reshape_volume

.. autofunction:: dtmm.data.rotate_director

.. autofunction:: dtmm.data.rot90_director

.. autofunction:: dtmm.data.sphere_mask

.. autofunction:: dtmm.data.validate_optical_data


Conversion functions
++++++++++++++++++++

.. autofunction:: dtmm.data.angles2director

.. autofunction:: dtmm.data.director2angles

.. autofunction:: dtmm.data.director2data

.. autofunction:: dtmm.data.director2order

.. autofunction:: dtmm.data.raw2director

.. autofunction:: dtmm.data.refind2eps

.. autofunction:: dtmm.data.uniaxial_order

IO functions
++++++++++++

.. autofunction:: dtmm.data.load_stack

.. autofunction:: dtmm.data.read_director

.. autofunction:: dtmm.data.read_raw

.. autofunction:: dtmm.data.save_stack


.. currentmodule:: dtmm.fft


FFT (``dtmm.fft``)
------------------

.. automodule:: dtmm.fft
   :members:


.. currentmodule:: dtmm.field


Field data (``dtmm.field``)
---------------------------

Creation functions
++++++++++++++++++

.. autofunction:: dtmm.field.illumination_betaphi

.. autofunction:: dtmm.field.illumination_data

.. autofunction:: dtmm.field.illumination_waves

.. autofunction:: dtmm.field.validate_field_data

Conversion functions
++++++++++++++++++++

.. autofunction:: dtmm.field.waves2field

.. autofunction:: dtmm.field.field2intensity

.. autofunction:: dtmm.field.field2specter

IO functions
++++++++++++

.. autofunction:: dtmm.field.load_field

.. autofunction:: dtmm.field.save_field



.. currentmodule:: dtmm.field_viewer

Field viewer (``dtmm.field_viewer``)
------------------------------------

.. automodule:: dtmm.field_viewer
   :members: field_viewer, FieldViewer, VIEWER_PARAMETERS

.. currentmodule:: dtmm.jones

Jones Calculus (``dtmm.jones``)
-------------------------------

.. automodule:: dtmm.jones

.. autofunction:: dtmm.jones.apply_polarizer_matrix

.. autofunction:: dtmm.jones.jonesvec

.. autofunction:: dtmm.jones.polarizer_matrix


4x4 linear algebra (``dtmm.linalg``)
------------------------------------

.. autofunction:: dtmm.linalg.inv4x4

.. currentmodule:: dtmm.transfer

Field transfer (``dtmm.transfer``)
----------------------------------

.. automodule:: dtmm.transfer
   :members: 


.. currentmodule:: dtmm.window

Windowing (``dtmm.window``)
---------------------------

.. automodule:: dtmm.window
   :members: aperture, blackman


