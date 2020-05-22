Introduction
============

``dtmm`` is an electro-magnetic field transmission and reflection calculation engine and visualizer. It can be used for calculation of transmission or reflection properties of layered homogeneous or inhomogeneous materials, such as confined liquid-crystals with homogeneous or inhomogeneous director profile. *DTMM* stands for Diffractive Transfer Matrix Method and is an adapted Berreman 4x4 transfer matrix method and an adapted 2x2 extended Jones method. Details of the method are given in *... some future paper*.

.. note::

   This package is still in its early stage of development, so it should be considered experimental. The core functionality has been defined and the package is ready for use.

License
-------

``dtmm`` is released under MIT license so you can use it freely. Please cite the package. See the DOI badge in the `repository`_.

Contributors
------------

I thank the following people for contributing:

* Alex Vasile

Highlights
----------

* Easy-to-use interface.
* Computes transmission and reflection from the material (with interference and diffraction).
* Biaxial, uniaxial and isotropic material supported.
* Exact calculation for homogeneous layers. Two different approximate methods for inhomogeneous layers.
* EMF visualizer (polarizing microscope simulator) - can be used with external computed data:

   * Polarizer/analizer rotation.
   * Sample rotation.
   * Refocusing.
   
   
Status and limitations
----------------------

``dtmm`` is a young (experimental) project. The package was developed mainly for light propagation through liquid crystals, and as such, other use cases have not yet been fully tested or implemented. Also, in the current version some limitations apply, which will hopefully be resolved in future versions:
 
* Limited color rendering functions and settings - no white balance correction of computed imaged.
* Non-dispersive material only. 
* Limited data IO functions.
* Two approximate methods for slowly varying medium:

   * An `effective` method : tunable accuracy (can be very fast) 
   * A `full` method : very slow, most accurate.

.. note::

   EMF field propagation calculation is exact for homogeneous layers, but it is only approximate for inhomogeneous layers. It works reasonably well for slowly varying (within the layer) refractive index material (e.g. confined liquid crystals with slowly varying director field).  

Other than that, the package is fully operational. Try the example below to get an impression on how it works.

Example
-------

.. doctest::

   >>> import dtmm
   >>> import numpy as np
   >>> NLAYERS, HEIGHT, WIDTH = (60, 96, 96)
   >>> WAVELENGTHS = np.linspace(380,780,9)

Build sample optical data:

.. doctest::

   >>> optical_data = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), 
   ...     radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)

Build illumination data (input EM field):

.. doctest::

   >>> field_data_in = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS,
   ...       pixelsize = 200) 

Transmit the field through the sample:

.. doctest::

   >>> field_data_out = dtmm.transfer_field(field_data_in, optical_data)

Visualize the transmitted field with matplotlib plot:

.. doctest::

   >>> viewer = dtmm.field_viewer(field_data_out)
   >>> viewer.set_parameters(sample = 0, polarizer = 0,
   ...      focus = -18, analyzer = 90)
   >>> fig, ax = viewer.plot() #creates matplotlib figure and axes
   >>> fig.show()


.. plot:: examples/hello_world.py

   Simulated optical polarizing microscope image of a nematic droplet with a radial nematic director profile (a point defect in the middle of the sphere). You can use sliders to change the focal plane, polarizer, sample rotation, analyzer, and light intensity.


Curious enough? Read the :ref:`quickstart`.

.. _repository: https://github.com/IJSComplexMatter/dtmm


