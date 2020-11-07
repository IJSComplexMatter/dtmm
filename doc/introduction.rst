Introduction
============

``dtmm`` is an electro-magnetic field transmission and reflection calculation engine and visualizer. It can be used for calculation of transmission or reflection properties of layered homogeneous or inhomogeneous materials, such as confined liquid-crystals with homogeneous or inhomogeneous director profile. *DTMM* stands for Diffractive Transfer Matrix Method and is an adapted Berreman 4x4 transfer matrix method and an adapted 2x2 extended Jones method. Details of the method are given in *... some future paper*.

.. seealso::
   
   You may also check `nemaktis`_, uses ``dtmm`` as one of the back-ends.

License
-------

``dtmm`` is released under MIT license so you can use it freely. Please cite the package. See the DOI badge in the `repository`_.

Contributors
------------

I thank the following people for contributing and for valuable discussions:

* Alex Vasile
* Guilhem Poy

Highlights
----------

* Easy-to-use interface.
* Nematic director, Q tensor and dielectric tensor import and conversion function.
* Computes transmission and reflection from the material (with interference and diffraction).
* Biaxial, uniaxial and isotropic material supported.
* Fast iterative algorithm for 3D data - with tunable accuracy.
* Non-iterative algorithm for 2D data - equivalent to the iterative algorithm with max accuracy settings. 
* Exact calculation for homogeneous layers (1D). 
* EM field visualizer (polarizing microscope simulator) allows you to simulate:

   * Light source intensity.
   * Polarizer/analyzer orientation and type (LCP, RCP or linear).
   * Phase retarders (lambda/4, lambda/2).
   * Sample rotation.
   * Focal plane adjustments.
   * Koehler illumination (field aperture).
   * Objective aperture.
   * Immersion or standard microscopes.
   * Cover glass aberration effects.

* Color rendering (RGB camera simulations based on CIE color matching functions). 
* Pre-defined spectral response for monochrome CMOS cameras. 
   
Status and limitations
----------------------

``dtmm`` was developed mainly for light propagation through liquid crystals, and as such, other use cases have not yet been fully tested or implemented. Also, in the current version some limitations apply, which will hopefully be resolved in future versions.
 
* Limited color rendering functions and settings - no white balance correction of computed images.
* Non-dispersive material only.

.. note::

   EM field propagation calculation based on the iterative and non-iterative approach for 2D and 3D is exact for homogeneous layers, but it is approximate for inhomogeneous layers. It works good for slowly varying refractive index material (e.g. confined liquid crystals with slowly varying director field). 

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

   >>> viewer = dtmm.pom_viewer(field_data_out)
   >>> viewer.set_parameters(sample = 0, polarizer = "h",
   ...      focus = -18, analyzer = "v")
   >>> fig, ax = viewer.plot() #creates matplotlib figure and axes
   >>> fig.show()


.. plot:: examples/hello_world.py

   Simulated optical polarizing microscope image of a nematic droplet with a radial nematic director profile (a point defect in the middle of the sphere). You can use sliders to change the focal plane, polarizer, sample rotation, analyzer, and light intensity.

Curious enough? Read the :ref:`quickstart`.

Contact
-------

Andrej {dot} Petelin {at} gmail {dot} com 

.. _repository: https://github.com/IJSComplexMatter/dtmm
.. _nemaktis: https://nemaktis.readthedocs.io



