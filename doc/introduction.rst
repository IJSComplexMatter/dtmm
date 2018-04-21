Introduction
============

The ``dtmm`` package is a simple-to-use light (electro-magnetic field) transmission and reflection calculation engine and visualizer. It can be used for calculation of transmission or reflection properties of layered homogeneous or inhomogeneous materials, such as liquid-crystal cells with homogeneous or inhomogeneous director profile. *DTMM* stands for Diffractive Transfer Matrix Method and is an adapted Berreman 4x4 transfer matrix method. Details of the method are given in *TBD*.

.. note::

   This package is still in its early stage of development, and it should be considered experimental. 

Highlights
----------

* Easy to use interface.
* Computes transmission and reflection from the material (Includes interference effects and diffraction).
* Exact calculation for homogeneous layers.
* EMF visualizer (polarizing microscope simulator) - can be used with external computed data:

   * Polarizer/analizer rotation.
   * Sample rotation.
   * Refocusing.
   
   
Status and limitations
----------------------

``dtmm`` is a young (experimental) project and no official release exists yet. The package was developed mainly for light propagation through liquid crystals, and as such, other use cases have not yet been tested or implemented. Also, in the current version, some limitations apply, which will be resolved in future versions:

* Uniaxial material only - biaxial material is not yet supported.
* Inhomogeneous layers with low birefringence material only - no double refractions. 
* Limited color rendering functions and settings - D65 illumination only.
* No absorption yet - real dielectric tensors only.
* Single call multi-wavelength calculation for non-dispersive material only. 

.. note::

   The package comes with a fast EMF field propagation calculation which is exact for homogeneous layers, and only approximate for inhomogeneous. It works reasonably well for slowly varying refractive index material (e.g. confined liquid crystals with slowly varying director field). A more accurate (and much slower) propagation is planned in future releases.


Example
-------


.. doctest::

   >>> import dtmm
   >>> NLAYERS, HEIGHT, WIDTH = (60, 96, 96)
   >>> WAVELENGTHS = range(380,780,40)

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

   >>> field_data_out = dtmm.transmit_field(field_data_in, optical_data)

Visualize the transmitted field with matplotlib plot:

.. doctest::

   >>> viewer = dtmm.field_viewer(field_data_out, sample = 0, polarizer = 0,
   ...      focus = -20, analizer = 90)
   >>> fig, ax = viewer.plot() #creates matplotlib figure and axes
   >>> fig.show()

.. plot:: ../examples/hello_world.py

   Simulated optical polarizing microscope image of a nematic droplet with a radial nematic director profile. You can use sliders to change the focal plane, polarizer,  sample rotation, analizer, and light intensity.

Curious enough? Read the :ref:`quickstart`.




