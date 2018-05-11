Introduction
============

``dtmm`` is a simple-to-use light (electro-magnetic field) transmission and reflection calculation engine and visualizer. It can be used for calculation of transmission or reflection properties of layered homogeneous or inhomogeneous materials, such as liquid-crystal with homogeneous or inhomogeneous director profile. *DTMM* stands for Diffractive Transfer Matrix Method and is an adapted Berreman 4x4 transfer matrix method. Details of the method are given in *... some future paper*.

.. note::

   This package is still in its early stage of development, so it should be considered experimental. No official release exists yet! The package and the documentation is actively being worked on, so please stay tuned. The core functionality has been defined and the package is ready for use and testing, but much needs to be done to documentation and code cleanup, etc.. Expect also some minor API changes in the near future.

License
-------

``dtmm`` will be released under MIT license so you will be able to use it freely, however, I will ask you to cite the *... some future paper*. In the mean time you can play with the current development version freely, but i kindly ask you not to redistribute the code or  publish data obtained with this package. **Please wait until the package is officially released!**

Highlights
----------

* Easy to use interface.
* Computes transmission and reflection from the material (Includes interference and diffraction).
* Exact calculation for homogeneous layers.
* EMF visualizer (polarizing microscope simulator) - can be used with external computed data:

   * Polarizer/analizer rotation.
   * Sample rotation.
   * Refocusing.
   
   
Status and limitations
----------------------

``dtmm`` is a young (experimental) project. The package was developed mainly for light propagation through liquid crystals, and as such, other use cases have not yet been fully tested or implemented. Also, in the current version some limitations apply, which will hopefully be resolved in future versions:

* Uniaxial material only - biaxial material is not yet supported.
* Inhomogeneous layers with low birefringence only - double refractions are neglected. 
* Limited color rendering functions and settings - D65 illumination only.
* No absorption yet - real dielectric tensors only.
* Non-dispersive material only. 
* Interference/reflections cannot be disabled.
* Limited data IO functions.

.. note::

   EMF field propagation calculation is exact for homogeneous layers, but it is only approximate for inhomogeneous layers. It works reasonably well for slowly varying refractive index material (e.g. confined liquid crystals with slowly varying director field). A more accurate (and much slower) propagation is planned in future releases.

Other than that, the package is fully operational. Try the example below to get an impression on how it works.

Example
-------

.. doctest::

   >>> import dtmm
   >>> import numpy as np
   >>> NLAYERS, HEIGHT, WIDTH = (60, 96, 96)
   >>> WAVELENGTHS = np.linspace(380,780,10)

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
   ...      focus = -20, analyzer = 90)
   >>> fig, ax = viewer.plot() #creates matplotlib figure and axes
   >>> fig.show()


.. plot:: examples/hello_world.py

   Simulated optical polarizing microscope image of a nematic droplet with a radial nematic director profile (a point defect in the middle of the sphere). You can use sliders to change the focal plane, polarizer, sample rotation, analyzer, and light intensity.


Curious enough? Read the :ref:`quickstart`.



