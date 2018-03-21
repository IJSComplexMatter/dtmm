Introduction
============

The `dtmm` package is a simple-to-use light (electro-magnetic field) transmission and reflection calculation engine and visualizer. It can be used for calculation of transmission or reflection properties of layered homogeneous or inhomogeneous materials, such as liquid-crystal cells with homogeneous or inhomogeneous director profile. *DTMM* stands for Diffractive Transfer Matrix Method and is an adapted Berreman 4x4 transfer matrix method. Details of the method are given in *TBD*. The package consist of:

* light and experiment setup functions,
* electro-magnetic field (EMF) transmission/reflection calculation functions, 
* EMF viewing/plotting functions.

.. note::

   This package is still in its early stage of development, and it should be considered experimental. 


Example
-------

.. doctest::

   >>> import dtmm
   >>> optical_data = dtmm.nematic_droplet_data((60, 128, 128), 
   ...     radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)

This creates some optical data, a nematic droplet of radius 30 pixels with radial director profile in a compute box of 60 layers and layer shape of 128x128. This is for testing only. In a real experiment, user has to construct the data with tools provided in the package. Next, define input light (input electro-magnetic field):

.. doctest::

   >>> field_waves, cmf = dtmm.illumination_data((128,128), range(380,780,40),
   ...      refind = 1.5, pixelsize = 400, diameter = 0.8) 

This creates light field, wavenumbers and color matching table between 380 nm and 780 nm in steps of 40 nm. This field is then propagated through the sample:

.. doctest::

   >>> out = dtmm.transmit_field(field_waves, data)

Finally, the transmitted field can be visualized:

.. doctest::

   >>> viewer = dtmm.field_viewer(out, cmf, refind = 1.5, 
   ...     sample = 0, polarizer = 0, focus = -30, analizer = 90)
   >>> fig = viewer.plot()
   >>> fig.show() #calls matplotlib show


.. plot:: pyplots/example1.py

   Simulated optical polarizing microscope image of a nematic droplet with a radial nematic director profile. You can use sliders to change the focal plane, polarizer,  sample rotation, analizer, and light intensity.

Curious enough? Read the :ref:`quickstart`.


