Introduction
============

The `dtmm` package is a simple-to-use light (electro-magnetic field) transmission calculation engine and visualizer. It can be used for calculation of transmission or reflection properties of layered homogeneous or inhomogeneous materials, such as liquid-crystal cells with homogeneous or inhomogeneous director profile. *DTMM* stands for Diffractive Transfer Matrix Method and is an adapted Berreman 4x4 transfer matrix method. Details of the method are given in *TBD*. The package consist of:

* light and experiment setup functions,
* electro-magnetic field (EMF) transmission/reflection calculation functions, 
* EMF viewing/plotting functions.

.. note::

   This package is still in its early stage of development, and it should be considered experimental. 


Example
-------

To get an idea, how this can be used, consider this::

   >>> import dtmm
   >>> stack, mask, material = dtmm.nematic_droplet_data((60, 128, 128), 
   ...    radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)

which creates some sample data, a nematic droplet of radius 30 pixels with radial director profile in a compute box of 60 layers and layer shape of 128x128. To define input light (input electro-magnetic field)::

   >>> field, wavenumbers, cmf = dtmm.illumination_data((128,128), range(380,780,40),
   ...      refind = 1.5, pixelsize = 400, diameter = 0.8) 

which defines light field, wavenumbers and color matching table between 380 nm and 780 nm in steps of 40 nm. This field is then  propagated through the sample::

   >>> out = dtmm.transmit_field(field, wavenumbers, stack, material, mask)

There is a also a simple visualizer of the field::

   >>> viewer = dtmm.field_viewer(out, wavenumbers, cmf, refind = 1.5, 
   ...     sample = 0, polarizer = 0, focus = -30, analizer = 90)
   >>> viewer.plot()
   >>> viewer.show() #calls matplotlib show


.. plot:: pyplots/example1.py

   Simulated optical polarizing microscope image of a radial nematic director profile in a nematic droplet.

Curious enough? Read the :ref:`quickstart`.


