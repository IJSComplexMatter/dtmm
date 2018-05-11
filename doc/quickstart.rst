.. _quickstart:

Quickstart Guide
================

Here you will learn how to construct optical data with the provided helper functions and how to perform calculations for the most typical use case that this package was designed for - propagation of light through a liquid-crystal cell with inhomogeneous director configuration, and visualization (simulation of the optical polarizing microscope imaging). If you are curious about the implementation details you are also advised to read the :ref:`data-model` first and then go through the examples below. More detailed examples  and tutorials are given in the :ref:`tutorial`. 

Importing Data
--------------

Most likely you have a director data stored in a raw or text file. Let us create a sample director data to work with.

.. doctest::
  
    >>> import dtmm
    >>> NLAYERS, HEIGHT, WIDTH = (60, 96, 96)
    >>> director = dtmm.nematic_droplet_director((NLAYERS, HEIGHT, WIDTH), radius = 30, profile = "r")
    >>> director.tofile("director.raw")

Here we have generated a director data array and saved it to a binary file written in system endianness called "director.raw". The data stored in this file is of shape (60,96,96,3), that is (NLAYERS, HEIGHT, WIDTH, 3). To load this data from file you can use the :func:`dtmm.read_director` helper function:

.. doctest::

    >>> director = dtmm.read_director("director.raw", (NLAYERS, HEIGHT, WIDTH ,3))

By default, data is assumed to be stored in single precision (float) and with "zyxn" data order and system endianness. If you have data in double precision and different order, these have to be specified. For instance, if data is in "xyzn" order, meaning that first axis is "x", and third axis is "z" coordinate (layer index) and the last axis is the director vector, and the data is in double precision little endianness, do::

    >>> director = dtmm.read_director("test.raw", (WIDTH,HEIGHT,NLAYERS,3),
    ...        order = "xyzn", dtype = "float64", endian = "little")

This reads director data and transposes it from (WIDTH, HEIGHT, NLAYERS,3) to shape (NLAYERS, HEIGHT, WIDTH, 3), a data format that is used internally for computation. Now we can build the optical data from the director array by providing the refractive indices of the liquid crystal material.

.. doctest::

   >>> optical_data = dtmm.director2data(director, no = 1.5, ne = 1.6)

This converts director to a valid :ref:`optical-data`. Director array should be an array of director vectors. Length of the vector should generally be 1. Director length is used to determine the dielectric tensor of the material. See :ref:`optical-data` for details. You can also set the director mask. For instance, a sphere mask of radius 30 pixels can be defined by

.. doctest::

   >>> mask = dtmm.sphere_mask((NLAYERS,HEIGHT,WIDTH),30)  
 
With this mask you can construct optical data of a nematic droplet in a host material with refractive index of 1.5:

.. doctest::

   >>> optical_data = dtmm.director2data(director, no = 1.5, ne = 1.6, mask = mask, nhost = 1.5)

Of course you can provide any mask, just that the shape of the mask must mach the shape of the bounding box of the director - (60,96,96) in our case. This way you can crop the director field to any volume shape and put it in a host material with the above helper function. 

.. note::

   For testing, there is a :func:`dtmm.nematic_droplet_data` function that you can call to construct a test data of nematic droplet data directly. See :ref:`optical-data` for details.

For a more complex data creation please refer to the :ref:`optical-data` format and tutorials.

Transmission Calculation
------------------------

In this part we will cover transmission calculation and light creation functions for simulating optical polarizing microscope images. First we will create and compute the transmission of a single plane wave and then show how to compute multiple rays (multiple plane waves with different ray directions) in order to simulate finite numerical aperture of the illuminating light field.

Single ray
++++++++++

Now that we have defined the sample data we need to construct initial (input) electro-magnetic field. Electro magnetic field is defined by an array of shape *(4,height,width)* where the first axis defines the component of the field, that is, an :math:`E_x`, :math:`H_y`, :math:`E_y` and :math:`H_x` components of the EM field specified at each of the (y,x) coordinates. Typically, you will calculate transmission spectra, so multiple  wavelengths need to be simulated. A multi-wavelength field has a shape of (n_wavelengths,4,height,width). You can define a multi-wavelength input light electro-magnetic field data with a :func:`dtmm.illumination_data` helper function. 

.. doctest::

   >>> import numpy as np
   >>> WAVELENGTHS = np.linspace(380,780,10)
   >>> field_data = dtmm.illumination_data((HEIGHT,WIDTH), WAVELENGTHS, pixelsize = 200, jones = (1,0)) 

Here we have defined an x-polarized light (we used jones vector of (1,0)). A left-handed circular polarized light light can be defined by:: 

   >>> jones = (1/2**0.5,1j/2**0.5)
   >>> field_data_in = dtmm.illumination_data((HEIGHT,WIDTH), WAVELENGTHS, pixelsize = 200, jones = jones) 

Typically, you will want input light to be non-polarized. A non-polarized light is taken to be a combination of *x* and *y* polarizations that are transmitted independently and the resulting intensity measured by the detector is an incoherent addition of both of the contributions from both of the two polarizations. So to simulate a non-polarized light, you have to compute both of the polarization states. The illumination_data function can be used to cunstruct such data. Just specify jones parameter to None or call the function without the jones parameter:

.. doctest::

   >>> field_data_in = dtmm.illumination_data((HEIGHT,WIDTH), WAVELENGTHS, pixelsize = 200) 

With the input light specified, you can now transfer this field through the stack

.. doctest::

   >>> field_data_out = dtmm.transfer_field(field_data_in, optical_data)


Multiple rays
+++++++++++++

If you want to simulate multiple rays (multiple plane waves), directions of these rays have to be defined. A simple approach is to use the illumination_betaphi helper function. This function returns beta values and phi values of the input rays for a specified numerical aperture of the illumination. 

.. note::

   Beta is a sine of ray angle towards the z axis. See :ref:`data-model` for details.

For numerical aperture of NA = 0.1 you can call

.. doctest::

   >>> beta, phi = dtmm.illumination_betaphi(0.1, 13)

which constructs direction parameters (beta, phi) of input rays of numerical aperture of 0.1 and with approximate number of rays of 13. In our case 

.. doctest::

   >>> len(beta)
   13
 
we have 13 rays evenly distributed in a cone of numerical aperture of 0.1. To calculate the transmitted field we now have to pass these ray parameters to the transmit_field function::

   >>> field_data_in = dtmm.illumination_data((HEIGHT,WIDTH), WAVELENGTHS, pixelsize = 200, beta = beta, phi = phi)
   >>> field_data_out = dtmm.transfer_field(field_data_in, optical_data, beta = beta, phi = phi)

.. warning::

   When doing multiple ray computation, the beta and phi parameters in the transmit_field function must match the beta and phi parameters that were used to generate input field. Do not forget to pass the beta, phi values to the appropriate functions.

Field Viewer
------------

Once the transmitted field has been calculated, we can simulate optical polarizing microscope image formation with the FieldViewer object. The output field is a calculated EM field at the exit surface of the optical stack. As such it can be further propagated and optical polarizing microscope image formation can be performed. Instead of doing full optical image formation calculation one can take the computed field and propagate it in space a little (forward or backward) from the initial position. This way one can calculate light intensities that would have been measured by a camera equipped microscope, had the field been propagated through an ideal microscope objective with a 1:1 magnifications and by not introducing any aberrations. Simply do:

Simply do:

.. doctest::

   >>> viewer = dtmm.field_viewer(field_data_out)

which returns a FieldViewer object. Now you can calculate transmission specter or obtain RGB image. Depending on how the illumination data was created (polarized/nonpolarized light, single/multiple ray) you can set different parameters. For instance, you can refocus the field

.. doctest::

   >>> viewer.focus = -20 

The calculated output field is defined at zero focus. To move the focus position more into the sample, you have to move focus to negative values. Next, you can set the analyzer.

.. doctest::

   >>> viewer.analyzer = 90 #in degrees - vertical direction

If you do not wish to use analyzer, simply remove it by specifying

.. doctest::

   >>> viewer.analyzer = None
   
To adjust the intensity of the input light you can set:

.. doctest::

   >>> viewer.intensity = 0.5

If input field was defined to be non polarized, you can set the polarizer

   >>> viewer.polarizer = 0. # horizontal

You can set all these parameters with a single call to:

.. doctest::

   >>> viewer.set_parameters(intensity = 1., polarizer = 0., analyzer = 90, focus = -20)

When you are done with setting the microscope parameters you can calculate the transmitted specter

.. doctest::

   >>> specter = viewer.calculate_specter()

or, if you want to obtain RGB image:

.. doctest::

   >>> image = viewer.calculate_image()

The viewer also allows you to play with the microscope settings dynamically. 

.. doctest::

   >>> fig, ax = viewer.plot()
   >>> fig.show()

.. plot:: examples/hello_world.py

   Microscope image formed by an ideal 1:1 objective.

.. note:: 

    For this to work you should not use the matplotlib figure inline option in your python development environment (e.g. Spyder, jupyterlab, notebook). Matpoltlib should be able to draw to a new figure widget for sliders to work. 

For more advanced image calculation, using windowing, reflection calculations, custom color matching functions please refer to the :ref:`Tutorial`
  




