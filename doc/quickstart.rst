.. _quickstart:

Quickstart Guide
================

This is a quickstart guide, you are advised to read this guide first and then go through the tutorials if needed. First, you will get familiar with the data model that is used internally for computations. Then you will learn how to construct the data with the provided helper functions and how to perform calculations for the most typical use case - propagation of light through a liquid-crystal cell with inhomogeneous director configuration.

.. _data-model:

Data Model
----------

There are two types of data, one describing the optical properties of the sample and the second describing the electro-magnetic field of the in-going and out-going light. We will call the first one an *optical_data* and the second one a *field_waves*. But first, let us define the coordinate system, units and conventions.

.. _conventions:

Coordinate system, units and conventions
++++++++++++++++++++++++++++++++++++++++

* Wavelengths are defined in nanometers.
* Dimensions (coordinates) are defined in pixel units. 
* In the computations, size of the pixel is defined as a parameter (in nanometers).
* Optical parameters (orientation of the optical axis) are defined relative to the reference laboratory coordinate frame *xyz*.  
* Propagation is said to be *forward propagation* if the wave vector has a positive *z* component. 
* Propagation is said to be *backward propagation* if the the wave vector has a negative *z* component.
* Light enters the material at *z=0*  at the bottom of the sample and exits at the top of the sample.
* The sample is defined by a single layer or sequence of layers - a *stack*. The first layer being the one at the bottom of the stack.
* Internally, optical parameters are stored in memory as a C-contiguous array. 
   * For 3D (*stack*) with axes (i,j,k,...) the axes are *z*, *y*, *x*, *parameter(s)*.
   * For 2D (*layer*) with axes (i,j,...) the axes are *y*, *x*, *parameter(s)*.
* Optical parameters data is stored in single precision "float32" and "complex64".
* Computation is done with double precision.

For uniaxial material, the orientation of the optical axis is defined with two angles. :math:`\theta_m` is an angle between the *z* axis and the optical axis  and :math:`\phi_m` is an angle between the projection of the optical axis vector on the *xy* plane and the *x* axis.

Direction of light propagation (wave vector) is defined with two parameters, :math:`\beta_x = n \sin\theta_k \cos\phi_k` and  :math:`\beta_y = n \sin\theta_k \sin\phi_k`, or equivalently :math:`\beta = \sqrt{\beta_x^2 + \beta_y^2} = n \sin\theta_k` and :math:`\phi_k = \arctan(\beta_y/\beta_x)`, where :math:`\theta_k` is an angle between the wave vector and the *z* axis, and :math:`\phi_k` between the projection of the wave vector on the *xy* plane and the *x* axis. 

Parameter :math:`\beta` is a fundamental parameter in transfer matrix method. This parameter is conserved when light is propagated through a stack of homogeneous layers.


.. note::

   In the current implementation, only isotropic and uniaxial material is implemented. Biaxial material is not yet supported.

.. _optical-data:

*optical_data*
++++++++++++++

.. doctest::

   >>> import dtmm
   >>> optical_data = dtmm.nematic_droplet_data((60, 128, 128), 
   ...    radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)

Here we have generated some test optical data, a nematic droplet with a radius of 30 pixels placed in a compute box of shape (60,128,128), that is, 60 layers of (height - *y*, width - *x*) of (128,128). Director profile is radial, with ordinary refractive index of 1.5 and extraordinary refractive index of 1.6 placed in an isotropic host with refractive index of 1.5. The optical data is a tuple of four arrays

.. doctest::

   >>> thickness, material_id, material_eps, eps_angles = optical_data

`thickness` describes the thickness of layer(s) in the optical data. It is a float (in case of a single layer) or array of floats (in case of a stack of layers). It is measured in pixel units. In our case it is an array of ones of length 60:

.. doctest::

   >>> import numpy as np
   >>> np.allclose(thickness, np.ones(shape = (60,)))
   True 

which means that layer thickness is the same as layer pixel size. `material_id` is an array of shape (60,128,128) of dtype "uint32" and describes the id index of the material. In our case we have two types of material, a host material (the one surrounding the droplet) with id of 0, and a liquid crystal material with material_id of 1. Let us plot the id index:

.. doctest::

   >>> fig = dtmm.plot_id(material_id, id = 1)
   >>> fig.show()

which plots dots at positions where liquid_crystal is defined. This plots a sphere centered in the conter of the compute box, as shown in Fig.

.. plot:: pyplots/plot_data_id.py

   LC is defined in a sphere 

`material_eps` is an array of shape (2,3) because we have in our data two types of material. Each material is defined by three complex dielectric tensor eigenvalues (refractive indices squared):

.. doctest::

   >>> material_eps
   array([[ 2.25+0.j,  2.25+0.j,  2.25+0.j],
          [ 2.25+0.j,  2.25+0.j,  2.56+0.j]])
   
The real part of the dielectric constant is the refractive index squared and the imaginary part determines absorption properties. The first element of `material_eps` correspond to diagonal epsilon tensor of the isotropic non-absorbing host material with refractive index of 1.5, and the second element is a non-absorbing LC material with refractive indices (no,no,ne).

`eps_angles` is an array of shape (60,128,128,3) and describe director angles in each point in the compute box. For isotropic material these are all zero, so outside of the sphere, these are all zero:

.. doctest::

   >>> eps_angles[0,0,0]
   array([ 0.,  0.,  0.], dtype=float32)

while inside of the sphere, these define the length of the director, theta, and phi angles. 


.. doctest::

   >>> eps_angles[30,64,64] #z=30, y = 64, x = 64
   array([ 0.99999994,  0.9553166 ,  0.78539819], dtype=float32)

The first element is always 1. because it defends the length of the director vector. the second value describes the :math:`\theta_m` angle and the last describes the :math:`\phi_m`  angle.

We can plot the director around the center (around the point defect) of the droplet by

.. doctest::

   >>> fig = dtmm.plot_angles(eps_angles, center = True, xlim = (-5,5), 
   ...              ylim = (-5,5), zlim = (-5,5))
   >>> fig.show()

.. note::

   matplotlib cannot handle quiver plot of large data sets, so here we limited dataset visualization to nearby center points.
    
.. plot:: pyplots/plot_data_angles.py

   LC director of the nematic droplet near the center of the sphere. Director is computed from director angles. 

Director length in the `eps_angles` data should normally be 1. However, you can set any value. This value is then used to compute the refractive indices of the material. In fact this value is treated as a nematic order parameter, which is used to compute the refractive indices from the following formula:

.. math:: 

   \epsilon_1 = \epsilon_{m} - 1/3  S  \epsilon_{a}

   \epsilon_2 = \epsilon_{m} - 1/3  S  \epsilon_{a}

   \epsilon_3 = \epsilon_{m} + 2/3  S  \epsilon_{a}
  

where :math:`\epsilon_{m}` is the mean value of dielectric tensor elements and :math:`\epsilon_{a} = \epsilon_{3}-\epsilon_{1}` is the anisotropy. 

.. _field-waves:

*field_waves*
+++++++++++++

This data describes the electro-magnetic field. . Let us describe it by an example:

.. doctest::

   >>> import numpy as np
   >>> pixelsize = 100
   >>> wavelengths = np.linspace(380,780,10)
   >>> shape = (128,128)
   >>> field_waves, cmf = dtmm.illumination_data(shape, wavelengths, 
   ...       pixelsize = pixelsize, refind = 1.5, diameter = 0.8, pol = None)

Here we used a :func:`.waves.illumination_data` convenience function that builds the field_waves data for us and also returns a color matching function as a `cmf` array. We will deal with colors later, now let us look at the field_waves data. It is a tuple of two `ndarrays` :

.. doctest::

   >>> field, wavenumbers = field_waves
   >>> np.allclose(wavenumbers, 2*np.pi/wavelengths * pixelsize)
   True

where the `wavenumbers` are computed from the wavelengths and pixel size and define the k-values of the EM field array. Now, the `field` array shape:

.. doctest::

   >>> field.shape
   (2, 10, 4, 128, 128)

Director Data IO 
----------------

There are several ways to create optical data. You can do it manually by setting all `optical_data` elements according to the data format explained in :ref:`optical-data`. However, there are helper functions to ease the data creation. We will cover creation of nematic cell optical data from file.

Most likely you have director data stored in a raw or text file. Let us create a sample director data (from the previous example) to work with. If you have some data prepared in a file, skip this step.

.. doctest::

    >>> director_sample = dtmm.nematic_droplet_director((60, 96, 128), radius = 30, profile = "r")
    >>> director_sample.tofile("director.raw")

Here we have generated a director data array an stored it to a binary file written in C-order and  system endianness called "director.raw". The data stored in this file is of shape (60,96,128,3). To load this data from file you can use the :func:`dtmm.read_director` helper function.

.. doctest::

    >>> director = dtmm.read_director("director.raw", (60,96,128,3), order = "zyxn")

By default, data is assumed to be stored in single precision (float) and with "zyxn" data order and system endianness. If you have data in double precision and different order, these have to be specified. For instance, if data is in "xyzn" order, meaning that first axis is "x", and third axis is "z" coordinate (layer index) and the last axis is the director vector, and the data is in double precision little endianness , do::

    >>> director = dtmm.read_director("test.raw", (128,96,60,3),
    ...        order = "xyzn", dtype = "float64", endian = "little")

This will read director data and transpose it to shape (60,96,128,3) 

   





Transmission Calculation
------------------------

Field Viewer
------------








