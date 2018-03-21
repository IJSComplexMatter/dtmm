.. _quickstart:

Quickstart Guide
================

This is a quickstart guide, for a more thorough review and explanation you should go through the tutorials.


Data Model
----------

First, we should get familiar with the data model that is used internally for computations. There are two types of data, one describing the optical properties of the sample and the second describing the electro-magnetic field. We will call the first data an *optical_data* and the second one a *field_waves*. Also, the geometry and the coordinate system are explained.

Coordinate system and units
+++++++++++++++++++++++++++



* Wavelengths are defined in nanometers.
* Dimensions (coordinates) are defined in pixel units. 

In the computations, size of the pixel is defined as a parameter.


*optical_data*
++++++++++++++

It is best to go with the example:

.. doctest::

   >>> import dtmm
   >>> optical_data = dtmm.nematic_droplet_data((60, 128, 128), 
   ...    radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)

Here we have generated some test data, a nematic droplet with a radius of 30 pixels placed in a compute box of shape (60,128,128), that is, 60 layers of shape (128,128). Director profile is radial, with ordinary refractive index of 1.5 and extraordinary refractive index of 1.6 placed in an isotropic host with refractive index of 1.5. The optical data is a tuple of four arrays

.. doctest::

   >>> thickness, material_id, material_eps, angles = optical_data

Here, `thickness` describes the thickness of layer(s) in the optical data. It is a float measured in pixel units, in our case it is simply an array of ones of length 60:

.. doctest::

   >>> import numpy as np
   >>> np.allclose(thickness, np.ones(shape = (60,)))
   True 

which means that layer thickness is same as layer cross-section pixel size. `material_id` is an array of shape (60,128,128) of dtype "uint32" and describes the id index of the material. In our case we have two types of material, a host material (the one surrounding the droplet) with id of 0, and a liquid crystal material with material_id of 1. Let us plot the id index:

.. doctest::

   >>> fig = dtmm.plot_id(material_id, id = 1)
   >>> fig.show()

which plots dots at positions where liquid_crystal is defined. This plots a sphere centered in the conter of the compute box, as shown in Fig.

.. plot:: pyplots/plot_data_id.py

   LC is defined in a sphere 

`material_eps` is an array of shape (2,3) because we have in our data two types of material. Each material is defined by three complex dielectric tensor values (refractive indices squared):

.. doctest::

   >>> material_eps
   array([[ 2.25+0.j,  2.25+0.j,  2.25+0.j],
          [ 2.25+0.j,  2.25+0.j,  2.56+0.j]])
   
the first element of `material_eps` correspond to diagonal epsilon tensor of the isotropic host material with refractive index of 1.5, and the second element is a LC material with refractive indices (no,no,ne), which correspond to above eps values.

`angles` is an array of shape (60,128,128,3) and describe director angles in each point in the compute box. For isotropic material these are all zero, so outside of the sphere, these are all zero, while inside of the sphere, these define the length of the director, theta, and phi angles. We can plot these angles 

.. doctest::

   >>> fig = dtmm.plot_angles(angles, center = True, xlim = (-5,5), 
   ...              ylim = (-5,5), zlim = (-5,5))
   >>> fig.show()

.. note::

   matplotlib cannot handle quiver plot of large data sets, so here we limited dataset visualization to nearby center points.
    

.. plot:: pyplots/plot_data_angles.py

   LC director of the nematic droplet near the center of the sphere. Director is computed from director angles. 

Director length should normally be 1. However, you can set any value. This value us then used to compute the refractive indices of the material. In fact this value is treated as a nematic order parameter, which is used to compute the refractive indices from the following formula.

.. math:: 

   \epsilon_1 = \epsilon_{m} - 1/3  S  \epsilon_{a}

   \epsilon_2 = \epsilon_{m} - 1/3  S  \epsilon_{a}

   \epsilon_3 = \epsilon_{m} + 2/3  S  \epsilon_{a}
  

where :math:`\epsilon_{m}` is the mean value of dielectric tensor elements and :math:`\epsilon_{a} = \epsilon_{3}-\epsilon_{1}` is the anisotropy. 

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

Here we used a :func:`.waves.illumination_data` convenience function that builds the field_waves data for us and also gives us color matching function as a `cmf` array. We will deal with colors later, now let us look at the field_waves data. It is a tuple of two `ndarrays` :

.. doctest::

   >>> field, wavenumbers = field_waves
   >>> np.allclose(wavenumbers, 2*np.pi/wavelengths * pixelsize)
   True

where the `wavenumbers` are computed from the wavelengths and pixel size and define the k-values of the EM field array. Now, the `field` array shape:

.. doctest::

   >>> field.shape
   (2, 10, 4, 128, 128)

Coordinate system
+++++++++++++++++






