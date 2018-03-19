.. _quickstart:

Quickstart Guide
================

This is a quickstart guide, for a more thorough review and explanation you should go through the tutorials.

Data Model
----------

First, we should get familiar with the data model that is used internally for computations. There are two types of data, one describing the optical properties of the sample and the second describing the electro-magnetic field. We will call the first data an *optical_data* and the second one a *field_waves*.

Units
+++++

Wavelengths are defined in nanometers, while dimensions (coordinates) are defined in pixel units. In the computation, size of the pixel is defined as a parameter, and the rest of the physical parameters (dimensions, wavenumbers) are defined in pixel units.


*optical_data*
++++++++++++++

It is best to go with the example::

   >>> import dtmm
   >>> optical_data = dtmm.nematic_droplet_data((60, 128, 128), 
   ...    radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)

Here we have generated some test data, a nematic droplet with a radius of 30 pixels placed in a compute box of shape (60,128,128), that is, 60 layers of shape (128,128). Director profile is radial, with ordinary refractive index of 1.5 and extraordinary refractive index of 1.6 placed in an isotropic host with refractive index of 1.5. The optical data is a tuple of a scalar and three arrays::

   >>> thickness, material_id, material_eps, angles = optical_data

Here, `thickness` describes the thickness of layer(s) in the optical data. It is a float measured in pixel units, in our case it is simply::

   >>> thickness
   1.0 

which means that layer thickness is same as layer cross-section pixel size. `material_id` is an array of shape (60,128,128) of dtype "uint32" and describes the id index of the material. In our case we have two types of material, a host material (the one surrounding the droplet) with id of 0, and a liquid crystal material with material_id of 1. Let us plot the id index::

   >>> fig = dtmm.plot_id(material_id, index = 1)
   >>> fig.show()

which plots dots at positions where liquid_crystal is defined.

.. plot:: pyplots/plot_data_id.py

   LC is defined in a sphere 

`material_eps` is an array of shape (2,3) because we have in our data two types of material::

   >>> material_eps
   array([[ 2.25+0.j,  2.25+0.j,  2.25+0.j],
       [ 2.25+0.j,  2.25+0.j,  2.89+0.j]])
   
the first element of `material_eps` correspond to diagonal epsilon tensor of the isotropic host material with refractive index of 1.5, and the second element is a LC material with refractive indices (no,no,ne), which correspond to above eps values.

`angles` is an array of shape (60,128,128,3) and describe director angles in each point in the compute box. For isotropic material these are all zero, so outside of the sphere, these are all zero, while inside of the sphere, these define the length of the director, theta, and phi angles. We can plot these angles. Let us first crop the data

   >>> center_region = angles[24:-24,58:-58,58:-58]

which gives us angles near the center of the sphere. We can now plot these angles with::

   >>> fig = dtmm.plot_angles(center_region)
   >>> fig.show()
    

.. plot:: pyplots/plot_data_angles.py

   LC director of the nematic droplet.
  
*field_waves*
+++++++++++++

This data describes the electro-magnetic field. . Let us describe it by an example::

   >>> import numpy as np
   >>> pixelsize = 100
   >>> wavelengths = np.linspace(380,780,10)
   >>> shape = (128,128)
   >>> field_waves, cmf = dtmm.illumination_data(shape, wavelengths, 
   ...       pixelsize = pixelsize, refind = 1.5, diameter = 0.8, pol = None)

Here we used a :func:`.waves.illumination_data` convenience function that builds the field_waves data for us and also gives us color matching function as a `cmf` array. We will deal with colors later, now let us look at the field_waves data. It is a tuple of two `ndarrays` ::

   >>> field, wavenumbers = field_waves
   >>> np.allclose(wavenumbers, 2*np.pi/wavelengths * pixelsize)
   True

where the `wavenumbers` are computed from the wavelengths and pixel size and define the k-values of the EM field array. Now, the `field` array shape::

   >>> field.shape
   (2,10,4,128,128)




