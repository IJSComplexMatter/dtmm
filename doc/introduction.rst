Introduction
============

The `dtmm` package stands for Diffractive Transfer Matrix Method. It is a simple-to-use light (electro-magnetic field) transmission calculation engine and visualizer. One of its main uses are for calculating transmission properties of liquid-crystal cells with homogeneous or inhomogeneous director profile. The method used is an adapted Berreman 4x4 transfer matrix method. Details of the method are given in ... The package consist of:

* light and experiment setup functions,
* electro-magnetic field (EMF) transmission/reflection calculation functions, 
* EMF viewing/plotting functions.

.. note::

   This package is still in its early stage of development. API is not yet stable, and the package can only be used for testing at this stage.



.. The method used is an adapted Berreman 4x4 transfer matrix method in which the input light electro-magnetic field is decomposed as a sum of plane waves. Two variations of light transmission calculation methods are available. In an exact approach, each of the input plane waves is transmitted through an inhomogeneous layer and then EMF is reconstructed after each layer, and the procedure is repeated layer-by-layer. In a faster approach, the material is assumed to be a layered material in which each of the layers is assumed to be a very thin phase plate that modifies the phase of the input wave front and a homogeneous layer through which the wave front is propagated. This is a much faster method, suitable for real-time calculations.

Example
-------

.. literalinclude:: pyplots/example1.py

.. plot:: pyplots/example1.py

   Example plot




