Introduction
============

Diffractive transfer matrix method was developed mainly for calculating light transmission through Liquid-Crystal cells with homogeneous or inhomogeneous director profile and for visualization of the calculated field to simulate optical polarizing microscope images in liquid-crystal studies. It consists of 

* light and experiment setup functions,
* functions for electro-magnetic field (EMF) transmission/reflection calculation, 
* EMF viewing/plotting functions.

The method used is an adapted Berreman 4x4 transfer matrix method in which input light is a sum of plane waves. Two variations of light transmission calculation methods are available. In an exact approach, each of the input plane waves is transmitted through an inhomogeneous layer and then EMF is reconstructed after each layer, and the procedure is repeated layer-by-layer. In a faster approach, the material is assumed to be a layered material in which each of the layers is assumed to be a very thin phase plate that modifies the phase of the input wave front and a homogeneous layer through which the wave front is propagated. This is a much faster method, suitable for real-time calculations.

Example
-------

example

.. plot:: pyplots/example1.py

   Example plot




