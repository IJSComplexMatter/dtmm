:mod:`dtmm.window`
==================

.. py:module:: dtmm.window

.. autoapi-nested-parse::

   Window functions.



Module Contents
---------------

.. function:: blackman(shape, out=None)

   Returns a blacman window of a given shape.

   :param shape: A shape of the 2D window.
   :type shape: (int,int)
   :param out: Output array.
   :type out: ndarray, optional

   :returns: **window** -- A Blackman window
   :rtype: ndarray


.. function:: gaussian(shape, waist)

   Gaussian amplitude window function.

   :param shape: A shape of the 2D window
   :type shape: (int,int)
   :param waist: Waist of the gaussian.
   :type waist: float

   :returns: **window** -- Gaussian beam window
   :rtype: ndarray


.. function:: gaussian_beam(shape, waist, k0, z=0.0, n=1)

   Returns gaussian beam window function.

   :param shape: A shape of the 2D window
   :type shape: (int,int)
   :param waist: Waist of the beam
   :type waist: float
   :param k0: Wavenumber
   :type k0: float
   :param z: The z-position of waist (0. by default)
   :type z: float, optional
   :param n: Refractive index (1. by default)
   :type n: float, optional
   :param out: Output array.
   :type out: ndarray, optional

   :returns: **window** -- Gaussian beam window
   :rtype: ndarray


.. function:: aperture(shape, diameter=1.0, smooth=0.05, out=None)

   Returns aperture window function.

   Aperture is basically a tukey filter with a given diameter

   :param shape: A shape of the 2D window
   :type shape: (int,int)
   :param diameter: Width of the aperture (1. for max height/width)
   :type diameter: float
   :param smooth: Smoothnes parameter - defines smoothness of the edge of the aperture
                  (should be between 0. and 1.; 0.05 by default)
   :type smooth: float
   :param out: Output array.
   :type out: ndarray, optional

   :returns: **window** -- Aperture window
   :rtype: ndarray


