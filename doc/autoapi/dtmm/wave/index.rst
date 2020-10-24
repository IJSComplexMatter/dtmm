:mod:`dtmm.wave`
================

.. py:module:: dtmm.wave

.. autoapi-nested-parse::

   Wave creation and wave characterization functions.



Module Contents
---------------

.. function:: betaphi(shape, k0, out=None)

   Returns beta and phi arrays of all possible plane eigenwaves.

   :param shape: Shape of the plane eigenwave.
   :type shape: (int,int)
   :param k0: Wavenumber in pixel units.
   :type k0: float
   :param out: Output arrays tuple
   :type out: (ndarray, ndarray), optional

   :returns: beta, phi arrays
   :rtype: array, array


.. function:: betaxy(shape, k0, out=None)

   Returns betax, betay arrays of plane eigenwaves.

   :param shape: Shape of the plane eigenwave.
   :type shape: (int,int)
   :param k0: Wavenumber in pixel units.
   :type k0: float
   :param out: Output arrays tuple
   :type out: (ndarray, ndarray), optional

   :returns: beta, phi arrays
   :rtype: array, array


.. function:: k0(wavelength, pixelsize=1.0)

   Calculate wave number in vacuum from a given wavelength and pixelsize

   :param wavelength: Wavelength in nm
   :type wavelength: float or array of floats
   :param pixelsize: Pixelsize in nm
   :type pixelsize: float

   :returns: Wavenumber array
   :rtype: array


.. function:: wavelengths(start=380, stop=780, count=9)

   Raturns wavelengths (in nanometers) equaly spaced in wavenumbers between
   start and stop.

   :param start: First wavelength
   :type start: float
   :param stop: Last wavelength
   :type stop: float
   :param count: How many wavelengths
   :type count: int

   :returns: **out** -- A wavelength array
   :rtype: ndarray


.. function:: eigenwave(shape, i, j, amplitude=None, out=None)

   Returns a planewave with a given fourier coefficient indices i and j.

   :param shape: Shape of the plane eigenwave.
   :type shape: (int,int)
   :param i: i-th index of the fourier coefficient
   :type i: int
   :param j: j-th index of the fourier coefficient
   :type j: float
   :param amplitude: Amplitude of the fourier mode.
   :type amplitude: complex
   :param out: Output array
   :type out: ndarray, optional

   :returns: Plane wave array.
   :rtype: array


.. function:: planewave(shape, k0, beta, phi, out=None)

   Returns a 2D planewave array with a given beta, phi, wave number k0.

   :param shape: Shape of the plane eigenwave.
   :type shape: (int,int)
   :param k0: Wavenumbers in pixel units.
   :type k0: float or array of floats
   :param beta: Beta parameter of the plane wave
   :type beta: float
   :param phi: Phi parameter of the plane wave
   :type phi: float
   :param out: Output arrays tuple
   :type out: (ndarray, ndarray), optional

   :returns: Plane wave array.
   :rtype: array


