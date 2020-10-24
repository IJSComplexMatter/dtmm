:mod:`dtmm.fft`
===============

.. py:module:: dtmm.fft

.. autoapi-nested-parse::

   Custom 2D FFT functions.

   numpy, scipy and mkl_fft do not have fft implemented such that output argument
   can be provided. This implementation adds the output argument for fft2 and
   ifft2 functions.

   Also, for mkl_fft and scipy, the computation can be performed in parallel using ThreadPool.



Module Contents
---------------

.. function:: fft2(a, out=None)

   Computes fft2 of the input complex array.

   :param a: Input array (must be complex).
   :type a: array_like
   :param out: Output array. Can be same as input for fast inplace transform.
   :type out: array or None, optional

   :returns: **out** -- Result of the transformation along the last two axes.
   :rtype: complex ndarray


.. function:: ifft2(a, out=None)

   Computes ifft2 of the input complex array.

   :param a: Input array (must be complex).
   :type a: array_like
   :param out: Output array. Can be same as input for fast inplace transform.
   :type out: array or None, optional

   :returns: **out** -- Result of the transformation along the last two axes.
   :rtype: complex ndarray


.. function:: mfft2(a, overwrite_x=False)

   Computes matrix fft2 on a matrix of shape (..., n,n,4,4).

   This is identical to np.fft2(a, axes = (-4,-3))

   :param a: Input array (must be complex).
   :type a: array_like
   :param overwrite_x: Specifies whether original array can be destroyed (for inplace transform)
   :type overwrite_x: bool

   :returns: **out** -- Result of the transformation along the (-4,-3) axes.
   :rtype: complex ndarray


.. function:: mifft2(a, overwrite_x=False)

   Computes matrix ifft2 on a matrix of shape (..., n,n,4,4).

   This is identical to np.ifft2(a, axes = (-4,-3))

   :param a: Input array (must be complex).
   :type a: array_like
   :param overwrite_x: Specifies whether original array can be destroyed (for inplace transform)
   :type overwrite_x: bool

   :returns: **out** -- Result of the transformation along the (-4,-3) axes.
   :rtype: complex ndarray


.. function:: mfft(a, overwrite_x=False)

   Computes matrix fft on a matrix of shape (..., n,4,4).

   This is identical to np.fft2(a, axis = -3)

   :param a: Input array (must be complex).
   :type a: array_like
   :param overwrite_x: Specifies whether original array can be destroyed (for inplace transform)
   :type overwrite_x: bool

   :returns: **out** -- Result of the transformation along the (-4,-3) axes.
   :rtype: complex ndarray


.. function:: fft(a, overwrite_x=False)

   Computes  fft on a matrix of shape (..., n).

   This is identical to np.fft2(a)

   :param a: Input array (must be complex).
   :type a: array_like
   :param overwrite_x: Specifies whether original array can be destroyed (for inplace transform)
   :type overwrite_x: bool

   :returns: **out** -- Result of the transformation along the (-4,-3) axes.
   :rtype: complex ndarray


.. function:: ifft(a, overwrite_x=False)

   Computes  ifft on a matrix of shape (..., n).

   This is identical to np.ifft2(a)

   :param a: Input array (must be complex).
   :type a: array_like
   :param overwrite_x: Specifies whether original array can be destroyed (for inplace transform)
   :type overwrite_x: bool

   :returns: **out** -- Result of the transformation along the (-4,-3) axes.
   :rtype: complex ndarray


