:mod:`dtmm.tmm2d`
=================

.. py:module:: dtmm.tmm2d

.. autoapi-nested-parse::

   4x4 and 2x2 transfer matrix method functions.



Module Contents
---------------

.. function:: layer_mat2d(k0, d, epsv, epsa, betay=0.0, method='4x4', mask=None)

   Computes characteristic matrix of a single layer M=F.P.Fi,

   Numpy broadcasting rules apply

   :param k0: A scalar or a vector of wavenumbers
   :type k0: float or sequence of floats
   :param d: Layer thickness
   :type d: array_like
   :param epsv: Epsilon eigenvalues.
   :type epsv: array_like
   :param epsa: Optical axes orientation angles (psi, theta, phi).
   :type epsa: array_like
   :param method: Either a 4x4 or 4x4_1
   :type method: str, optional

   :returns: **cmat** -- Characteristic matrix of the layer.
   :rtype: ndarray


.. function:: system_mat2d(fmatin, cmat, fmatout)

   Computes a system matrix from a characteristic matrix Fin-1.C.Fout


.. function:: transmit2d(fvecin, fmatin, rmat, fmatout, fvecout=None)

   Transmits field vector using 4x4 method.

   This functions takes a field vector that describes the input field and
   computes the output transmited field vector and also updates the input field
   with the reflected waves.


