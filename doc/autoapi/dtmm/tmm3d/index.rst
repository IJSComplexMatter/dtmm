:mod:`dtmm.tmm3d`
=================

.. py:module:: dtmm.tmm3d

.. autoapi-nested-parse::

   4x4 and 2x2 transfer matrix method functions.



Module Contents
---------------

.. function:: layer_mat3d(k0, d, epsv, epsa, mask=None, method='4x4')

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


.. function:: fmat3d(fmat)

   Converts a sequence of 4x4 matrices to a single large matrix


.. function:: system_mat3d(fmatin, cmat, fmatout)

   Computes a system matrix from a characteristic matrix Fin-1.C.Fout


.. function:: transmit3d(fvecin, fmatin, rmat, fmatout, fvecout=None)

   Transmits field vector using 4x4 method.

   This functions takes a field vector that describes the input field and
   computes the output transmited field vector and also updates the input field
   with the reflected waves.


