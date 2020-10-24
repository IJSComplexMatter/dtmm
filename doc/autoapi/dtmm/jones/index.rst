:mod:`dtmm.jones`
=================

.. py:module:: dtmm.jones

.. autoapi-nested-parse::

   Jones calculus helper functions.



Module Contents
---------------

.. function:: jonesvec(pol, phi=0.0, out=None)

   Returns a normalized jones vector from an input length 2 vector., Additionaly,
   you can use this function to view the jones vector in rotated coordinate frame,
   defined with a rotation angle phi.

   Numpy broadcasting rules apply.

   :param pol: Input jones vector. Does not need to be normalized.
   :type pol: (...,2) array
   :param phi: If jones vector must be view in the rotated frame, this parameter
               defines the rotation angle.
   :type phi: float or (...,1) array
   :param out: Output array in which to put the result; if provided, it
               must have a shape that the inputs broadcast to.
   :type out: ndarray

   :returns: **vec** -- Normalized jones vector
   :rtype: ndarray

   .. rubric:: Example

   >>> jonesvec((1,1j)) #left-handed circuar polarization
   array([0.70710678+0.j        , 0.        +0.70710678j])

   In rotated frame

   >>> j1,j2 = jonesvec((1,1j), (np.pi/2,np.pi/4))
   >>> np.allclose(j1, (0.70710678j,-0.70710678))
   True
   >>> np.allclose(j2, (0.5+0.5j,-0.5+0.5j))
   True


.. function:: polarizer(jones, out=None)

   Returns jones polarizer matrix.

   Numpy broadcasting rules apply.

   :param jones: Input normalized jones vector. Use :func:`.jonesvec` to generate jones vector
   :type jones: (...,2) array
   :param out: Output array in which to put the result; if provided, it
               must have a shape that the inputs broadcast to.
   :type out: ndarray, optional

   :returns: **mat** -- Output jones matrix.
   :rtype: ndarray

   .. rubric:: Examples

   >>> pmat = polarizer(jonesvec((1,1))) #45 degree linear polarizer
   >>> np.allclose(pmat, linear_polarizer(np.pi/4)+0j )
   True
   >>> pmat = polarizer(jonesvec((1,-1))) #-45 degree linear polarizer
   >>> np.allclose(pmat, linear_polarizer(-np.pi/4)+0j )
   True
   >>> pmat = polarizer(jonesvec((1,1j))) #left handed circular polarizer
   >>> np.allclose(pmat, circular_polarizer(1) )
   True


.. function:: linear_polarizer(angle, out=None)

   Return jones matrix for a polarizer.

   Numpy broadcasting rules apply.

   :param angle: Orientation of the polarizer.
   :type angle: float or array
   :param out: Output array in which to put the result; if provided, it
               must have a shape that the inputs broadcast to.
   :type out: ndarray, optional

   :returns: **mat** -- Output jones matrix.
   :rtype: ndarray


.. function:: circular_polarizer(hand, out=None)

   Returns circular polarizer matrix.

   Numpy broadcasting rules apply.

   :param hand: Handedness +1 (left-hand) or -1 (right-hand).
   :type hand: int or (...,1) array
   :param out: Output array in which to put the result; if provided, it
               must have a shape that the inputs broadcast to.
   :type out: ndarray, optional

   :returns: **mat** -- Output jones matrix.
   :rtype: ndarray


.. function:: as4x4(jonesmat, out=None)

   Converts jones 2x2 matrix to eigenfield 4x4 matrix.

   :param jonesmat: Jones matrix
   :type jonesmat: (...,2,2) array
   :param out: Output array in which to put the result; if provided, it
               must have a shape that the inputs broadcast to.
   :type out: ndarray, optional

   :returns: **mat** -- Output jones matrix.
   :rtype: (...,4,4) ndarray


