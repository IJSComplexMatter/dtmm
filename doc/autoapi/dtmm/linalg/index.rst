:mod:`dtmm.linalg`
==================

.. py:module:: dtmm.linalg

.. autoapi-nested-parse::

   Numba optimized linear algebra functions for 4x4 matrices and 2x2 matrices.



Module Contents
---------------

.. function:: eig(matrix, overwrite_x=False)

   Computes eigenvalues and eigenvectors of 3x3 matrix using numpy.linalg.eig.
   Eigenvalues are sorted so that eig[2] is the most distinct (extraordinary).

   :param matrix: A 3x3 matrix.
   :type matrix: (...,3,3) array
   :param overwrite_x: Ifset, the function will write eigenvectors as rows in the input array.
   :type overwrite_x: bool, optional

   :returns: * **w** (*(..., 3) array*) -- The eigenvalues, each repeated according to its multiplicity.
               The eigenvalues are ordered so that the third eigenvalue is most
               distinct and first two are least distinct
             * **v** (*(..., 3, 3) array*) -- The normalized (unit "length") eigenvectors, such that the
               column ``v[:,i]`` is the eigenvector corresponding to the
               eigenvalue ``w[i]``.


.. function:: tensor_eig(tensor, overwrite_x=False)

   Computes eigenvalues and eigenvectors of a tensor.

   Eigenvalues are sorted so that eig[2] is the most distinct (extraordinary).

   If tensor is provided as a length 6 matrix, the elements are
   a[0,0], a[1,1], a[2,2], a[0,1], a[0,2], a[1,2]. If provided as a (3x3)
   matrix a, the rest of the elements are silently ignored.

   :param tensor: A length 6 array or 3x3 matrix
   :type tensor: (...,6) or (...,3,3) array
   :param overwrite_x: If tensor is (...,3,3) array, the function will write eigenvectors
                       as rows in the input array.
   :type overwrite_x: bool, optional

   :returns: * **w** (*(..., 3) array*) -- The eigenvalues, each repeated according to its multiplicity.
               The eigenvalues are ordered so that the third eigenvalue is most
               distinct and first two are least distinct
             * **v** (*(..., 3, 3) array*) -- The normalized (unit "length") eigenvectors, such that the
               column ``v[:,i]`` is the eigenvector corresponding to the
               eigenvalue ``w[i]``.


.. function:: inv(mat, out)

   inv(mat), gufunc

   Calculates inverse of a 4x4 complex matrix or 2x2 complex matrix

   :param mat: Input array
   :type mat: ndarray

   .. rubric:: Examples

   >>> a = np.random.randn(4,4) + 0j
   >>> ai = inv4x4(a)

   >>> from numpy.linalg import inv
   >>> ai2 = inv(a)

   >>> np.allclose(ai2,ai)
   True


.. function:: dotmf(a, b, out=None)

   dotmf(a, b)

   Computes a dot product of an array of 4x4 (or 2x2) matrix with
   a field array or an E-array (in case of 2x2 matrices).


.. function:: dotmm(a, b, out)

   dotmm(a, b)

   Computes an efficient dot product of a 4x4,  2x2
   or a less efficient general matrix multiplication.


.. function:: dotmd(a, d, out)

   dotmd(a, d)

   Computes a dot product of a 4x4 (or 2x2) matrix with a diagonal
   matrix represented by a vector of shape 4 (or 2).


.. function:: dotmv(a, b, out)

   dotmv(a, b)

   Computes a dot product of a 4x4 or 2x2 matrix with a vector.


.. function:: dotmdm(a, d, b, out)

   dotmdm(a, d, b)

   Computes a dot product of a 4x4 (or 2x2) matrix with a diagonal matrix (4- or 2-vector)
   and another 4x4 (or 2x2) matrix.


.. function:: multi_dot(arrays, axis=0, reverse=False)

   Computes dot product of multiple 2x2 or 4x4 matrices. If reverse is
   specified, it is performed in reversed order. Axis defines the axis over
   which matrices are multiplied.


