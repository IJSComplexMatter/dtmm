:mod:`dtmm.rotation`
====================

.. py:module:: dtmm.rotation

.. autoapi-nested-parse::

   Rotation matrices



Module Contents
---------------

.. function:: rotation_vector2(angle, out=None)

   Coverts the provided angle into a rotation vector

   :param angle: Array containing the angle of rotation at different points in space
   :type angle: array
   :param out: Rotation, represented as a 2D vector at every point in space
   :type out: array


.. function:: rotation_matrix2(angle, out=None)

   Returns 2D rotation matrix.
   Numpy broadcasting rules apply.


.. function:: rotation_matrix_z(angle, out=None)

   Calculates a rotation matrix for rotations around the z axis.

   Numpy broadcasting rules apply.


.. function:: rotation_matrix_y(angle, out=None)

   Calculates a rotation matrix for rotations around the y axis.

   Numpy broadcasting rules apply.


.. function:: rotation_matrix_x(angle, out=None)

   Calculates a rotation matrix for rotations around the x axis.

   Numpy broadcasting rules apply.


.. function:: rotation_matrix(angles, out)

   rotation_matrix(angles, out)

   Calculates a general rotation matrix for rotations z-y-z psi, theta, phi.
   If out is specified.. it should be 3x3 float matrix.

   :param angles: A length 3 vector of the three angles
   :type angles: array_like

   .. rubric:: Examples

   >>> a = rotation_matrix([0.12,0.245,0.7])

   The same can be obtained by:

   >>> Ry = rotation_matrix_z(0.12)
   >>> Rt = rotation_matrix_y(0.245)
   >>> Rf = rotation_matrix_z(0.78)

   >>> b = np.dot(Rf,np.dot(Rt,Ry))
   >>> np.allclose(a,b)
   True


.. function:: rotate_diagonal_tensor(R, diagonal, output=None)

   Rotates a diagonal tensor, based on the rotation matrix provided

   >>> R = rotation_matrix((0.12,0.245,0.78))
   >>> diag = np.array([1.3,1.4,1.5], dtype = CDTYPE)
   >>> tensor = rotate_diagonal_tensor(R, diag)
   >>> matrix = tensor_to_matrix(tensor)

   The same can be obtained by:

   >>> Ry = rotation_matrix_z(0.12)
   >>> Rt = rotation_matrix_y(0.245)
   >>> Rf = rotation_matrix_z(0.78)
   >>> R = np.dot(Rf,np.dot(Rt,Ry))

   >>> diag = np.diag([1.3,1.4,1.5]) + 0j
   >>> matrix2 = np.dot(R,np.dot(diag, R.transpose()))

   >>> np.allclose(matrix2,matrix)
   True


.. function:: rotate_tensor(R, tensor, output=None)

   Rotates a tensor, based on the rotation matrix provided

   >>> R = rotation_matrix((0.12,0.245,0.78))
   >>> tensor = np.array([1.3,1.4,1.5,0.1,0.2,0.3], dtype = CDTYPE)
   >>> tensor = rotate_tensor(R, tensor)
   >>> matrix = tensor_to_matrix(tensor)


.. function:: rotate_vector(rotation_matrix, vector, out)

   Rotates vector <vector> using rotation matrix <rotation_matrix>
   rotate_vector(R, vector)

   Calculates out = R.vector of a vector

   :param rotation_matrix:
   :param vector:
   :param out:


.. function:: tensor_to_matrix(tensor, output=None)

   Converts a symmetric tensor of shape (6,) to matrix of shape (3,3).

   :param tensor: The symmetric tensor to represent as a matrix
   :type tensor: array
   :param output: The (3, 3) matrix representation of <tensor>
   :type output: array


.. function:: diagonal_tensor_to_matrix(tensor, output=None)

   Converts diagonal tensor of shape (3,) to matrix of shape (3,3).

   :param tensor: The diagonal tensor to represent as a matrix
   :type tensor: array
   :param output: The (3, 3) matrix representation of <tensor>
   :type output: array


