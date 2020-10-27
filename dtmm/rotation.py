"""Rotation matrices"""

from __future__ import absolute_import, print_function, division

import numpy as np


#import math,cmath
from math import cos, sin
from numba import jit

import numba as nb


from dtmm.conf import NCDTYPE, NFDTYPE, NF32DTYPE, NF64DTYPE, NC64DTYPE,NC128DTYPE, \
       CDTYPE, FDTYPE, NUMBA_TARGET, NUMBA_CACHE, NUMBA_FASTMATH


# def _check_matrix(mat, shape, dtype):
#     """
#     Checks that <mat> is of the correct shape and data type.

#     Parameters
#     ----------
#     mat : array
#         The matrix to check
#     shape : tuple
#         The shape that <mat> should have.
#     dtype : dtype
#         The data type that <mat> should have.

#     Returns
#     -------

#     """
#     if not (mat.shape == shape and mat.dtype == dtype):
#         raise TypeError('Input matrix must be a numpy array of shape %s and %s dtype' % (shape, dtype))     


# def _output_matrix(mat, shape, dtype):
#     """
#     Ensures that the output matrix <mat> exists and is of the correct data type.
#     Checks that <mat> is of the correct shape.

#     Parameters
#     ----------
#     mat : array
#         The matrix to check
#     shape : tuple
#         The shape that <mat> should have.
#     dtype : dtype
#         The data type that <mat> should have.

#     Returns
#     -------

#     """
#     if mat is None:
#         mat = np.empty(shape, dtype=dtype)
#     else:
#         _check_matrix(mat, shape, dtype)

#     return mat


# def _input_matrix(mat, shape, dtype):
#     """
#     Ensures that the matrix is of the correct data type and checks that
#     it is of the right shape.

#     Parameters
#     ----------
#     mat : array
#         The matrix to check
#     shape : tuple
#         The shape that <mat> should have.
#     dtype : dtype
#         The data type that <mat> should have.

#     Returns
#     -------

#     """
#     # Ensure matrix is a numpy ndarray
#     if not isinstance(mat, np.ndarray):
#         mat = np.array(mat, dtype=dtype)

#     # Ensure that the matrix is of the correct shape and type
#     _check_matrix(mat, shape, dtype)

#     return mat


def rotation_vector2(angle, out=None):
    """
    Converts the provided angle into a rotation vector (cos, sin).

    Parameters
    ----------
    angle : array
         Array containing the angle of rotation at different points in space
    out : array
        Rotation, represented as a 2D vector at every point in space

    Returns
    -------

    """
    # calculate rotations from angle
    c, s = np.cos(angle), np.sin(angle)

    # Create <out> if not provided
    if out is None:
        out = np.empty(shape=c.shape + (2,), dtype=FDTYPE)

    # Store values
    out[..., 0] = c
    out[..., 1] = s

    return out

def rotation_matrix2(angle, out = None):
    """Returns 2D rotation matrix.
    
    Numpy broadcasting rules apply.
    """
    c,s = np.cos(angle), np.sin(angle)
    if out is None:
        out = np.empty(shape = c.shape + (2,2), dtype = FDTYPE)
    out[...,0,0] = c
    out[...,1,1] = c
    out[...,0,1] = -s
    out[...,1,0] = s    
    return out

def rotation_matrix_z(angle, out = None):
    """Calculates a rotation matrix for rotations around the z axis.
    
    Numpy broadcasting rules apply.
    """
    c,s = np.cos(angle), np.sin(angle)
    if out is None:
        out = np.zeros(shape = c.shape + (3,3), dtype = FDTYPE)
    out[...,0,0] = c
    out[...,0,1] = -s
    out[...,1,0] = s
    out[...,1,1] = c
    out[...,2,2] = 1.
    return out

def rotation_matrix_y(angle, out = None):
    """Calculates a rotation matrix for rotations around the y axis.
    
    Numpy broadcasting rules apply.
    """
    c,s = np.cos(angle), np.sin(angle)
    if out is None:
        out = np.zeros(shape = c.shape + (3,3), dtype = FDTYPE)
    out[...,0,0] = c
    out[...,0,2] = s
    out[...,1,1] = 1.
    out[...,2,0] = -s
    out[...,2,2] = c    
    return out

def rotation_matrix_x(angle, out = None):
    """Calculates a rotation matrix for rotations around the x axis.
    
    Numpy broadcasting rules apply.
    """
    c,s = np.cos(angle), np.sin(angle)
    if out is None:
        out = np.zeros(shape = c.shape + (3,3), dtype = FDTYPE)
    out[...,0,0] = 1.
    out[...,1,1] = c
    out[...,1,2] = -s
    out[...,2,1] = s
    out[...,2,2] = c    
    return out
           
@jit([(NF32DTYPE,NF32DTYPE,NF32DTYPE,NF32DTYPE[:,:]),
      (NF64DTYPE,NF64DTYPE,NF64DTYPE,NFDTYPE[:,:])],nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH) 
def _rotation_matrix(psi,theta,phi, R):
    """Fills rotation matrix values R = Rphi.Rtheta.Rpsi, where rphi and Rpsi are 
    rotations around y and Rtheta around z axis. 
    """
    cospsi = cos(psi)
    sinpsi = sin(psi)
    costheta = cos(theta)
    sintheta = sin(theta)
    cosphi = cos(phi)
    sinphi = sin(phi)

    sinphi_sinpsi = sinphi * sinpsi
    sinphi_cospsi = sinphi * cospsi    

    cosphi_sinpsi = cosphi * sinpsi
    cosphi_cospsi = cosphi * cospsi
    
    R[0,0] = costheta * cosphi_cospsi - sinphi_sinpsi
    R[0,1] = - costheta * cosphi_sinpsi - sinphi_cospsi
    R[0,2] = cosphi * sintheta
    R[1,0] = costheta * sinphi_cospsi + cosphi_sinpsi
    R[1,1] = cosphi_cospsi - costheta * sinphi_sinpsi
    R[1,2] = sintheta * sinphi
    R[2,0] = - cospsi * sintheta
    R[2,1] = sintheta*sinpsi
    R[2,2] = costheta
    
@jit([(NF32DTYPE[:,:],NF32DTYPE[:]),
      (NF64DTYPE[:,:],NFDTYPE[:])],nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)     
def _rotation_angles(R, out):
    """Computes the three Euler angles from the rotation matrix"""
    theta = np.arccos(R[2,2])
    #if sin(theta) == 0., then R[1,2] and R[0,2] are zero
    if R[1,2] == 0. and R[0,2] == 0. or theta == 0.:
        #it does not matter what psi is, so set to zero
        psi = 0.
        #np.arccos(R[1,1]) is phi -psi, but since we set psi ti zero we may set this to phi.
        phi = np.arccos(R[1,1])
    else:
        phi = np.arctan2(R[1,2],R[0,2])
        psi = np.arctan2(R[2,1],-R[2,0])
    out[0] = psi
    out[1] = theta
    out[2] = phi
    
    
@jit([(NF32DTYPE,NF32DTYPE,NF32DTYPE[:,:]),
      (NF64DTYPE,NF64DTYPE,NFDTYPE[:,:])],nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH) 
def _rotation_matrix_uniaxial(theta,phi, R):
    """Fills rotation matrix values R = Rphi.Rtheta, where phi is
    rotations around y and Rtheta around z axis. 
    """
    costheta = cos(theta)
    sintheta = sin(theta)
    cosphi = cos(phi)
    sinphi = sin(phi)
 
    R[0,0] = costheta * cosphi
    R[0,1] =  - sinphi 
    R[0,2] = cosphi * sintheta
    R[1,0] = costheta * sinphi  
    R[1,1] = cosphi
    R[1,2] = sintheta * sinphi
    R[2,0] = -sintheta
    R[2,1] = 0.
    R[2,2] = costheta
    

@nb.guvectorize([(NF32DTYPE[:],NF32DTYPE[:,:]),
                 (NF64DTYPE[:],NFDTYPE[:,:])], "(n)->(n,n)", 
                 target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def rotation_matrix(angles, out):
    """rotation_matrix(angles, out)
    
    Calculates a general rotation matrix for rotations z-y-z psi, theta, phi.
    If out is specified.. it should be 3x3 float matrix.
    
    Parameters
    ----------
    angles : array_like
        A length 3 vector of the three angles
        
    Examples
    --------
    
    >>> a = rotation_matrix([0.12,0.245,0.7])
    
    The same can be obtained by:
        
    >>> Ry = rotation_matrix_z(0.12)
    >>> Rt = rotation_matrix_y(0.245)
    >>> Rf = rotation_matrix_z(0.78)
      
    >>> b = np.dot(Rf,np.dot(Rt,Ry))
    >>> np.allclose(a,b)
    True
    """   
    if len(angles) != 3:
        raise ValueError("Invalid input data shape")
    _rotation_matrix(angles[0],angles[1],angles[2], out)


@nb.guvectorize([(NF32DTYPE[:,:],NF32DTYPE[:]),
                 (NF64DTYPE[:,:],NFDTYPE[:])], "(n,n)->(n)", 
                 target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def rotation_angles(matrix, out):
    if len(matrix) != 3:
        raise ValueError("Invalid input data shape")
    _rotation_angles(matrix, out)

def rotate_diagonal_tensor(R,diagonal,out = None):
    """Rotates a diagonal tensor, based on the rotation matrix provided
    
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
    """

    tensor = np.empty(shape = diagonal.shape[:-1] + (6,), dtype = diagonal.dtype)
    tensor[...,0:3] = diagonal
    tensor[...,3:] = 0
    return rotate_tensor(R,tensor,out)


        
@jit([NCDTYPE[:](NFDTYPE[:,:],NCDTYPE[:],NCDTYPE[:])],nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _rotate_diagonal_tensor(R,diagonal,out):
    """Calculates out = R.diagonal.RT of a diagonal tensor"""
    for i in range(3):
        out[i] = diagonal[0]*R[i,0]*R[i,0] + diagonal[1]*R[i,1]*R[i,1] + diagonal[2]*R[i,2]*R[i,2]
    out[3] = diagonal[0]*R[0,0]*R[1,0] + diagonal[1]*R[0,1]*R[1,1] + diagonal[2]*R[0,2]*R[1,2]
    out[4] = diagonal[0]*R[0,0]*R[2,0] + diagonal[1]*R[0,1]*R[2,1] + diagonal[2]*R[0,2]*R[2,2]          
    out[5] = diagonal[0]*R[1,0]*R[2,0] + diagonal[1]*R[1,1]*R[2,1] + diagonal[2]*R[1,2]*R[2,2]
    return out


@nb.guvectorize([(NFDTYPE[:,:],NCDTYPE[:],NCDTYPE[:])], "(m,m),(n)->(n)", 
                 target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def rotate_tensor(R,tensor,out):
    """Calculates out = R.tensor.RT of a tensor"
    
    >>> R = rotation_matrix((0.12,0.245,0.78))
    >>> tensor = np.array([1.3,1.4,1.5,0.1,0.2,0.3], dtype = CDTYPE)
    >>> tensor = rotate_tensor(R, tensor)
    >>> matrix = tensor_to_matrix(tensor)
    """
    r11 = R[0,0]
    r12 = R[0,1]
    r13 = R[0,2]
    r21 = R[1,0]
    r22 = R[1,1]
    r23 = R[1,2]
    r31 = R[2,0]
    r32 = R[2,1]
    r33 = R[2,2]
    t1 = tensor[0]
    t2 = tensor[1]
    t3 = tensor[2]
    t4 = tensor[3]
    t5 = tensor[4]
    t6 = tensor[5]    
    
    a1 = (r11 * t1 + r12 * t4 + r13 * t5)
    a2 = (r13 * t3 + r11 * t5 + r12 * t6)
    a3 = (r12 * t2 + r11 * t4 + r13 * t6)
    
    out[0] = r11 * a1 + r13 * a2 + r12 * a3 
    out[3] = r21 * a1 + r23 * a2 + r22 * a3  
    out[4] = r31 * a1 + r33 * a2 + r32 * a3 
    
    a1 = r21 * t1 + r22 * t4 + r23 * t5 
    a2 = r23 * t3 + r21 * t5 + r22 * t6  
    a3 = r22 * t2 + r21 * t4 + r23 * t6
    
    out[1] =  r21 * a1 + r23 * a2 + r22 * a3
    out[5] =  r31 * a1 + r33 * a2 + r32 * a3
    
    out[2] = r31 * (r31 * t1 + r32 * t4 + r33 * t5) + r33 * (r33 * t3 + r31 * t5 + r32 * t6) + r32 * (r32 * t2 + r31 * t4 + r33 * t6)
    

@jit([(NF32DTYPE[:, :], NF32DTYPE[:], NF32DTYPE[:]),
      (NF64DTYPE[:, :], NF64DTYPE[:], NFDTYPE[:])], nopython=True, cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)
def _rotate_vector(rotation_matrix, vector, out):
    """
    Rotates vector <vector> using rotation matrix <rotation_matrix>

    Parameters
    ----------
    rotation_matrix : array
        3x3 rotation matrix
    vector : array
        Input 3-vector to rotate
    out : array
        Output rotated 3-vector

    Returns
    -------

    """
    # Rotate x values
    out0 = vector[0] * rotation_matrix[0, 0] + vector[1] * rotation_matrix[0, 1] + vector[2] * rotation_matrix[0, 2]
    # Rotate y values
    out1 = vector[0] * rotation_matrix[1, 0] + vector[1] * rotation_matrix[1, 1] + vector[2] * rotation_matrix[1, 2]
    # Rotate z values
    out2 = vector[0] * rotation_matrix[2, 0] + vector[1] * rotation_matrix[2, 1] + vector[2] * rotation_matrix[2, 2]

    # Set outputs, return by reference
    out[0] = out0
    out[1] = out1
    out[2] = out2


@nb.guvectorize([(NF32DTYPE[:, :], NF32DTYPE[:], NF32DTYPE[:]),
                 (NF64DTYPE[:, :], NF64DTYPE[:], NFDTYPE[:]), ], "(n,n),(n)->(n)", cache=NUMBA_CACHE,
                fastmath=NUMBA_FASTMATH, target=NUMBA_TARGET)
def rotate_vector(rotation_matrix, vector, out):
    """
    Rotates vector <vector> using rotation matrix <rotation_matrix>
    rotate_vector(R, vector)

    Calculates out = R.vector of a vector.

    Parameters
    ----------
    rotation_matrix : array
        3x3 rotation matrix
    vector : array
        Input 3-vector to rotate
    out : array, optional
        Output rotated 3-vector

    Returns
    -------
    vector : ndarray
        Rotated vector.
    """
    # Ensure size is correct size
    if len(vector) != 3:
        raise ValueError("Invalid input data shape")

    # Perform rotation
    _rotate_vector(rotation_matrix, vector, out)
    

#dotrv = rotate_vector


# def tensor_to_matrix(tensor, output=None):
#     """
#     Converts a symmetric tensor of shape (6,) to matrix of shape (3,3).

#     Parameters
#     ----------
#     tensor : array
#         The symmetric tensor to represent as a matrix
#     output : array
#         The (3, 3) matrix representation of <tensor>

#     Returns
#     -------

#     """
#     # Check that output matrix exists and is good
#     output = _output_matrix(output, (3, 3), CDTYPE)
#     # Check the input matrix is good
#     tensor = _input_matrix(tensor, (6,), CDTYPE)
#     # Convert tensor to matrix
#     _tensor_to_matrix(tensor, output)

#     return output


# def diagonal_tensor_to_matrix(tensor, output=None):
#     """
#     Converts diagonal tensor of shape (3,) to matrix of shape (3,3).

#     Parameters
#     ----------
#     tensor : array
#         The diagonal tensor to represent as a matrix
#     output : array
#         The (3, 3) matrix representation of <tensor>

#     Returns
#     -------

#     """
#     # Check that output matrix exists and is good
#     output = _output_matrix(output, (3, 3), CDTYPE)
#     # Check the input matrix is good
#     tensor = _input_matrix(tensor, (3,), CDTYPE)
#     # Convert tensor to matrix
#     _diagonal_tensor_to_matrix(tensor, output)

#     return output

# _TENS_DECL = [NF32DTYPE[:,:](NF32DTYPE[:],NF32DTYPE[:,:]), 
#               NFDTYPE[:,:](NF64DTYPE[:],NFDTYPE[:,:]), 
#               NC64DTYPE[:,:](NC64DTYPE[:],NC64DTYPE[:,:]), 
#               NCDTYPE[:,:](NC128DTYPE[:],NCDTYPE[:,:]), 
#              ]


# #@jit([NCDTYPE[:,:](NCDTYPE[:],NCDTYPE[:,:]), NFDTYPE[:,:](NFDTYPE[:],NFDTYPE[:,:])],nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
# @jit(_TENS_DECL,nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
# def _tensor_to_matrix(tensor, matrix):
#     matrix[0,0] = tensor[0]
#     matrix[1,1] = tensor[1]
#     matrix[2,2] = tensor[2]
#     matrix[0,1] = tensor[3]
#     matrix[1,0] = tensor[3]
#     matrix[0,2] = tensor[4]
#     matrix[2,0] = tensor[4]
#     matrix[1,2] = tensor[5]
#     matrix[2,1] = tensor[5]
#     return matrix

# #@jit([NCDTYPE[:,:](NCDTYPE[:],NCDTYPE[:,:]),NFDTYPE[:,:](NFDTYPE[:],NFDTYPE[:,:])],nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
# @jit(_TENS_DECL,nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
# def _diagonal_tensor_to_matrix(tensor, matrix):
#     matrix[0,0] = tensor[0]
#     matrix[1,1] = tensor[1]
#     matrix[2,2] = tensor[2]
#     matrix[0,1] = 0.
#     matrix[1,0] = 0.
#     matrix[0,2] = 0.
#     matrix[2,0] = 0.
#     matrix[1,2] = 0.
#     matrix[2,1] = 0.
#     return matrix

@jit([NFDTYPE[:,:](NFDTYPE,NFDTYPE[:],NFDTYPE[:,:])],nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH) 
def _calc_rotations_uniaxial(phi0,element,R):
    theta = element[1]
    phi = element[2] -phi0
    _rotation_matrix_uniaxial(theta,phi, R)
    return R    

#@jit([NFDTYPE[:,:](NFDTYPE,NFDTYPE[:,:])],nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH) 
#def _calc_rotations_isotropic(phi0,R):
#    theta = np.pi/2
#    #theta = 0.
#    phi = -phi0
#    _rotation_matrix_uniaxial(theta,phi, R)
#    return R  

@jit([NFDTYPE[:,:](NFDTYPE,NFDTYPE[:],NFDTYPE[:,:])],nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH) 
def _calc_rotations(phi0,element,R):
    psi = element[0]
    theta = element[1]
    phi = element[2] -phi0
    _rotation_matrix(psi,theta,phi, R)
    return R   

if __name__ == "__main__":
    import doctest
    doctest.testmod()