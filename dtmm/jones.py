"""
Jones calculus functions. Ju can use this module for jones calculus

2x2 matrices
------------

* :func:`.jonesvec` : jones vector
* :func:`.mirror` : mirror matrix
* :func:`.polarizer` : polarizer matrix
* :func:`.retarder` : retarder matrix
* :func:`.quarter_waveplate` : quarter-wave plate matrix
* :func:`.half_waveplate` : half-wave plate  matrix
* :func:`.circular_polarizer` : circular polarizer matrix
* :func:`.linear_polarizer` : linear polarizer matrix

Conversion functions
--------------------

* :func:`.as4x4` : convert to 4x4 matrix for (...,4) field vector.
* :func:`.rotated_matrix` : to view the matrix in rotated frame.

Examples
--------

>>> j_in = jonesvec((1,1)) #input light polarized at 45
>>> r = retarder(0.2,0.4)
>>> p = linear_polarizer((1,0)) # x polarization
>>> c = circular_polarizer(+1) # left handed
>>> m = multi_dot([r,p,c])
>>> j_out = dotmv(m,j_in) #output light E field
>>> i = jones_intensity(j_out) #intensity
"""

from dtmm.conf import CDTYPE,FDTYPE
import numpy as np
from dtmm.rotation import rotation_matrix2
from dtmm.linalg import dotmv, dotmm, multi_dot

def jones_intensity(jones):
    """Computes intensity of the jones vector
    
    Parameters
    ----------
    jones : (...,2) array
        Jones vector.
    
    Returns
    -------
    intensity : (...) float array
        Computed intensity.
    """
    return (jones * np.conj(jones)).sum(-1)

def jonesvec(pol, phi = 0., out = None):
    """Returns a normalized jones vector from an input length 2 vector., Additionaly,
    you can use this function to view the jones vector in rotated coordinate frame, 
    defined with a rotation angle phi. 
    
    Numpy broadcasting rules apply.
    
    Parameters
    ----------
    pol : (...,2) array
        Input jones vector. Does not need to be normalized. 
    phi : float or (...,1) array
        If jones vector must be view in the rotated frame, this parameter
        defines the rotation angle.
    out : ndarray
        Output array in which to put the result; if provided, it
        must have a shape that the inputs broadcast to.
        
    Returns
    -------
    vec : ndarray
        Normalized jones vector
    
    Example
    -------
    
    >>> jonesvec((1,1j)) #left-handed circuar polarization
    array([0.70710678+0.j        , 0.        +0.70710678j])
    
    In rotated frame
    
    >>> j1,j2 = jonesvec((1,1j), (np.pi/2,np.pi/4)) 
    >>> np.allclose(j1, (0.70710678j,-0.70710678))
    True
    >>> np.allclose(j2, (0.5+0.5j,-0.5+0.5j))
    True
    
    """
    pol = np.asarray(pol, CDTYPE)
    phi = np.asarray(phi)
    if pol.shape[-1] != 2:
        raise ValueError("Invalid input shape")
    norm = (pol[...,0] * pol[...,0].conj() + pol[...,1] * pol[...,1].conj())**0.5
    pol = pol/norm[...,np.newaxis]
    pol = np.asarray(pol, CDTYPE)
    r = rotation_matrix2(-phi)
    return dotmv(r, pol, out)

def mirror(out = None):
    """Returns jones mirror matrix.
    
    Parameters
    ----------
    out : ndarray, optional
        Output array in which to put the result; if provided, it
        must have a shape that the inputs broadcast to.
        
    Returns
    -------
    mat : ndarray
        Output jones matrix.      
    """
    m = np.array(((1,0),(0,-1)), dtype = CDTYPE)
    if out is not None:
        out[...,:,:] = m
        m = out
    return m

def retarder(phase, phi = 0., out = None):
    """Returns jones matrix of general phase retarder
    
    Numpy broadcasting rules apply.
    
    Parameters
    ----------
    phase : float or array
        Phase difference between the fast and slow axis of the retarder.
    phi : float or array 
        Fast axis orientation
    out : ndarray, optional
        Output array in which to put the result; if provided, it
        must have a shape that the inputs broadcast to.
        
    Returns
    -------
    mat : ndarray
        Output jones matrix.
    """
    phi = np.asarray(phi)
    phase = np.asarray(phase)
    
    c = np.cos(phi*2)
    s = np.sin(phi*2)
    
    cp = np.cos(phase/2.)
    sp = np.sin(phase/2.)
    
    m00 = (cp + 1j*sp*c) 
    m11 = (cp - 1j*sp*c) 
    
    m01 = 1j*sp * s 
    
    shape = m00.shape + (2,2)
    if out is None:
        out = np.empty(shape = shape, dtype = CDTYPE)
    else:
        assert out.shape == shape 
    
    out[...,0,0] = m00
    out[...,0,1] = m01
    out[...,1,0] = m01
    out[...,1,1] = m11
    return out     

def quarter_waveplate(phi = 0., out = None):
    """Returns jones quarter-wave plate matrix.
    
    Numpy broadcasting rules apply.
    
    Parameters
    ----------
    phi : float or array 
        Fast axis orientation
    out : ndarray, optional
        Output array in which to put the result; if provided, it
        must have a shape that the inputs broadcast to.
        
    Returns
    -------
    mat : ndarray
        Output jones matrix.        
    """
    return retarder(np.pi/2, phi,  out)  

def half_waveplate(phi = 0., out = None):
    """Returns jones half-wave plate matrix.
    
    Numpy broadcasting rules apply.
    
    Parameters
    ----------
    phi : float or array 
        Fast axis orientation
    out : ndarray, optional
        Output array in which to put the result; if provided, it
        must have a shape that the inputs broadcast to.
        
    Returns
    -------
    mat : ndarray
        Output jones matrix.    
    """
    return retarder(np.pi, phi, out)         

def polarizer(jones, out = None):
    """Returns jones polarizer matrix.
    
    Numpy broadcasting rules apply.
    
    Parameters
    ----------
    jones : (...,2) array
        Input normalized jones vector. Use :func:`.jonesvec` to generate jones vector
    out : ndarray, optional
        Output array in which to put the result; if provided, it
        must have a shape that the inputs broadcast to.
        
    Returns
    -------
    mat : ndarray
        Output jones matrix.
    
    Examples
    --------
    >>> pmat = polarizer(jonesvec((1,1))) #45 degree linear polarizer
    >>> np.allclose(pmat, linear_polarizer(np.pi/4)+0j )
    True
    >>> pmat = polarizer(jonesvec((1,-1))) #-45 degree linear polarizer
    >>> np.allclose(pmat, linear_polarizer(-np.pi/4)+0j )
    True
    >>> pmat = polarizer(jonesvec((1,1j))) #left handed circular polarizer
    >>> np.allclose(pmat, circular_polarizer(1) )
    True
    """
    jones = np.asarray(jones)
    assert jones.shape[-1] == 2
    shape = jones.shape + (2,)
    if out is None:
        out = np.empty(shape = shape, dtype = CDTYPE)
    else:
        assert out.shape == shape 
    c,s = jones[...,0], jones[...,1]
    
    out[...,0,0] = c*c.conj()
    out[...,0,1] = c*s.conj()
    out[...,1,0] = s*c.conj()
    out[...,1,1] = s*s.conj()
    return out    
    
def linear_polarizer(angle, out = None):
    """Return jones matrix for a polarizer.
    
    Numpy broadcasting rules apply.
    
    Parameters
    ----------
    angle : float or array
        Orientation of the polarizer.
    out : ndarray, optional
        Output array in which to put the result; if provided, it
        must have a shape that the inputs broadcast to.
        
    Returns
    -------
    mat : ndarray
        Output jones matrix.
    """
    angle = np.asarray(angle)
    shape = angle.shape + (2,2)
    if out is None:
        out = np.empty(shape = shape, dtype = FDTYPE)
    else:
        assert out.shape == shape 
    c = np.cos(angle)
    s = np.sin(angle)
    cs = c*s
    out[...,0,0] = c*c
    out[...,1,0] = cs
    out[...,0,1] = cs
    out[...,1,1] = s*s
    return out

polarizer_matrix = linear_polarizer

def circular_polarizer(hand, out = None):
    """Returns circular polarizer matrix.
    
    Numpy broadcasting rules apply.
    
    Parameters
    ----------
    hand : int or (...,1) array
        Handedness +1 (left-hand) or -1 (right-hand).
    out : ndarray, optional
        Output array in which to put the result; if provided, it
        must have a shape that the inputs broadcast to.
        
    Returns
    -------
    mat : ndarray
        Output jones matrix.
    """
    hand = np.asarray(hand)*0.5
    shape = hand.shape + (2,2)
    if out is None:
        out = np.empty(shape = shape, dtype = CDTYPE)
    else:
        assert out.shape == shape 
    out[...,0,0] = 0.5
    out[...,0,1] = -1j*hand
    out[...,1,0] = 1j*hand
    out[...,1,1] = 0.5   
    return out

def rotated_matrix(jmat, phi):
    """jones matrix viewed in the rotated frame, where phi is the rotation angle
    
    Performs (R.T).jmat.R where R is the rotation matrix.
    
    Numpy broadcasting rules apply.
    
    Parameters
    ----------
    jmat : (...,2,2) array
        Jones matrix
    phi : float
        Rotation angle.
        
    Returns
    -------
    out : (...,2,2) array
        Rotated jones matrix.
    
    """
    r = np.asarray(rotation_matrix2(phi),CDTYPE)
    jmat = np.asarray(jmat)
    rT = np.swapaxes(r,-1,-2)
    return multi_dot([rT,jmat,r])

def as4x4(jmat,  out = None):
    """Converts jones 2x2 matrix to eigenfield 4x4 matrix.
    
    Parameters
    ----------
    jmat : (...,2,2) array
        Jones matrix
    out : ndarray, optional
        Output array in which to put the result; if provided, it
        must have a shape that the inputs broadcast to.
        
    Returns
    -------
    mat : (...,4,4) ndarray
        Output jones matrix.
    """
    
    if out is None:
        shape = jmat.shape[:-2] + (4,4)
        out = np.zeros(shape, dtype = jmat.dtype)
    else:
        out[...] = 0.
    out[...,0::2,0::2] = jmat
    
    jmatT = np.swapaxes(jmat, -2,-1)
    
    #for back propagating waves, jones matrix is conjugate
    out[...,1::2,1::2] = np.conj(jmat)

    return out

if __name__ == "__main__":
    import doctest
    doctest.testmod()