"""
Some helper function for jones calculus.
"""

from dtmm.conf import CDTYPE,FDTYPE,NFDTYPE, NCDTYPE, NUMBA_TARGET, NUMBA_CACHE, NUMBA_FASTMATH
import numpy as np
from dtmm.rotation import rotation_matrix2
from dtmm.linalg import dotmv

def jonesvec(pol, phi = 0.):
    """Returns a normalized jones vector from an input length 2 vector., Additionaly,
    you can use this function to view the jones vactor in rotated coordinate frame, 
    defined with a rotation angle phi. 
    
    Numpy broadcasting rules apply.
    
    Example
    -------
    
    >>> jonesvec((1,1j)) #left-handed circuar polarization
    array([0.70710678+0.j        , 0.        +0.70710678j])
    
    """
    pol = np.asarray(pol, CDTYPE)
    assert pol.shape[-1] == 2
    norm = (pol[...,0] * pol[...,0].conj() + pol[...,1] * pol[...,1].conj())**0.5
    pol = pol/norm[...,np.newaxis]
    pol = np.asarray(pol, CDTYPE)
    r = rotation_matrix2(-phi)
    return dotmv(r, pol)


def polarizer(jones, out = None):
    """Returns jones polarizer matrix from a jones vector describing the output
    polarizartion state.
    
    Numpy broadcasting rules apply
    
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
    """Return jones matrix for a polarizer. Angle is the polarizer angle.
    
    Broadcasting rules apply.
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
    """Returns circular polarizee matrix. Hand is an integer  +1 (left-hand)
    or -1 (right-hand).
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

#def as4x4(jonesmat, out = None):
#    """Converts jones 2x2 matrix to field 4x4 matrix"""
#    if out is None:
#        shape = jonesmat.shape[:-2] + (4,4)
#        out = np.zeros(shape, dtype = jonesmat.dtype)
#    else:
#        out[...] = 0.
#    out[...,0,0] = jonesmat[...,0,0]
#    out[...,0,2] = jonesmat[...,0,1]
#    out[...,1,1] = jonesmat[...,0,0]
#    out[...,1,3] = -jonesmat[...,0,1]
#    out[...,2,0] = jonesmat[...,1,0]
#    out[...,2,2] = jonesmat[...,1,1]
#    out[...,3,1] = -jonesmat[...,1,0]
#    out[...,3,3] = jonesmat[...,1,1]  
#    return out

def as4x4(jonesmat,  out = None):
    """Converts jones 2x2 matrix to eigenfield 4x4 matrix."""
    
    if out is None:
        shape = jonesmat.shape[:-2] + (4,4)
        out = np.zeros(shape, dtype = jonesmat.dtype)
    else:
        out[...] = 0.
    out[...,0::2,0::2] = jonesmat
    out[...,1::2,1::2] = jonesmat

    return out


#@nb.guvectorize([(NCDTYPE[:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],"(m,m),(n,k,l)->(n,k,l)", 
#                 target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
#def apply_jones_matrix(pmat, field, out):
#    """Multiplies 2x2 jones polarizer matrix with 4 x n x m field array"""
#    for i in range(field.shape[1]):
#        for j in range(field.shape[2]):
#            Ex = field[0,i,j] * pmat[0,0] + field[2,i,j] * pmat[0,1]
#            Hy = field[1,i,j] * pmat[0,0] - field[3,i,j] * pmat[0,1]
#            Ey = field[0,i,j] * pmat[1,0] + field[2,i,j] * pmat[1,1]
#            Hx = -field[1,i,j] * pmat[1,0] + field[3,i,j] * pmat[1,1]
#            out[0,i,j] = Ex
#            out[1,i,j] = Hy
#            out[2,i,j] = Ey
#            out[3,i,j] = Hx
#            
#            
#@nb.guvectorize([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],"(k,l,m,m),(n,k,l)->(n,k,l)", 
#                 target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
#def apply_jones_matrix2(pmat, field, out):
#    """Multiplies 2x2 jones polarizer matrix with 4 x n x m field array"""
#    for i in range(field.shape[1]):
#        for j in range(field.shape[2]):
#            Ex = field[0,i,j] * pmat[i,j,0,0] + field[2,i,j] * pmat[i,j,0,1]
#            Hy = field[1,i,j] * pmat[i,j,0,0] - field[3,i,j] * pmat[i,j,0,1]
#            Ey = field[0,i,j] * pmat[i,j,1,0] + field[2,i,j] * pmat[i,j,1,1]
#            Hx = -field[1,i,j] * pmat[i,j,1,0] + field[3,i,j] * pmat[i,j,1,1]
#            out[0,i,j] = Ex
#            out[1,i,j] = Hy
#            out[2,i,j] = Ey
#            out[3,i,j] = Hx            
#            
__all__ = ["polarizer","circular_polarizer","linear_polarizer", "jonesvec", "as4x4"]
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()