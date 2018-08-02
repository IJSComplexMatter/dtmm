"""
Some helper function for jones calculus.
"""

from dtmm.conf import CDTYPE,FDTYPE,NFDTYPE, NCDTYPE, NUMBA_TARGET, NUMBA_CACHE, NUMBA_FASTMATH
import numba as nb
import numpy as np

def jonesvec(pol):
    """Returns normalized jones vector from an input length 2 vector. 
    Numpy broadcasting rules apply.
    
    Example
    -------
    
    >>> jonesvec((1,1j)) #left-handed circuar polarization
    array([0.70710678+0.j        , 0.        +0.70710678j])
    
    """
    pol = np.asarray(pol)
    assert pol.shape[-1] == 2
    norm = (pol[...,0] * pol[...,0].conj() + pol[...,1] * pol[...,1].conj())**0.5
    return pol/norm[...,np.newaxis]

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
    """Returns circular polarizee matrix. Hand is an integer  +1 (left-) or -1 
    (right-hand).
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

@nb.guvectorize([(NCDTYPE[:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],"(m,m),(n,k,l)->(n,k,l)", 
                 target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def apply_jones_matrix(pmat, field, out):
    """Multiplies 2x2 jones polarizer matrix with 4 x n x m field array"""
    for i in range(field.shape[1]):
        for j in range(field.shape[2]):
            Ex = field[0,i,j] * pmat[0,0] + field[2,i,j] * pmat[0,1]
            Hy = field[1,i,j] * pmat[0,0] - field[3,i,j] * pmat[0,1]
            Ey = field[0,i,j] * pmat[1,0] + field[2,i,j] * pmat[1,1]
            Hx = -field[1,i,j] * pmat[1,0] + field[3,i,j] * pmat[1,1]
            out[0,i,j] = Ex
            out[1,i,j] = Hy
            out[2,i,j] = Ey
            out[3,i,j] = Hx
            
__all__ = ["polarizer","circular_polarizer","linear_polarizer", "apply_jones_matrix", "jonesvec", ]
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()