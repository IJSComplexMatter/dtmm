"""
Field vector and field data polarization and phase retardation matrix functions and handling.

Polarization and retardation handling 
+++++++++++++++++++++++++++++++++++++

* :func:`.polarizer4x4` creates a polarizer matrix.
* :func:`.jonesmat4x4` creates a general jones 4x4 matrix from 2x2 jones patrix. 
* :func:`.mode_jonesmat4x4` creates a general jones 4x4 matrix for field data in fft space.  
* :func:`.ray_jonesmat4x4` creates a general jones 4x4 matrix for field data in real space.  

Convenience functions
+++++++++++++++++++++

* :func:`.apply_mode_jonesmat4x4` to apply the matrix to field data in fft space
* :func:`.apply_ray_jonesmat4x4` to apply the matrix to field data in real space
* :func:`.apply_jonesmat` to apply the matrix to field vector (1D).

"""
from __future__ import absolute_import, print_function, division

from dtmm.conf import cached_result, BETAMAX, FDTYPE, CDTYPE
from dtmm.wave import betaphi
from dtmm.tmm import alphaf,normalize_f
from dtmm.linalg import  dotmf, inv, dotmm
from dtmm.fft import fft2, ifft2
#make this module a drop-in replacement for jones module
from dtmm.jones import polarizer, as4x4, rotated_matrix
from dtmm.jones import * 
import numpy as np

from dtmm.diffract import diffraction_alphaf

def polarizer4x4(jones, fmat, out = None):
    """Returns a 4x4 polarizer matrix for applying in eigenframe.
    
    Numpy broadcasting rules apply.
    
    Parameters
    ----------
    jones : (...,2) array
        A length two array describing the jones vector in the eigenframe. 
        Jones vector should be normalized.
    fmat : (...,4,4) array
        A field matrix array of the isotropic medium.
    out : ndarray, optional
        Output array

    Returns
    -------
    jmat : (...,4,4) array
        A 4x4 matrix for field vector manipulation in the eigenframe.   
        
    See Also
    --------
    ray_jonesmat4x4 : for applying a general matrix in the laboratory frame.
    
    Examples
    --------
    >>> f = dtmm.tmm.f_iso(n = 1.) 
    >>> jones = dtmm.jones.jonesvec((1,0)) 
    >>> pol_mat = polarizer4x4(jones, f) #x polarizer matrix

    """
    jmat = polarizer(jones)
    return jonesmat4x4(jmat, fmat, out)

def jonesmat4x4(jmat, fmat, out = None):
    """Returns a 4x4 jones matrix for applying in eigenframe.
    
    Numpy broadcasting rules apply.
    
    Parameters
    ----------
    jmat : (...,2,2) array
        A 2x2 jones matrix in the eigenframe. Any of matrices in :mod:`dtmm.jones` 
        can be used.
    fmat : (...,4,4) array
        A field matrix array of the isotropic medium.
    out : ndarray, optional
        Output array   
        
    Returns
    -------
    jmat : (...,4,4) array
        A 4x4 matrix for field vector manipulation in the eigenframe.  
        
    See Also
    --------
    ray_jonesmat4x4 : for applying the jones matrix in the laboratory frame.
    """
    fmat = normalize_f(fmat)
    fmati = inv(fmat)
    pmat = as4x4(jmat)    
    m = dotmm(fmat,dotmm(pmat,fmati, out = out), out = out)
    return m

@cached_result
def mode_jonesmat4x4(shape, k, jmat, epsv = (1.,1.,1.), 
                            epsa = (0.,0.,0.), betamax = BETAMAX):
    """Returns a mode polarizer for fft of the field data in the laboratory frame.
    
    This is the most general set of jones matrices for field data. It is meant
    to be used in FFT space.
    
    Parameters
    ----------
    shape : (int,int)
        Shape of the 2D crossection of the field data.
    k : float or array of floats
        Wavenumber at which to compute the mode matrices.
    jmat : (2,2) array
        A 2x2 jones matrix that needs to be converted to 4x4 mode matrices.
    epsv : array
        Medium epsilon eigenvalues
    epsa : array
        Medium epsilon euler angles
    betamax : float
        The betamax parameter
        
    Returns
    -------
    pmat : (:,:,4,4) array
        Output matrix. Shape of the matirx is shape + (4,4) or len(ks) + shape + (4,4) 
        if k is an array. 
        
    See Also
    --------
    ray_jonesmat4x4 : for applying the jones matrix in the real space.
    jonesmat4x4 : for applying the jones matrix in the eigenframe.
    """
    ks = np.asarray(k, FDTYPE)
    ks = abs(ks)
    epsv = np.asarray(epsv, CDTYPE)
    epsa = np.asarray(epsa, FDTYPE)
    beta, phi = betaphi(shape,ks)
    alpha, f = diffraction_alphaf(shape, ks, epsv = epsv, 
                            epsa = epsa, betamax = betamax)

    beta, phi = betaphi(shape,ks)
    
    #matrix viewed in the eigenframe
    jmat = rotated_matrix(jmat, phi)
    pmat = jonesmat4x4(jmat, f)
    return pmat

@cached_result
def ray_jonesmat4x4(jmat, beta = 0, phi = 0, epsv = (1.,1.,1.), 
                            epsa = (0.,0.,0.),):
    """Returns a ray polarizer for field data in the laboratory frame.
    
    Numpy broadcasting rules apply.
    
    Parameters
    ----------
    jmat : (...,2,2) array
        A 2x2 jones matrix that needs to be converted to 4x4 mode matrices.
    beta : float
        The beta parameter of the beam.
    phi : float
        The phi parameter of the beam.
    epsv : (...,3) array
        Medium epsilon eigenvalues
    epsa : (...,3) array
        Medium epsilon euler angles
    betamax : float
        The betamax parameter
        
    Returns
    -------
    pmat : (...,4,4) array
        Output matrix.
        
    See Also
    --------
    mode_jonesmat4x4 : for applying the jones matrix in the fft space.
    jonesmat4x4 : for applying the jones matrix in the eigenframe.
    """
    epsv = np.asarray(epsv, CDTYPE)
    epsa = np.asarray(epsa, FDTYPE)
    beta = np.asarray(beta, FDTYPE)
    phi = np.asarray(phi, FDTYPE)
    
    alpha, f = alphaf(beta, phi, epsv, epsa)
    
    jmat = rotated_matrix(jmat, phi)
    pmat = jonesmat4x4(jmat, f)

    return pmat

def apply_mode_jonesmat4x4(pmat,field, out = None):
    """Multiplies matrix with field data in fft space.
    
    Parameters
    ----------
    pmat : (...,:,:,4,4) array
        A 4x4 jones matrix for field data
    field : (...,4,:,:) array
        Field data array.
    out : ndarray, optional
        If specified, the results are written here.
        
    Returns
    -------
    out : ndarray
        Computed field array of shape (...,4,:,:).
    """
    fft = fft2(field, out = out)
    pfft = dotmf(pmat, fft ,out  = fft)
    return ifft2(fft, out = pfft)

def apply_ray_jonesmat4x4(pmat,field, out = None):
    """Multiplies matrix with field data in real space.
    
    Parameters
    ----------
    pmat : (...,4,4) array
        A 4x4 jones matrix for field data
    field : (...,4,:,:) array
        Field data array.
    out : ndarray, optional
        If specified, the results are written here.
        
    Returns
    -------
    out : ndarray
        Computed field array of shape (...,4,:,:).
    """
    return dotmf(pmat, field ,out  = out)

def apply_jonesmat(pmat, field, out = None):
    """Multiplies a (2x2) or (4x4) jones matrix with field vector
    
    Parameters
    ----------
    pmat : (...,4,4) array
        A 4x4 jones matrix for field data
    field : (...,4) array
        Field vector array.
    out : ndarray, optional
        If specified, the results are written here.
        
    Returns
    -------
    out : ndarray
        Computed field array of shape (...,4).
    """
    return dotmm(pmat, field ,out  = out)
    
 