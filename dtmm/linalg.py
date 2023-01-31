"""
Numba optimized linear algebra functions for 4x4 matrices and 2x2 matrices.
"""

from __future__ import absolute_import, print_function, division
from dtmm.conf import NCDTYPE, NFDTYPE, NUMBA_TARGET,NUMBA_PARALLEL, NUMBA_CACHE, NUMBA_FASTMATH, CDTYPE, FDTYPE
from numba import njit, prange, guvectorize, boolean
import numpy as np

if not NUMBA_PARALLEL:
    prange = range
    

@njit([NFDTYPE(NFDTYPE[:]),NFDTYPE(NCDTYPE[:])], cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)        
def _vecabs2(v):
    """Computes vec.(vec.conj)"""
    out = 0.
    for i in range(len(v)):
        out = out + v[i].real**2 + v[i].imag**2
    return out

@njit([NFDTYPE(NFDTYPE[:]),NCDTYPE(NCDTYPE[:])], cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)        
def _vnorm2(v):
    """Computes vec.vec"""
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2]

@njit([NFDTYPE[:](NFDTYPE[:],NFDTYPE[:],NFDTYPE[:]),NCDTYPE[:](NCDTYPE[:],NCDTYPE[:],NCDTYPE[:]), ], cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)        
def _cross(v1,v2,v3):
    """performs vector cross product"""
    v30 = v1[1] * v2[2] - v1[2] * v2[1]
    v31 = v1[2] * v2[0] - v1[0] * v2[2]
    v32 = v1[0] * v2[1] - v1[1] * v2[0]
    v3[0] = v30
    v3[1] = v31
    v3[2] = v32
    return v3

_eigvec0_decl = [
                 (NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:]),
                 (NCDTYPE,NCDTYPE,NCDTYPE,NCDTYPE,NCDTYPE,NCDTYPE,NCDTYPE,NCDTYPE[:],NCDTYPE[:],NCDTYPE[:],NCDTYPE[:])
                 ]

@njit(_eigvec0_decl, cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)            
def _eigvec0(a00,a11,a22,a01,a02,a12,eval0,vec0, tmp1, tmp2, tmp3):
    """The first eigenvector"""
    
    row0 = vec0
    row0[0] = a00 - eval0
    row0[1] = a01
    row0[2] = a02
   
    row1 = tmp1
    row1[0] = a01 
    row1[1] = a11 - eval0
    row1[2] = a12
    
    row2 = tmp2
    row2[0] = a02 
    row2[1] = a12
    row2[2] = a22 - eval0
    
    r0xr1 = _cross(row0, row1, tmp3 ) #tmp[0] is empty
    r1xr2 = _cross(row1, row2, tmp1 ) #row1 is in tmp1, but it can be deleted
    r0xr2 = _cross(row0, row2, tmp2 )
    
    d0 = _vecabs2(r0xr1)
    d1 = _vecabs2(r0xr2)
    d2 = _vecabs2(r1xr2)
    
    dmax = d0
    imax = 0
    
    if d1 > dmax:
        dmax = d1
        imax = 1
    if d2 > dmax:
        imax = 2
        
    if imax == 0:
        norm = 1/(_vnorm2(r0xr1)**0.5)
        vec0[:] = r0xr1[:] 
    elif imax == 1:
        norm = 1/(_vnorm2(r0xr2)**0.5)
        vec0[:] = r0xr2[:] 
    else:
        norm = 1/(_vnorm2(r1xr2)**0.5)
        vec0[:] = r1xr2[:] 
        
    vec0 *= norm

@njit([(NFDTYPE[:],NFDTYPE[:],NFDTYPE[:]),(NCDTYPE[:],NCDTYPE[:],NCDTYPE[:])], cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH) 
def _orthogonal_basis(w,u,v):
    """build an orthogonal basis based on normalized input vector w"""
    if abs(w[0]) > abs(w[1]):
        u[0] = (-w[2])
        u[1] = 0
        u[2] = w[0]
        scale = 1 / (_vecabs2(u)**0.5)
        u *= scale
    else:
        u[0] = 0
        u[1] = w[2]
        u[2] = (-w[1])
        scale = 1 / (_vecabs2(u)**0.5)
        u *= scale   
    _cross(w,u,v)

    
_eigvec1_decl = [
                 (NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE,NFDTYPE[:],NFDTYPE,NFDTYPE[:],NFDTYPE[:],NFDTYPE[:]),
                 (NCDTYPE,NCDTYPE,NCDTYPE,NCDTYPE,NCDTYPE,NCDTYPE,NCDTYPE[:],NCDTYPE,NCDTYPE[:],NCDTYPE[:],NCDTYPE[:])
                 ]   

@njit(_eigvec1_decl, cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)            
def _eigvec1(a00,a11,a22,a01,a02,a12,vec0,eval1, vec1, tmp1, tmp2):
    """The second eigenvector"""
    u = vec1
    v = tmp1
    AU = tmp2
    
    _orthogonal_basis(vec0, u, v)
    
    AU[0] = a00 * u[0] + a01 * u[1] + a02 * u[2]
    AU[1] = a01 * u[0] + a11 * u[1] + a12 * u[2]
    AU[2] = a02 * u[0] + a12 * u[1] + a22 * u[2]
    
    m00 = u[0] * AU[0] + u[1] * AU[1] + u[2] * AU[2] - eval1
    
    AV = tmp2
   
    AV[0] = a00 * v[0] + a01 * v[1] + a02 * v[2]
    AV[1] = a01 * v[0] + a11 * v[1] + a12 * v[2]
    AV[2] = a02 * v[0] + a12 * v[1] + a22 * v[2]    
    
    m01 = u[0] * AV[0] + u[1] * AV[1] + u[2] * AV[2]
    m11 = v[0] * AV[0] + v[1] * AV[1] + v[2] * AV[2] - eval1
    
    absm00 = abs(m00)
    absm01 = abs(m01)
    absm11 = abs(m11)
    
    if absm00 >= absm11:
        m = max(absm00,absm01)
        if m > 0:
            if absm00 >= absm01:
                m01 /= m00 
                m00 = 1/((1+m01**2)**0.5)
                m01 *= m00
            else:
                m00 /= m01
                m01 = 1/((1+m00**2)**0.5) 
                m00 *= m01
            for i in range(3):
                vec1[i] = m01* u[i] - m00 * v[i] #vec1 is u so we are doing vec1 = m01 * u
    else:
        m = max(absm11,absm01)
        if m > 0:
            if absm11 >= absm01:
                m01 /= m11
                m11= 1/((1+m01**2)**0.5)
                m01 *= m11
            else:
                m11 /= m01
                m01 = 1/((1+m11**2)**0.5) 
                m11 *= m01
                
            for i in range(3):
                vec1[i] = m11* u[i] - m01 * v[i]   
 
    
@njit([(NFDTYPE[:],NFDTYPE[:,:],NFDTYPE[:],NFDTYPE[:,:]),(NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath=NUMBA_FASTMATH)
def _sort_eigvec(eps,r, epsout, rout):
    """Eigen modes sorting based on eigenvalues. Finds extraordinary axis,
    performs cyclic permutation of axes to move the extraordinary axis to 3"""
    e0 = eps[0]#.real #real part of the refractive index
    e1 = eps[1]#.real
    e2 = eps[2]#.real
    
    m2 = np.abs(e1-e0)
    m1 = np.abs(e2-e0)
    m0 = np.abs(e2-e1)
    
    #assume extraordinary is 2    
    i,j,k = 0,1,2
    
    #if we need to move eigenavlue to axis 2... do a cyclical permutation, to preserve right-handed coordinate system
    if m2 > m1 or m2 > m0:
        if m1 > m0:
            #extraordinary is 0
            i,j,k = 1,2,0
        else:
            #extraordinary is 1
            i,j,k = 2,0,1
            
    #perform sorting

    eps0 = eps[i]
    eps1 = eps[j]
    eps2 = eps[k]
    
    r00 = r[i,0]
    r01 = r[i,1]
    r02 = r[i,2]
    
    r10 = r[j,0]
    r11 = r[j,1]
    r12 = r[j,2]
    
    r20 = r[k,0]
    r21 = r[k,1]
    r22 = r[k,2]
    
    rout[0] = (r00,r01,r02)
    rout[1] = (r10,r11,r12)
    rout[2] = (r20,r21,r22)
          
    epsout[0] = eps0
    epsout[1] = eps1
    epsout[2] = eps2
        
    
_EIG_DECL = [(NFDTYPE[:,:],boolean[:],NFDTYPE[:],NFDTYPE[:,:]), (NCDTYPE[:,:],boolean[:],NCDTYPE[:], NCDTYPE[:,:])]         

@guvectorize(_EIG_DECL, '(m,m),()->(m),(m,m)', target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)   
def _tensor_eig(tensor, is_real, eig,vec):
    """Computes eigenvalues of a tensor using analytical noniterative algorithm
    adapted from A Robust Eigensolver for 3 × 3 Symmetric Matrices by 
    David Eberly @ Geometric Tools.
    
    Input array is overwritten.
    """
    
    #if norm of the offdiagonal elements is smaller than this, treat the
    #tensor as a diagonal tensor. This improves handling of real diagonal epsilon 
    #data treated as complex data, where the complex part introduces unwanted
    # rotations of the eigenframe, in particular when working with floats.
    normtol = 1e-12
    
    m1 = max(abs(tensor[0,0]),abs(tensor[1,1]))
    m2 = max(abs(tensor[2,2]),abs(tensor[0,1]))
    m3 = max(abs(tensor[0,2]),abs(tensor[1,2]))
    
    scale = max(max(m1,m2),m3)
    
    if scale == 0.:
        #zero matrix.. set eigenvalues to zero
        eig[0] = 0.
        eig[1] = 0.
        eig[2] = 0.
        vec[0] = (1,0,0)
        vec[1] = (0,1,0)
        vec[2] = (0,0,1)
    else:
        #precondition the matrix to avoid floating-point overflow
        scalem = 1 / scale
        a00 = tensor[0,0]*scalem
        a11 = tensor[1,1]*scalem
        a22 = tensor[2,2]*scalem
        a01 = tensor[0,1]*scalem
        a02 = tensor[0,2]*scalem
        a12 = tensor[1,2]*scalem
        
        q = (a00 + a11 + a22)/3
        
        b00 = a00 - q
        b11 = a11 - q
        b22 = a22 - q
            
        norm = a01**2 + a02**2 + a12**2
        
        if abs(norm) > normtol:
            p = ((b00**2 + b11**2 +b22**2 + 2 * norm)/6)**0.5
        
            c00 = b11 * b22 - a12 * a12
            c01 = a01 * b22 - a12 * a02
            c02 = a01 * a12 - b11 * a02
            
            half_det = (b00 * c00 - a01 * c01 + a02 * c02) / (p * p * p * 2)
            
            if is_real[0] == True:
                #for real data, half_det is between -1 and 1, to avoid possible rounding error, clamp it, just to make sure
                half_det = min(max(half_det.real, -1),1)
                
            phi = np.arccos(half_det)/3.
            
            #this number is closest to 2*np.pi/3 in float or double
            twoThirdsPi = NFDTYPE(2.09439510239319549)
            
            eig0 = np.cos(phi + twoThirdsPi) * 2
            eig2 = np.cos(phi) * 2
            eig1 = -(eig0 + eig2)
            
            #eigenvalues sorted by increasing 
            eig0 = (eig0 * p + q) 
            eig1 = (eig1 * p + q) 
            eig2 = (eig2 * p + q) 
            
            #now compute the eigenvectors
            if half_det.real >= 0:
                _eigvec0(a00,a11,a22,a01,a02,a12,eig2,vec[2], vec[1],vec[0], eig)    
                _eigvec1(a00,a11,a22,a01,a02,a12,vec[2],eig1,vec[1],vec[0],eig)
                _cross(vec[1],vec[2],vec[0])
            else:
                _eigvec0(a00,a11,a22,a01,a02,a12,eig0,vec[0],vec[1],vec[2],eig) 
                _eigvec1(a00,a11,a22,a01,a02,a12,vec[0],eig1,vec[1],vec[2],eig) 
                _cross(vec[0],vec[1],vec[2])
                
            eig[0] = (eig0 *  scale) 
            eig[1] = (eig1 *  scale) 
            eig[2] = (eig2 *  scale) 
        else:
            #matrix is diagonal
            eig[0] = a00 *  scale
            eig[1] = a11 *  scale
            eig[2] = a22 *  scale
            vec[0] = (1,0,0)
            vec[1] = (0,1,0)
            vec[2] = (0,0,1)
        
        _sort_eigvec(eig,vec,eig,vec)

_EIG_DECL = [(NFDTYPE[:,:],NFDTYPE[:],NFDTYPE[:,:]), (NCDTYPE[:,:],NCDTYPE[:], NCDTYPE[:,:])]         


@guvectorize(_EIG_DECL, '(m,m)->(m),(m,m)', target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)   
def _eig(tensor, eig,vec):
    e,v = np.linalg.eig(tensor)
    #_sort_eigvec uses rows for eigvecs.. so we need to transpose v and vec
    v = v.transpose()
    _sort_eigvec(e,v,eig,vec.transpose())
    
def eig(matrix,overwrite_x = False):
    """Computes eigenvalues and eigenvectors of 3x3 matrix using numpy.linalg.eig.
    Eigenvalues are sorted so that eig[2] is the most distinct (extraordinary).

    Parameters
    ----------
    matrix: (...,3,3) array 
        A 3x3 matrix.
    overwrite_x : bool, optional
        Ifset, the function will write eigenvectors as rows in the input array.
        
    Returns
    -------
    w : (..., 3) array
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are ordered so that the third eigenvalue is most
        distinct and first two are least distinct
    
    v : (..., 3, 3) array
        The normalized (unit "length") eigenvectors, such that the
        column ``v[:,i]`` is the eigenvector corresponding to the
        eigenvalue ``w[i]``.
    """
    matrix = np.asarray(matrix)
    if matrix.shape[-2:] != (3,3):
        raise ValueError("Only 3x3 matrices supported")
    if overwrite_x == True:
        out = (np.empty(matrix.shape[:-1],matrix.dtype), matrix)
        return _eig(matrix,out[0], out[1])
    else:
        return _eig(matrix)   
        
def tensor_eig(tensor, overwrite_x = False):
    """Computes eigenvalues and eigenvectors of a tensor.
    
    Eigenvalues are sorted so that eig[2] is the most distinct (extraordinary).
    
    If tensor is provided as a length 6 matrix, the elements are 
    a[0,0], a[1,1], a[2,2], a[0,1], a[0,2], a[1,2]. If provided as a (3x3) 
    matrix a, the rest of the elements are silently ignored.
    
    Parameters
    ----------
    tensor : (...,6) or (...,3,3) array 
        A length 6 array or 3x3 matrix
    overwrite_x : bool, optional
        If tensor is (...,3,3) array, the function will write eigenvectors
        as rows in the input array.
        
    Returns
    -------
    w : (..., 3) array
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are ordered so that the third eigenvalue is most
        distinct and first two are least distinct
    
    v : (..., 3, 3) array
        The normalized (unit "length") eigenvectors, such that the
        column ``v[:,i]`` is the eigenvector corresponding to the
        eigenvalue ``w[i]``.
    """
    tensor = np.asarray(tensor)
    matrix = tensor
    out = None
    
    _is_real = (np.iscomplexobj(matrix) == False)
    
    _eig = _tensor_eig
    #_eig = np.linalg.eig
    
    if tensor.shape[-1] == 6:
        matrix = np.empty(tensor.shape[0:-1] + (3,3),tensor.dtype)
        matrix[...,0,0] = tensor[...,0]
        matrix[...,1,1] = tensor[...,1]
        matrix[...,2,2] = tensor[...,2]
        matrix[...,0,1] = tensor[...,3]
        matrix[...,0,2] = tensor[...,4]
        matrix[...,1,2] = tensor[...,5]
        # we can overwrite temporary matrix data
        out = (np.empty(tensor.shape[:-1] + (3,),tensor.dtype), matrix)
    elif matrix.shape[-2:] != (3,3):
        raise ValueError("Not a matrix of size (...,3,3) or (...,6)")
    else:
        if overwrite_x == True:
            out = (np.empty(matrix.shape[:-1],matrix.dtype), matrix)
    if out is None:
        #eigv, vec = _eig(matrix)
        eigv, vec = _eig(matrix,_is_real)
    else:
        #eigv, vec = _eig(matrix)
        eigv, vec = _eig(matrix, _is_real, out[0], out[1])
        
    return eigv, np.swapaxes(vec, -1,-2) #as returned by np.linalg.eig, eigenvectors are in columns, not rows
    

@njit([(NCDTYPE[:, :], NCDTYPE[:, :])], cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH)
def _inv2x2(src, dst):
    """

    Parameters
    ----------
    src : array
        (2, 2) input array to invert
    dst : array
        (2, 2) output array to store the invert input array


    """
    # Extract individual elements
    a = src[0, 0]
    b = src[0, 1]
    c = src[1, 0]
    d = src[1, 1]

    # Calculate determinant
    det = a * d - b * c
    if det == 0.:
        det = 0.
    else:
        det = 1. / det

    # Calcualte inverse
    dst[0, 0] =  d * det
    dst[0, 1] = -b * det
    dst[1, 0] = -c * det
    dst[1, 1] =  a * det

@njit([NCDTYPE[:,:](NCDTYPE[:,:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _inv4x4(src,dst):
    
    #calculate pairs for first 8 elements (cofactors)
    tmp0 = src[2,2] * src[3,3]
    tmp1 = src[3,2] * src[2,3]
    tmp2 = src[1,2] * src[3,3]
    tmp3 = src[3,2] * src[1,3]
    tmp4 = src[1,2] * src[2,3]
    tmp5 = src[2,2] * src[1,3]
    tmp6 = src[0,2] * src[3,3]
    tmp7 = src[3,2] * src[0,3]
    tmp8 = src[0,2] * src[2,3]
    tmp9 = src[2,2] * src[0,3]
    tmp10 = src[0,2] * src[1,3]
    tmp11 = src[1,2] * src[0,3]
    # calculate first 8 elements (cofactors)
    dst[0,0] = tmp0*src[1,1] + tmp3*src[2,1] + tmp4*src[3,1] -tmp1*src[1,1] - tmp2*src[2,1] - tmp5*src[3,1]
    dst[0,1] = tmp1*src[0,1] + tmp6*src[2,1] + tmp9*src[3,1] - tmp0*src[0,1] - tmp7*src[2,1] - tmp8*src[3,1]
    dst[0,2] = tmp2*src[0,1] + tmp7*src[1,1] + tmp10*src[3,1] - tmp3*src[0,1] - tmp6*src[1,1] - tmp11*src[3,1]
    dst[0,3] = tmp5*src[0,1] + tmp8*src[1,1] + tmp11*src[2,1] - tmp4*src[0,1] - tmp9*src[1,1] - tmp10*src[2,1]
    
    dst[1,0] = tmp1*src[1,0] + tmp2*src[2,0] + tmp5*src[3,0] - tmp0*src[1,0] - tmp3*src[2,0] - tmp4*src[3,0]
    dst[1,1] = tmp0*src[0,0] + tmp7*src[2,0] + tmp8*src[3,0] - tmp1*src[0,0] - tmp6*src[2,0] - tmp9*src[3,0]
    dst[1,2] = tmp3*src[0,0] + tmp6*src[1,0] + tmp11*src[3,0] - tmp2*src[0,0] - tmp7*src[1,0] - tmp10*src[3,0]
    dst[1,3] = tmp4*src[0,0] + tmp9*src[1,0] + tmp10*src[2,0] - tmp5*src[0,0] - tmp8*src[1,0] - tmp11*src[2,0]
    # calculate pairs for second 8 elements (cofactors) 
    tmp0 = src[2,0]*src[3,1]
    tmp1 = src[3,0]*src[2,1]
    tmp2 = src[1,0]*src[3,1]
    tmp3 = src[3,0]*src[1,1]
    tmp4 = src[1,0]*src[2,1]
    tmp5 = src[2,0]*src[1,1]
    tmp6 = src[0,0]*src[3,1]
    tmp7 = src[3,0]*src[0,1]
    tmp8 = src[0,0]*src[2,1]
    tmp9 = src[2,0]*src[0,1]
    tmp10 = src[0,0]*src[1,1]
    tmp11 = src[1,0]*src[0,1]

    dst[2,0] = tmp0*src[1,3] + tmp3*src[2,3] + tmp4*src[3,3] - (tmp1*src[1,3] + tmp2*src[2,3] + tmp5*src[3,3])
    dst[2,1] = tmp1*src[0,3] + tmp6*src[2,3] + tmp9*src[3,3] - (tmp0*src[0,3] + tmp7*src[2,3] + tmp8*src[3,3])
    dst[2,2] = tmp2*src[0,3] + tmp7*src[1,3] + tmp10*src[3,3] - (tmp3*src[0,3] + tmp6*src[1,3] + tmp11*src[3,3])
    dst[2,3] = tmp5*src[0,3] + tmp8*src[1,3] + tmp11*src[2,3] - (tmp4*src[0,3] + tmp9*src[1,3] + tmp10*src[2,3])
    dst[3,0] = tmp2*src[2,2] + tmp5*src[3,2] + tmp1*src[1,2] - (tmp4*src[3,2] + tmp0*src[1,2] + tmp3*src[2,2])
    dst[3,1] = tmp8*src[3,2] + tmp0*src[0,2] + tmp7*src[2,2] - (tmp6*src[2,2] + tmp9*src[3,2] + tmp1*src[0,2])
    dst[3,2] = tmp6*src[1,2] + tmp11*src[3,2] + tmp3*src[0,2] - (tmp10*src[3,2] + tmp2*src[0,2] + tmp7*src[1,2])
    dst[3,3] = tmp10*src[2,2] + tmp4*src[0,2] + tmp9*src[1,2] - (tmp8*src[1,2] + tmp11*src[2,2] + tmp5*src[0,2])
    #/* calculate determinant */
    det=src[0,0]*dst[0,0]+src[1,0]*dst[0,1]+src[2,0]*dst[0,2]+src[3,0]*dst[0,3]
    #/* calculate matrix inverse */
    if det == 0:
        det = 0.
    else:
        det = 1./det
    for i in range(4):
        for j in range(4):
            dst[i,j] = dst[i,j]*det
            
    return dst


@guvectorize([(NCDTYPE[:,:], NCDTYPE[:,:])], '(n,n)->(n,n)', target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def inv(mat, out):
    """inv(mat), gufunc
    
    Calculates inverse of a 4x4 complex matrix or 2x2 complex matrix
    
    Parameters
    ----------
    mat : ndarray
       Input array
    
    Examples
    --------
    
    >>> a = np.random.randn(4,4) + 0j
    >>> ai = inv(a)
    
    >>> from numpy.linalg import inv
    >>> ai2 = inv(a)
    
    >>> np.allclose(ai2,ai)
    True
    """
    if mat.shape[0] == 2:
        _inv2x2(mat,out)
    elif mat.shape[0] == 4:
        _inv4x4(mat,out)
    else:
        inv = np.linalg.inv(mat)
        out[...] = inv


@njit([(NCDTYPE[:,:],NCDTYPE[:,:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def _dotmm4(a,b,out):
    a0 = a[0,0]*b[0,0]+a[0,1]*b[1,0]+a[0,2]*b[2,0]+a[0,3]*b[3,0]
    a1 = a[0,0]*b[0,1]+a[0,1]*b[1,1]+a[0,2]*b[2,1]+a[0,3]*b[3,1]
    a2 = a[0,0]*b[0,2]+a[0,1]*b[1,2]+a[0,2]*b[2,2]+a[0,3]*b[3,2]
    a3 = a[0,0]*b[0,3]+a[0,1]*b[1,3]+a[0,2]*b[2,3]+a[0,3]*b[3,3] 

    b0 = a[1,0]*b[0,0]+a[1,1]*b[1,0]+a[1,2]*b[2,0]+a[1,3]*b[3,0]
    b1 = a[1,0]*b[0,1]+a[1,1]*b[1,1]+a[1,2]*b[2,1]+a[1,3]*b[3,1]
    b2 = a[1,0]*b[0,2]+a[1,1]*b[1,2]+a[1,2]*b[2,2]+a[1,3]*b[3,2]
    b3 = a[1,0]*b[0,3]+a[1,1]*b[1,3]+a[1,2]*b[2,3]+a[1,3]*b[3,3] 

    c0 = a[2,0]*b[0,0]+a[2,1]*b[1,0]+a[2,2]*b[2,0]+a[2,3]*b[3,0]
    c1 = a[2,0]*b[0,1]+a[2,1]*b[1,1]+a[2,2]*b[2,1]+a[2,3]*b[3,1]
    c2 = a[2,0]*b[0,2]+a[2,1]*b[1,2]+a[2,2]*b[2,2]+a[2,3]*b[3,2]
    c3 = a[2,0]*b[0,3]+a[2,1]*b[1,3]+a[2,2]*b[2,3]+a[2,3]*b[3,3]
    
    d0 = a[3,0]*b[0,0]+a[3,1]*b[1,0]+a[3,2]*b[2,0]+a[3,3]*b[3,0]
    d1 = a[3,0]*b[0,1]+a[3,1]*b[1,1]+a[3,2]*b[2,1]+a[3,3]*b[3,1]
    d2 = a[3,0]*b[0,2]+a[3,1]*b[1,2]+a[3,2]*b[2,2]+a[3,3]*b[3,2]
    d3 = a[3,0]*b[0,3]+a[3,1]*b[1,3]+a[3,2]*b[2,3]+a[3,3]*b[3,3]     
 
    out[0,0] = a0
    out[0,1] = a1
    out[0,2] = a2
    out[0,3] = a3

    out[1,0] = b0
    out[1,1] = b1
    out[1,2] = b2
    out[1,3] = b3  

    out[2,0] = c0
    out[2,1] = c1
    out[2,2] = c2
    out[2,3] = c3 
    
    out[3,0] = d0
    out[3,1] = d1
    out[3,2] = d2
    out[3,3] = d3 
    
@njit([(NCDTYPE[:,:],NCDTYPE[:,:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def _dotmm2(a,b,out):
    a0 = a[0,0]*b[0,0]+a[0,1]*b[1,0]
    a1 = a[0,0]*b[0,1]+a[0,1]*b[1,1]

    b0 = a[1,0]*b[0,0]+a[1,1]*b[1,0]
    b1 = a[1,0]*b[0,1]+a[1,1]*b[1,1] 
   
    out[0,0] = a0
    out[0,1] = a1

    out[1,0] = b0
    out[1,1] = b1
    
# @njit([(NCDTYPE[:,::1],NCDTYPE[:,::1],NCDTYPE[:,::1])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
# def _dotmm(a,b,out):
#     np.dot(a,b, out)
 
@njit([(NCDTYPE[:,:],NCDTYPE[:,:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def _dotmm(a,b,out):
    tmp = np.zeros_like(out)
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            #tmp = NCDTYPE(0)
            for k in range(a.shape[1]):
                tmp[i,j] += a[i,k] *b[k,j]
                
    out[...] = tmp


@njit([(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def _dotmv(a,b,out):
    tmp = np.zeros_like(out)
    for i in range(a.shape[0]):
        for k in range(a.shape[1]):
            tmp[i] += a[i,k] *b[k]
                
    out[...] = tmp

#@njit([(NCDTYPE[:,:],NFDTYPE[:],NCDTYPE[:,:])], cache = NUMBA_CACHE)    
#def _dotmr2(a,b,out):
#    a0 = a[0,0]*b[0]-a[0,2]*b[1]
#    a1 = a[0,1]*b[0]+a[0,3]*b[1]
#    a2 = a[0,0]*b[1]+a[0,2]*b[0]
#    a3 = -a[0,1]*b[1]+a[0,3]*b[0] 
#
#    b0 = a[1,0]*b[0]-a[1,2]*b[1]
#    b1 = a[1,1]*b[0]+a[1,3]*b[1]
#    b2 = a[1,0]*b[1]+a[1,2]*b[0]
#    b3 = -a[1,1]*b[1]+a[1,3]*b[0] 
#
#    c0 = a[2,0]*b[0]-a[2,2]*b[1]
#    c1 = a[2,1]*b[0]+a[2,3]*b[1]
#    c2 = a[2,0]*b[1]+a[2,2]*b[0]
#    c3 = -a[2,1]*b[1]+a[2,3]*b[0] 
#    
#    d0 = a[3,0]*b[0]-a[3,2]*b[1]
#    d1 = a[3,1]*b[0]+a[3,3]*b[1]
#    d2 = a[3,0]*b[1]+a[3,2]*b[0]
#    d3 = -a[3,1]*b[1]+a[3,3]*b[0]     
# 
#    out[0,0] = a0
#    out[0,1] = a1
#    out[0,2] = a2
#    out[0,3] = a3
#
#    out[1,0] = b0
#    out[1,1] = b1
#    out[1,2] = b2
#    out[1,3] = b3  
#
#    out[2,0] = c0
#    out[2,1] = c1
#    out[2,2] = c2
#    out[2,3] = c3 
#    
#    out[3,0] = d0
#    out[3,1] = d1
#    out[3,2] = d2
#    out[3,3] = d3 
#    
    
@njit([(NFDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def _dotr2m(r,a,out):
    a0 = a[0,0]*r[0]-a[2,0]*r[1]
    a1 = a[0,1]*r[0]-a[2,1]*r[1]
    a2 = a[0,2]*r[0]-a[2,2]*r[1]
    a3 = a[0,3]*r[0]-a[2,3]*r[1] 

    b0 = a[1,0]*r[0]+a[3,0]*r[1]
    b1 = a[1,1]*r[0]+a[3,1]*r[1]
    b2 = a[1,2]*r[0]+a[3,2]*r[1]
    b3 = a[1,3]*r[0]+a[3,3]*r[1] 

    c0 = a[0,0]*r[1]+a[2,0]*r[0]
    c1 = a[0,1]*r[1]+a[2,1]*r[0]
    c2 = a[0,2]*r[1]+a[2,2]*r[0]
    c3 = a[0,3]*r[1]+a[2,3]*r[0] 
    
    d0 = -a[1,0]*r[1]+a[3,0]*r[0]
    d1 = -a[1,1]*r[1]+a[3,1]*r[0]
    d2 = -a[1,2]*r[1]+a[3,2]*r[0]
    d3 = -a[1,3]*r[1]+a[3,3]*r[0]   
 
    out[0,0] = a0
    out[0,1] = a1
    out[0,2] = a2
    out[0,3] = a3

    out[1,0] = b0
    out[1,1] = b1
    out[1,2] = b2
    out[1,3] = b3  

    out[2,0] = c0
    out[2,1] = c1
    out[2,2] = c2
    out[2,3] = c3 
    
    out[3,0] = d0
    out[3,1] = d1
    out[3,2] = d2
    out[3,3] = d3 
    
    
@njit([(NFDTYPE[:],NCDTYPE[:],NCDTYPE[:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def _dotr2v(r,a,out):
    out[0]= a[0]*r[0]-a[2]*r[1]
    out[1]= a[1]*r[0]+a[3]*r[1]
    out[2]= a[0]*r[1]+a[2]*r[0]    
    out[3]= -a[1]*r[1]+a[3]*r[0]

    
@njit([(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmv4(a, b, out):
    out0 = a[0,0] * b[0] + a[0,1] * b[1] + a[0,2] * b[2] +a[0,3] * b[3]
    out1 = a[1,0] * b[0] + a[1,1] * b[1] + a[1,2] * b[2] +a[1,3] * b[3]
    out2 = a[2,0] * b[0] + a[2,1] * b[1] + a[2,2] * b[2] +a[2,3] * b[3]
    out3 = a[3,0] * b[0] + a[3,1] * b[1] + a[3,2] * b[2] +a[3,3] * b[3]
    out[0]= out0
    out[1]= out1
    out[2]= out2
    out[3]= out3
    
@njit([(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmv2(a, b, out):
    out0 = a[0,0] * b[0] + a[0,1] * b[1] 
    out1 = a[1,0] * b[0] + a[1,1] * b[1]
    out[0]= out0
    out[1]= out1
    

    
@njit([(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmd4(a, b, out):
    a0 = a[0,0]*b[0]
    a1 = a[0,1]*b[1]
    a2 = a[0,2]*b[2]
    a3 = a[0,3]*b[3] 
    
    out[0,0] = a0
    out[0,1] = a1
    out[0,2] = a2
    out[0,3] = a3

    a0 = a[1,0]*b[0]
    a1 = a[1,1]*b[1]
    a2 = a[1,2]*b[2]
    a3 = a[1,3]*b[3] 
    
    out[1,0] = a0
    out[1,1] = a1
    out[1,2] = a2
    out[1,3] = a3  

    a0 = a[2,0]*b[0]
    a1 = a[2,1]*b[1]
    a2 = a[2,2]*b[2]
    a3 = a[2,3]*b[3] 
    
    out[2,0] = a0
    out[2,1] = a1
    out[2,2] = a2
    out[2,3] = a3    
 
    a0 = a[3,0]*b[0]
    a1 = a[3,1]*b[1]
    a2 = a[3,2]*b[2]
    a3 = a[3,3]*b[3] 
    
    out[3,0] = a0
    out[3,1] = a1
    out[3,2] = a2
    out[3,3] = a3
    
@njit([(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmd2(a, b, out):
    a0 = a[0,0]*b[0]
    a1 = a[0,1]*b[1]
    
    out[0,0] = a0
    out[0,1] = a1

    a0 = a[1,0]*b[0]
    a1 = a[1,1]*b[1]
  
    out[1,0] = a0
    out[1,1] = a1

#@njit([(NCDTYPE[:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],parallel = NUMBA_PARALLEL, cache = NUMBA_CACHE)
#def _dotm1f4(a, b, out):
#    for i in prange(b.shape[1]):
#        for j in range(b.shape[2]):
#            b0 = b[0,i,j]
#            b1 = b[1,i,j]
#            b2 = b[2,i,j]
#            b3 = b[3,i,j]
#            
#            out0 = a[0,0] * b0 + a[0,1] * b1 + a[0,2] * b2 +a[0,3] * b3
#            out1 = a[1,0] * b0 + a[1,1] * b1 + a[1,2] * b2 +a[1,3] * b3
#            out2 = a[2,0] * b0 + a[2,1] * b1 + a[2,2] * b2 +a[2,3] * b3
#            out3 = a[3,0] * b0 + a[3,1] * b1 + a[3,2] * b2 +a[3,3] * b3
#            
#            out[0,i,j]= out0
#            out[1,i,j]= out1
#            out[2,i,j]= out2
#            out[3,i,j]= out3
#            
#@njit([(NCDTYPE[:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],parallel = NUMBA_PARALLEL, cache = NUMBA_CACHE)
#def _dotm1f2(a, b, out):
#    for i in prange(b.shape[1]):
#        for j in range(b.shape[2]):
#            b0 = b[0,i,j]
#            b1 = b[1,i,j]
#            
#            out0 = a[0,0] * b0 + a[0,1] * b1
#            out1 = a[1,0] * b0 + a[1,1] * b1 
#            
#            out[0,i,j]= out0
#            out[1,i,j]= out1
#            
@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmf4(a, b, out):
    for i in range(b.shape[1]):
        for j in range(b.shape[2]):
            b0 = b[0,i,j]
            b1 = b[1,i,j]
            b2 = b[2,i,j]
            b3 = b[3,i,j]
            
            out0 = a[i,j,0,0] * b0 + a[i,j,0,1] * b1 + a[i,j,0,2] * b2 +a[i,j,0,3] * b3
            out1 = a[i,j,1,0] * b0 + a[i,j,1,1] * b1 + a[i,j,1,2] * b2 +a[i,j,1,3] * b3
            out2 = a[i,j,2,0] * b0 + a[i,j,2,1] * b1 + a[i,j,2,2] * b2 +a[i,j,2,3] * b3
            out3 = a[i,j,3,0] * b0 + a[i,j,3,1] * b1 + a[i,j,3,2] * b2 +a[i,j,3,3] * b3
            
            out[0,i,j]= out0
            out[1,i,j]= out1
            out[2,i,j]= out2
            out[3,i,j]= out3  
            
@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmf2(a, b, out):
    for i in range(b.shape[1]):
        for j in range(b.shape[2]):
            b0 = b[0,i,j]
            b1 = b[1,i,j]
            
            out0 = a[i,j,0,0] * b0 + a[i,j,0,1] * b1 
            out1 = a[i,j,1,0] * b0 + a[i,j,1,1] * b1 
            
            out[0,i,j]= out0
            out[1,i,j]= out1


#@njit([(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:])],cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
#def _dotmdmv4(a, d, b, f,out):
#    f0 = f[0]
#    f1 = f[1]
#    f2 = f[2]
#    f3 = f[3,]
#    
#    out0 = b[0,0] * f0 + b[0,1] * f1 + b[0,2] * f2 +b[0,3] * f3
#    out1 = b[1,0] * f0 + b[1,1] * f1 + b[1,2] * f2 +b[1,3] * f3
#    out2 = b[2,0] * f0 + b[2,1] * f1 + b[2,2] * f2 +b[2,3] * f3
#    out3 = b[3,0] * f0 + b[3,1] * f1 + b[3,2] * f2 +b[3,3] * f3
#    
#    b0 = out0*d[0]
#    b1 = out1*d[1]
#    b2 = out2*d[2]
#    b3 = out3*d[3]
#    
#    out[0]= a[0,0] * b0 + a[0,1] * b1 + a[0,2] * b2 +a[0,3] * b3
#    out[1]= a[1,0] * b0 + a[1,1] * b1 + a[1,2] * b2 +a[1,3] * b3
#    out[2]= a[2,0] * b0 + a[2,1] * b1 + a[2,2] * b2 +a[2,3] * b3
#    out[3]= a[3,0] * b0 + a[3,1] * b1 + a[3,2] * b2 +a[3,3] * b3   
#
#
#@njit([(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
#def _dotmdmv2(a, d, b, f,out):
#    f0 = f[0]
#    f1 = f[1]
#    
#    out0 = b[0,0] * f0 + b[0,1] * f1
#    out1 = b[1,0] * f0 + b[1,1] * f1
#
#    
#    b0 = out0*d[0]
#    b1 = out1*d[1]
#    
#    out[0]= a[0,0] * b0 + a[0,1] * b1
#    out[1]= a[1,0] * b0 + a[1,1] * b1 
 
            
@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmdmf4(a, d, b, f,out):
    for i in range(f.shape[1]):
        for j in range(f.shape[2]):
            f0 = f[0,i,j]
            f1 = f[1,i,j]
            f2 = f[2,i,j]
            f3 = f[3,i,j]
            
            out0 = b[i,j,0,0] * f0 + b[i,j,0,1] * f1 + b[i,j,0,2] * f2 +b[i,j,0,3] * f3
            out1 = b[i,j,1,0] * f0 + b[i,j,1,1] * f1 + b[i,j,1,2] * f2 +b[i,j,1,3] * f3
            out2 = b[i,j,2,0] * f0 + b[i,j,2,1] * f1 + b[i,j,2,2] * f2 +b[i,j,2,3] * f3
            out3 = b[i,j,3,0] * f0 + b[i,j,3,1] * f1 + b[i,j,3,2] * f2 +b[i,j,3,3] * f3
            
            b0 = out0*d[i,j,0]
            b1 = out1*d[i,j,1]
            b2 = out2*d[i,j,2]
            b3 = out3*d[i,j,3]
            
            out[0,i,j]= a[i,j,0,0] * b0 + a[i,j,0,1] * b1 + a[i,j,0,2] * b2 +a[i,j,0,3] * b3
            out[1,i,j]= a[i,j,1,0] * b0 + a[i,j,1,1] * b1 + a[i,j,1,2] * b2 +a[i,j,1,3] * b3
            out[2,i,j]= a[i,j,2,0] * b0 + a[i,j,2,1] * b1 + a[i,j,2,2] * b2 +a[i,j,2,3] * b3
            out[3,i,j]= a[i,j,3,0] * b0 + a[i,j,3,1] * b1 + a[i,j,3,2] * b2 +a[i,j,3,3] * b3   

@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmdmf2(a, d, b, f,out):
    for i in range(f.shape[1]):
        for j in range(f.shape[2]):
            f0 = f[0,i,j]
            f1 = f[1,i,j]
            
            out0 = b[i,j,0,0] * f0 + b[i,j,0,1] * f1 
            out1 = b[i,j,1,0] * f0 + b[i,j,1,1] * f1 
            
            b0 = out0*d[i,j,0]
            b1 = out1*d[i,j,1]
            
            out[0,i,j]= a[i,j,0,0] * b0 + a[i,j,0,1] * b1 
            out[1,i,j]= a[i,j,1,0] * b0 + a[i,j,1,1] * b1  
                        

#@guvectorize([(NCDTYPE[:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],"(n,n),(n,m,k)->(n,m,k)",target = "cpu", cache = NUMBA_CACHE)
#def dotm1f(a, b, out):
#    if b.shape[0] == 2:
#        _dotm1f2(a, b, out)
#    else:
#        assert a.shape[0] >= 4 #make sure it is not smaller than 4
#        _dotm1f4(a, b, out)

@guvectorize([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],"(m,k,n,n),(n,m,k)->(n,m,k)",target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmf(a, b, out):
    """dotmf(a, b)
    
Computes a dot product of an array of 4x4 (or 2x2) matrix with 
a field array or an E-array (in case of 2x2 matrices).
"""
    if b.shape[0] == 2:
        _dotmf2(a, b, out)
    else:
        assert b.shape[0] >= 4 #make sure it is not smaller than 4
        _dotmf4(a, b, out)


def broadcast_m(m, field):
    """Broadcasts matrix m to match field spatial indices (the last two axes) so
    that it can be used in dot functions."""
    shape = m.shape[:-4]+ field.shape[-2:] + m.shape[-2:]
    return np.broadcast_to(m, shape)

def broadcast_d(d, field):
    """Broadcasts diagonal matrix d to match field spatial indices (the last two axes) so
    that it can be used in dot functions."""
    shape = d.shape[:-3]+ field.shape[-2:] + d.shape[-1:]
    return np.broadcast_to(d, shape)

def dotmf(a,b, out = None):
    """dotmf(a, b)
    
Computes a dot product of an array of 4x4 (or 2x2) matrix with 
a field array or an E-array (in case of 2x2 matrices).
"""
    try:
        return _dotmf(a, b, out)
    except:
        a = np.asarray(a)
        b = np.asarray(b)
        a = broadcast_m(a, b)  
        return _dotmf(a, b, out)

       
@guvectorize([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],"(m,k,n,n),(m,k,n),(m,k,n,n),(n,m,k)->(n,m,k)",target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmdmf(a, d,b,f, out):
    if f.shape[0] == 2:
        _dotmdmf2(a, d,b,f, out)
    else:    
        assert f.shape[0] >= 4 #make sure it is not smaller than 4
        _dotmdmf4(a,d, b, f,out)
        
def dotmdmf(a,d,b,f, out = None):
    """dotmdmf(a, d, b, f)
    
Computes a dot product of an array of 4x4 (or 2x2) matrices, array of diagonal matrices, 
another array of matrices and a field array or an E-array (in case of 2x2 matrices).

Notes
-----
This is equivalent to

>>> dotmf(dotmdm(a,d,b),f)
"""
    try:
        return _dotmdmf(a, d,b,f, out)
    except:
        m = dotmdm(a,d,b)
        return dotmf(m,f, out)
        
#    a = broadcast_m(a, f)
#    d = broadcast_d(d, f)
#    b = broadcast_m(b, f)
#    return _dotmdmf(a, d,b,f, out)
    

#@guvectorize([(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:])],"(n,n),(n),(n,n),(n)->(n)",target =NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
#def dotmdmv(a, d,b,f, out):
#    """dotmdmv(a, d, b, f)
#    
#Computes a dot product of an array of 4x4 (or 2x2) matrices, array of diagonal matrices, 
#another array of matrices and a vector.
#
#Notes
#-----
#This is equivalent to
#
#>>> dotmv(dotmdm(a,d,b),f)
#"""
#    if f.shape[0] == 2:
#        _dotmdmv2(a, d,b,f, out)
#    else:    
#        assert f.shape[0] >= 4 #make sure it is not smaller than 4
#        _dotmdmv4(a,d, b, f,out)
                
                
@guvectorize([(NCDTYPE[:,:],NCDTYPE[:,:],NCDTYPE[:,:])],"(n,k),(k,m)->(n,m)",target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def dotmm(a, b, out):
    """dotmm(a, b)
    
Computes an efficient dot product of a 4x4,  2x2 
or a less efficient general matrix multiplication.
"""
    if a.shape[0] == 2 and  a.shape[1] == 2:
        _dotmm2(a, b, out)
    elif a.shape[0] == 4 and a.shape[1] == 4:
        _dotmm4(a, b, out)
    else:
        _dotmm(a, b, out)
        
# def dotmm(a,b,out = None):
#     a = np.asarray(a)
#     b = np.asarray(b)
#     shape = np.broadcast_shapes(a.shape[0:-2],b.shape[0:-2])
#     x = np.broadcast_to(a,shape + a.shape[-2:]).copy()
#     y = np.broadcast_to(b,shape + b.shape[-2:]).copy()
#     if not x.data.c_contiguous: 
#         x = x.copy()
#     if not y.data.c_contiguous:
#         y = y.copy()
#     return _dotmm_vec(x,y,out)
    
    
# #
# @guvectorize([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:,:])],"(n,n,k,k),(n,n,k,k)->(n,n,k,k)", cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)        
# def bdotmm3(m1,m2, out):
#     """Performs a dot product of two nxn block matrices of blocks 4x4.
#     Matrices must be of shape nxnx4x4 that describe two mxm matrices
#     (m = 4*n) of blocks of size 4x4
#     """
#     assert m1.shape[3] == 4
    
#     tmp = np.empty(out.shape[2:],out.dtype)

#     n = m1.shape[0]    
#     for i in prange(n):
#         for j in range(n):
#             for k in range(n):
#                 if k == 0:
#                     _dotmm4(m1[i,k], m2[k,j], out[i,j] )
#                 else:
#                     _dotmm4(m1[i,k], m2[k,j], tmp)
#                     out[i,j] += tmp

# @njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:,:])], parallel = NUMBA_PARALLEL)        
# def bdotmm2(m1,m2, out):
#     """Performs a dot product of two nxn block matrices of blocks 4x4.
#     Matrices must be of shape nxnx4x4 that describe two mxm matrices
#     (m = 4*n) of blocks of size 4x4
#     """
#     assert m1.shape[3] == 4
    
#     tmp = np.empty(out.shape[2:],out.dtype)

#     n = m1.shape[0]    
#     for i in prange(n):
#         for j in range(n):
#             for k in range(n):
#                 if k == 0:
#                     _dotmm4(m1[i,k], m2[k,j], out[i,j] )
#                 else:
#                     _dotmm4(m1[i,k], m2[k,j], tmp)
#                     out[i,j] += tmp


def bdotmm_old(m1,m2, out = None):
    """Performs a dot product of two nxn block matrices of blocks of size 4x4
    or 2x2. Matrices must be of shape nxnx4x4  or nxnx2x2 that describe two 
    mxm matrices (m = 4*n or m = 2*n) of blocks of size 4x4 (or 2x2). 
    """
    assert m1.shape == m2.shape
    assert m2 is not out
    if out is None:
        out = np.empty(shape = m1.shape, dtype = CDTYPE)
 
    tmp = np.empty(shape = m1.shape, dtype = CDTYPE)
    for j in range(m1.shape[0]):
        # for some reason it is much more efficient to copy. This may be resolved in future numba versions.
        # TODO: inspect this issue and file a bug report
        #b = np.broadcast_to(m1[j][:,None,:,:],m2.shape).copy()
        tmp[...] = m1[j][:,None,:,:]
        b = tmp
        dotmm(b,m2, out = tmp)
        #dotmm(m1[j][:,None,:,:],m2, out = tmp)
        #out[j] = tmp.sum(-4, out = out[j])
        tmp.sum(-4, out = out[j])
    return out

def bdotmm(m1,m2):
    
    new_shape = m1.shape[:-4] + (m1.shape[-4] * m1.shape[-2], m1.shape[-3] * m1.shape[-1])
    _m1 = np.moveaxis(m1,-2,-3).reshape(new_shape)
    _m2 = np.moveaxis(m2,-2,-3).reshape(new_shape)
    
    new_shape = m1.shape[:-4] + (m1.shape[-4], m1.shape[-2], m1.shape[-3], m1.shape[-1])
    out = np.matmul(_m1,_m2).reshape(new_shape)
    return np.moveaxis(out,-2,-3)

def _bdotmm_ref(m1,m2, out = None):
    """same as bdotmm, but slower, for testing"""
    m1,m2 = np.broadcast_arrays(m1,m2)
    if out is None:
        out = np.zeros(m1.shape, CDTYPE)
    else:
        out[...] = 0.
    n = m1.shape[-3] 
    assert n == m1.shape[-4]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                out[...,i,j,:,:] += dotmm(m1[...,i,k,:,:],m2[...,k,j,:,:])
    return out


def bdotmd(m1,m2, out = None):
    """Performs a dot product of two nxn block matrices of blocks 4x4 or 2x2.
    The second matrix is block diagonal matrix of shape nx4x4 or nx2x2,
    The first matrix must be of shape nxnx4x4 or nxnx2x2 that describe two mxm matrices
    (m = 4*n or m = 2*n) of blocks of size 4x4 or 2x2.
    """
    if out is None:
        out = np.empty(shape = m1.shape, dtype = CDTYPE)

    for j in range(m1.shape[-3]):
        dotmm(m1[j],m2,out[j])
    return out

def _bdotmd_ref(m1,d):
    """same as bdotmd, but slower, for testing"""
    out = np.zeros_like(m1)
    for i in range(m1.shape[0]):
        for k in range(m1.shape[0]):
            out[i,k] +=  np.dot(m1[i,k],d[k])
    return out


def bdotdm(m1,m2, out = None):
    """Performs a dot product of two nxn block matrices of blocks 4x4 or 2x2.
    The first matrix is block diagonal matrix of shape nx4x4 or nx2x2,
    The second matrix must be of shape nxnx4x4 or nxnx2x2 that describe two mxm matrices
    (m = 4*n or m = 2*n) of blocks of size 4x4 or 2x2.
    """
    assert m2 is not out
    if out is None:
        out = np.empty(shape = m2.shape, dtype = CDTYPE)

    for j in range(m2.shape[-3]):
        dotmm(m1,m2[:,j],out[:,j])
    return out

def _bdotdm_ref(d,m):
    """Same as bdotdm, but slower, for testing"""
    out = np.zeros_like(m)
    for j in range(m.shape[0]):
        for k in range(m.shape[0]):
            out[k,j] +=  np.dot(d[k],m[k,j])
    return out


# def bdotmv(m,v, out = None):
#     """
#     """
#     tmp = dotmv(m,v)
#     return tmp.sum(-2, out = out)


# def _bdotmv_ref(m,v):    
#     m = np.moveaxis(m,-2,-3).copy()
#     shape = v.shape
#     m = m.reshape(m.shape[0]*4,m.shape[0]*4).copy()
#     v = v.reshape(v.shape[0]*4).copy()
#     out = dotmv(m,v).copy()
#     out = out.reshape(shape).copy()
#     return out



        
@guvectorize([(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])],"(n,n),(n)->(n,n)",target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def dotmd(a, d, out):
     """dotmd(a, d)
     
Computes a dot product of a 4x4 (or 2x2) matrix with a diagonal
matrix represented by a vector of shape 4 (or 2).
"""
     if a.shape[0]== 2:
         _dotmd2(a, d, out)
     else:
         assert a.shape[0] >= 4 #make sure it is not smaller than 4
         _dotmd4(a, d, out)

@guvectorize([(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:])],"(n,n),(n)->(n)",target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def dotmv(a, b, out):
    """dotmv(a, b)
    
Computes a dot product of a 4x4 or 2x2 matrix with a vector.
"""
    if a.shape[0] == 2:
        _dotmv2(a, b, out)
    elif a.shape[0] == 4:
        assert a.shape[0] >= 4 #make sure it is not smaller than 4
        _dotmv4(a, b, out)
    else:
        _dotmv(a, b, out)
    
@guvectorize([(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])],"(n,n),(n),(n,n)->(n,n)",target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def dotmdm(a, d, b, out):
    """dotmdm(a, d, b)
    
Computes a dot product of a 4x4 (or 2x2) matrix with a diagonal matrix (4- or 2-vector) 
and another 4x4 (or 2x2) matrix.
"""
    #assert a.shape[0] >= 4 #make sure it is not smaller than 4 
    if a.shape[0] == 2:
        _dotmd2(a, d, out)
        _dotmm2(out,b,out)
    else:
        assert a.shape[0] >= 4 #make sure it is not smaller than 4
        _dotmd4(a, d, out)
        _dotmm4(out,b,out)        
    

def multi_dot(arrays,  axis = 0, reverse = False):
    """Computes dot product of multiple 2x2 or 4x4 matrices. If reverse is 
    specified, it is performed in reversed order. Axis defines the axis over 
    which matrices are multiplied."""
    out = None
    if axis != 0:
        arrays = np.asarray(arrays)
        indices = range(arrays.shape[axis])
        #arrays = np.rollaxis(arrays, axis)
        arrays = np.moveaxis(arrays, axis, 0)
    else:
        indices = range(len(arrays))
    if reverse == True:
        indices = reversed(indices)
    for i in indices:
        if out is None:
            out = np.asarray(arrays[i]).copy()
        else:
            b = np.broadcast(out, arrays[i])
            if b.shape == out.shape:
                out = dotmm(out, arrays[i], out = out)
            else:
                out = dotmm(out, arrays[i])
    return out
    
    
__all__ = ["inv", "dotmm","dotmf","dotmv","dotmdm","dotmd","multi_dot","eig","tensor_eig"]

