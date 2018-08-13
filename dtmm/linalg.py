"""
Numba optimized linear algebra functions for 4x4 matrices and 2x2 martrices.
"""

from __future__ import absolute_import, print_function, division
from dtmm.conf import NC128DTYPE, NC64DTYPE, NCDTYPE, NFDTYPE, NUMBA_TARGET,NUMBA_PARALLEL, NUMBA_CACHE, NUMBA_FASTMATH

import numpy as np


from numba import njit, prange, guvectorize, vectorize

if NUMBA_PARALLEL == False:
    prange = range

#numba.config.DUMP_ASSEMBLY = 1

@njit([(NCDTYPE[:,:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _inv2x2(src,dst):
    a = src[0,0]
    b = src[0,1]
    c = src[1,0]
    d = src[1,1]
    det = a*d-b*c
    if det == 0:
        det = 0.#np.nan
    else:
        det = 1./det
    dst[0,0] = d*det
    dst[0,1] = -b*det
    dst[1,0] = -c*det
    dst[1,1] = a*det

@njit([NCDTYPE[:,:](NCDTYPE[:,:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _inv4x4(src,dst):
    
    #calculate pairs for first 8 elements (cofactors) */
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
    #/* calculate first 8 elements (cofactors) */
    dst[0,0] = tmp0*src[1,1] + tmp3*src[2,1] + tmp4*src[3,1] -tmp1*src[1,1] - tmp2*src[2,1] - tmp5*src[3,1]
    dst[0,1] = tmp1*src[0,1] + tmp6*src[2,1] + tmp9*src[3,1] - tmp0*src[0,1] - tmp7*src[2,1] - tmp8*src[3,1]
    dst[0,2] = tmp2*src[0,1] + tmp7*src[1,1] + tmp10*src[3,1] - tmp3*src[0,1] - tmp6*src[1,1] - tmp11*src[3,1]
    dst[0,3] = tmp5*src[0,1] + tmp8*src[1,1] + tmp11*src[2,1] - tmp4*src[0,1] - tmp9*src[1,1] - tmp10*src[2,1]
    
    
    dst[1,0] = tmp1*src[1,0] + tmp2*src[2,0] + tmp5*src[3,0] - tmp0*src[1,0] - tmp3*src[2,0] - tmp4*src[3,0]
    dst[1,1] = tmp0*src[0,0] + tmp7*src[2,0] + tmp8*src[3,0] - tmp1*src[0,0] - tmp6*src[2,0] - tmp9*src[3,0]
    dst[1,2] = tmp3*src[0,0] + tmp6*src[1,0] + tmp11*src[3,0] - tmp2*src[0,0] - tmp7*src[1,0] - tmp10*src[3,0]
    dst[1,3] = tmp4*src[0,0] + tmp9*src[1,0] + tmp10*src[2,0] - tmp5*src[0,0] - tmp8*src[1,0] - tmp11*src[2,0]
    #/* calculate pairs for second 8 elements (cofactors) */
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
    #if det == 0:
    #    return 0
    #else:
    #    det = 1./det
    #    for i in range(4):
    #        for j in range(4):
    #            out[i,j] = dst[j + 4*i]*det
    #    return 1
    if det == 0:
        det = 0.#np.nan
    else:
        det = 1./det
    for i in range(4):
        for j in range(4):
            dst[i,j] = dst[i,j]*det
            
    return dst


@guvectorize([(NCDTYPE[:,:], NCDTYPE[:,:])], '(n,n)->(n,n)', target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def inv(mat, output):
    """inv(mat)
    
Calculates inverse of a 4x4 complex matrix or 2x2 complex matrix

Parameters
----------
mat : array_like
    A 4x4 or 2x2 complex matrix.

Examples
--------

>>> a = np.random.randn(4,4) + 0j
>>> ai = inv4x4(a)

>>> from numpy.linalg import inv
>>> ai2 = inv(a)

>>> np.allclose(ai2,ai)
True
    """
    if mat.shape[0] == 2:
        _inv2x2(mat,output)
    else:
        assert mat.shape[0] >= 4
        _inv4x4(mat,output)
        
inv4x4 = inv
inv2x2 = inv

@njit([(NCDTYPE[:,:],NCDTYPE[:,:],NCDTYPE[:,:])], cache = NUMBA_CACHE)    
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
    
@njit([(NCDTYPE[:,:],NCDTYPE[:,:],NCDTYPE[:,:])], cache = NUMBA_CACHE)    
def _dotmm2(a,b,out):
    a0 = a[0,0]*b[0,0]+a[0,1]*b[1,0]
    a1 = a[0,0]*b[0,1]+a[0,1]*b[1,1]

    b0 = a[1,0]*b[0,0]+a[1,1]*b[1,0]
    b1 = a[1,0]*b[0,1]+a[1,1]*b[1,1] 
   
    out[0,0] = a0
    out[0,1] = a1

    out[1,0] = b0
    out[1,1] = b1
    
@njit([(NCDTYPE[:,:],NFDTYPE[:],NCDTYPE[:,:])], cache = NUMBA_CACHE)    
def _dotmr2(a,b,out):
    a0 = a[0,0]*b[0]-a[0,2]*b[1]
    a1 = a[0,1]*b[0]+a[0,3]*b[1]
    a2 = a[0,0]*b[1]+a[0,2]*b[0]
    a3 = -a[0,1]*b[1]+a[0,3]*b[0] 

    b0 = a[1,0]*b[0]-a[1,2]*b[1]
    b1 = a[1,1]*b[0]+a[1,3]*b[1]
    b2 = a[1,0]*b[1]+a[1,2]*b[0]
    b3 = -a[1,1]*b[1]+a[1,3]*b[0] 

    c0 = a[2,0]*b[0]-a[2,2]*b[1]
    c1 = a[2,1]*b[0]+a[2,3]*b[1]
    c2 = a[2,0]*b[1]+a[2,2]*b[0]
    c3 = -a[2,1]*b[1]+a[2,3]*b[0] 
    
    d0 = a[3,0]*b[0]-a[3,2]*b[1]
    d1 = a[3,1]*b[0]+a[3,3]*b[1]
    d2 = a[3,0]*b[1]+a[3,2]*b[0]
    d3 = -a[3,1]*b[1]+a[3,3]*b[0]     
 
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
    
    
@njit([(NFDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])], cache = NUMBA_CACHE)    
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
    
@njit([(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:])], cache = NUMBA_CACHE)
def _dotmv4(a, b, out):
    out0 = a[0,0] * b[0] + a[0,1] * b[1] + a[0,2] * b[2] +a[0,3] * b[3]
    out1 = a[1,0] * b[0] + a[1,1] * b[1] + a[1,2] * b[2] +a[1,3] * b[3]
    out2 = a[2,0] * b[0] + a[2,1] * b[1] + a[2,2] * b[2] +a[2,3] * b[3]
    out3 = a[3,0] * b[0] + a[3,1] * b[1] + a[3,2] * b[2] +a[3,3] * b[3]
    out[0]= out0
    out[1]= out1
    out[2]= out2
    out[3]= out3
    
@njit([(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:])], cache = NUMBA_CACHE)
def _dotmv2(a, b, out):
    out0 = a[0,0] * b[0] + a[0,1] * b[1] 
    out1 = a[1,0] * b[0] + a[1,1] * b[1]
    out[0]= out0
    out[1]= out1
    
@njit([(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])], cache = NUMBA_CACHE)
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
    
@njit([(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])], cache = NUMBA_CACHE)
def _dotmd2(a, b, out):
    a0 = a[0,0]*b[0]
    a1 = a[0,1]*b[1]
    
    out[0,0] = a0
    out[0,1] = a1

    a0 = a[1,0]*b[0]
    a1 = a[1,1]*b[1]
  
    out[1,0] = a0
    out[1,1] = a1

    
@njit([(NCDTYPE[:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],parallel = NUMBA_PARALLEL, cache = NUMBA_CACHE)
def _dotm1f4(a, b, out):
    for i in prange(b.shape[1]):
        for j in range(b.shape[2]):
            b0 = b[0,i,j]
            b1 = b[1,i,j]
            b2 = b[2,i,j]
            b3 = b[3,i,j]
            
            out0 = a[0,0] * b0 + a[0,1] * b1 + a[0,2] * b2 +a[0,3] * b3
            out1 = a[1,0] * b0 + a[1,1] * b1 + a[1,2] * b2 +a[1,3] * b3
            out2 = a[2,0] * b0 + a[2,1] * b1 + a[2,2] * b2 +a[2,3] * b3
            out3 = a[3,0] * b0 + a[3,1] * b1 + a[3,2] * b2 +a[3,3] * b3
            
            out[0,i,j]= out0
            out[1,i,j]= out1
            out[2,i,j]= out2
            out[3,i,j]= out3
            
@njit([(NCDTYPE[:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],parallel = NUMBA_PARALLEL, cache = NUMBA_CACHE)
def _dotm1f2(a, b, out):
    for i in prange(b.shape[1]):
        for j in range(b.shape[2]):
            b0 = b[0,i,j]
            b1 = b[1,i,j]
            
            out0 = a[0,0] * b0 + a[0,1] * b1
            out1 = a[1,0] * b0 + a[1,1] * b1 
            
            out[0,i,j]= out0
            out[1,i,j]= out1
            
@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],parallel = NUMBA_PARALLEL, cache = NUMBA_CACHE)
def _dotmf4(a, b, out):
    for i in prange(b.shape[1]):
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
            
@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],parallel = NUMBA_PARALLEL, cache = NUMBA_CACHE)
def _dotmf2(a, b, out):
    for i in prange(b.shape[1]):
        for j in range(b.shape[2]):
            b0 = b[0,i,j]
            b1 = b[1,i,j]
            
            out0 = a[i,j,0,0] * b0 + a[i,j,0,1] * b1 
            out1 = a[i,j,1,0] * b0 + a[i,j,1,1] * b1 
            
            out[0,i,j]= out0
            out[1,i,j]= out1
            
@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],parallel = NUMBA_PARALLEL, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmdmf4(a, d, b, f,out):
    for i in prange(f.shape[1]):
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

@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],parallel = NUMBA_PARALLEL, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmdmf2(a, d, b, f,out):
    for i in prange(f.shape[1]):
        for j in range(f.shape[2]):
            f0 = f[0,i,j]
            f1 = f[1,i,j]
            
            out0 = b[i,j,0,0] * f0 + b[i,j,0,1] * f1 
            out1 = b[i,j,1,0] * f0 + b[i,j,1,1] * f1 
            
            b0 = out0*d[i,j,0]
            b1 = out1*d[i,j,1]
            
            out[0,i,j]= a[i,j,0,0] * b0 + a[i,j,0,1] * b1 
            out[1,i,j]= a[i,j,1,0] * b0 + a[i,j,1,1] * b1  
                        

@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],parallel = NUMBA_PARALLEL, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmtmf4(a, d, b, f,out):
    for i in prange(f.shape[1]):
        for j in range(f.shape[2]):
            f0 = f[0,i,j]
            f1 = f[1,i,j]
            f2 = f[2,i,j]
            f3 = f[3,i,j]
            
            out0 = b[i,j,0,0] * f0 + b[i,j,0,1] * f1 + b[i,j,0,2] * f2 +b[i,j,0,3] * f3
            out2 = b[i,j,2,0] * f0 + b[i,j,2,1] * f1 + b[i,j,2,2] * f2 +b[i,j,2,3] * f3
            
            b0 = out0*d[i,j,0]
            b2 = out2*d[i,j,2]
            
            out[0,i,j]= a[i,j,0,0] * b0  + a[i,j,0,2] * b2 
            out[1,i,j]= a[i,j,1,0] * b0  + a[i,j,1,2] * b2 
            out[2,i,j]= a[i,j,2,0] * b0  + a[i,j,2,2] * b2 
            out[3,i,j]= a[i,j,3,0] * b0  + a[i,j,3,2] * b2    
            

#@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:,:],NFDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],parallel = True)
#def _dotmdmrf(a, d, b, r,f,out):
#    for i in prange(f.shape[1]):
#        for j in range(f.shape[2]):
#            f0 = f[0,i,j]*r[i,j,0,0] + f[2,i,j]*r[i,j,0,1]
#            f1 = f[3,i,j]*r[i,j,1,0] + f[1,i,j]*r[i,j,1,1]
#            f2 = f[0,i,j]*r[i,j,1,0] + f[2,i,j]*r[i,j,1,1]
#            f3 = f[3,i,j]*r[i,j,0,0] + f[1,i,j]*r[i,j,0,1]
#            
#            out0 = b[i,j,0,0] * f0 + b[i,j,0,1] * f1 + b[i,j,0,2] * f2 +b[i,j,0,3] * f3
#            out1 = b[i,j,1,0] * f0 + b[i,j,1,1] * f1 + b[i,j,1,2] * f2 +b[i,j,1,3] * f3
#            out2 = b[i,j,2,0] * f0 + b[i,j,2,1] * f1 + b[i,j,2,2] * f2 +b[i,j,2,3] * f3
#            out3 = b[i,j,3,0] * f0 + b[i,j,3,1] * f1 + b[i,j,3,2] * f2 +b[i,j,3,3] * f3
#            
#            b0 = out0*d[i,j,0]
#            b1 = 0#out1*d[i,j,1]
#            b2 = out2*d[i,j,2]
#            b3 = 0#out3*d[i,j,3]
#                            
#            
#            out0 = a[i,j,0,0] * b0 + a[i,j,0,1] * b1 + a[i,j,0,2] * b2 +a[i,j,0,3] * b3
#            out1 = a[i,j,1,0] * b0 + a[i,j,1,1] * b1 + a[i,j,1,2] * b2 +a[i,j,1,3] * b3
#            out2 = a[i,j,2,0] * b0 + a[i,j,2,1] * b1 + a[i,j,2,2] * b2 +a[i,j,2,3] * b3
#            out3 = a[i,j,3,0] * b0 + a[i,j,3,1] * b1 + a[i,j,3,2] * b2 +a[i,j,3,3] * b3   
#            
#            out[0,i,j]= out0*r[i,j,0,0] + out2*r[i,j,1,0]
#            out[1,i,j]= out3*r[i,j,0,1] + out1*r[i,j,1,1]
#            out[2,i,j]= out0*r[i,j,0,1] + out2*r[i,j,1,1]
#            out[3,i,j]= out3*r[i,j,0,0] + out1*r[i,j,1,0]

            
@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NFDTYPE, NCDTYPE[:,:,:])],parallel = NUMBA_PARALLEL, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _transmit(a, d, b, f,kd,out):
    for i in prange(f.shape[1]):
        for j in range(f.shape[2]):
            f0 = f[0,i,j]
            f1 = f[1,i,j]
            f2 = f[2,i,j]
            f3 = f[3,i,j]
            
            out0 = b[i,j,0,0] * f0 + b[i,j,0,1] * f1 + b[i,j,0,2] * f2 +b[i,j,0,3] * f3
            out1 = b[i,j,1,0] * f0 + b[i,j,1,1] * f1 + b[i,j,1,2] * f2 +b[i,j,1,3] * f3
            out2 = b[i,j,2,0] * f0 + b[i,j,2,1] * f1 + b[i,j,2,2] * f2 +b[i,j,2,3] * f3
            out3 = b[i,j,3,0] * f0 + b[i,j,3,1] * f1 + b[i,j,3,2] * f2 +b[i,j,3,3] * f3
            
            b0 = out0*np.exp(1j*kd*d[i,j,0].real)
            b1 = out1*np.exp(1j*kd*d[i,j,1].real)
            b2 = out2*np.exp(1j*kd*d[i,j,2].real)
            b3 = out3*np.exp(1j*kd*d[i,j,3].real)
                                        
            out[0,i,j]= a[i,j,0,0] * b0 + a[i,j,0,1] * b1 + a[i,j,0,2] * b2 +a[i,j,0,3] * b3
            out[1,i,j]= a[i,j,1,0] * b0 + a[i,j,1,1] * b1 + a[i,j,1,2] * b2 +a[i,j,1,3] * b3
            out[2,i,j]= a[i,j,2,0] * b0 + a[i,j,2,1] * b1 + a[i,j,2,2] * b2 +a[i,j,2,3] * b3
            out[3,i,j]= a[i,j,3,0] * b0 + a[i,j,3,1] * b1 + a[i,j,3,2] * b2 +a[i,j,3,3] * b3  

@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NFDTYPE, NCDTYPE[:,:,:])],parallel = NUMBA_PARALLEL, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _ftransmit(a, d, b, f,kd,out):
    for i in prange(f.shape[1]):
        for j in range(f.shape[2]):
            f0 = f[0,i,j]
            f1 = f[1,i,j]
            f2 = f[2,i,j]
            f3 = f[3,i,j]
            
            out0 = b[i,j,0,0] * f0 + b[i,j,0,1] * f1 + b[i,j,0,2] * f2 +b[i,j,0,3] * f3
            out2 = b[i,j,2,0] * f0 + b[i,j,2,1] * f1 + b[i,j,2,2] * f2 +b[i,j,2,3] * f3
            
            b0 = out0*np.exp(1j*kd*d[i,j,0].real)
            b2 = out2*np.exp(1j*kd*d[i,j,2].real)
                                        
            out[0,i,j]= a[i,j,0,0] * b0 + a[i,j,0,2] * b2 
            out[1,i,j]= a[i,j,1,0] * b0 + a[i,j,1,2] * b2 
            out[2,i,j]= a[i,j,2,0] * b0 + a[i,j,2,2] * b2 
            out[3,i,j]= a[i,j,3,0] * b0 + a[i,j,3,2] * b2  



#@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:,:],NFDTYPE[:,:],NCDTYPE[:,:,:],NFDTYPE, NCDTYPE[:,:,:])],parallel = NUMBA_PARALLEL)
#def _transmitr(a, d, b, r,f,kd,out):
#    for i in prange(f.shape[1]):
#        for j in range(f.shape[2]):
#            f0 = f[0,i,j]*r[0,0] + f[2,i,j]*r[0,1]
#            f1 = f[3,i,j]*r[1,0] + f[1,i,j]*r[1,1]
#            f2 = f[0,i,j]*r[1,0] + f[2,i,j]*r[1,1]
#            f3 = f[3,i,j]*r[0,0] + f[1,i,j]*r[0,1]
#            
#            out0 = b[i,j,0,0] * f0 + b[i,j,0,1] * f1 + b[i,j,0,2] * f2 +b[i,j,0,3] * f3
#            out1 = b[i,j,1,0] * f0 + b[i,j,1,1] * f1 + b[i,j,1,2] * f2 +b[i,j,1,3] * f3
#            out2 = b[i,j,2,0] * f0 + b[i,j,2,1] * f1 + b[i,j,2,2] * f2 +b[i,j,2,3] * f3
#            out3 = b[i,j,3,0] * f0 + b[i,j,3,1] * f1 + b[i,j,3,2] * f2 +b[i,j,3,3] * f3
#            
#            b0 = out0*np.exp(1j*kd*d[i,j,0].real)
#            b1 = 0#out1*np.exp(1j*kd*d[i,j,1].real)
#            b2 = out2*np.exp(1j*kd*d[i,j,2].real)
#            b3 = 0#out3*np.exp(1j*kd*d[i,j,3].real)
#            
#            out0 = a[i,j,0,0] * b0 + a[i,j,0,1] * b1 + a[i,j,0,2] * b2 +a[i,j,0,3] * b3
#            out1 = a[i,j,1,0] * b0 + a[i,j,1,1] * b1 + a[i,j,1,2] * b2 +a[i,j,1,3] * b3
#            out2 = a[i,j,2,0] * b0 + a[i,j,2,1] * b1 + a[i,j,2,2] * b2 +a[i,j,2,3] * b3
#            out3 = a[i,j,3,0] * b0 + a[i,j,3,1] * b1 + a[i,j,3,2] * b2 +a[i,j,3,3] * b3   
#            
#            out[0,i,j]= out0*r[0,0] + out2*r[1,0]
#            out[1,i,j]= out3*r[0,1] + out1*r[1,1]
#            out[2,i,j]= out0*r[0,1] + out2*r[1,1]
#            out[3,i,j]= out3*r[0,0] + out1*r[1,0]

#
#@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],parallel = True)
#def _dotFf(a, f,out):
#    for i in prange(f.shape[1]):
#        for j in range(f.shape[2]):
#            
#            f0 = f[0,i,j]
#            f1 = f[1,i,j]
#            f2 = f[2,i,j]
#            f3 = f[3,i,j]
#            
#            out0 = a[i,j,1,0] * f0 + a[i,j,1,1] * f1 + a[i,j,1,2] * f2 +a[i,j,1,3] * f3
#            out1 = a[i,j,2,0] * f0 + a[i,j,2,1] * f1 + a[i,j,2,2] * f2 +a[i,j,2,3] * f3
#            out2 = a[i,j,3,0] * f0 + a[i,j,3,1] * f1 + a[i,j,3,2] * f2 +a[i,j,3,3] * f3
#            out3 = a[i,j,4,0] * f0 + a[i,j,4,1] * f1 + a[i,j,4,2] * f2 +a[i,j,4,3] * f3
#            
#            b0 = out0*np.exp(a[i,j,0,0])
#            b1 = out1*np.exp(a[i,j,0,1])
#            b2 = out2*np.exp(a[i,j,0,2])
#            b3 = out3*np.exp(a[i,j,0,3])
#                            
#            
#            out[0,i,j]= a[i,j,5,0] * b0 + a[i,j,5,1] * b1 + a[i,j,5,2] * b2 +a[i,j,5,3] * b3
#            out[1,i,j]= a[i,j,6,0] * b0 + a[i,j,6,1] * b1 + a[i,j,6,2] * b2 +a[i,j,6,3] * b3
#            out[2,i,j]= a[i,j,7,0] * b0 + a[i,j,7,1] * b1 + a[i,j,7,2] * b2 +a[i,j,7,3] * b3
#            out[3,i,j]= a[i,j,8,0] * b0 + a[i,j,8,1] * b1 + a[i,j,8,2] * b2 +a[i,j,8,3] * b3  
           
#@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],parallel = True)
#def _dotmf0(a, b, out):
#    for i in prange(b.shape[0]):
#        for j in range(b.shape[1]):
#            b0 = b[i,j,0]
#            b1 = b[i,j,1]
#            b2 = b[i,j,2]
#            b3 = b[i,j,3]
#            
#            out0 = a[i,j,0,0] * b0 + a[i,j,0,1] * b1 + a[i,j,0,2] * b2 +a[i,j,0,3] * b3
#            out1 = a[i,j,1,0] * b0 + a[i,j,1,1] * b1 + a[i,j,1,2] * b2 +a[i,j,1,3] * b3
#            out2 = a[i,j,2,0] * b0 + a[i,j,2,1] * b1 + a[i,j,2,2] * b2 +a[i,j,2,3] * b3
#            out3 = a[i,j,3,0] * b0 + a[i,j,3,1] * b1 + a[i,j,3,2] * b2 +a[i,j,3,3] * b3
#            
#            out[0,i,j]= out0
#            out[1,i,j]= out1
#            out[2,i,j]= out2
#            out[3,i,j]= out3  
            
#@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],parallel = True)
#def _dotmf2(a, b, out):
#    for i in prange(b.shape[1]):
#        for j in range(b.shape[2]):
#            b0 = b[0,i,j]
#            b1 = b[1,i,j]
#            b2 = b[2,i,j]
#            b3 = b[3,i,j]
#            
#            out0 = a[0,0,i,j] * b0 + a[0,1,i,j] * b1 + a[0,2,i,j] * b2 +a[0,3,i,j] * b3
#            out1 = a[1,0,i,j] * b0 + a[1,1,i,j] * b1 + a[1,2,i,j] * b2 +a[1,3,i,j] * b3
#            out2 = a[2,0,i,j] * b0 + a[2,1,i,j] * b1 + a[2,2,i,j] * b2 +a[2,3,i,j] * b3
#            out3 = a[3,0,i,j] * b0 + a[3,1,i,j] * b1 + a[3,2,i,j] * b2 +a[3,3,i,j] * b3
#            
#            out[0,i,j]= out0
#            out[1,i,j]= out1
#            out[2,i,j]= out2
#            out[3,i,j]= out3   

@guvectorize([(NCDTYPE[:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],"(n,n),(n,m,k)->(n,m,k)",target = "cpu", cache = NUMBA_CACHE)
def dotm1f(a, b, out):
    """Computes a dot product of a 4x4 matrix with a 4x4 matrix"""
    if a.shape[0] == 2:
        _dotm1f2(a, b, out)
    else:
        assert a.shape[0] >= 4 #make sure it is not smaller than 4
        _dotm1f4(a, b, out)

@guvectorize([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],"(m,k,n,n),(n,m,k)->(n,m,k)",target = "cpu", cache = NUMBA_CACHE)
def dotmf(a, b, out):
    """Computes a dot product of a 4x4 matrix with a 4x4 matrix"""
    if b.shape[0] == 2:
        _dotmf2(a, b, out)
    else:
        assert b.shape[0] >= 4 #make sure it is not smaller than 4
        _dotmf4(a, b, out)
 
        
@guvectorize([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],"(m,k,n,n),(m,k,n),(m,k,n,n),(n,m,k)->(n,m,k)",target = "cpu", cache = NUMBA_CACHE)
def dotmdmf(a, d,b,f, out):
    """Computes a dot product of a 4x4 matrix with a 4x4 matrix"""
    if f.shape[0] == 2:
        _dotmdmf2(a, d,b,f, out)
    else:    
        assert f.shape[0] >= 4 #make sure it is not smaller than 4
        _dotmdmf4(a,d, b, f,out)
        
@guvectorize([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],"(m,k,n,n),(m,k,n),(m,k,n,n),(n,m,k)->(n,m,k)",target = "cpu", cache = NUMBA_CACHE)
def dotmtmf(a, t,b,f, out):
    """Computes a dot product of a 4x4 matrix with a 4x4 matrix"""
    assert f.shape[0] >= 4 #make sure it is not smaller than 4
    _dotmtmf4(a,t, b, f,out)

@guvectorize([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NFDTYPE[:], NCDTYPE[:,:,:])],"(m,k,n,n),(m,k,n),(m,k,n,n),(n,m,k),()->(n,m,k)",target = "cpu", cache = NUMBA_CACHE)
def transmit( a, d, b, f,kd, out):
    _transmit( a, d, b, f, kd[0], out)
    
@guvectorize([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NFDTYPE[:], NCDTYPE[:,:,:])],"(m,k,n,n),(m,k,n),(m,k,n,n),(n,m,k),()->(n,m,k)",target = "cpu", cache = NUMBA_CACHE)
def ftransmit( a, d, b, f,kd, out):
    _ftransmit( a, d, b, f, kd[0], out)
        
@guvectorize([(NCDTYPE[:,:],NCDTYPE[:,:],NCDTYPE[:,:])],"(n,n),(n,n)->(n,n)",target = NUMBA_TARGET, cache = NUMBA_CACHE)
def dotmm(a, b, out):
    """Computes a dot product of a 4x4 matrix with a 4x4 matrix"""
    if a.shape[0] == 2:
        _dotmm2(a, b, out)
    else:
        assert a.shape[0] >= 4 #make sure it is not smaller than 4
        _dotmm4(a, b, out)
        
@guvectorize([(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:])],"(n,n),(n)->(n,n)",target = NUMBA_TARGET, cache = NUMBA_CACHE)
def dotmd(a, b, out):
     """Computes a dot product of a 4x4 matrix with a diagonal matrix"""
     if a.shape[0]== 2:
         _dotmd2(a, b, out)
     else:
         assert a.shape[0] >= 4 #make sure it is not smaller than 4
         _dotmd4(a, b, out)

@guvectorize([(NCDTYPE[:,:],NFDTYPE[:],NCDTYPE[:,:])],"(n,n),(m)->(n,n)",target = NUMBA_TARGET, cache = NUMBA_CACHE)
def dotmr2(a, b, out):
     assert a.shape[0] >= 4 #make sure it is not smaller than 4
     assert b.shape[0] >= 2 #make sure it is not smaller than 2
     _dotmr2(a, b, out)

@guvectorize([(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:])],"(n,n),(n)->(n)",target = NUMBA_TARGET, cache = NUMBA_CACHE)
def dotmv(a, b, out):
    """Computes a dot product of a 4x4 or 2x2 matrix with a vector"""
    if a.shape[0] == 2:
        _dotmv2(a, b, out)
    else:
        assert a.shape[0] >= 4 #make sure it is not smaller than 4
        _dotmv4(a, b, out)
    
@guvectorize([(NCDTYPE[:,:],NCDTYPE[:],NCDTYPE[:,:],NCDTYPE[:,:])],"(n,n),(n),(n,n)->(n,n)",target = NUMBA_TARGET, cache = NUMBA_CACHE)
def dotmdm(a, d, b, out):
    """Computes a dot product of a 4x4 (or 2x2) matrix with a diagonal matrix (4- or 2-vector) 
    and another 4x4 (or 2x2) matrix"""
    #assert a.shape[0] >= 4 #make sure it is not smaller than 4 
    if a.shape[0] == 2:
        _dotmd2(a, d, out)
        _dotmm2(out,b,out)
    else:
        assert a.shape[0] >= 4 #make sure it is not smaller than 4
        _dotmd4(a, d, out)
        _dotmm4(out,b,out)        
    


__all__ = ["inv4x4", "dotmm","dotmf","dotmv","dotmdm", "ftransmit", "transmit"]