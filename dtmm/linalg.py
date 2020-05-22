"""
Numba optimized linear algebra functions for 4x4 matrices and 2x2 martrices.
"""

from __future__ import absolute_import, print_function, division
from dtmm.conf import NCDTYPE, NFDTYPE, NUMBA_TARGET,NUMBA_PARALLEL, NUMBA_CACHE, NUMBA_FASTMATH, CDTYPE
from numba import njit, prange, guvectorize
import numpy as np

if NUMBA_PARALLEL == False:
    prange = range


@njit([(NCDTYPE[:,:],NCDTYPE[:,:])], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _inv2x2(src,dst):
    a = src[0,0]
    b = src[0,1]
    c = src[1,0]
    d = src[1,1]
    det = a*d-b*c
    if det == 0.:
        det = 0.#np.nan
    else:
        det = 1./det
    dst[0,0] = d*det
    dst[0,1] = -b*det
    dst[1,0] = -c*det
    dst[1,1] = a*det

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
def inv(mat, output):
    """inv(mat)
    
Calculates inverse of a 4x4 complex matrix or 2x2 complex matrix

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
    elif mat.shape[0] == 4:
        _inv4x4(mat,output)
    else:
        inv = np.linalg.inv(mat)
        output[...] = inv


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
@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],parallel = NUMBA_PARALLEL, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
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
            
@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],parallel = NUMBA_PARALLEL, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _dotmf2(a, b, out):
    for i in prange(b.shape[1]):
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
                        

#@guvectorize([(NCDTYPE[:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],"(n,n),(n,m,k)->(n,m,k)",target = "cpu", cache = NUMBA_CACHE)
#def dotm1f(a, b, out):
#    if b.shape[0] == 2:
#        _dotm1f2(a, b, out)
#    else:
#        assert a.shape[0] >= 4 #make sure it is not smaller than 4
#        _dotm1f4(a, b, out)

@guvectorize([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],"(m,k,n,n),(n,m,k)->(n,m,k)",target = "cpu", cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
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
    a = np.asarray(a)
    b = np.asarray(b)
    a = broadcast_m(a, b)
    return _dotmf(a, b, out)

       
@guvectorize([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:],NCDTYPE[:,:,:])],"(m,k,n,n),(m,k,n),(m,k,n,n),(n,m,k)->(n,m,k)",target = "cpu", cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
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
#
#@guvectorize([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:,:])],"(n,n,k,k),(n,n,k,k)->(n,n,k,k)", cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)        
#def bdotmm3(m1,m2, out):
#    """Performs a dot product of two nxn block matrices of blocks 4x4.
#    Matrices must be of shape nxnx4x4 that describe two mxm matrices
#    (m = 4*n) of blocks of size 4x4
#    """
#    assert m1.shape[3] == 4
#    
#    tmp = np.empty(out.shape[2:],out.dtype)
#
#    n = m1.shape[0]    
#    for i in prange(n):
#        for j in range(n):
#            for k in range(n):
#                if k == 0:
#                    _dotmm4(m1[i,k], m2[k,j], out[i,j] )
#                else:
#                    _dotmm4(m1[i,k], m2[k,j], tmp)
#                    out[i,j] += tmp
#
#@njit([(NCDTYPE[:,:,:,:],NCDTYPE[:,:,:,:],NCDTYPE[:,:,:,:])], parallel = NUMBA_PARALLEL)        
#def bdotmm2(m1,m2, out):
#    """Performs a dot product of two nxn block matrices of blocks 4x4.
#    Matrices must be of shape nxnx4x4 that describe two mxm matrices
#    (m = 4*n) of blocks of size 4x4
#    """
#    assert m1.shape[3] == 4
#    
#    tmp = np.empty(out.shape[2:],out.dtype)
#
#    n = m1.shape[0]    
#    for i in prange(n):
#        for j in range(n):
#            for k in range(n):
#                if k == 0:
#                    _dotmm4(m1[i,k], m2[k,j], out[i,j] )
#                else:
#                    _dotmm4(m1[i,k], m2[k,j], tmp)
#                    out[i,j] += tmp


def bdotmm(m1,m2, out = None):
    """Performs a dot product of two nxn block matrices of blocks of size 4x4.
    Matrices must be of shape nxnx4x4 that describe two mxm matrices
    (m = 4*n) of blocks of size 4x4. 
    """
    assert m1.shape == m2.shape
    assert m2 is not out
    if out is None:
        out = np.empty(shape = m1.shape, dtype = CDTYPE)
        
    tmp = np.empty(shape = m1.shape, dtype = CDTYPE)

    for j in range(m1.shape[0]):
        dotmm(m1[j][:,None,:,:],m2, out = tmp)
        out[j] = tmp.sum(-4, out = out[j])
    return out

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
    """Performs a dot product of two nxn block matrices of blocks 4x4.
    The second matrix is block diagonal matrix of shape nx4x4,
    The first matrix must be of shape nxnx4x4 that describe two mxm matrices
    (m = 4*n) of blocks of size 4x4.
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
    """Performs a dot product of two nxn block matrices of blocks 4x4.
    The second matrix is block diagonal matrix of shape nx4x4,
    The first matrix must be of shape nxnx4x4 that describe two mxm matrices
    (m = 4*n) of blocks of size 4x4.
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


def bdotmv(m,v, out = None):
    """Performs a dot product of two nxn block matrices of blocks 4x4.
    The second matrix is block diagonal matrix of shape nx4x4,
    The first matrix must be of shape nxnx4x4 that describe two mxm matrices
    (m = 4*n) of blocks of size 4x4.
    """
    tmp = dotmv(m,v)
    return tmp.sum(-2, out = out)


def _bdotmv_ref(m,v):
    """Performs a dot product of two nxn block matrices of blocks 4x4.
    Matrices must be of shape nxnx4x4 that describe two mxm matrices
    (m = 4*n) of blocks of size 4x4
    """
    
    m = np.moveaxis(m,-2,-3).copy()
    shape = v.shape
    m = m.reshape(m.shape[0]*4,m.shape[0]*4).copy()
    v = v.reshape(v.shape[0]*4).copy()
    out = dotmv(m,v).copy()
    out = out.reshape(shape).copy()
    return out



        
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
        arrays = np.rollaxis(arrays, axis)
        #arrays = np.moveaxis(arrays, axis)
    else:
        indices = range(len(arrays))
    if reverse == True:
        indices = reversed(indices)
    for i in indices:
        if out is None:
            out = np.asarray(arrays[i]).copy()
        else:
            out = dotmm(out, arrays[i], out = out)
    return out
    
    
__all__ = ["inv", "dotmm","dotmf","dotmv","dotmdm","dotmd","multi_dot"]