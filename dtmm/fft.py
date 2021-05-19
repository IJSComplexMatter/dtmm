"""
Custom 2D FFT functions.

numpy, scipy and mkl_fft do not have fft implemented such that output argument 
can be provided. This implementation adds the output argument for fft2 and 
ifft2 functions. 

Also, for mkl_fft and scipy, the computation can be performed in parallel using ThreadPool.

"""
from __future__ import absolute_import, print_function, division

from dtmm.conf import DTMMConfig, CDTYPE, MKL_FFT_INSTALLED, SCIPY_INSTALLED, PYFFTW_INSTALLED
import numpy as np

import numpy.fft as npfft

from multiprocessing.pool import ThreadPool

from functools import reduce

if MKL_FFT_INSTALLED:
    import mkl_fft
    
if SCIPY_INSTALLED:
    import scipy.fft as spfft
    
if PYFFTW_INSTALLED:
    import pyfftw


POOL = {}

from threading import Thread
from queue import Queue

class PoolWorker():
    """Mimics the object returned by ThreadPool.apply_async method."""
    def __init__(self, queue):
        self.queue = queue
        
    def get(self):
        return self.queue.get()

class Pool:
    """A multiprocessing.pool.ThreadPool -like object.
    
    Implements only necessary part of ThreadPool API.
    """
    def __init__(self,nthreads):
        def worker(i, inputs, results):
            #print("Thread ", i, "started")
            while True:
                data = inputs.get()
                if data is None:
                    results.put(None)
                    break
                else:
                    func, args, kwargs = data
                    out = func(*args,**kwargs)
                    results.put((i,out))
            #print("Thread ", i, "stopped")
            
            
        self.nthreads = nthreads
        self.results = [Queue() for i in range(nthreads)]
        self.inputs = [Queue() for i in range(nthreads)]
        self.threads = [Thread(target = worker, args = (i,self.inputs[i],self.results[i]), daemon = True) for i in range(nthreads)]
        [t.start() for t in self.threads]
        self.nruns = 0
        
    def apply_async(self,func, args = (), kwds = {}):
        index = self.nruns % self.nthreads
        self.inputs[index].put((func,args,kwds))
        self.nruns += 1
        return PoolWorker(self.results[index])
              
    def close(self):
        for q in self.inputs:
            q.put(None)
        for q in self.results:
            while q.get(timeout = 1) != None:
                pass
        for t in self.threads:
            t.join()
                 
    def __del__(self):
        self.close()

def clear_pool():
    """Clears thread pool. Deletes all Pool objects, which terminates all
    running background threads."""
    
    POOL.clear()

def _set_out_mkl(a,out):
    if out is not a:
        if out is None:
            out = a.copy()
        else:
            out[...] = a 
    return out

def _set_out(a,out):
    if out is not a:
        if out is None:
            out = np.empty_like(a)
    return out

def _copy(x,y):
    y[...] = x
    return y

_copy_if_needed = lambda x,y: _copy(x,y) if x is not y else x
    
def _sequential_inplace_fft(fft,array,**kwargs):
    [fft(d,overwrite_x = True,**kwargs) for d in array] 

def _sequential_fft(fft,array, out, overwrite_x = False, **kwargs):
    if out is array:
        [_copy_if_needed(fft(d, overwrite_x = True, **kwargs),out[i]) for i,d in enumerate(array)] 
    else:
        [_copy_if_needed(fft(d, overwrite_x = overwrite_x, **kwargs),out[i]) for i,d in enumerate(array)] 

def _optimal_workers(size, nthreads):
    if size%nthreads == 0:
        return nthreads
    else:
        return _optimal_workers(size, nthreads-1)
    
def _reshape(a, optimize = True, dim = 2):
    shape = a.shape
    newshape = reduce((lambda x,y: x*y), shape[:-dim] or [1])
    if DTMMConfig.thread_pool == True and DTMMConfig.fft_threads > 1 and optimize == True:
        n = _optimal_workers(newshape,DTMMConfig.fft_threads)
        newshape = (n,newshape//n,) + shape[-dim:]
    else:
        newshape = (newshape,) + shape[-dim:]
    a = a.reshape(newshape)
    return shape, a    

def _reshape_fftw(a, dim = 3):
    shape = a.shape
    newshape = reduce((lambda x,y: x*y), shape[:-dim] or [1])

    n = _optimal_workers(newshape,DTMMConfig.fft_threads)
    newshape = (newshape//n,n) + shape[-dim:]

    a = a.reshape(newshape)
    return shape, a    

def __mkl_fft(fft,a,out):
    out = _set_out_mkl(a,out)
    shape, out = _reshape(out)
    if DTMMConfig.thread_pool == True and DTMMConfig.fft_threads > 1:
        try:
            pool = POOL[DTMMConfig.fft_threads]
        except KeyError:
            pool = Pool(DTMMConfig.fft_threads)
            POOL[DTMMConfig.fft_threads] = pool
        #pool = ThreadPool(DTMMConfig.fft_threads)
        workers = [pool.apply_async(_sequential_inplace_fft, args = (fft,d)) for d in out] 
        results = [w.get() for w in workers]
        #pool.close()
    else:
        _sequential_inplace_fft(fft,out)
    return out.reshape(shape)    

def _mkl_fft2(a,out = None):
    return __mkl_fft(mkl_fft.fft2,a,out)

def _mkl_ifft2(a,out = None):
    return __mkl_fft(mkl_fft.ifft2,a,out)

def __sp_fft(fft,a,out, overwrite_x = False):
    out = _set_out(a,out)
    shape, a = _reshape(a)
    shape, out = _reshape(out)
    if DTMMConfig.thread_pool == True and DTMMConfig.fft_threads > 1:
        try:
            pool = POOL[DTMMConfig.fft_threads]
        except KeyError:
            pool = Pool(DTMMConfig.fft_threads)
            POOL[DTMMConfig.fft_threads] = pool
        #pool = ThreadPool(DTMMConfig.fft_threads)
        workers = [pool.apply_async(_sequential_fft, args = (fft,d,out[i]), kwds = {"overwrite_x" : overwrite_x, "workers" : 1}) for i,d in enumerate(a)] 
        results = [w.get() for w in workers]
        #pool.close()
    else:
        _sequential_fft(fft,a,out,overwrite_x, workers = DTMMConfig.fft_threads)
    return out.reshape(shape)   
       
def _sp_fft2(a, out = None):
    return __sp_fft(spfft.fft2, a, out)
    
def _sp_ifft2(a, out = None):
    return __sp_fft(spfft.ifft2, a, out)   

FFTW_CACHE = {}

FFTW_PLANNERS = {0 :"FFTW_ESTIMATE",
                 1: "FFTW_MEASURE", 
                 2: "FFTW_PATIENT",
                 3: "FFTW_EXHAUSTIVE"}

def clear_cache():
    """Clears fft cache data"""
    FFTW_CACHE.clear()
    
def clear_planner():
    """Clears fft planners (pyfftw)"""
    if PYFFTW_INSTALLED:
        pyfftw.forget_wisdom()
        
def _fftw_fft2(a, out = None):
    planner = FFTW_PLANNERS.get(DTMMConfig.fft_planner, "FFTW_MEASURE")
    if out is None:
        out = np.empty_like(a)

    if out is not a:
        key = ("fft2o",a.dtype,DTMMConfig.fft_threads) + a.strides + out.strides
    else:
        key = ("fft2i",a.dtype,DTMMConfig.fft_threads) + a.strides + out.strides
    try:
        fft = FFTW_CACHE[key]
    except KeyError:
        a0 = a.copy()
        test_array = a
        if out is a:
            fft = pyfftw.FFTW(test_array,test_array, axes = (-2,-1), threads = DTMMConfig.fft_threads, flags = ["FFTW_DESTROY_INPUT", planner])
        else:
            fft = pyfftw.FFTW(test_array,out, axes = (-2,-1), threads = DTMMConfig.fft_threads, flags = [planner])
        a[...] = a0   
        FFTW_CACHE[key] = fft
    fft(a,out)    
    return out

def _fftw_ifft2(a, out = None):
    planner = FFTW_PLANNERS.get(DTMMConfig.fft_planner, "FFTW_MEASURE")
    if out is None:
        out = np.empty_like(a)
    if out is not a:
        key = ("ifft2o",a.dtype,DTMMConfig.fft_threads) + a.strides + out.strides
    else:
        key = ("ifft2i",a.dtype,DTMMConfig.fft_threads) + a.strides + out.strides
    try:
        fft = FFTW_CACHE[key]
    except KeyError:
        a0 = a.copy()
        test_array = a
        if out is a:
            fft = pyfftw.FFTW(test_array,test_array, axes = (-2,-1), threads = DTMMConfig.fft_threads, direction = "FFTW_BACKWARD", flags = ["FFTW_DESTROY_INPUT", planner])
        else:
            fft = pyfftw.FFTW(test_array,out, axes = (-2,-1), threads = DTMMConfig.fft_threads, direction = "FFTW_BACKWARD", flags = [planner])
        a[...] = a0
        FFTW_CACHE[key] = fft
    fft(a,out,normalise_idft=True)    
    return out

        
def __np_fft(fft,a,out):
    if out is None:
        return np.asarray(fft(a),a.dtype)
    else:
        out[...] = fft(a)
        return out
        
def _np_fft2(a, out = None):
    return __np_fft(npfft.fft2, a, out)
    
def _np_ifft2(a, out = None):
    return __np_fft(npfft.ifft2, a, out)       

def fft2(a, out = None):
    """Computes fft2 of the input complex array.
    
    Parameters
    ----------
    a : array_like
        Input array (must be complex).
    out : array or None, optional
        Output array. Can be same as input for fast inplace transform.
       
    Returns
    -------
    out : complex ndarray
        Result of the transformation along the last two axes.
    """
    a = np.asarray(a, dtype = CDTYPE)
    libname = DTMMConfig["fftlib"]
    if libname == "mkl_fft":
        return _mkl_fft2(a, out)
    elif libname == "scipy":
        return _sp_fft2(a, out)
    elif libname == "numpy":
        return _np_fft2(a, out) 
    elif libname == "pyfftw":
        return _fftw_fft2(a, out)   
    else:#default implementation is numpy
        return _np_fft2(a, out) 

    
def ifft2(a, out = None): 
    """Computes ifft2 of the input complex array.
    
    Parameters
    ----------
    a : array_like
        Input array (must be complex).
    out : array or None, optional
        Output array. Can be same as input for fast inplace transform.
       
    Returns
    -------
    out : complex ndarray
        Result of the transformation along the last two axes.
    """
    a = np.asarray(a, dtype = CDTYPE)      
    libname = DTMMConfig["fftlib"]
    if libname == "mkl_fft":
        return _mkl_ifft2(a, out)
    
    elif libname == "scipy":
        return _sp_ifft2(a, out)
    elif libname == "pyfftw":
        return _fftw_ifft2(a, out) 
    elif libname == "numpy":
        return _np_ifft2(a, out)  
    elif libname == "pyfftw":
        return _fftw_ifft2(a, out) 
    else: #default implementation is numpy
        return _np_ifft2(a, out)   

  
def mfft2(a, overwrite_x = False):
    """Computes matrix fft2 on a matrix of shape (..., n,n,4,4).
    
    This is identical to np.fft2(a, axes = (-4,-3))
    
    Parameters
    ----------
    a : array_like
        Input array (must be complex).
    overwrite_x : bool
        Specifies whether original array can be destroyed (for inplace transform)
       
    Returns
    -------
    out : complex ndarray
        Result of the transformation along the (-4,-3) axes.    
    """
    a = np.asarray(a, dtype = CDTYPE)      
    libname = DTMMConfig["fftlib"]    
    if libname == "mkl_fft":
        return mkl_fft.fft2(a, axes = (-4,-3), overwrite_x = overwrite_x)
    elif libname == "scipy":
        return spfft.fft2(a, axes = (-4,-3), overwrite_x = overwrite_x)
    elif libname == "numpy":
        return npfft.fft2(a, axes = (-4,-3))
    elif libname == "pyfftw":
        return pyfftw.interfaces.scipy_fft.fft2(a, axes = (-4,-3), overwrite_x = overwrite_x)
    else: #default implementation is numpy
        return npfft.fft2(a, axes = (-4,-3))

def mifft2(a, overwrite_x = False):
    """Computes matrix ifft2 on a matrix of shape (..., n,n,4,4).
    
    This is identical to np.ifft2(a, axes = (-4,-3))
    
    Parameters
    ----------
    a : array_like
        Input array (must be complex).
    overwrite_x : bool
        Specifies whether original array can be destroyed (for inplace transform)
       
    Returns
    -------
    out : complex ndarray
        Result of the transformation along the (-4,-3) axes.    
    """
    a = np.asarray(a, dtype = CDTYPE)      
    libname = DTMMConfig["fftlib"]    
    if libname == "mkl_fft":
        return mkl_fft.ifft2(a, axes = (-4,-3), overwrite_x = overwrite_x)
    elif libname == "scipy":
        return spfft.ifft2(a, axes = (-4,-3), overwrite_x = overwrite_x)
    elif libname == "numpy":
        return npfft.ifft2(a, axes = (-4,-3))
    elif libname == "pyfftw":
        return pyfftw.interfaces.scipy_fft.ifft2(a, axes = (-4,-3), overwrite_x = overwrite_x)
    else: #default implementation is numpy
        return npfft.ifft2(a, axes = (-4,-3))    
    
def mfft(a, overwrite_x = False):
    """Computes matrix fft on a matrix of shape (..., n,4,4).
    
    This is identical to np.fft2(a, axis = -3)
    
    Parameters
    ----------
    a : array_like
        Input array (must be complex).
    overwrite_x : bool
        Specifies whether original array can be destroyed (for inplace transform)
       
    Returns
    -------
    out : complex ndarray
        Result of the transformation along the (-4,-3) axes.    
    """
    a = np.asarray(a, dtype = CDTYPE)      
    libname = DTMMConfig["fftlib"]    
    if libname == "mkl_fft":
        return mkl_fft.fft(a, axis = -3, overwrite_x = overwrite_x)
    elif libname == "scipy":
        return spfft.fft(a, axis = -3, overwrite_x = overwrite_x)
    elif libname == "numpy":
        return npfft.fft(a, axis = -3)
    elif libname == "pyfftw":
        return pyfftw.interfaces.scipy_fft.fft(a, axis = -3, overwrite_x = overwrite_x)
    else: #default implementation is numpy
        return npfft.fft(a, axis = -3)
    
def fft(a, overwrite_x = False):
    """Computes  fft on a matrix of shape (..., n).
    
    This is identical to np.fft2(a)
    
    Parameters
    ----------
    a : array_like
        Input array (must be complex).
    overwrite_x : bool
        Specifies whether original array can be destroyed (for inplace transform)
       
    Returns
    -------
    out : complex ndarray
        Result of the transformation along the (-4,-3) axes.    
    """
    a = np.asarray(a, dtype = CDTYPE)      
    libname = DTMMConfig["fftlib"]    
    if libname == "mkl_fft":
        return mkl_fft.fft(a, overwrite_x = overwrite_x)
    elif libname == "scipy":
        return spfft.fft(a,  overwrite_x = overwrite_x)
    elif libname == "numpy":
        return npfft.fft(a)
    elif libname == "pyfftw":
        return pyfftw.interfaces.scipy_fft.fft(a, overwrite_x = overwrite_x)
    else: #default implementation is numpy
        return npfft.fft(a)
    
def ifft(a, overwrite_x = False):
    """Computes  ifft on a matrix of shape (..., n).
    
    This is identical to np.ifft2(a)
    
    Parameters
    ----------
    a : array_like
        Input array (must be complex).
    overwrite_x : bool
        Specifies whether original array can be destroyed (for inplace transform)
       
    Returns
    -------
    out : complex ndarray
        Result of the transformation along the (-4,-3) axes.    
    """
    a = np.asarray(a, dtype = CDTYPE)      
    libname = DTMMConfig["fftlib"]    
    if libname == "mkl_fft":
        return mkl_fft.ifft(a, overwrite_x = overwrite_x)
    elif libname == "scipy":
        return spfft.ifft(a,  overwrite_x = overwrite_x)
    elif libname == "numpy":
        return npfft.ifft(a)
    elif libname == "pyfftw":
        return pyfftw.interfaces.scipy_fft.ifft(a, overwrite_x = overwrite_x)
    else: #default implementation is numpy
        return npfft.ifft(a)