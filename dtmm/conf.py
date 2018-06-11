#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration and constants
"""

from __future__ import absolute_import, print_function, division

import numpy as np
from functools import wraps
import os, warnings, shutil


try:
    from configparser import ConfigParser
except:
    #python 2.7
    from ConfigParser import ConfigParser


DATAPATH = os.path.dirname(__file__)


#os.environ["NUMBA_INTEL_SVML"] = "1"
#os.environ['NUMBA_ENABLE_AVX'] = '1'
#os.environ['NUMBA_LOOP_VECTORIZE'] = '1'
#os.environ['NUMBA_OPT'] = '3'

def read_environ_variable(name, default = "1"):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)
        warnings.warn("Environment variable {0:s} was found, but its value is not valid!".format(name))



def get_home_dir():
    """
    Return user home directory
    """
    try:
        path = os.path.expanduser('~')
    except:
        path = ''
    for env_var in ('HOME', 'USERPROFILE', 'TMP'):
        if os.path.isdir(path):
            break
        path = os.environ.get(env_var, '')
    if path:
        return path
    else:
        raise RuntimeError('Please define environment variable $HOME')

#: home directory
HOMEDIR = get_home_dir()

DTMM_CONFIG_DIR = os.path.join(HOMEDIR, ".dtmm")
NUMBA_CACHE_DIR = os.path.join(DTMM_CONFIG_DIR, "numba_cache")

if not os.path.exists(DTMM_CONFIG_DIR):
    try:
        os.makedirs(DTMM_CONFIG_DIR)
    except:
        warnings.warn("Could not create folder in user's home directory! Is it writeable?")
        NUMBA_CACHE_DIR = ""
        

CONF = os.path.join(DTMM_CONFIG_DIR, "dtmm.ini")
CONF_TEMPLATE = os.path.join(DATAPATH, "dtmm.ini")

config = ConfigParser()

if not os.path.exists(CONF):
    try:
        shutil.copy(CONF_TEMPLATE, CONF)
    except:
        warnings.warn("Could not copy config file in user's home directory! Is it writeable?")
        CONF = CONF_TEMPLATE
config.read(CONF)
    
def _readconfig(func, section, name, default):
    try:
        return func(section, name)
    except:
        return default

BETAMAX = _readconfig(config.getfloat, "core", "betamax", 0.8)
    
if NUMBA_CACHE_DIR != "":
    os.environ["NUMBA_CACHE_DIR"] = NUMBA_CACHE_DIR #set it to os.environ.. so that numba can use it

if read_environ_variable("DTMM_TARGET_PARALLEL","0") or \
   _readconfig(config.getboolean, "numba", "parallel", False):
    NUMBA_TARGET = "parallel"
    NUMBA_PARALLEL = True
else:
    NUMBA_TARGET = "cpu"
    NUMBA_PARALLEL = False

NUMBA_CACHE = False   
if read_environ_variable("DTMM_NUMBA_CACHE","1") or \
   _readconfig(config.getboolean, "numba", "cache", True):
    if NUMBA_PARALLEL == False:
        NUMBA_CACHE = True    

if read_environ_variable("DTMM_FASTMATH","0") or \
   _readconfig(config.getboolean, "numba", "fastmath", False):        
    NUMBA_FASTMATH = True
else:
    NUMBA_FASTMATH = False

def is_module_installed(name):
    """Checks whether module with name 'name' is istalled or not"""
    try:
        __import__(name)
        return True
    except ImportError:
        return False    
        
NUMBA_INSTALLED = is_module_installed("numba")
MKL_FFT_INSTALLED = is_module_installed("mkl_fft")

#reference to all cashed functions - for automatic cache clearing with clear_cache.
_cache = set()

def clear_cache(func = None):
    """Clears compute cache.
    
    Parameters
    ----------
    func : function
        A cached function of which cache results are to be cleared (removed
        from cache). If not provided (default) all cache data is cleared from 
        all registred cached function - functions returned by the 
        :func:`cached_function` decorator"""
    
    if func is not None:
        func.cache.clear()
    else:
        for func in _cache:
            func.cache.clear()
        
            
def cached_function(f):
    """A decorator that converts a function into a cached function. 
    
    The function needs to be a function that returns a numpy array as a result.
    This result is then cached and future function calls with same arguments
    return result from the cache. Function arguments must all be hashable, or
    are small numpy arrays. The function can also take "out" keyword argument for
    an output array in which the resulting array is copied to.
    
    Notes
    -----
    When caching is enabled, cached numpy arrayes have a read-only attribute. 
    You need to copy first, or provide an output array if you need to write to 
    the result.
    """
    
    
    def add_result_to_cache(result, key, cache):
        try:
            cache.pop(next(iter(cache.keys())))
        except StopIteration:
            pass     
        cache[key] = result

    def to_key(arg):
        from dtmm.hashing import hash_buffer
        if isinstance(arg, np.ndarray):
            return (arg.shape, arg.dtype, arg.strides, hash_buffer(arg))
            #return (arg.shape, arg.dtype, tuple(arg.flat))
        else:
            return arg
        
    def copy(result,out):
        if out is not None:
            if isinstance(result, tuple):
                for o,a in zip(out,result):
                    o[...] = a
            else:
                out[...] = result
            return out
        else:
            return result   
        
    def set_readonly(result):
        if isinstance(result, tuple):
            for a in result:
                a.setflags(write = False)
        else:
            result.setflags(write = False)     
            
    def delete(f):
        f.cache.clear()
        _cache.remove(f)
        f.cache = None
   
     
    @wraps(f)
    def _f(*args,**kwargs):
        try_read_from_cache = (kwargs.pop("cache",True) == True) and (DTMMConfig.cache != 0) and _f.cache is not None
        if try_read_from_cache:
            key = tuple((to_key(arg) for arg in args)) + tuple((to_key(arg) for arg in kwargs.values()))
            out = kwargs.pop("out",None)    
            try:
                result = _f.cache[key]
                return copy(result,out)
            except KeyError:
                result = f(*args,**kwargs)
                set_readonly(result)
                add_result_to_cache(result,key, _f.cache)
                return copy(result,out)
        else:
            return f(*args,**kwargs)
    _f.cache = {}
    
    _cache.add(_f)
    _f.delete = delete
    return _f    
    

def detect_number_of_cores():
    """
    detect_number_of_cores()

    Detect the number of cores in this system.

    Returns
    -------
    out : int
        The number of cores in this system.

    """
    import os
    # Linux, Unix and MacOS:
    if hasattr(os, "sysconf"):
        if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
            # Linux & Unix:
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:  # OSX:
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    # Windows:
    if "NUMBER_OF_PROCESSORS" in os.environ:
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            return ncpus
    return 1  # Default

F32DTYPE = np.dtype("float32")
F64DTYPE = np.dtype("float64")
C64DTYPE = np.dtype("complex64")
C128DTYPE = np.dtype("complex128")
U32DTYPE = np.dtype("uint32")

if read_environ_variable("DTMM_DOUBLE_PRECISSION"):
    FDTYPE = F64DTYPE
    CDTYPE = C128DTYPE
    UDTYPE = U32DTYPE
else:
    FDTYPE = F32DTYPE
    CDTYPE = C64DTYPE
    UDTYPE = U32DTYPE
    
#FDTYPE = F32DTYPE
#CDTYPE = C64DTYPE
#UDTYPE = U32DTYPE

class DTMMConfig(object):
    """DTMM settings are here. You should use the set_* functions in the
    conf.py module to set these values"""
    def __init__(self):
        self.numba = NUMBA_INSTALLED
        self.mkl_fft = MKL_FFT_INSTALLED
        if MKL_FFT_INSTALLED:
            self.fftlib = "mkl_fft"
        else:
            self.fftlib = "numpy"
        self.ncores = detect_number_of_cores()
        self.nthreads = 1
        self.verbose = 0
        self.cdtype = CDTYPE
        self.fdtype = FDTYPE
        self.cache = 1
        
    def __getitem__(self, item):
        return self.__dict__[item]
        
    def __repr__(self):
        return repr(self.__dict__)

#: a singleton holding user configuration    
DTMMConfig = DTMMConfig()

def print_config():
    """Prints configurtion parameters and settings."""
    print(DTMMConfig)

#setter functions for DDMConfig
def set_verbose(level):
    """Sets verbose level (0-2) used by compute functions."""
    out = DTMMConfig.verbose
    DTMMConfig.verbose = max(0,int(level))
    return out
    
def set_nthreads(num):
    """Sets number of threads used by fft functions."""
    out = DTMMConfig.nthreads
    DTMMConfig.nthreads = max(1,int(num))
    return out
    
#def set_precision(precision):
#    """Sets internal precision It can be either 'single' (default) or 'double'"""
#    out = DDMConfig.precision
#    values = ("single", "double")
#    if precision in values:
#        DDMConfig.precision = precision
#        if precision == "single":
#            DDMConfig.cdtype = np.dtype(np.complex64)
#            DDMConfig.fdtype = np.dtype(np.float32)
#        else:
#            DDMConfig.cdtype = np.dtype(np.complex128)
#            DDMConfig.fdtype = np.dtype(np.float64)            
#    else:
#        raise ValueError("Precision argument must be 'single' or 'double'")
#    return out
#        
#def set_numba(ok):
#    """Sets numba acceleration on or off. Returns previous setting."""
#    out, ok = DDMConfig.numba, bool(ok) 
#    if NUMBA_INSTALLED and ok:
#        DDMConfig.numba = ok
#        return out
#    elif ok:
#        import warnings
#        warnings.warn("Numba is not installed so it can not be used! Please install numba.")
#    DDMConfig.numba = False
#    return out 
#    
def set_cache(level):
    out = DTMMConfig.cache
    level = max(int(level),0)
    if level > 1:
        warnings.warn("Cache levels higher than 1 not supported yet!")
    DTMMConfig.cache = level
    return out
    
          
def set_fftlib(name = "numpy.fft"):
    """Sets fft library. Returns previous setting."""
    out, name = DTMMConfig.fftlib, str(name) 
    if name == "mkl_fft":
        if MKL_FFT_INSTALLED: 
            DTMMConfig.fftlib = name
        else:
            warnings.warn("MKL FFT is not installed so it can not be used! Please install mkl_fft.")            
    elif name == "scipy.fftpack" or name == "scipy":
        DTMMConfig.fftlib = "scipy"
    elif name == "numpy.fft" or name == "numpy":
        DTMMConfig.fftlib = "numpy"
    else:
        raise ValueError("Unsupported fft library!")
    return out    


import numba

NF32DTYPE = numba.float32
NF64DTYPE = numba.float64
NC64DTYPE = numba.complex64
NC128DTYPE = numba.complex128
NU32DTYPE = numba.uint32

if read_environ_variable("DTMM_DOUBLE_PRECISSION"):

    NFDTYPE = NF64DTYPE
    NCDTYPE = NC128DTYPE
    NUDTYPE = NU32DTYPE
else:
    NFDTYPE = NF32DTYPE
    NCDTYPE = NC64DTYPE
    NUDTYPE = NU32DTYPE
#    
#NFDTYPE = NF32DTYPE
#NCDTYPE = NC64DTYPE
#NUDTYPE = NU32DTYPE    
    
