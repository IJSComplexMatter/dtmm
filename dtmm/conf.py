"""
Configuration and constants
===========================

dtmm configuration functions and constants

"""

from __future__ import absolute_import, print_function, division

import numpy as np
from functools import wraps
import os, warnings, shutil

warnings.simplefilter('always', DeprecationWarning)

try:
    from configparser import ConfigParser
except:
    #python 2.7
    from ConfigParser import ConfigParser
    
try:
    from inspect import signature
except:
    #python 2.7
    from funcsigs import signature
    
    
    
#These will be defined later at runtime... here we hold reference to disable warnings in autoapi generation
I = None
CDTYPE = None 
IDTYPE = None
C = None
NUMBA_PARALLEL = None
NCDTYPE = None
NFDTYPE = None


DATAPATH = os.path.dirname(__file__)

def read_environ_variable(name, default):
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

# NUMBA_CACHE_DIR = os.path.join(DTMM_CONFIG_DIR, "numba_cache")


# if not os.path.exists(DTMM_CONFIG_DIR):
#     try:
#         os.makedirs(DTMM_CONFIG_DIR)
#     except:
#         warnings.warn("Could not create folder in user's home directory! Is it writeable?",stacklevel=2)
#         NUMBA_CACHE_DIR = ""

#FILE_LOCK = os.path.join(DTMM_CONFIG_DIR, "lock")        
# if os.path.exists(NUMBA_CACHE_DIR):
#     #we have compiled functions.. make sure it is safe to read from this cache folder
#     if os.path.exists(FILE_LOCK):
#         #it is not safe
#         warnings.warn("There appears to be another instance of dtmm running, so caching is disabled. Try removing .dtmm/numba_cache folder if this message persists.",stacklevel=2)
#         NUMBA_CACHE_DIR = ""  
#     else:
#         #it is safe, make a file lock

#         f = open(FILE_LOCK, "w")
#         f.close()

# import atexit

# @atexit.register
# def cleanup():
#     if NUMBA_CACHE_DIR != "":
#         os.remove(FILE_LOCK)      

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
        #warnings.warn("There was a problem parsing config file while reading section {} and option {}; using default value instead".format(section, name))
        return default
    
    
def is_module_installed(name):
    """Checks whether module with name 'name' is istalled or not"""
    try:
        __import__(name)
        return True
    except ImportError:
        return False  
    except:
        import warnings
        warnings.warn("There appears to be an error in '{}'.".format(name),UserWarning)
        return False
        
NUMBA_INSTALLED = is_module_installed("numba")
MKL_FFT_INSTALLED = is_module_installed("mkl_fft")
SCIPY_INSTALLED = is_module_installed("scipy")
PYFFTW_INSTALLED = is_module_installed("pyfftw")

BETAMAX = _readconfig(config.getfloat, "core", "betamax", 0.8)
SMOOTH = _readconfig(config.getfloat, "core", "smooth", 0.1)


#setting environment variables does not seem to work properly in numba...
#disable cache dir until I figure out how to do it properly.
    
# if NUMBA_CACHE_DIR != "":
#     os.environ["NUMBA_CACHE_DIR"] = NUMBA_CACHE_DIR #set it to os.environ.. so that numba can use it

# def wait_until_cache_is_set(timeout = 1):
#     import subprocess, time
#     t = time.time()
#     while True:
#         print()
#         out = subprocess.run(["printenv"], capture_output=True, text = True)
#         if out.find("NUMBA_CACHE_DIR") != -1:
#             break
#         if time.time()-t > timeout:
#             NUMBA_CACHE_DIR = ""


if read_environ_variable("DTMM_TARGET_PARALLEL",
            default = _readconfig(config.getboolean, "numba", "parallel", False)):
    NUMBA_TARGET = "parallel"
    NUMBA_PARALLEL = True
else:
    NUMBA_TARGET = "cpu"
    NUMBA_PARALLEL = False



_matplotlib_3_4_or_greater = False


try:
    import matplotlib
    major, minor = matplotlib.__version__.split(".")[0:2]
    if int(major) >= 3 and int(minor) >=4:
        _matplotlib_3_4_or_greater = True
except:
    print("Could not determine matplotlib version you are using, assuming < 3.4")

NUMBA_CACHE = False   

if read_environ_variable("DTMM_NUMBA_CACHE",
            default = _readconfig(config.getboolean, "numba", "cache", True)):
    NUMBA_CACHE = True

if read_environ_variable("DTMM_FASTMATH",
        default = _readconfig(config.getboolean, "numba", "fastmath", False)):        
    NUMBA_FASTMATH = True
else:
    NUMBA_FASTMATH = False

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
    
    def pop_fifo_result_from_cache(cache):
        try:
            return cache.pop(next(iter(cache.keys())))
        except StopIteration:
            return None   
        
    def add_result_to_cache(result, key, cache):
        pop_fifo_result_from_cache(cache)   
        cache[key] = result

    def to_key(arg, name = None):
        from dtmm.hashing import hash_buffer 
        
        if isinstance(arg, np.ndarray):
            arg = (arg.shape, arg.dtype, arg.strides, hash_buffer(arg))
            #return (arg.shape, arg.dtype, tuple(arg.flat))
        elif isinstance(arg, list):
            arg = tuple(arg)
        if name is None:
            return arg
        else:
            return (name, arg)
        
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
 
    def unset_readonly(result):
        if isinstance(result, tuple):
            for a in result:
                a.setflags(write = True)
        else:
            result.setflags(write = True)     
           
#    def delete(f):
#        f.cache.clear()
#        _cache.remove(f)
#        f.cache = None
   
     
    @wraps(f)
    def _f(*args,**kwargs):
        try_read_from_cache = (kwargs.pop("cache",True) == True) and (DTMMConfig.cache != 0) and _f.cache is not None
        if try_read_from_cache:
            if _f.has_out: 
                out = kwargs.pop("out",None)
                if len(args) == _f.nargs:
                    #no keyword arguments... this means that the last argument is out
                    out = args[-1] 
                    args = args[:-1]
            else:
                out = None
            key = tuple((to_key(arg) for arg in args)) + tuple((to_key(arg, name = key) for key,arg in kwargs.items()))
            try:
                result = _f.cache[key]
                return copy(result,out)
            except KeyError:
                if kwargs.pop("reuse",False):
                    result = pop_fifo_result_from_cache(_f.cache)
                    unset_readonly(result)
                    kwargs["out"] = result 
                result = f(*args,**kwargs)
                set_readonly(result)
                add_result_to_cache(result,key, _f.cache)
                return copy(result,out)
        else:
            return f(*args,**kwargs)
    _f.cache = {}
    
    parameters = signature(f).parameters
    
    _f.nargs = len(parameters)
    _f.has_out = parameters.get("out") is not None
    
    _cache.add(_f)
    #_f.delete = delete
    return _f    
 
def cached_result(f):
    """A decorator that converts a function into a cached result function. 
    
    The function needs to be a function that returns any result.
    Function arguments must all be hashable, or
    are small numpy arrays. 
    """
    
    def pop_fifo_result_from_cache(cache):
        try:
            return cache.pop(next(iter(cache.keys())))
        except StopIteration:
            return None   
        
    def add_result_to_cache(result, key, cache):
        pop_fifo_result_from_cache(cache)   
        cache[key] = result

    def to_key(arg, name = None):
        from dtmm.hashing import hash_buffer 
        if isinstance(arg, np.ndarray):
            arg = (arg.shape, arg.dtype, arg.strides, hash_buffer(arg))
            #return (arg.shape, arg.dtype, tuple(arg.flat))
        if name is None:
            return arg
        else:
            return (name, arg)

    def set_readonly(result):
        if isinstance(result, tuple):
            for a in result:
                try:
                    a.setflags(write = False)
                except AttributeError:
                    pass
        else:
            try:
                result.setflags(write = False)  
            except AttributeError:
                pass
            
    @wraps(f)
    def _f(*args,**kwargs):
        try_read_from_cache = (kwargs.pop("cache",True) == True) and (DTMMConfig.cache != 0) and _f.cache is not None
        if try_read_from_cache:
            key = tuple((to_key(arg) for arg in args)) + tuple((to_key(arg, name = key) for key,arg in kwargs.items()))  
            try:
                result = _f.cache[key]
                return result
            except KeyError:
                result = f(*args,**kwargs)
                set_readonly(result)
                add_result_to_cache(result,key, _f.cache)
                return result
        else:
            return f(*args,**kwargs)
    _f.cache = {}
    
    _cache.add(_f)
    #_f.delete = delete
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


if read_environ_variable("DTMM_DOUBLE_PRECISION",
  default =  _readconfig(config.getboolean, "core", "double_precision", True)):
    FDTYPE = F64DTYPE
    CDTYPE = C128DTYPE
    UDTYPE = U32DTYPE
    PRECISION = "double"
else:
    FDTYPE = F32DTYPE
    CDTYPE = C64DTYPE
    UDTYPE = U32DTYPE
    PRECISION = "single"
    
def disable_mkl_threading():
    """Disables mkl threading."""
    try:
        import mkl
        mkl.set_num_threads(1)
    except ImportError:
        warnings.warn("Cannot disable mkl threading, because 'mkl' module is not installed!")

def enable_mkl_threading():
    """Enables mkl threading."""
    try:
        import mkl
        mkl.set_num_threads(detect_number_of_cores())
    except ImportError:
        warnings.warn("Cannot enable mkl threading, because 'mkl' module is not installed!")

class DTMMConfig(object):
    """DTMM settings are here. You should use the set_* functions in the
    conf.py module to set these values"""
    def __init__(self):
        
        
        if MKL_FFT_INSTALLED:
            self.fftlib = "mkl_fft"
        elif PYFFTW_INSTALLED:
            self.fftlib = "pyfftw"
        elif SCIPY_INSTALLED:
            self.fftlib = "scipy"
        else:
            self.fftlib = "numpy"
        
        if _readconfig(config.getboolean, "fft", "parallel", False):
            import warnings
            warnings.warn("parallel option in conf.ini is deprecated, use thread_pool = True instead", DeprecationWarning)
        
        self.thread_pool = _readconfig(config.getboolean, "fft", "thread_pool", False)
        self.fft_threads = 1
        self.fft_planner = 1#_readconfig(config.getboolean, "fft", "planner", 1)
        self.numba_threads = 1
        if _readconfig(config.getboolean, "core", "cache", True):
            self.cache = 1
        else:
            self.cache = 0
        self.verbose = _readconfig(config.getint, "core", "verbose", 0)
        
        self.gray =  _readconfig(config.getboolean, "viewer", "gray", False)
        self.show_ticks = _readconfig(config.getboolean, "viewer", "show_ticks", None)
        self.show_scalebar = _readconfig(config.getboolean, "viewer", "show_scalebar", False)
        self.show_sliders = _readconfig(config.getboolean, "viewer", "show_sliders", True)
        gamma = _readconfig(config.getfloat, "viewer", "gamma", None)
        if gamma is None:
            gamma = _readconfig(config.getboolean, "viewer", "gamma", True)
        self.gamma = gamma
        
        self.n_cover = _readconfig(config.getfloat, "viewer", "n_cover", 1.5)
        self.d_cover = _readconfig(config.getfloat, "viewer", "d_cover", 0.)
        self.immersion = _readconfig(config.getboolean, "viewer", "immersion", False)
        self.NA = _readconfig(config.getfloat, "viewer", "NA", 0.7)
        self.cmf = _readconfig(config.get, "viewer", "cmf", "CIE1931")
        
        self.diffraction = _readconfig(config.getint, "transfer", "diffraction", 1)
        self.nin = _readconfig(config.getfloat, "transfer", "nin", self.n_cover)
        self.nout = _readconfig(config.getfloat, "transfer", "nout", self.n_cover)
        self.method = _readconfig(config.get, "transfer", "method", "2x2")
        self.npass = _readconfig(config.getint, "transfer", "npass", 1)
        self.reflection = _readconfig(config.getint, "transfer", "reflection", None)
        self.eff_data = _readconfig(config.getint, "transfer", "eff_data", 0)        
        self.betamax = _readconfig(config.getfloat, "core", "betamax", np.inf)
    
    @property
    def nthreads(self):
        import warnings
        warnings.warn("deprecated, use fft_threads instead", DeprecationWarning)
        return self.fft_threads
        
    def __getitem__(self, item):
        return self.__dict__[item]
        
    def __repr__(self):
        return repr(self.__dict__)
    

#: a singleton holding user configuration    
DTMMConfig = DTMMConfig()
if DTMMConfig.thread_pool == True:
    disable_mkl_threading()

    
def get_default_config_option(name, value = None):
    """Returns default config option specified with 'name', if value is not None,
    returns value instead"""
    return DTMMConfig[name] if value is None else value
    
CMF = _readconfig(config.get, "viewer", "cmf", "CIE1931")

def print_config():
    """Prints all compile-time and run-time configurtion parameters and settings."""
    options = {"PRECISION" : PRECISION, "BETAMAX": BETAMAX, "SMOOTH" : SMOOTH,
               "NUMBA_FASTMATH" :NUMBA_FASTMATH, "NUMBA_PARALLEL" : NUMBA_PARALLEL,
           "NUMBA_CACHE" : NUMBA_CACHE, "NUMBA_FASTMATH" : NUMBA_FASTMATH,
           "NUMBA_TARGET" : NUMBA_TARGET}
    options.update(DTMMConfig.__dict__)
    print(options)

#setter functions for DTMMConfig    
def set_verbose(level):
    """Sets verbose level (0-2) used by compute functions."""
    out = DTMMConfig.verbose
    DTMMConfig.verbose = max(0,int(level))
    return out
   
def set_cache(level):
    """Sets compute cache level."""
    out = DTMMConfig.cache
    level = max(int(level),0)
    if level > 1:
        warnings.warn("Cache levels higher than 1 not supported yet!")
    DTMMConfig.cache = level
    return out

def set_fftlib(name = "numpy.fft"):
    """Sets fft library name."""
    out, name = DTMMConfig.fftlib, str(name) 
    if name == "mkl_fft":
        if MKL_FFT_INSTALLED: 
            DTMMConfig.fftlib = name
        else:
            warnings.warn("MKL FFT is not installed so it can not be used! Please install mkl_fft.")            
    
    elif name in ("scipy.fftpack", "scipy.fft", "scipy"):
        if SCIPY_INSTALLED:
            DTMMConfig.fftlib = "scipy"
        else:
            warnings.warn("Scipy is not installed so it can not be used! Please install scipy.") 
    elif name == "pyfftw":
        if PYFFTW_INSTALLED: 
            DTMMConfig.fftlib = name
        else:
            warnings.warn("Pyfftw is not installed so it can not be used! Please install pyfftw.")            
    elif name == "numpy.fft" or name == "numpy":
        DTMMConfig.fftlib = "numpy"
    else:
        raise ValueError("Unsupported fft library!")
    return out    

def set_betamax(value):
    """Sets betamax value."""
    DTMMConfig.betamax = float(value)

set_fftlib(_readconfig(config.get, "fft", "fftlib", DTMMConfig.fftlib))

def set_fft_threads(n):
    """Sets number of threads used in fft functions."""
    out = DTMMConfig.fft_threads
    n = int(n)
    DTMMConfig.fft_threads = n
    if DTMMConfig.thread_pool == True:
        _set_external_fft_threads(1)
    else:
        _set_external_fft_threads(n)
    return out

def set_fft_planner(n):
    """Sets fft planner effort (pyfftw) 0-3, higher means more planning effort"""
    out = DTMMConfig.fft_planner
    DTMMConfig.fft_planner = max(min(int(n),3),0)
    return out
    
      
def _set_external_fft_threads(n):
    if DTMMConfig.thread_pool:
        #disable native threading if we are to use threadpool
        num = 1
    else:
        num = n
    try:
        import mkl
        mkl.set_num_threads(num)
    except ImportError:
        pass
    try:
        import pyfftw 
        #threadpool does not seem to work properly with pyfftw, so we use n and disable it in fft.py
        pyfftw.config.NUM_THREADS = n
    except ImportError:
        pass

def set_thread_pool(ok):
    """Sets or unsets ThreadPool. If set to True, a ThreadPoll is used for threading.
    If set to False, threading is defined by the fft library."""
    out = DTMMConfig.thread_pool
    ok = bool(ok)
    DTMMConfig.thread_pool = ok
    if ok == True:
        _set_external_fft_threads(1)
    else:
        _set_external_fft_threads(DTMMConfig.fft_threads)
    return out
    
import numba
    
def set_numba_threads(n):
    """Sets number of threads used in numba-accelerated functions."""
    out = DTMMConfig.numba_threads
    numba.set_num_threads(n)
    n = numba.get_num_threads()
    DTMMConfig.numba_threads = int(n)
    return out

def set_nthreads(num):
    """Sets number of threads used by numba and fft functions."""
    out1 = set_numba_threads(num)
    out2 = set_fft_threads(num)
    if out1 == out2:
        return out1
     #if not the same no return... 

try:
    set_fft_threads(_readconfig(config.getint, "fft", "nthreads", detect_number_of_cores()))
except:
    #in case something wents wrong, we do not want dtmm to fail loading.
    import warnings
    warnings.warn("Could not set fft threads", UserWarning)
try:
    set_numba_threads(_readconfig(config.getint, "numba", "nthreads", detect_number_of_cores()))
except:
    import warnings
    #in case something wents wrong, we do not want dtmm to fail loading.
    warnings.warn("Could not set numba threads", UserWarning)



NF32DTYPE = numba.float32
NF64DTYPE = numba.float64
NC64DTYPE = numba.complex64
NC128DTYPE = numba.complex128
NU32DTYPE = numba.uint32

if PRECISION == "double":
    NFDTYPE = NF64DTYPE
    NCDTYPE = NC128DTYPE
    NUDTYPE = NU32DTYPE
else:
    NFDTYPE = NF32DTYPE
    NCDTYPE = NC64DTYPE
    NUDTYPE = NU32DTYPE
  
    
