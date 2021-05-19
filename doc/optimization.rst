.. _optimization:

Configuration & Tips
====================

In the :mod:`dtmm.conf` there are a few configuration options that you can use for custom configuration, optimization and tuning. The package relies heavily on numba-optimized code. Default numba compilation options are used. For fine-tuning you can of course use custom compilation options that numba provides (See numba_ compilation options). There are also a few numba related environment variables that you can set in addition to numba_ compilation options. These are explained below.


Verbosity
---------

By default compute functions do not print to stdout. You can set printing of progress bar and messages with:

.. doctest::

   >>> import dtmm
   >>> dtmm.conf.set_verbose(1) #level 1 messages
   0
   >>> dtmm.conf.set_verbose(2) #level 2 messages (more info)
   1

To disable verbosity, set verbose level to zero:

.. doctest::

   >>> dtmm.conf.set_verbose(0) #disable printing to stdout
   2

.. note:: 

   The setter functions in the :mod:`dtmm.conf` module return previous defined setting.


Numba multithreading
--------------------

Most computationally expensive numerical algorithms were implemented using @vectorize or @guvecgorize and can be compiled with target="parallel" option. By default, parallel execution is disabled. You can enable parallel target for numba functions by setting the *DTMM_TARGET_PARALLEL* environment variable. This has to be set prior to importing the package.

.. doctest::

   >>> import os
   >>> os.environ["DTMM_TARGET_PARALLEL"] = "1"
   >>> import dtmm #parallel enabled dtmm

Another option is to modify the configuration file (see below). Depending on the number of cores in your system, you should be able to notice an increase in the computation speed.

.. note:

   Full transmission calculation consists of matrix creations and multiplications and 2D FFT computations. The *parallel* target will speed up matrix computations, but it will not have an impact on FFT speed. If you are using mkl_fft, FFT's are already multithreaded by default - but see below.

To set the desired number of threads used in the calculation::

   >>> dtmm.conf.set_numba_threads(2)
   4
   
Numba cache
-----------

Numba allows caching of compiled functions. For debugging purposes, you can enable/disable caching with *DTMM_NUMBA_CACHE* environment variable. To disable caching (enabled by default):

.. doctest::

   >>> os.environ["DTMM_NUMBA_CACHE"]  = "0"

To enable/disable caching you can modify the configuration file (see below). The cached files are stored in `__pycache__` folder in the package's root directory.

FFT optimization
----------------

The package was intended to work with mkl_fft or pyfftw FFT library. In stock numpy or spicy, there are no inplace FFT transform and FFT implementation is not optimized. Although the package works without the intel or fftw library, you are advised to install mkl_fft or pyfftw for best performance.

You can select FFT library ("mkl_fft", "pyfftw", "numpy", or "scipy") with the following::

   >>> dtmm.conf.set_fftlib("mkl_fft")
   'mkl_fft'

For mkl_fft and scipy there is an additional optimization step. Intel's FFT implementation is multithreaded for single FFT computation, which works well for large sized arrays, but there is a very small increase in speed when computing smaller arrays (say 256x256 and smaller). In light transmission calculation, for each wavelength, each polarization, or ray direction there are four 2D FFT and four 2D IFFT computations performed per layer. Instead of parallelizing each of the transforms it is better to make all these transforms in parallel. 

FFT functions in the :mod:`dtmm.fft` can be parallelized using a ThreadPool. By default, this parallelization is disabled and you can enable ThreadPool parallelization of FFTs with:

.. doctest::

   >>> dtmm.conf.set_thread_pool(True)
   False

.. note::

   Creating a ThreadPool in python adds some overhead (a few miliseconds). It makes sense to perform multithreading if computational complexity is high enough. MKL's threading works well for large arrays, but for large number of computations of small arrays, (as in multi-ray computations) ThreadPool should be faster. 

You can also define number of threads used in fft. This number is independent of the number of threads used for numba-compiled functions.::

   >>> dtmm.conf.set_fft_threads(4)
   2
   
If you choose to use pyfftw, you can define fft planner effort with::

   >>> dtmm.conf.set_fft_planner(2) #int 0-3
   1

Which may find a faster version of fft. For pyfftw, the created FFTW plans are stored in :obj:`dtmm.fft.FFTW_CACHE`. If you are running lots of computations on different fft sizes, you may be forced to clear resources to free memory::

   >>> dtmm.fft.clear_cache()
   
After clearing the cache, pyfftw will have to prepare new plan for each new fft transform. Transform is new, if input and output arrays have different shape and stride parameters. If you call fft functions repeatedly with same input/output arrays fftw collects the plan from the cache, which speeds up the computation. 

Default threading options can also be set in the configuration file (see below).

Precision
---------

By default, computation is performed in double precision. You may disable double precision if you are low on memory, and to gain some speed in computation. 

.. doctest::

   >>> os.environ["DTMM_DOUBLE_PRECISION"] = "0"

Please note that if you work with single precision you also have to prepare optical data and field data in single precision. You can also use *fastmath* option in numba compilation to gain some small speed by reducing the computation accuracy when using MKL.

   >>> os.environ["DTMM_FASTMATH"] = "1"

Default values can also be set the configuration file (see below).

DTMM cache
----------

DTMM package uses results cache internally. You can disable caching of results by:

.. doctest::
    
   >>> dtmm.conf.set_cache(0)
   1

If you are running out of memory you should probably disable cashing. To clear cached data you can call:

.. doctest::
    
   >>> dtmm.conf.clear_cache()

Default option can also be set the configuration file (see below).

DTMM configuration file
-----------------------

You can also edit the configuration file *.dtmm/dtmm.ini* in user's home directory to define default settings. This file is automatically generated from a template if it does not exist in the directory. To create the default configuration file, remove the configuration file and import the library in python.

.. literalinclude:: dtmm.ini


.. _numba: https://numba.pydata.org/numba-doc/latest/reference/envvars.html
