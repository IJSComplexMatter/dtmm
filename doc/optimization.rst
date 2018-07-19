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

Most computationally expensive numerical algorithms were implemented using @vectorize or @guvecgorize and can be compiled with target="parallel" option. By default, parallel execution is disabled for two reasons. In parallel mode, the functions have to be compiled at runtime. This adds significant compilation time overhead when importing the package. Secondly, automatic parallelization of vectorized functions is a new feature in numba and is still experimental and not supported on all platforms according to numba documentation.

You can enable parallel target for numba functions by setting the *DTMM_TARGET_PARALLEL* environment variable. This has to be set prior to importing the package.

.. doctest::

   >>> import os
   >>> os.environ["DTMM_TARGET_PARALLEL"] = "1"
   >>> import dtmm #parallel enabled dtmm

Another option is to modify the configuration file (see below). Depending on the number of cores in your system, you should be able to notice an increase  in the computation speed.

.. note:

   Full transmission calculation consists of matrix creations and multiplications and 2D FFT computations. The *parallel* target will speed up matrix computations, but it will not have an impact on FFT speed. If you are using mkl_fft, FFT's are already multithreaded by default - but see below.


Numba cache
-----------

Numba allows caching of compiled functions. If *DTMM_TARGET_PARALLEL* environment variable is not defined, all compiled functions are cached and stored in your home directory for faster import by default. For debugging purposes, you can enable/disable caching with *DTMM_NUMBA_CACHE* environment variable. To disable caching (enabled by default):

.. doctest::

   >>> os.environ["DTMM_NUMBA_CACHE"]  = "0"

Cached files are stored in *.dtmm/numba_cache*  in user's home directory. You can remove this folder to force recompilation. To enable/disable caching you can modify the configuration file (see below).

FFT optimization
----------------

The package was intended to work with mkl_fft FFT library. In stock numpy or spicy, there are no inplace FFT transform and FFT implementation is not optimized. Although the package works without the intel library, you are advised to install mkl_fft for best performance.

You can select FFT library ("mkl_fft", "numpy", or "scipy") with the following::

   >>> dtmm.conf.set_fftlib("mkl_fft")
   'mkl_fft'

For mkl_fft there is an additional optimization step. Intel's FFT implementation is multithreaded for single FFT computation, which works well for large sized arrays, but there is a very small increase in speed when computing smaller arrays (say 256x256 and smaller). In light transmission calculation, for each wavelength, each polarization, or ray direction there are four 2D FFT and four 2D IFFT computations performed per layer. Instead of parallelizing each of the transforms it is better to make all these transforms in parallel. 

FFT functions in the :mod:`dtmm.fft` can be parallelized using a ThreadPool. By default, this parallelization is disabled and you can enable ThreadPool parallelization of FFTs with:

.. doctest::

   >>> dtmm.conf.set_nthreads(4)
   1

It is important that you disable MKL's multithreading by setting the *MKL_NUM_THREADS* environment variable to "1", or if you have mkl-services installed try:

   >>> import mkl
   >>> mkl.set_num_threads(1)

You must experiment with settings a little. Depending on the size of the field_data, number of cores, the ThreadPool version may work faster or it may work slower than mkl_fft version. If you are not sure what to use, stick with stock MKL threading and default setting of:

.. doctest::

   >>> dtmm.conf.set_nthreads(1)
   4
   
.. note::

   Creating a ThreadPool in python adds some overhead (a few miliseconds). It makes sense to perform multithreading if computational complexity is high enough. MKL's threading works well for large arrays, but for large number of computations of small arrays, ThreadPool  should be faster. As a rule of a thumb, layer computation time has to be greater than 10ms to make it feasible to use ThreadPools, otherwise, stick with defaults. 

Default threading options can also be set in the configuration file (see below).

Precision
---------

By default, computation is performed in double precision. You may disable double precision if you are low on memory, and to gain some speed in FFT computation. 

.. doctest::

   >>> os.environ["DTMM_DOUBLE_PRECISION"] = "0"

You can also use *fastmath* option in numba compilation to gain some small speed by reducing the computation accuracy when using MKL.

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

You can also edit the configuration file *.dtmm/dtmm.ini* in user's home directory to define default settings. This file is automatically generated from a template if it does not exist in the directory.

.. literalinclude:: dtmm.ini


.. _numba: https://numba.pydata.org/numba-doc/latest/reference/envvars.html

