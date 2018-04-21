Optimization Tips
=================

In the :mod:`dtmm.conf` there are several configuration options

Parallelization
---------------

All computationally expensive numerical algorithms have been implemented in numba using @vectorize or @guvecgorize and can be performed with target="parallel" option. By default, parallel execution is disabled for two reasons. In parallel mode, the functions have to be compiled at runtime. This adds significant time overhead when importing the package. Secondly, automatic parallelization of vectorized functions is a new feature in numba and is still experimental according to numba documentation.

You can enable parallel target for numba functions by setting the *DTMM_TARGET_PARALLEL* environment variable. This has to be set prior to importing the package::

   >>> import os
   >>> os.environ["DTMM_TARGET_PARALLEL"] = "1"
   >>> import dtmm #parallel enabled dtmm

Depending on the number of cores in your system, you may gain increase in the computation speed.

FFT optimization
----------------

The package was intended to work with mkl_fft FFT library. In stock numpy or spicy, the FFT implementation does not have an inlace FFT transform. Although the package works without the intel library, you are advised to install mkl_fft for best performance.

You can select FFT library ("mkl_fft", "numpy", or "scipy") with the following::

   >>> dtmm.conf.set_fftlib("mkl_fft")
   'mkl_fft'

For mkl_fft there is an additional optimization step. Intels FFT implementation is parallelized for single fft computation, therefore , gains are achieved when computing large size arrays (larger than 256x256). In light transmission calculation for each wavelength, for each polarization or ray direction there are four FFT and four IFFT computations performed per layer. Instead of parallelizing each of the transforms it is better to make all these transforms in parallel. 

You can enable ThreadPool parallelization of FFTs in :func:`dtmm.fft.fft2` and :func:`dtmm.fft.ifft2`::


   >>> dtmm.conf.set_nthreads(4)
   1

.. note:: 

   The setter functions in the :mod:`dtmm.conf` module return previous defined setting.

It is important that you disable MKL's multithreading by setting the *MKL_NUM_THREADS* environment variable to "1", or if you have mkl-services installed try:

   >>> import mkl
   >>> mkl.set_num_threads(1)

You must experiment with settings a little. Depending on the size of the field_data, number of cores, the ThreadPool version may work faster or it may work slower than mkl_fft version. If you are not sure what to use, stick with stock MKL threading and default setting of::

   >>> dtmm.conf.set_nthreads(1)
   4
   

.. note::

   Creating a ThreadPool in python adds some overhead (a few miliseconds). It makes sense to perform multithreading if computational complexity is high enough. MKL's threading works well for large arrays, but for multiple computations of small arrays, ThreadPool version of fft should be faster. As a rule of a thumb, layer computation time has to be greater than 10ms to make it feasible to use ThreadPools, otherwise, stick with MKL threading.

Verbose messages
----------------

By default compute functions do not print to stdout. You can set printing of progress bar and messages with:

.. doctest::

   >>> dtmm.conf.set_verbose(1) #level 1 messages
   0
   >>> dtmm.conf.set_verbose(2) #level 2 messages (more info)
   1

To disable verbosity, set verbose level to zero:

.. doctest::

   >>> dtmm.conf.set_verbose(0) #disable printing to stdout
   2



Numba cache
-----------

The package internally uses numba for numerical work. This increases import time when the package is loaded. Therefore, when *DTMM_TARGET_PARALLEL* environment variable is not defined, all compiled functions are cached and stored in your home directory for faster import by default. For debugging purposes, you can enable/disable caching with *DTMM_NUMBA_CACHE* environment variable. To disable caching (enabled by default):

.. doctest::

   >>> os.environ["DTMM_NUMBA_CACHE"]  = "0"

Cached files are stored in *.dtmm/numba_cache*  in users home directory. You can remove this folder to force recompilation of numba functions.


DTMM cache
----------



