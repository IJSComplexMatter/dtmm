:mod:`dtmm.conf`
================

.. py:module:: dtmm.conf

.. autoapi-nested-parse::

   Configuration and constants



Module Contents
---------------

.. function:: get_home_dir()

   Return user home directory


.. function:: is_module_installed(name)

   Checks whether module with name 'name' is istalled or not


.. function:: clear_cache(func=None)

   Clears compute cache.

   :param func: A cached function of which cache results are to be cleared (removed
                from cache). If not provided (default) all cache data is cleared from
                all registred cached function - functions returned by the
                :func:`cached_function` decorator
   :type func: function


.. function:: cached_function(f)

   A decorator that converts a function into a cached function.

   The function needs to be a function that returns a numpy array as a result.
   This result is then cached and future function calls with same arguments
   return result from the cache. Function arguments must all be hashable, or
   are small numpy arrays. The function can also take "out" keyword argument for
   an output array in which the resulting array is copied to.

   .. rubric:: Notes

   When caching is enabled, cached numpy arrayes have a read-only attribute.
   You need to copy first, or provide an output array if you need to write to
   the result.


.. function:: cached_result(f)

   A decorator that converts a function into a cached result function.

   The function needs to be a function that returns any result.
   Function arguments must all be hashable, or
   are small numpy arrays.


.. function:: detect_number_of_cores()

   detect_number_of_cores()

   Detect the number of cores in this system.

   :returns: **out** -- The number of cores in this system.
   :rtype: int


.. function:: disable_mkl_threading()

   Disables mkl threading.


.. function:: enable_mkl_threading()

   Enables mkl threading.


.. py:class:: DTMMConfig

   Bases: :class:`object`

   DTMM settings are here. You should use the set_* functions in the
   conf.py module to set these values


.. function:: print_config()

   Prints all compile-time and run-time configurtion parameters and settings.


.. function:: set_verbose(level)

   Sets verbose level (0-2) used by compute functions.


.. function:: set_nthreads(num)

   Sets number of threads used by fft functions.


.. function:: set_cache(level)

   Sets compute cache level.


.. function:: set_fftlib(name='numpy.fft')

   Sets fft library. Returns previous setting.


