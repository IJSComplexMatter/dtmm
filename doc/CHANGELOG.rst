Release Notes
-------------

V0.7.0 (May 25 2021)
++++++++++++++++++++

This release improves multi-threading computation speed and adds several new features.
It add support for pyfftw and matplotlib>=3.4.

New features
////////////

* We introduce new optical data model. Legacy data model, which is in this version refered to as an optical block, is still a valid optical data. New optical data model is now a list of optical blocks. It is now possible to mix homogeneous layers with inhomogeneous layers and it simplifies creation of the optical stack.
* New solver.py module with two objects for easier usage of non-iterative matrix-based simulations
* New hybrid solver for heterogeneous data, which is triggered when `create_matrix` argument is set. See :func:`dtmm.transfer.tranfer_field` for details.
* Several new options for computation optimization in dtmm.ini. See the documentation for details.
* Implements a new thread poll for parallel fft computation, which is faster than native threading in mkl_fft or scipy. This feature is still in beta, so it is disabled by defauly and you must activate it using a configuration file or using :func:`dtmm.conf.set_thread_pool`.
* Support for pyfftw. You can enable it by calling :func:`dtmm.conf.set_fftlib` with 'pyfftw' as an argument.
* New RangeSlider for condenser aperture setting, allowing you to set annular aperture in matplotlib figure (matplotlib 3.4 only).
* Added an option to specify annular aperture in illumination_data and in the viewer's aperture atrribute.

Fixes
/////

* Fixes BulkViewer (that got broken in 0.6 release).
* Fixes CustomRadioButtons, which stopped working in matplotlib 3.4.
* In mode grouping (When working with diffraction > 1) the effective beam parameter is now more accurately determined. The results using low diffraction e.g 2,3,4 are now more accurate.
* When working with single precision, atol and rtol values are now less strict when checking for real eigenvector output in eps2epsva function.
* Cached functions now support list argument.
* When using numpy for fft and in single precision, ouput of fft2 is checked to be of correct type ("complex64"). In previous version this check was not made, resulting in a possible calculation error if the underlying fft2 implementation in numpy chose "complex128" as an otput.

Changes
///////

* New optical data format. In previous versions, optical data was a tuple of three elements. Now,  optical data is a list of tuples. Each element in the list represents an optical block, which was called optical data in previous releases. Users should adopt the new optical data format in their code.
* In previous versions, the :func:set_nthreads changed the number of fft threads and activated the thread pool. Now it changes number of threads used both for fft and numba computation, but it does not activate the thread pool. You must explicitly call dtmm.conf.set_thread_pool(True).
* Drops support for older versions of numba. Minimum version of numba is now 0.45.0
* Drops support for older versions of numpy. Minimum version of numpy is now 1.20

V0.6.1 (Nov 10 2020)
++++++++++++++++++++

This is a bugfix release, focusing on documentation/examples improvements:

* New jones caculus example.

Fixes
/////

* linalg.dot_multi now works with input matrices of different shapes
* jones.jones_intensity now returns float instead of complex.
* pom_viewer now correctly converts field to jones, assuming `n_cover` as the refractive index (instead of `n` - the output medium).
* data.illumination_data now uses n = `n_cover` as the default medium (instead of n = 1).

V0.6.0 (Nov 6 2020)
+++++++++++++++++++

This release adds many new features.  

Please note that this release also partially breaks backward compatibility because of code refactoring. High-level functions remain backward compatible, but some low-level functions were renamed, omitted or replaced with a different implementation. See *Changes* for details.

New features
////////////

* New :func:`dtmm.field_viewer.pom_viewer` for more accurate optical microscope simulations for experiments done with thick cover glass.
* Full support for tensorial input data handling (both for the Q tensor or for the eps tensor, real or complex valued epsilon). 
* RadioButtons for FieldViewer, with options for LCP and RCP polarizations and retardation settings (lambda/2 and lambda/4) plates.
* Added show_scalebar option in FieldViewer.plot().
* New CMOS spectral response function to allow simulations using grayscale cameras.
* Simplified tcmf data generation for custom spectral data using `load_tcmf` and `load_specter`
* New :func:`dtmm.data.effective_data` function to simplify effective data construction.
* The eff_data argument of :func:`dtmm.transfer.transfer_field` can now take strings "isotropic", "uniaxial" or "biaxial" to simplify creation of effective medium.
* New jones4.py module for creation of 4x4 jones-like matrices to simplify polarization handling of field data.
* Extended configuration options in dtmm.ini.

Changes
///////

* Removed tensor_to_matrix function fromm rotation.py, added tensor2matrix and matrix2tensor functions in data.py
* Moved polarizer4x4 and jonesmat4x4 from tmm.py to jones4.py
* Removed polarization.py in favor of jones4.py.
* New defaults for transfer_field's `nin` and `nout` arguments. These now default to the newly introduced `n_cover` parameter and a configuration parameter inside dtmm.ini. You can override this behavior by setting `nin` and `not` options in dtmm.ini file.
* Removed the NUMBA_CACHE_DIR option in conf.py, which appears to fix the segfault error.

Fixes
/////

* segfault error due to numba caching. 

V0.5.0 (Oct 20 2020)
++++++++++++++++++++

Initial support for non-iterative 4x4 calculation with reflections (for 2d data)


V0.4.0 (May 22 2020)
++++++++++++++++++++

Initial official release.
