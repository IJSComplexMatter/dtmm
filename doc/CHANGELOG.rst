Release Notes
-------------

V0.6.1 (Nov 10 200)
+++++++++++++++++++

This is a bugfix release, focusing on documentation/examples improvements:

* New jones caculus example.

Fixes
/////

* linalg.dot_multi now works with input matrices of different shapes
* jones.jones_intensity now returns float instead of complex.
* pom_viewer now correctly converts field to jones, assuming n_cover as the refractive index (instead of n). 


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
