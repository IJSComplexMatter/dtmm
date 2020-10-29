Release Notes
-------------

V0.6.0 (In development)
+++++++++++++++++++++++

General improvements to code readibility & documentation of tmm.py module. 
Refactoring of field_viewer.py. Because of code refactoring some lovel level functions
were renamed or ommited, so this release partially break backward compatibility.
See *changes* for details.

New features
////////////

* Added RadioButtons for FieldViewer, with options for LCP and RCP polarizations
  and retardation settings (labda/2 and lambda/4) plates.
* Added full support for tensorial input data handling (both for the Q tensor or 
  for the eps tensor). Added several new functions in data.py module.
* Added calculate_pom_field function to field_viewer.py
* Added show_scalebar option in FieldViewer.plot().
* Added CMOS spectral response function to allow simulations using grayscale cameras.
* Added simplified tcmf data generation for custom spectral data using `load_tcmf` and `load_specter`
* Added `effective_data` function to simplify effective data construction.
* Added an option to add illuminant data as a table to load_tcmf function to 
  allow for custom illuminant spectra.

Enhancements
////////////

* Improved speed in tensor diagonalization procedure when working with tensor input data.
* Improvements in the effective medium handling for the difraction step in the propagation calculation.

Changes
///////

* removed tensor_to_matrix function fromm rotation.py, added tensor2matrix and matrix2tensor 
  functions in data.py


V0.5.0 (October 20 2020)
++++++++++++++++++++++++

Initial support for non-iterative 4x4 calculation with reflections (for 2d data)


V0.4.0 (May 22 2020)
++++++++++++++++++++

Initial official release.
