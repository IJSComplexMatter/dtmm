Release Notes
-------------

V0.6.0 (In development)
+++++++++++++++++++++++

General improvements to code readibility & documentation of tmm.py module. 
Refactoring of field_viewer.py.

New features
////////////

* Added RadioButtons for FieldViewer, with options for LCP and RCP polarizations
  and retardation settings (labda/2 and lambda/4) plates.
* Added full support for tensorial input data handling (both for the Q tensor or 
  for the eps tensor). Added several new functions in data.py module.
* Added calculate_pom_field function to field_viewer.py
* Added show_scalebar option in FieldViewer.plot().

Enhancements
////////////

* Improved speed in tensor diagonalization procedure when working with tensor input data.

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
