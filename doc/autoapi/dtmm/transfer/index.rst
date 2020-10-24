:mod:`dtmm.transfer`
====================

.. py:module:: dtmm.transfer

.. autoapi-nested-parse::

   Main top level calculation functions for light propagation through optical data.



Module Contents
---------------

.. function:: total_intensity(field)

   Calculates total intensity of the field.
   Computes intesity and sums over pixels.


.. function:: transmitted_field(field, wavenumbers, n=1, betamax=BETAMAX, out=None)

   Computes transmitted (forward propagating) part of the field.

   :param field: Input field array
   :type field: ndarray
   :param wavenumbers: Wavenumbers of the field
   :type wavenumbers: array_like
   :param n: Refractive index of the media (1 by default)
   :type n: float, optional
   :param betamax: Betamax perameter used.
   :type betamax: float, optional
   :param out: Output array
   :type out: ndarray, optinal

   :returns: **out** -- Transmitted field.
   :rtype: ndarray


.. function:: reflected_field(field, wavenumbers, n=1, betamax=BETAMAX, out=None)

   Computes reflected (backward propagating) part of the field.

   :param field: Input field array
   :type field: ndarray
   :param wavenumbers: Wavenumbers of the field
   :type wavenumbers: array_like
   :param n: Refractive index of the media (1 by default)
   :type n: float, optional
   :param betamax: Betamax perameter used.
   :type betamax: float
   :param norm: Whether to normalize field so that power spectrum of the output field
                remains the same as that of the input field.
   :type norm: bool, optional
   :param out: Output array
   :type out: ndarray, optinal

   :returns: **out** -- Reflected field.
   :rtype: ndarray


.. function:: transfer_field(field_data, optical_data, beta=None, phi=None, nin=1.0, nout=1.0, npass=1, nstep=1, diffraction=1, reflection=None, method='2x2', multiray=False, norm=DTMM_NORM_FFT, betamax=BETAMAX, smooth=SMOOTH, split_rays=False, split_diffraction=False, split_wavelengths=False, eff_data=None, ret_bulk=False, out=None)

   Tranfers input field data through optical data.

   This function calculates transmitted field and possibly (when npass > 1)
   updates input field with reflected waves.


   :param field_data: Input field data tuple
   :type field_data: Field data tuple
   :param optical_data: Optical data tuple through which input field is transfered.
   :type optical_data: Optical data tuple
   :param beta: Beta parameter of the input field. If it is a 1D array, beta[i] is the
                beta parameter of the field_data[0][i] field array.f not provided, beta
                is caluclated from input data (see also multiray option).
   :type beta: float or 1D array_like of floats, optional
   :param phi: Phi angle of the input light field. If it is a 1D array, phi[i] is the
               phi parameter of the field_data[0][i] field array. If not provided, phi
               is caluclated from input data (see also multiray option).
   :type phi: float or 1D array_like of floats, optional
   :param nin: Refractive index of the input (bottom) surface (1. by default). Used
               in combination with npass > 1 to determine reflections from input layer,
               or in combination with reflection = True to include Fresnel reflection
               from the input surface.
   :type nin: float, optional
   :param nout: Refractive index of the output (top) surface (1. by default). Used
                in combination with npass > 1 to determine reflections from output layer,
                or in combination with reflection = True to include Fresnel reflection
                from the output surface.
   :type nout: float, optional
   :param npass: How many passes (iterations) to perform. For strongly reflecting elements
                 this should be set to a higher value. If npass > 1, then input field data is
                 overwritten and adds reflected light from the sample (defaults to 1).
   :type npass: int, optional
   :param nstep: Specifies layer propagation computation steps (defaults to 1). For thick
                 layers you may want to increase this number. If layer thickness is greater
                 than pixel size, you should increase this number.
   :type nstep: int or 1D array_like of ints
   :param diffraction: Defines how diffraction is calculated. Setting this to False or 0 will
                       disable diffraction calculation. Diffraction is enabled by default.
                       If specified as an integer, it defines diffraction calculation quality.
                       1 for simple (fast) calculation, higher numbers increase accuracy
                       and decrease computation speed. You can set it to np.inf or -1 for max
                       (full) diffraction calculation and very slow computation.
   :type diffraction: bool or int, optional
   :param reflection: Reflection calculation mode for '2x2' method. It can be either
                      0 or False for no reflections, 1 (default) for reflections in fft space
                      (from effective layers), or 2 for reflections in real space
                      (from individual layers). If this argument is not provided it is
                      automatically set to 0 if npass == 1 and 1 if npass > 1 and diffraction
                      == 1 and to 2 if npass > 1 and diffraction > 1. See documentation for details.
   :type reflection: bool or int or None, optional
   :param method: Specifies which method to use, either '2x2' (default) or '4x4'.
   :type method: str, optional
   :param multiray: If specified it defines if first axis of the input data is treated as multiray data
                    or not. If beta and phi are not set, you must define this if your data
                    is multiray so that beta and phi values are correctly determined.
   :type multiray: bool, optional
   :param norm: Normalization mode used when calculating multiple reflections with
                npass > 1 and 4x4 method. Possible values are 0, 1, 2, default value is 1.
   :type norm: int, optional
   :param smooth: Smoothing parameter when calculating multiple reflections with
                  npass > 1 and 4x4 method. Possible values are values above 0.Setting this
                  to higher values > 1 removes noise but reduces convergence speed. Setting
                  this to < 0.1 increases convergence, but it increases noise.
   :type smooth: float
   :param split_diffraction: In diffraction > 1 calculation this option specifies whether to split
                             computation over single beam to consume less temporary memory storage.
                             For large diffraction values this option should be set.
   :type split_diffraction: bool, optional
   :param split_rays: In multi-ray computation this option specifies whether to split
                      computation over single rays to consume less temporary memory storage.
                      For large multi-ray datasets this option should be set.
   :type split_rays: bool, optional
   :param eff_data: Optical data tuple of homogeneous layers through which light is diffracted
                    in the diffraction calculation when diffraction >= 1. If not provided,
                    an effective data is build from optical_data by taking an average
                    isotropic refractive index of the material.
   :type eff_data: Optical data tuple or None
   :param ret_bulk: Whether to return bulk field instead of the transfered field (default).
   :type ret_bulk: bool, optional


.. function:: transfer_4x4(field_data, optical_data, beta=0.0, phi=0.0, eff_data=None, nin=1.0, nout=1.0, npass=1, nstep=1, diffraction=True, reflection=1, multiray=False, norm=DTMM_NORM_FFT, smooth=SMOOTH, betamax=BETAMAX, ret_bulk=False, out=None)

   Transfers input field data through optical data. See transfer_field.


.. function:: transfer_2x2(field_data, optical_data, beta=None, phi=None, eff_data=None, nin=1.0, nout=1.0, npass=1, nstep=1, diffraction=True, reflection=True, multiray=False, split_diffraction=False, betamax=BETAMAX, ret_bulk=False, out=None)

   Tranfers input field data through optical data using the 2x2 method
   See transfer_field for documentation.


