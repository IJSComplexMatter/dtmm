:mod:`dtmm.denoise`
===================

.. py:module:: dtmm.denoise

.. autoapi-nested-parse::

   Field denoising functions.



Module Contents
---------------

.. function:: denoise_field(field, wavenumbers, beta, smooth=1, filter_func=exp_notch_filter, out=None)

   Denoises field by attenuating modes around the selected beta parameter.


.. function:: denoise_fftfield(ffield, wavenumbers, beta, smooth=1, filter_func=exp_notch_filter, out=None)

   Denoises fourier transformed field by attenuating modes around the selected beta parameter.


