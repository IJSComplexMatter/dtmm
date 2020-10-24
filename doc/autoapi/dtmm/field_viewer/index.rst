:mod:`dtmm.field_viewer`
========================

.. py:module:: dtmm.field_viewer

.. autoapi-nested-parse::

   Field visualizer (polarizing miscroscope simulator)



Module Contents
---------------

.. py:class:: CustomRadioButtons(ax, labels, active=0, activecolor='blue', size=49, orientation='horizontal', **kwargs)

   Bases: :class:`matplotlib.widgets.RadioButtons`

   A GUI neutral radio button.

   For the buttons to remain responsive you must keep a reference to this
   object.

   Connect to the RadioButtons with the :meth:`on_clicked` method.

   .. attribute:: ax

      The containing `~.axes.Axes` instance.

   .. attribute:: activecolor

      The color of the selected button.

   .. attribute:: labels

      A list of `~.text.Text` instances containing the button labels.

   .. attribute:: circles

      A list of `~.patches.Circle` instances defining the buttons.

   .. attribute:: value_selected

      The label text of the currently selected button.

      :type: str


.. function:: bulk_viewer(field_data, cmf=None, window=None, **parameters)

   Returns a FieldViewer object for optical microscope simulation

   :param field_data: Input field data
   :type field_data: tuple[np.ndarray]
   :param cmf: Color matching function (table). If provided as an array, it must match
               input field wavelengths. If provided as a string, it must match one of
               available CMF names or be a valid path to tabulated data. See load_tcmf.
   :type cmf: str, ndarray or None, optional
   :param window: Window function by which the calculated field is multiplied. This can
                  be used for removing artefact from the boundaries.
   :type window: ndarray, optional
   :param parameters: Extra parameters passed directly to the :meth:`FieldViewer.set_parameters`
   :type parameters: kwargs, optional

   :returns: **out** -- A :class:`BulkViewer` viewer object
   :rtype: BulkViewer


.. function:: field_viewer(field_data, cmf=None, bulk_data=False, n=1.0, mode=None, window=None, diffraction=True, polarization_mode='normal', betamax=BETAMAX, beta=None, **parameters)

   Returns a FieldViewer object for optical microscope simulation

   :param field_data: Input field data
   :type field_data: tuple[np.ndarray]
   :param cmf: Color matching function (table). If provided as an array, it must match
               input field wavelengths. If provided as a string, it must match one of
               available CMF names or be a valid path to tabulated data. See load_tcmf.
   :type cmf: str, ndarray or None, optional
   :param bulk_data: # TODO: I don't know what this value is
   :type bulk_data: bool
   :param n: Refractive index of the output material.
   :type n: float, optional
   :param mode: Viewer mode 't' for transmission mode, 'r' for reflection mode None for
                as is data (no projection calculation - default).
   :type mode: [ 't' | 'r' | None], optional
   :param window: Window function by which the calculated field is multiplied. This can
                  be used for removing artefact from the boundaries.
   :type window: ndarray, optional
   :param diffraction: Specifies whether field is treated as diffractive field or not (if it
                       was calculated by diffraction > 0 algorithm or not). If set to False
                       refocusing is disabled.
   :type diffraction: bool, optional
   :param polarization_mode: Defines polarization mode. That is, how the polarization of the light is
                             treated after passing the analyzer. By default, polarizer is applied
                             in real space (`normal`) which is good for normal (or mostly normal)
                             incidence light. You can use `mode` instead of `normal` for more
                             accurate, but slower computation. Here polarizers are applied to
                             mode coefficients in fft space.
   :type polarization_mode: str, optional
   :param betamax: Betamax parameter used in the diffraction calculation function. With this
                   you can simulate finite NA of the microscope (NA = betamax).
   :type betamax: float
   :param parameters: Extra parameters passed directly to the :meth:`FieldViewer.set_parameters`
   :type parameters: kwargs, optional

   :returns: **out** -- A :class:`FieldViewer` or :class:`BulkViewer` viewer object
   :rtype: FieldViewer


.. py:class:: FieldViewer(field, ks, cmf, mode=None, n=1.0, polarization='normal', window=None, diffraction=True, betamax=BETAMAX, beta=None)

   Bases: :class:`object`

   Base viewer

   .. method:: focus(self)
      :property:

      Focus position, relative to the calculated field position.


   .. method:: ffield(self)
      :property:

      Fourier transform of the field


   .. method:: cols(self)
      :property:

      Number of columns used (for periodic tructures)


   .. method:: sample(self)
      :property:

      Sample rotation angle


   .. method:: sample_angle(self)
      :property:

      Sample rotation angle in degrees in float


   .. method:: aperture(self)
      :property:

      Illumination field aperture


   .. method:: polarizer(self)
      :property:

      Polarizer angle. Can be 'h','v', 'lcp', 'rcp', 'none', angle float or a jones vector


   .. method:: analyzer(self)
      :property:

      Analyzer angle. Can be 'h','v', 'lcp', 'rcp', 'none', angle float or a jones vector


   .. method:: intensity(self)
      :property:

      Input light intensity


   .. method:: set_parameters(self, **kwargs)

      Sets viewer parameters. Any of the :attr:`.VIEWER_PARAMETERS`


   .. method:: get_parameters(self)

      Returns viewer parameters as dict


   .. method:: plot(self, fig=None, ax=None, sliders=None, show_sliders=True, **kwargs)

      Plots field intensity profile. You can set any of the below listed
      arguments. Additionaly, you can set any argument that imshow of
      matplotlib uses (e.g. 'interpolation = "sinc"').

      :param fmin: Minimimum value for the focus setting.
      :type fmin: float, optional
      :param fmax: Maximum value for the focus setting.
      :type fmax: float, optional
      :param imin: Minimimum value for then intensity setting.
      :type imin: float, optional
      :param imax: Maximum value for then intensity setting.
      :type imax: float, optional
      :param pmin: Minimimum value for the polarizer angle.
      :type pmin: float, optional
      :param pmax: Maximum value for the polarizer angle.
      :type pmax: float, optional
      :param smin: Minimimum value for the sample rotation angle.
      :type smin: float, optional
      :param smax: Maximum value for the sample rotation angle.
      :type smax: float, optional
      :param amin: Minimimum value for the analyzer angle.
      :type amin: float, optional
      :param amax: Maximum value for the analyzer angle.
      :type amax: float, optional
      :param namin: Minimimum value for the numerical aperture.
      :type namin: float, optional
      :param namax: Maximum value for the numerical aperture.
      :type namax: float, optional


   .. method:: calculate_specter(self, recalc=False, **params)

      Calculates field specter.

      :param recalc: If specified, it forces recalculation. Otherwise, result is calculated
                     only if calculation parameters have changed.
      :type recalc: bool, optional
      :param params: Any additional keyword arguments that are passed dirrectly to
                     set_parameters method.
      :type params: kwargs, optional


   .. method:: calculate_image(self, recalc=False, **params)

      Calculates RGB image.

      :param recalc: If specified, it forces recalculation. Otherwise, result is calculated
                     only if calculation parameters have changed.
      :type recalc: bool, optional
      :param params: Any additional keyword arguments that are passed dirrectly to
                     set_parameters method.
      :type params: keyword arguments


   .. method:: save_image(self, fname, origin='lower', **kwargs)

      Calculates and saves image to file using matplotlib.image.imsave.

      :param fname: Output filename or file object.
      :type fname: str
      :param origin: Indicates whether the (0, 0) index of the array is in the upper left
                     or lower left corner of the axes. Defaults to 'lower'
      :type origin: [ 'upper' | 'lower' ]
      :param kwargs: Any extra keyword argument that is supported by matplotlib.image.imsave
      :type kwargs: optional


   .. method:: update_plot(self)

      Triggers plot redraw


   .. method:: show(self)

      Shows plot



.. py:class:: BulkViewer(field, ks, cmf, mode=None, n=1.0, polarization='normal', window=None, diffraction=True, betamax=BETAMAX, beta=None)

   Bases: :class:`dtmm.field_viewer.FieldViewer`

   Base viewer

   .. method:: ffield(self)
      :property:

      Fourier transform of the field


   .. method:: focus(self)
      :property:

      Focus position, relative to the calculated field position.



