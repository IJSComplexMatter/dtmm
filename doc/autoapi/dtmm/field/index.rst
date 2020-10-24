:mod:`dtmm.field`
=================

.. py:module:: dtmm.field

.. autoapi-nested-parse::

   Field creation, transformation in IO functions



Module Contents
---------------

.. function:: illumination_rays(NA, diameter=5.0, smooth=0.1)

   Returns beta, phi, intensity values for illumination.

   This function can be used to define beta,phi,intensiy arrays that can be used to
   construct illumination data with the :func:`illumination_data` function.
   The resulting beta,phi parameters define directions of rays for the input
   light with a homogeneous angular distribution of rays - input
   light with a given numerical aperture.

   :param NA: Approximate numerical aperture of the illumination.
   :type NA: float
   :param diameter: Field aperture diaphragm diameter in pixels. Approximate number of rays
                    is np.pi*(diameter/2)**2
   :type diameter: int
   :param smooth: Smoothness of diaphragm edge between 0. and 1.
   :type smooth: float, optional

   :returns: **beta,phi, intensity** -- Ray parameters
   :rtype: ndarrays


.. function:: illumination_data(shape, wavelengths, pixelsize=1.0, beta=0.0, phi=0.0, intensity=1.0, n=1.0, focus=0.0, window=None, backdir=False, jones=None, diffraction=True, eigenmodes=None, betamax=BETAMAX)

   Constructs forward (or backward) propagating input illumination field data.

   :param shape: Shape of the illumination
   :type shape: (int,int)
   :param wavelengths: A list of wavelengths.
   :type wavelengths: array_like
   :param pixelsize: Size of the pixel in nm.
   :type pixelsize: float, optional
   :param beta: Beta parameter(s) of the illumination. (Default 0. - normal incidence)
   :type beta: float or array_like of floats, optional
   :param phi: Azimuthal angle(s) of the illumination.
   :type phi: float or array_like of floats, optional
   :param n: Refractive index of the media that this illumination field is assumed to
             be propagating in (default 1.)
   :type n: float, optional
   :param focus: Focal plane of the field. By default it is set at z=0.
   :type focus: float, optional
   :param window: If None, no window function is applied. This window function
                  is multiplied with the constructed plane waves to define field diafragm
                  of the input light. See :func:`.window.aperture`.
   :type window: array or None, optional
   :param backdir: Whether field is bacward propagating, instead of being forward
                   propagating (default)
   :type backdir: bool, optional
   :param jones: If specified it has to be a valid jones vector that defines polarization
                 of the light. If not given (default), the resulting field will have two
                 polarization components. See documentation for details and examples.
   :type jones: jones vector or None, optional
   :param diffraction: Specifies whether field is diffraction limited or not. By default, the
                       field is filtered so that it has only propagating waves. You can disable
                       this by specifying diffraction = False.
   :type diffraction: bool, optional
   :param eigenmodes: If set to True (sefault value when `window` = None.), it defines whether
                      to build eigenmodes from beta and phi values. In this case, beta and phi
                      are only approximate values. If set to False (default when `window` != None),
                      true beta and phi values are set.
   :type eigenmodes: bool or None
   :param betamax: The betamax parameter of the propagating field.
   :type betamax: float, optional

   :returns: **illumination** -- A field data tuple.
   :rtype: field_data


.. function:: field2intensity(field, out)

   field2intensity(field)

   Converts field array of shape [...,4,height,width] to intensity array
   of shape [...,height,width]. For each pixel element, a normal
   component of the Poynting vector is computed.

   :param field: Input field array
   :type field: array_like
   :param cmf: Color matching function
   :type cmf: array_like

   :returns: **spec** -- Computed intensity array
   :rtype: ndarray


.. function:: field2specter(field, out)

   field2specter(field)

   Converts field array of shape [...,nwavelengths,4,height,width] to specter array
   of shape [...,height,width,nwavelengths]. For each pixel element, a normal
   componentof Poynting vector is computed

   :param field: Input field array
   :type field: array_like
   :param cmf: Color matching function
   :type cmf: array_like

   :returns: **spec** -- Computed specter array
   :rtype: ndarray


.. function:: validate_field_data(data)

   Validates field data.

   This function inspects validity of the field data, and makes proper data
   conversions to match the field data format. In case data is not valid and
   it cannot be converted to a valid data it raises an exception (ValueError).

   :param data: A valid field data tuple.
   :type data: tuple of field data

   :returns: **data** -- Validated field data tuple.
   :rtype: tuple


.. function:: save_field(file, field_data)

   Saves field data to a binary file in ``.dtmf`` format.

   :param file: File or filename to which the data is saved.  If file is a file-object,
                then the filename is unchanged.  If file is a string, a ``.dtmf``
                extension will be appended to the file name if it does not already
                have one.
   :type file: file, str, or pathlib.Path
   :param field_data: A valid field data tuple
   :type field_data: (field,wavelengths,pixelsize)


.. function:: load_field(file)

   Load field data from file

   :param file: The file or filenam to read.
   :type file: file, str


