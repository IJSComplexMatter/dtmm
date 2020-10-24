:mod:`dtmm.color`
=================

.. py:module:: dtmm.color

.. autoapi-nested-parse::

   Color conversion functions and utilities.



Module Contents
---------------

.. function:: specter2color(spec, cmf, norm=False, gamma=True, gray=False, out=None)

   Converts specter data to RGB data (color or gray).

   Specter shape must be [...,k], where wavelengths are in the last axis. cmf
   must be a valid color matchin function array of size [k,3].

   :param spec: Specter data of shape [..., n] where each data element in the array has
                n wavelength values
   :type spec: array
   :param cmf: A color matching function (array of shape [n,3]) that converts the specter data
               to a XYZ color.
   :type cmf: array
   :param norm: If set to False, no data normalization is performed (default). If True,
                internally, xyz data is normalized in the range [0,1.], so that no clipping occurs.
                If it is a float, data is normalized to this value.
   :type norm: bool or float, optional
   :param gamma: If gamma is True srgb gamma function is applied (default). If float is
                 provided, standard gamma factor is applied with a given gamma value. If False,
                 no gamma correction is performed.
   :type gamma: bool or float, optional
   :param gray: Whether gray output is calculated (color by default)
   :type gray: bool, optional
   :param out: Output array
   :type out: array, optional

   :returns: **rgb** -- A computed RGB value.
   :rtype: ndarray

   .. rubric:: Notes

   Numpy broadcasting rules apply to spec and cmf.

   .. rubric:: Example

   >>> cmf = load_tcmf()
   >>> specter2color([1]*81, cmf)#should be close to 1,1,1
   array([ 0.99994901,  1.        ,  0.99998533])


.. function:: load_tcmf(wavelengths=None, illuminant='D65', cmf=CMF, norm=True, retx=False, single_wavelength=False)

   Loads transmission color matching function.

   This functions loads a CIE XYZ color matching function and transforms it
   to a transmission color matching function for a given illuminant. Resulting
   CMF matrix will transform unity into white color.

   :param wavelengths: Wavelengths at which data is computed. If not specified (default), original
                       data from the 5nm tabulated data is returned.
   :type wavelengths: array_like, optional
   :param illuminant: Name of the standard illuminant or path to illuminant data.
   :type illuminant: str, optional
   :param cmf: Name or path to the cmf function. Can be 'CIE1931' for CIE 1931 2-deg
               5nm tabulated data, 'CIE1964' for CIE1964 10-deg 5nm tabulatd data, or
               'CIE2006-2' or 'CIE2006-10' for a proposed CIE 2006 2- or 10-deg 5nm
               tabulated data.
   :type cmf: str, optional
   :param norm: By default cmf is normalized so that unity transmission value over the
                full spectral range of the illuminant is converted to XYZ color with Y=1.
   :type norm: bool, optional
   :param retx: Should the selected wavelengths be returned as well.
   :type retx: bool, optional
   :param single_wavelength: If specified, color matching function for single wavelengths specter is
                             calculated by interpolation. By default, specter is assumed to be a
                             piece-wise linear function and continuous between the specified
                             wavelengts, and data is integrated instead.
   :type single_wavelength: bool, optional

   :returns: **cmf** -- Color matching function array of shape [n,3] or a tuple of (x,cmf)
             if retx is specified.
   :rtype: array

   .. rubric:: Example

   >>> cmf = load_tcmf()
   >>> specter2color([1]*81, cmf) #should be close to 1,1,1
   array([ 0.99994901,  1.        ,  0.99998533])


.. function:: load_specter(wavelengths=None, illuminant='D65', retx=False)

   Loads illuminant specter data from file.

   :param wavelengths: Wavelengths at which data is interpolated
   :type wavelengths: array_like, optional
   :param illuminant: Name of the standard illuminant or filename
   :type illuminant: str, optional
   :param retx: Should the selected wavelengths be returned as well.
   :type retx: bool, optional

   :returns: **specter** -- Specter array of shape [num] or a tuple of (x,specter)
             if retx is specified
   :rtype: array


.. function:: load_cmf(wavelengths=None, cmf=CMF, retx=False, single_wavelength=False)

   Load XYZ Color Matching function as an array.

   This function loads 5nm tabulated data and re-calculates xyz array on a given range of
   wavelength values.

   See also load_tcmf.

   :param wavelengths: A 1D array of wavelengths at which data is computed. If not specified
                       (default), original data from the 5nm tabulated data is returned.
   :type wavelengths: array_like, optional
   :param cmf: Name or path to the cmf function. Can be 'CIE1931' for CIE 1931 2-deg
               5nm tabulated data, 'CIE1964' for CIE1964 10-deg 5nm tabulated data, or
               'CIE2006-2' or 'CIE2006-10' for a proposed CIE 2006 2- or 10-deg 5nm
               tabulated data.
   :type cmf: str, optional
   :param retx: Should the selected wavelengths be returned as well.
   :type retx: bool, optional
   :param single_wavelength: If specified, color matching function for single wavelengths specter is
                             calculated by interpolation. By default, specter is assumed to be a
                             piece-wise linear function and continuous between the specified
                             wavelengts, and data is integrated instead.
   :type single_wavelength: bool, optional

   :returns: **cmf** -- Color matching function array of shape [n,3] or a tuple of (x,cmf)
             if retx is specified.
   :rtype: array


