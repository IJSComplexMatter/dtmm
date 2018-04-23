Tutorial
========

Reflections and Interference
----------------------------

Color Conversion
----------------

In this tutorial you will learn how to transform specter to RGB colors using `CIE 1931`_ standard observer color matching function (see `CIE 1931`_ wiki pages for details on XYZ color space).

In the :mod:`dtmm.color` there is a limited set of functions for converting computed specters to RGB images. It is not a full color engine, so only a few color conversion functions are implemented. The specter is converted to color using a `CIE 1931`_ color matching function (CMF). Conversion to color is performed as follows. Specter data is first converted to XYZ color space using the `CIE 1931`_ standard observer (5 nm tabulated) color matching function data. Then the image is converted to RGB color space (using a D65 reference white point) as specified in the `sRGB`_ standard (see `sRGB`_ wiki pages for details on sRGB color space). Data values are then clipped to (0.,1.) and finally, sRGB gamma transfer function is applied.


CIE 1931 standard observer
++++++++++++++++++++++++++

`CIE 1931`_ color matching function can be loaded from table with.

.. doctest::
   
   >>> import dtmm.color as dc
   >>> import numpy as np
   >>> cmf = dc.load_cmf()
   >>> cmf.shape
   (81, 3)

It is a 5nm tabulated data (between 380 and 780 nm) of 2-deg *XYZ* tristimulus values - a numerical representation of human vision system with three cones. This table is used to convert specter data to *XYZ* color space.

.. plot:: examples/color_cmf.py

   XYZ tristimulus values.

D65 standard illuminant
+++++++++++++++++++++++

CIE also defines several standard illuminants. We will work with a D65 standard illuminant, which represents natural daylight. Its XYZ tristimulus value is used as a reference white color in the `sRGB`_ standard.

.. doctest::
   
   >>> spec = dc.load_specter()

.. plot:: examples/color_D65.py

   D65 color specter from 5nm tabulated data.

XYZ Color Space
+++++++++++++++

The CMF table and D65 specter are defined so that resulting RGB image gives a white color.  To convert specter to XYZ color space the specter dimensions has to match CMF table dimensions. CIE 1931 CMF is defined between 380 and 780 nm, while the D65 specter is defined between 300 and 830 nm. Let us match the specter to CMF by interpolating D65 tabulated data at CMF wavelengths:

.. doctest::

   >>> wavelengths, cmf = dc.load_cmf(retx = True)
   >>> spec = dc.load_specter(wavelengths)

Now we can convert the specter to XYZ value with:

.. doctest::

   >>> dc.spec2xyz(spec,cmf)
   array([ 2008.69027494,  2113.45495097,  2301.13095117])

Typically you will want to work with a normalized specter:

.. doctest::

   >>> spec = dc.normalize_specter(spec,cmf)
   >>> xyz = dc.spec2xyz(spec,cmf)
   >>> xyz
   array([ 0.95042966,  1.        ,  1.08880057])

Here we have normalized the specter so that the resulting XYZ value has the Y component equal to 1 (full brightness). 

SRGB Color Space
++++++++++++++++

Resulting XYZ can be converted to sRGB (using sRGB color primaries) with

.. doctest::

   >>> linear_rgb = dc.xyz2srgb(xyz)
   >>> linear_rgb
   array([ 0.99988402,  1.00003784,  0.99996664])
  
Because we have used a D65 specter data to compute the XYZ tristimulus values, the resulting RGB equals full brightness white color [1,1,1] (small deviation comes from the numerical precision of the XYZ2RGB color matrix transform). Note that Color matrices in the standard are defined for 8bit transformation. When converting float values to unsigned integer (8bit mode), these values have to be multiplied with 255 and clipped to a range of [0,255]. Finally, we have to apply sRGB gamma curve to have this linear data ready to display on a sRGB monitor.

.. doctest::

   >>> rgb = dc.apply_srgb_gamma(linear_rgb)

Since conversion to sRGB color space (from the input specter values) is a standard operation, there is a helper function to perform this transformation in a single call:

.. doctest::

   >>> rgb2 = dc.specter2color(spec,cmf)
   >>> np.allclose(rgb,rgb2)
   True

Transmission CMF
++++++++++++++++

We can define a transmission color matching function. The idea is to have the CMF function defined for a transmission coefficients for a specific illumination so that the transmission computation becomes independent on the actual light spectra used in the experiment. For example, say we have computed transmission coefficients for a given set of wavelengths

.. doctest::

   >>> wavelengths = [380,480,580,680,780]
   >>> coefficients = [1,1,1,1,1]

We would like to construct a color matching function that will convert these coefficient to color, assuming a given light spectrum. We can build a transmission color matching function with

.. doctest::

   >>> tcmf = dc.cmf2tcmf(cmf, spec)

or we could have loaded this directly with:

.. doctest::

   >>> tcmf2 = dc.load_tcmf()
   >>> np.allclose(tcmf,tcmf2)
   True

.. plot:: examples/color_tcmf.py

   D65-normalized XYZ tristimulus values.

this way we defined a new CMF function that converts unity transmission curve to bright white color (We are using D65 illuminant here).

.. doctest::

   >>> rgb3 = dc.specter2color([1]*81,tcmf)
   >>> import numpy as np
   >>> np.allclose(rgb,rgb3)
   True

All fair, but we would not like to compute transmission coefficients at all 81 wavelengths defined in the original CMF data. We need to integrate the CMF function 


.. doctest::

   >>> itcmf = dc.integrate_data(wavelengths, np.linspace(380,780,81), tcmf)

which results in a new CMF function applicable to transmission coefficients defined at new  (different) wavelengths

We could have built this data directly by:

.. doctest::

   >>> itcmf = dc.load_tcmf(wavelengths)

Now we can compute 

   >>> rgb4 = dc.specter2color(coefficients,itcmf)
   >>> import numpy as np
   >>> np.allclose(rgb,rgb4)
   True

Color Rendering
+++++++++++++++

Not all colors can be displayed on a sRGB monitor. Colors that are out of gamut (R,G,B) chanels are larger than 1. or smaller than 0. are clipped. For instance, a D65 light that gives (R,G,B) = (1,1,1)* intensity filtered with a 150 nm band-pass filter already has colors clipped at some higher values of intensities. These colors are more vivid and saturated at light intensity of 1. 


.. plot:: examples/color_bandpass_filter.py
   
   An example of color rendering of a D65 illuminant filtered with a band-pass filter. If the illuminant is too bright, color clipping may occur. 

Also, with sRGB color space we cannot render all colors, especially in the green part of the spectrum. For example, let us compute RGB values of a D65 light filtered with a band-pass filter between 500 and 550 nm.

.. doctest::

   >>> tcmf = dc.load_tcmf([500,550])
   >>> xyz = dc.spec2xyz([1.,1.], tcmf)
   >>> rgb = dc.xyz2srgb(xyz)
   >>> rgb
   array([-0.37267476,  0.67704885, -0.0234957 ])

gives a strong negative value in the red channel, which shows that the color is too saturated to be displayed in a sRGB color space. After we apply gamma (which clips the RGB channels to (0,1.)) we get

.. doctest::

   >>> dc.apply_srgb_gamma(rgb)
   array([ 0.        ,  0.84176254,  0.        ])

with the blue and red channel clipped. We should have used wide-gamut color space and a monitor capable of displaying wider gamuts to display this color properly. As stated already, this package was not intended to be a full color management system and you should use your own CMS system if you need more complex color transforms and rendering.

.. _`CIE 1931`: https://en.wikipedia.org/wiki/CIE_1931_color_space
.. _`sRGB`: https://en.wikipedia.org/wiki/SRGB


