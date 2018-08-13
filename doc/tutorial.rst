.. _tutorial:

Tutorial
========

Interference and reflections
----------------------------

Background
++++++++++

Dealing with interference can be tricky. The `DTMM` method is an adapted 4x4 transfer 
matrix method which includes interference, however, one needs to perform multiple passes (iterations) to compute the reflections off the material. With transfer matrix method one computes the output field (the forward propagating part and the backward propagating part) given the defined input field. In a typical experiment, the forward propagating part of the input field is known - this is the input illumination light. However, the backward propagating part of the input field is not know (the reflection off the surface of the material) and it must be determined. 

The procedure to calculate reflections is as follows. First the method computes the output field by assuming zero reflections, so that the input field has no back propagating part. When light is transferred through the material, we compute both the forward and the backward propagating part of the output field. The back propagating part of the output field cancels all reflected waves from the material and the input light has no back propagating component of the field. To calculate reflections off the surface one needs to do it iteratively:

* Initially input light is defined with zero reflections and transferred through the stack to obtain the output field.
* After the first pass (first field transfer calculation), output light is modified so that the back propagating part of the field is completely removed. Then this modified light is transferred again through the stack in backward direction to obtain the modified input light which includes reflections.
* After the second pass. Input light is modified to so that forward propagating part of the field matches the initial field, and the field is transferred through the stack again..

For low reflective material, three passes are usually enough to obtain a reasonable accurate reflection and transmission values. However, in highly reflective media (cholesterics) more passes are needed.

The calculation is done by setting the `npass` and `norm` arguments::

>>> field_data_out = dtmm.transfer_field(field_data_in, optical_data, npass = 3, norm = 2)

The `npass` argument defines number of passes (field transfers). You are advised to use odd number of passes when dealing with reflections. With odd passes you can inspect any residual back propagating field left in the output field, to make sure that the method has converged.

In highly reflective media, the solution may not converge. You must play with the `norm` argument, which defines how the output field is modified after each even pass. 

* with `norm` = 0 the back propagating part is simply removed, and the total intensity of the forward propagating part is rescaled to conserve total power flow. This method works well for weak reflections.
* with `norm` = 1 the back propagating part is removed, and the amplitude of the fourier coefficients of the forward propagating part are modified so that power flow of each of the modes is conserved. This is how reflections should be treated in homogeneous layers (or nearly homogeneous layers). This method work well in most cases, especially when reflections come from mostly the top and bottom surfaces.
* with `norm` = 2, during each even step, a reference non-reflecting and non-interfering wave is transferred through the stack. This reference wave is then used to normalize the forward propagating part of the output field. Because of the additional reference wave calculation this procedure is slower, but it was found to work well in any material (even cholesterics).

Examples
++++++++

Surface reflections
'''''''''''''''''''

In this example we calculate reflection and transmission of a spatially narrow light beam (bandpassed white light 500-600 nm) that passes a two micron thick isotropic layer of high refractive index of n = 4 at an angle of beta = 0.4. Here `norm` = 1 works best. Already at three passes, the residual data is almost gone.

One clearly sees beam walking and multiple reflections and interference from both surfaces. See the `examples/reflection_isolayer.py` for details.

.. plot:: examples/reflection_isolayer.py

   Reflection and transmission of an off-axis (beta = 0.4) light beam from a single layer of two micron thick high refractive index material (n=4). Intensity is increased to a value of 100, to see the multiple reflected waves,


Cholesterics
''''''''''''

In this example, we use multiple passes to compute reflections of the cholesteric
droplet. For cholesterics one should take the `norm` = 2 argument in the
computation of the tranfered field.

The droplet is a left-handed cholesteric with pitch of 350 nm, which results in a strong reflection of right-handed light of wavelength 520 nm (350*1.5 nm). Already with `npass` = 3, the residual field has almost vanished.

In the example below, we simulated propagation of right-handed light with beta parameter `beta` = 0.2. See the `examples/cholesteric_droplet.py` for details.


.. plot:: examples/cholesteric_droplet.py

   Reflection and transmission properties of a cholesterol droplet.



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
   array([2008.69027494, 2113.45495097, 2301.13095117])

Typically you will want to work with a normalized specter:

.. doctest::

   >>> spec = dc.normalize_specter(spec,cmf)
   >>> xyz = dc.spec2xyz(spec,cmf)
   >>> xyz
   array([0.95042966, 1.        , 1.08880057])

Here we have normalized the specter so that the resulting XYZ value has the Y component equal to 1 (full brightness). 

SRGB Color Space
++++++++++++++++

Resulting XYZ can be converted to sRGB (using sRGB color primaries) with

.. doctest::

   >>> linear_rgb = dc.xyz2srgb(xyz)
   >>> linear_rgb
   array([0.99988402, 1.00003784, 0.99996664])
  
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
   array([0.        , 0.84176254, 0.        ])

with the blue and red channel clipped. We should have used wide-gamut color space and a monitor capable of displaying wider gamuts to display this color properly. As stated already, this package was not intended to be a full color management system and you should use your own CMS system if you need more complex color transforms and rendering.



.. _`CIE 1931`: https://en.wikipedia.org/wiki/CIE_1931_color_space
.. _`sRGB`: https://en.wikipedia.org/wiki/SRGB


