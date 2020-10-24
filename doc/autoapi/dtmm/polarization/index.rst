:mod:`dtmm.polarization`
========================

.. py:module:: dtmm.polarization

.. autoapi-nested-parse::

   Field vector polarization matrix functions.



Module Contents
---------------

.. function:: mode_jonesmat4x4(shape, ks, jmat, epsv=(1.0, 1.0, 1.0), epsa=(0.0, 0.0, 0.0), betamax=BETAMAX)

   Returns a mode polarizer that should be applied in fft space. This is the
   most general polarizer that does not introduce any reflections for any kind
   of field data.


.. function:: ray_jonesmat4x4(jmat, beta=0, phi=0, epsv=(1.0, 1.0, 1.0), epsa=(0.0, 0.0, 0.0))

   Returns a ray polarizer that should be applied in real space. Good for
   beams that can be approximated with a single wave vector and with a direction of
   ray propagation beta and phi parameters.

   See also mode_polarizer, which is for non-planewave field data.


.. function:: normal_polarizer(jones=(1, 0))

   A 4x4 polarizer for normal incidence light. It works reasonably well also
   for off-axis light, but it introduces weak reflections and depolarization.

   For off-axis planewaves you should use ray_polarizer instead of this.


.. function:: apply_mode_polarizer(pmat, field, out=None)

   Multiplies mode polarizer with field data in fft space.


