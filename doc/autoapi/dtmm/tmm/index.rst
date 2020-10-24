:mod:`dtmm.tmm`
===============

.. py:module:: dtmm.tmm

.. autoapi-nested-parse::

   4x4 and 2x2 transfer matrix method functions for 1D calculation.

   The implementation is based on standard formulation of 4x4 transfer matrix method.

   4x4 method
   ----------

   Layers are stacked in the z direction, field vectors describing the field are
   f = (Ex,Hy,Ey,Hx), Core functionality is defined by field matrix calculation
   functions:

   Field vector creation/conversion functions
   ++++++++++++++++++++++++++++++++++++++++++

   * :func:`.avec` for amplitude vector (eigenmode amplitudes).
   * :func:`.fvec` for field vector creation,
   * :func:`.avec2fvec` for amplitude to field conversion.
   * :func:`.fvec2avec` for field to amplitude conversion.

   Field matrix functions
   ++++++++++++++++++++++

   * :func:`.f_iso` for input and output field matrix caluclation.
   * :func:`.ffi_iso` computes the inverse of the field matrix.
   * :func:`.alphaf` for general field vectors and field coefficents calcualtion.
   * :func:`.alphaffi` computes the inverse of the field matrix.
   * :func:`.phase_mat` for phase matrix calculation.

   Layer/stack computation
   +++++++++++++++++++++++

   * :func:`.layer_mat` for layer matrix calculation Mi=Fi.Pi.Fi^-1
   * :func:`.stack_mat` for stack matrix caluclation M = M1.M2.M3....
   * :func:`.system_mat` for system matrix calculation Fin^-1.M.Fout

   Transmission/reflection calculation
   +++++++++++++++++++++++++++++++++++

   * :func:`.transmit4x4` to work with the computed system  matrix
   * :func:`.transfer4x4` or :func:`.transfer` for higher level interface

   Polarization handling and analysis
   ++++++++++++++++++++++++++++++++++

   * :func:`.polarizer4x4` to apply polarizer.
   * :func:`.jonesmat4x4` to apply general jones matrix.

   Intensity and Ez Hz field
   +++++++++++++++++++++++++

   * :func:`.poynting` the z component of the Poynting vector.
   * :func:`.intensity` the absolute value of the Poytning vector.
   * :func:`.EHz` for calculation of the z component of the E and H fields.

   2x2 method
   ----------

   todo..



Module Contents
---------------

.. function:: alphaf(beta=None, phi=None, epsv=None, epsa=None, out=None)

   Computes alpha and field arrays (eigen values and eigen vectors arrays).

   Broadcasting rules apply.

   :param beta: The beta parameter of the field (defaults to 0.)
   :type beta: float, optional
   :param phi: The phi parameter of the field (defaults to 0.)
   :type phi: float, optional
   :param epsv: Dielectric tensor eigenvalues array (defaults to unity).
   :type epsv: (...,3) array, optional
   :param epsa: Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
   :type epsa: (...,3) array, optional
   :param out: Output arrays.
   :type out: (ndarray,ndarray), optional

   :returns: **alpha, fieldmat** -- Eigen values and eigen vectors arrays.
   :rtype: (ndarray, ndarray)


.. function:: alphaffi(beta=None, phi=None, epsv=None, epsa=None, out=None)

   Computes alpha and field arrays (eigen values and eigen vectors arrays)
   and inverse of the field array. See :func:`alphaf` for details

   Broadcasting rules apply.

   :param beta: The beta parameter of the field (defaults to 0.)
   :type beta: float, optional
   :param phi: The phi parameter of the field (defaults to 0.)
   :type phi: float, optional
   :param epsv: Dielectric tensor eigenvalues array (defaults to unity).
   :type epsv: (...,3) array, optional
   :param epsa: Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
   :type epsa: (...,3) array, optional
   :param out: Output arrays.
   :type out: (ndarray,ndarray,ndarray), optional

   :returns: **alpha, field, ifield** -- Eigen values and eigen vectors arrays and its inverse
   :rtype: (ndarray, ndarray, ndarray)

   .. rubric:: Examples

   This is equivalent to

   >>> alpha,field = alphaf(0,0, [2,2,2], [0.,0.,0.])
   >>> ifield = inv(field)


.. function:: phase_mat(alpha, kd, mode=None, out=None)

   Computes phase a 4x4 or 2x2 matrix from eigenvalue matrix alpha
   and wavenumber

   :param alpha: The eigenvalue alpha array.
   :type alpha: (...,4) array
   :param kd: The kd phase value (layer thickness times wavenumber in vacuum).
   :type kd: float
   :param mode: Either +1, for forward propagating mode, or -1 for negative propagating mode.
   :type mode: int
   :param out: Output array where results are written.
   :type out: ndarray, optional


.. function:: poynting(fvec, out)

   Calculates a z-component of the poynting vector from the field vector

   :param fvec: Field matrix array.
   :type fvec: (...,4,4) array
   :param out: Output array where results are written.
   :type out: ndarray, optional


.. function:: intensity(fvec, out=None)

   Calculates absolute value of the z-component of the poynting vector

   :param fvec: Field matrix array.
   :type fvec: (...,4,4) array
   :param out: Output array where results are written.
   :type out: ndarray, optional


.. function:: EHz(fvec, beta=None, phi=None, epsv=None, epsa=None, out=None)

   Constructs the z component of the electric and magnetic fields

   :param fvec: Field matrix array.
   :type fvec: (...,4,4) array
   :param beta: The beta parameter of the field (defaults to 0.)
   :type beta: float, optional
   :param phi: The phi parameter of the field (defaults to 0.)
   :type phi: float, optional
   :param epsv: Dielectric tensor eigenvalues array (defaults to unity).
   :type epsv: (...,3) array, optional
   :param epsa: Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
   :type epsa: (...,3) array, optional
   :param out: Output arrays where results are written.
   :type out: (ndarray,ndarray), optional

   :returns: **Ez,Hz** -- Ez and Hz arrays of shape (...,4)
   :rtype: (ndarray,ndarray)


.. function:: f_iso(beta=0.0, phi=0.0, n=1.0)

   Returns field matrix for isotropic layer of a given refractive index
   and beta, phi parameters

   :param beta: The beta parameter of the field (defaults to 0.)
   :type beta: float, optional
   :param phi: The phi parameter of the field (defaults to 0.)
   :type phi: float, optional
   :param n: Refractive index of the medium (1. by default).
   :type n: float


.. function:: ffi_iso(beta=0.0, phi=0.0, n=1)

   Returns field matrix and inverse of the field matrix for isotropic layer
   of a given refractive index and beta, phi parameters

   :param beta: The beta parameter of the field (defaults to 0.)
   :type beta: float, optional
   :param phi: The phi parameter of the field (defaults to 0.)
   :type phi: float, optional
   :param n: Refractive index of the medium (1. by default).
   :type n: float


.. function:: layer_mat(kd, epsv, epsa, beta=0, phi=0, cfact=0.1, method='4x4', fmatin=None, retfmat=False, out=None)

   Computes characteristic matrix of a single layer M=F.P.Fi,

   Numpy broadcasting rules apply

   :param kd: A sequence of phase values (layer thickness times wavenumber in vacuum).
              len(kd) must match len(epsv) and len(epsa).
   :type kd: float
   :param epsv: Dielectric tensor eigenvalues array (defaults to unity).
   :type epsv: (...,3) array, optional
   :param epsa: Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
   :type epsa: (...,3) array, optional
   :param beta: The beta parameter of the field (defaults to 0.)
   :type beta: float, optional
   :param phi: The phi parameter of the field (defaults to 0.)
   :type phi: float, optional
   :param cfact: Coherence factor, only used in combination with `4x4_2` method.
   :type cfact: float, optional
   :param method: One of 4x4 (4x4 berreman - trasnmittance + reflectance),
                  2x2 (2x2 jones - transmittance only),
                  4x4_1 (4x4, single reflections - transmittance only),
                  2x2_1 (2x2, single reflections - transmittance only)
                  4x4_2 (4x4, partially coherent reflections - transmittance only)
   :type method: str
   :param fmatin: Used in compination with 2x2_1 method. Itspecifies the field matrix of
                  the input media in order to compute fresnel reflections. If not provided
                  it reverts to 2x2 with no reflections.
   :type fmatin: ndarray, optional
   :param out:
   :type out: ndarray, optional

   :returns: **cmat** -- Characteristic matrix of the layer.
   :rtype: ndarray


.. function:: stack_mat(kd, epsv, epsa, beta=0, phi=0, cfact=0.01, method='4x4', out=None)

   Computes a stack characteristic matrix M = M_1.M_2....M_n if method is
   4x4, 4x2(2x4) and a characteristic matrix M = M_n...M_2.M_1 if method is
   2x2.

   Note that this function calls :func:`layer_mat`, so numpy broadcasting
   rules apply to kd[i], epsv[i], epsa[i], beta and phi.

   :param kd: A sequence of phase values (layer thickness times wavenumber in vacuum).
              len(kd) must match len(epsv) and len(epsa).
   :type kd: array_like
   :param epsv: Dielectric tensor eigenvalues array (defaults to unity).
   :type epsv: (...,3) array, optional
   :param epsa: Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
   :type epsa: (...,3) array, optional
   :param beta: The beta parameter of the field (defaults to 0.)
   :type beta: float, optional
   :param phi: The phi parameter of the field (defaults to 0.)
   :type phi: float, optional
   :param cfact: Coherence factor, only used in combination with `4x4_r` and `4x4_2` methods.
   :type cfact: float
   :param method: One of 4x4 (4x4 berreman), 2x2 (2x2 jones),
                  4x4_1 (4x4, single reflections), 2x2_1 (2x2, single reflections)
                  4x4_r (4x4, incoherent to compute reflection) or
                  4x4_t (4x4, incoherent to compute transmission)
   :type method: str
   :param out:
   :type out: ndarray, optional

   :returns: **cmat** -- Characteristic matrix of the stack.
   :rtype: ndarray


.. function:: system_mat(cmat=None, fmatin=None, fmatout=None, fmatini=None, out=None)

   Computes a system matrix from a characteristic matrix Fin-1.C.Fout

   :param cmat: Characteristic matrix.
   :type cmat: (...,4,4) array
   :param fmatin: Input field matrix array.
   :type fmatin: (...,4,4) array
   :param fmatout: Output field matrix array.
   :type fmatout: (...,4,4) array
   :param fmatini: Inverse of the input field matrix array.
   :type fmatini: (...,4,4) array
   :param out: Output array where results are written.
   :type out: ndarray, optional


.. function:: transmit4x4(fvec_in, cmat=None, fmatin=None, fmatout=None, fmatini=None, fmatouti=None, fvec_out=None)

   Transmits field vector using 4x4 method.

   This functions takes a field vector that describes the input field and
   computes the output transmited field and also updates the input field
   with the reflected waves.

   :param fvec_in: Input field vector array. This function will update the input array
                   with the calculated reflected field
   :type fvec_in: (...,4) array
   :param cmat: Characteristic matrix.
   :type cmat: (...,4,4) array
   :param fmatin: Input field matrix array.
   :type fmatin: (...,4,4) array
   :param fmatout: Output field matrix array.
   :type fmatout: (...,4,4) array
   :param fmatini: Inverse of the input field matrix array.
   :type fmatini: (...,4,4) array
   :param fmatouti: Inverse of the output field matrix array. If not provided, it is computed
                    from `fmatout`.
   :type fmatouti: (...,4,4) array, optional
   :param fvec_out: The ouptut field vector array. This function will update the output array
                    with the calculated transmitted field.
   :type fvec_out: (...,4) array, optional


.. function:: transfer4x4(fvec_in, kd, epsv, epsa, beta=0.0, phi=0.0, nin=1.0, nout=1.0, method='4x4', reflect_in=False, reflect_out=False, fvec_out=None)

   tranfers 4x4 field

   :param fvec_in: Input field vector array. This function will update the input array
                   with the calculated reflected field
   :type fvec_in: (...,4) array
   :param kd: The kd phase value (layer thickness times wavenumber in vacuum).
   :type kd: float
   :param epsv: Dielectric tensor eigenvalues array (defaults to unity).
   :type epsv: (...,3) array, optional
   :param epsa: Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
   :type epsa: (...,3) array, optional
   :param beta: The beta parameter of the field (defaults to 0.)
   :type beta: float, optional
   :param phi: The phi parameter of the field (defaults to 0.)
   :type phi: float, optional
   :param nin: Input layer refractive index.
   :type nin: float
   :param nout: Output layer refractive index.
   :type nout: float
   :param method: Any of 4x4, 4x4_1, 4x4_2, 4x4_r.
   :type method: str
   :param reflect_in: Defines how to treat reflections from the input media and the first layer.
                      If specified it does an incoherent reflection from the first interface.
   :type reflect_in: bool
   :param reflect_out: Defines how to treat reflections from the last layer and the output media.
                       If specified it does an incoherent reflection from the last interface.
   :type reflect_out: bool
   :param fvec_out: The ouptut field vector array. This function will update the output array
                    with the calculated transmitted field.
   :type fvec_out: (...,4) array, optional


.. function:: transfer(fvec_in, kd, epsv, epsa, beta=0.0, phi=0.0, nin=1.0, nout=1.0, method='2x2', reflect_in=False, reflect_out=False, fvec_out=None)

   Transfer input field vector through a layered material specified by the propagation
   constand k*d, eps tensor (epsv, epsa) and input and output isotropic media.

   :param fvec_in: Input field vector array. This function will update the input array
                   with the calculated reflected field
   :type fvec_in: (...,4) array
   :param kd: The kd phase value (layer thickness times wavenumber in vacuum).
   :type kd: float
   :param epsv: Dielectric tensor eigenvalues array (defaults to unity).
   :type epsv: (...,3) array, optional
   :param epsa: Euler rotation angles (psi, theta, phi) (defaults to (0.,0.,0.)).
   :type epsa: (...,3) array, optional
   :param beta: The beta parameter of the field (defaults to 0.)
   :type beta: float, optional
   :param phi: The phi parameter of the field (defaults to 0.)
   :type phi: float, optional
   :param nin: Input layer refractive index.
   :type nin: float
   :param nout: Output layer refractive index.
   :type nout: float
   :param method: Any of 4x4, 2x2, 2x2_1 or 4x4_1, 4x4_2, 4x4_r
   :type method: str
   :param reflect_in: Defines how to treat reflections from the input media and the first layer.
                      If specified it does an incoherent reflection from the first interface.
   :type reflect_in: bool
   :param reflect_out: Defines how to treat reflections from the last layer and the output media.
                       If specified it does an incoherent reflection from the last interface.
   :type reflect_out: bool
   :param fvec_out: The ouptut field vector array. This function will update the output array
                    with the calculated transmitted field.
   :type fvec_out: (...,4) array, optional


.. function:: polarizer4x4(jvec, fmat, out=None)

   Returns a polarizer matrix from a given jones vector and a field matrix.

   Numpy broadcasting rules apply.

   :param jvec: A length two array describing the jones vector. Jones vector should
                be normalized.
   :type jvec: array_like
   :param fmat: A field matrix array of the isotropic medium.
   :type fmat: array_like
   :param out: Output array
   :type out: ndarray, optional

   .. rubric:: Examples

   >>> f = f_iso(n = 1.)
   >>> jvec = dtmm.jones.jonesvec((1,0))
   >>> pol_mat = polarizer4x4(jvec, f) #x polarizer matrix


.. function:: jonesmat4x4(jmat, fmat, out=None)

   Returns a 4x4 jones matrix from a given 2x2 jones matrix and a field matrix.

   Numpy broadcasting rules apply.

   :param jmat: A 2x2 jones matrix. Any of matrices in :mod:`dtmm.jones` can be used.
   :type jmat: (...,2,2) array
   :param fmat: A field matrix array of the isotropic medium.
   :type fmat: (...,4,4) array
   :param out: Output array
   :type out: ndarray, optional


.. function:: avec(jvec=(1, 0), amplitude=1.0, mode=+1, out=None)

   Constructs amplitude vector.

   Numpy broadcasting rules apply for jones, and amplitude parameters

   :param jvec: A jones vector, describing the polarization state of the field.
   :type jvec: jonesvec
   :param amplitude: Amplitude of the field.
   :type amplitude: complex
   :param mode: Either +1, for forward propagating mode, or -1 for negative propagating mode.
   :type mode: int
   :param out: Output array where results are written.
   :type out: ndarray, optional

   :returns: **avec** -- Amplitude vector of shape (4,).
   :rtype: ndarray

   .. rubric:: Examples

   X polarized light with amplitude = 1
   >>> avec()
   array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])

   X polarized light with amplitude 1 and y polarized light with amplitude 2.
   >>> b = avec(jones = ((1,0),(0,1)),amplitude = (1,2))
   >>> b[0]
   array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
   >>> b[1]
   array([0.+0.j, 0.+0.j, 2.+0.j, 0.+0.j])


.. function:: fvec(fmat, jvec=(1, 0), amplitude=1.0, mode=+1, out=None)

   Build field vector form a given polarization state, amplitude and mode.

   This function calls avec and then avec2fvec, see avec for details.

   :param fmat: Field matrix array.
   :type fmat: (...,4,4) array
   :param jvec: A jones vector, describing the polarization state of the field.
   :type jvec: jonesvec
   :param amplitude: Amplitude of the field.
   :type amplitude: complex
   :param mode: Either +1, for forward propagating mode, or -1 for negative propagating mode.
   :type mode: int
   :param out: Output array where results are written.
   :type out: ndarray, optional

   .. rubric:: Examples

   X polarized light traveling at beta = 0.4 and phi = 0.2 in medium with n = 1.5

   >>> fmat = f_iso(beta = 0.4, phi = 0.2, n = 1.5)
   >>> m = fvec(fmat, jones = jonesvec((1,0), phi = 0.2))

   This is equivalent to

   >>> a = avec(jones = jonesvec((1,0), phi = 0.2))
   >>> ma = avec2fvec(a,fmat)
   >>> np.allclose(ma,m)
   True


.. function:: fvec2avec(fvec, fmat, normalize_fmat=True, out=None)

   Converts field vector to amplitude vector

   :param fvec: Input field vector
   :type fvec: ndarray
   :param fmat: Field matrix
   :type fmat: ndarray
   :param normalize_fmat: Setting this to false will not normalize the field matrix. In this case
                          user has to make sure that the normalization of the field matrix has
                          been performed prior to calling this function by calling normalize_f.
   :type normalize_fmat: bool, optional
   :param out: Output array
   :type out: ndarray, optional

   :returns: **avec** -- Amplitude vector
   :rtype: ndarray


.. function:: avec2fvec(avec, fmat, normalize_fmat=True, out=None)

   Converts amplitude vector to field vector

   :param avec: Input amplitude vector
   :type avec: ndarray
   :param fmat: Field matrix
   :type fmat: ndarray
   :param normalize_fmat: Setting this to false will not normalize the field matrix. In this case
                          user has to make sure that the normalization of the field matrix has
                          been performed prior to calling this function by calling normalize_f.
   :type normalize_fmat: bool, optional
   :param out: Output array
   :type out: ndarray, optional

   :returns: **fvec** -- Field vector.
   :rtype: ndarray


