:mod:`dtmm.data`
================

.. py:module:: dtmm.data

.. autoapi-nested-parse::

   Director and optical data creation and IO functions.



Module Contents
---------------

.. function:: read_director(file, shape, dtype=FDTYPE, sep='', endian=sys.byteorder, order='zyxn', nvec='xyz')

   Reads raw director data from a binary or text file.

   A convinient way to read director data from file.

   :param file: Open file object or filename.
   :type file: str or file
   :param shape: Shape of the data array, e.g., ``(50, 24, 34, 3)``
   :type shape: sequence of ints
   :param dtype: Data type of the raw data. It is used to determine the size of the items
                 in the file.
   :type dtype: data-type
   :param sep: Separator between items if file is a text file.
               Empty ("") separator means the file should be treated as binary.
               Spaces (" ") in the separator match zero or more whitespace characters.
               A separator consisting only of spaces must match at least one
               whitespace.
   :type sep: str
   :param endian: Endianess of the data in file, e.g. 'little' or 'big'. If endian is
                  specified and it is different than sys.endian, data is byteswapped.
                  By default no byteswapping is done.
   :type endian: str, optional
   :param order: Data order. It can be any permutation of 'xyzn'. Defaults to 'zyxn'. It
                 describes what are the meaning of axes in data.
   :type order: str, optional
   :param nvec: Order of the director data coordinates. Any permutation of 'x', 'y' and
                'z', e.g. 'yxz', 'zxy' ...
   :type nvec: str, optional


.. function:: read_Q(file, shape, dtype=FDTYPE, sep='', endian=sys.byteorder, order='zyxn')

   Reads raw Q tensor data from a binary or text file.

   A convinient way to read Q tensor data from file. Q tensor is assumed to be


   :param file: Open file object or filename.
   :type file: str or file
   :param shape: Shape of the data array, e.g., ``(50, 24, 34, 6)``
   :type shape: sequence of ints
   :param dtype: Data type of the raw data. It is used to determine the size of the items
                 in the file.
   :type dtype: data-type
   :param sep: Separator between items if file is a text file.
               Empty ("") separator means the file should be treated as binary.
               Spaces (" ") in the separator match zero or more whitespace characters.
               A separator consisting only of spaces must match at least one
               whitespace.
   :type sep: str
   :param endian: Endianess of the data in file, e.g. 'little' or 'big'. If endian is
                  specified and it is different than sys.endian, data is byteswapped.
                  By default no byteswapping is done.
   :type endian: str, optional
   :param order: Data order. It can be any permutation of 'xyzn'. Defaults to 'zyxn'. It
                 describes what are the meaning of axes in data.
   :type order: str, optional


.. function:: rotate_director(rmat, data, method='linear', fill_value=(0.0, 0.0, 0.0), norm=True, out=None)

   Rotate a director field around the center of the compute box by a specified
   rotation matrix. This rotation is lossy, as datapoints are interpolated.
   The shape of the output remains the same.

   :param rmat: A 3x3 rotation matrix.
   :type rmat: array_like
   :param data: Array specifying director field with ndim = 4
   :type data: array_like
   :param method: Interpolation method "linear" or "nearest"
   :type method: str
   :param fill_value: If provided, the values (length 3 vector) to use for points outside of the
                      interpolation domain. Defaults to (0.,0.,0.).
   :type fill_value: numbers, optional
   :param norm: Whether to normalize the length of the director to 1. after rotation
                (interpolation) is performed. Because of interpolation error, the length
                of the director changes slightly, and this options adds a constant
                length constraint to reduce the error.
   :type norm: bool,
   :param out: Output array.
   :type out: ndarray, optional

   :returns: **y** -- A rotated director field
   :rtype: ndarray

   .. seealso::

      :func:`data.rot90_director`
          a lossless rotation by 90 degrees.


.. function:: rot90_director(data, axis='+x', out=None)

   Rotate a director field by 90 degrees around the specified axis.

   :param data: Array specifying director field with ndim = 4.
   :type data: array_like
   :param axis: Axis around which to perform rotation. Can be in the form of
                '[s][n]X' where the optional parameter 's' can be "+" or "-" decribing
                the sign of rotation. [n] is an integer describing number of rotations
                to perform, and 'X' is one of 'x', 'y' 'z', and defines rotation axis.
   :type axis: str
   :param out: Output array.
   :type out: ndarray, optional

   :returns: **y** -- A rotated director field
   :rtype: ndarray

   .. seealso::

      :func:`data.rotate_director`
          a general rotation for arbitrary angle.


.. function:: director2data(director, mask=None, no=1.5, ne=1.6, nhost=None, thickness=None)

   Builds optical data from director data. Director length is treated as
   an order parameter. Order parameter of S=1 means that refractive indices
   `no` and `ne` are set as the material parameters. With S!=1, a
   :func:`uniaxial_order` is used to calculate actual material parameters.

   :param director: A 4D array describing the director
   :type director: ndarray
   :param mask: If provided, this mask must be a 3D bolean mask that define voxels where
                nematic is present. This mask is used to define the nematic part of the sample.
                Volume not defined by the mask is treated as a host material. If mask is
                not provided, all data points are treated as a director.
   :type mask: ndarray, optional
   :param no: Ordinary refractive index
   :type no: float
   :param ne: Extraordinary refractive index
   :type ne: float
   :param nhost: Host refracitve index (if mask is provided)
   :type nhost: float
   :param thickness: Thickness of layers (in pixels). If not provided, this defaults to ones.
   :type thickness: ndarray


.. function:: validate_optical_data(data, homogeneous=False)

   Validates optical data.

   This function inspects validity of the optical data, and makes proper data
   conversions to match the optical data format. In case data is not valid and
   it cannot be converted to a valid data it raises an exception (ValueError).

   :param data: A valid optical data tuple.
   :type data: tuple of optical data
   :param homogeneous: Whether data is for a homogenous layer. (Inhomogeneous by defult)
   :type homogeneous: bool, optional

   :returns: **data** -- Validated optical data tuple.
   :rtype: tuple


.. function:: raw2director(data, order='zyxn', nvec='xyz')

   Converts raw data to director array.

   :param data: Data array
   :type data: array
   :param order: Data order. It can be any permutation of 'xyzn'. Defaults to 'zyxn'. It
                 describes what are the meaning of axes in data.
   :type order: str, optional
   :param nvec: Order of the director data coordinates. Any permutation of 'x', 'y' and
                'z', e.g. 'yxz', 'zxy'. Defaults to 'xyz'
   :type nvec: str, optional

   :returns: **director** -- A new array or same array (if no trasposing and data copying was made)
   :rtype: array

   .. rubric:: Example

   >>> a = np.random.randn(10,11,12,3)
   >>> director = raw2director(a, "xyzn")


.. function:: read_raw(file, shape, dtype, sep='', endian=sys.byteorder)

   Reads raw data from a binary or text file.

   :param file: Open file object or filename.
   :type file: str or file
   :param shape: Shape of the data array, e.g., ``(50, 24, 34, 3)``
   :type shape: sequence of ints
   :param dtype: Data type of the raw data. It is used to determine the size of the items
                 in the file.
   :type dtype: data-type
   :param sep: Separator between items if file is a text file.
               Empty ("") separator means the file should be treated as binary.
               Spaces (" ") in the separator match zero or more whitespace characters.
               A separator consisting only of spaces must match at least one
               whitespace.
   :type sep: str
   :param endian: Endianess of the data in file, e.g. 'little' or 'big'. If endian is
                  specified and it is different than sys.endian, data is byteswapped.
                  By default no byteswapping is done.
   :type endian: str, optional


.. function:: sphere_mask(shape, radius, offset=(0, 0, 0))

   Returns a bool mask array that defines a sphere.

   The resulting bool array will have ones (True) insede the sphere
   and zeros (False) outside of the sphere that is centered in the compute
   box center.

   :param shape: A tuple of (nlayers, height, width) defining the bounding box of the sphere.
   :type shape: (int,int,int)
   :param radius: Radius of the sphere in pixels.
   :type radius: int
   :param offset: Offset of the sphere from the center of the bounding box. The coordinates
                  are (x,y,z).
   :type offset: (int, int, int), optional

   :returns: **out** -- Bool array defining the sphere.
   :rtype: array


.. function:: nematic_droplet_director(shape, radius, profile='r', retmask=False)

   Returns nematic director data of a nematic droplet with a given radius.

   :param shape: (nz,nx,ny) shape of the output data box. First dimension is the
                 number of layers, second and third are the x and y dimensions of the box.
   :type shape: tuple
   :param radius: radius of the droplet.
   :type radius: float
   :param profile: Director profile type. It can be a radial profile "r", or homeotropic
                   profile with director orientation specified with the parameter "x", "y",
                   or "z", or as a director tuple e.g. (np.sin(0.2),0,np.cos(0.2)). Note that
                   director length  defines order parameter (S=1 for this example).
   :type profile: str, optional
   :param retmask: Whether to output mask data as well
   :type retmask: bool, optional

   :returns: **out** -- A director data array, or tuple of director mask and director data arrays.
   :rtype: array or tuple of arrays


.. function:: cholesteric_director(shape, pitch, hand='left')

   Returns a cholesteric director data.

   :param shape: (nz,nx,ny) shape of the output data box. First dimension is the
                 number of layers, second and third are the x and y dimensions of the box.
   :type shape: tuple
   :param pitch: Cholesteric pitch in pixel units.
   :type pitch: float
   :param hand: Handedness of the pitch; either 'left' (default) or 'right'
   :type hand: str, optional

   :returns: **out** -- A director data array
   :rtype: ndarray


.. function:: nematic_droplet_data(shape, radius, profile='r', no=1.5, ne=1.6, nhost=1.5)

   Returns nematic droplet optical_data.

   This function returns a thickness,  material_eps, angles, info tuple
   of a nematic droplet, suitable for light propagation calculation tests.

   :param shape: (nz,nx,ny) shape of the stack. First dimension is the number of layers,
                 second and third are the x and y dimensions of the compute box.
   :type shape: tuple
   :param radius: radius of the droplet.
   :type radius: float
   :param profile: Director profile type. It can be a radial profile "r", or homeotropic
                   profile with director orientation specified with the parameter "x",
                   "y", or "z".
   :type profile: str, optional
   :param no: Ordinary refractive index of the material (1.5 by default)
   :type no: float, optional
   :param ne: Extraordinary refractive index (1.6 by default)
   :type ne: float, optional
   :param nhost: Host material refractive index (1.5 by default)
   :type nhost: float, optional

   :returns: **out** -- A (thickness, material_eps, angles) tuple of three arrays
   :rtype: tuple of length 3


.. function:: cholesteric_droplet_data(shape, radius, pitch, hand='left', no=1.5, ne=1.6, nhost=1.5)

   Returns cholesteric droplet optical_data.

   This function returns a thickness,  material_eps, angles, info tuple
   of a cholesteric droplet, suitable for light propagation calculation tests.

   :param shape: (nz,nx,ny) shape of the stack. First dimension is the number of layers,
                 second and third are the x and y dimensions of the compute box.
   :type shape: tuple
   :param radius: radius of the droplet.
   :type radius: float
   :param pitch: Cholesteric pitch in pixel units.
   :type pitch: float
   :param hand: Handedness of the pitch; either 'left' (default) or 'right'
   :type hand: str, optional
   :param no: Ordinary refractive index of the material (1.5 by default)
   :type no: float, optional
   :param ne: Extraordinary refractive index (1.6 by default)
   :type ne: float, optional
   :param nhost: Host material refractive index (1.5 by default)
   :type nhost: float, optional

   :returns: **out** -- A (thickness, material_eps, angles) tuple of three arrays
   :rtype: tuple of length 3


.. function:: director2order(data, out)

   Converts director data to order parameter (length of the director)


.. function:: director2angles(data, out)

   Converts director data to angles (yaw, theta phi)


.. function:: angles2director(data, out)

   Converts angles data (yaw,theta,phi) to director (nx,ny,nz)


.. function:: expand(data, shape, xoff=None, yoff=None, zoff=None, fill_value=0.0)

   Creates a new scalar or vector field data with an expanded volume.
   Missing data points are filled with fill_value. Output data shape
   must be larger than the original data.

   :param data: Input vector or scalar field data
   :type data: array_like
   :param shape: A scalar or length 3 vector that defines the volume of the output data
   :type shape: array_like
   :param xoff: Data offset value in the x direction. If provided, original data is
                copied to new data starting at this offset value. If not provided, data
                is copied symmetrically (default).
   :type xoff: int, optional
   :param yoff, int, optional: Data offset value in the x direction.
   :param zoff, int, optional: Data offset value in the z direction.
   :param fill_value: A length 3 vector of default values for the border volume data points.
   :type fill_value: array_like

   :returns: **y** -- Expanded ouput data
   :rtype: array_like


.. function:: refind2eps(refind)

   Converts refractive index to epsilon


.. function:: uniaxial_order(order, eps, out)

   uniaxial_order(order, eps)

   Calculates uniaxial dielectric tensor of a material with a given orientational order parameter
   from a diagonal dielectric (eps) tensor of the same material with perfect order (order = 1)

   >>> uniaxial_order(0,[1,2,3.])
   array([ 2.+0.j,  2.+0.j,  2.+0.j])
   >>> uniaxial_order(1,[1,2,3.])
   array([ 1.5+0.j,  1.5+0.j,  3.0+0.j])


.. function:: save_stack(file, optical_data)

   Saves optical data to a binary file in ``.dtms`` format.

   :param file: File or filename to which the data is saved.  If file is a file-object,
                then the filename is unchanged.  If file is a string, a ``.dtms``
                extension will be appended to the file name if it does not already
                have one.
   :type file: file, str
   :param optical_data: A valid optical data
   :type optical_data: optical data tuple


.. function:: load_stack(file)

   Load optical data from a file.

   :param file: The file to read.
   :type file: file, str


.. function:: director2Q(director)

   Computes Q tensor form the uniaxial director. The length of the director is
   the order parameter


.. function:: Q2director(qtensor, qlength=False)

   Computes the director form the traceless q tensor


.. function:: Q2eps(qtensor, no=1.5, ne=1.6, scale_factor=1.0, out=None)

   Converts Q tensor to epsilon tensor


.. function:: eps2epsva(eps)

   Computes epsilon eigenvalues (epsv) and rotation angles (epsa) from
   epsilon tensor of shape (...,6) or represented as a (...,3,3)

   :param eps:
   :type eps: (...,3,3) or (...,6) symmetric tensor

   :returns: **epsv, epsa** -- Eigenvalues and Euler angles arrays.
   :rtype: ndarray, ndarray


