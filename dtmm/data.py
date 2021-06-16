"""
Director and optical data creation and IO functions.

Conversion functions
--------------------

* :func:`.director2data` generates optical block data from director.
* :func:`.Q2data` generates optical block data from the Q tensor.
* :func:`.director2order` computes order parameter from directo length.
* :func:`.director2Q` computes uniaxial Q tensor.
* :func:`.Q2director` computes an effective uniaxial director from a biaxial Q tensor.
* :func:`.director2angles` computes Euler rotation angles from the director.
* :func:`.angles2director` computes the director from Euler rotation angles.
* :func:`.Q2eps` converts Q tensor to epsilon tensor.
* :func:`.eps2epsva` converts epsilon tensor to epsilon eigenvalues and Euler angles.
* :func:`.epsva2eps` converts epsilon eigenvalues and Euler angles to epsilon tensor.
* :func:`.tensor2matrix` convert tensor to matrix.
* :func:`.matrix2tensor` convert matrix to tensor.
* :func:`.refind2eps` convert refractive index eigenvalues to epsilon eigenvalues.
* :func:`.uniaxial_order` creates uniaxial tensor from a biaxial eigenvalues.
* :func:`.sellmeier2eps` computes epsilon from Sellmeier coefficients.
* :func:`.cauchy2eps` computes epsilon from Cauchy coefficients.

IO functions
------------

* :func:`.read_director` reads 3D director data.
* :func:`.read_tensor` reads 3D tensor data.
* :func:`.read_raw` reads any raw data.
* :func:`.load_stack` loads optical data.
* :func:`.save_stack` saves optical data.

Data creation
-------------

* :func:`.validate_optical_block` validates data.
* :func:`.validate_optical_data` validates data.
* :func:`.evaluate_optical_block` evaluates data.
* :func:`.evaluate_optical_data` evaluates data.
* :func:`.eig_symmetry` creates effective tensor of given symmetry.
* :func:`.effective_block` computes effective (mean layers) 1D data from 3D data.
* :func:`.effective_data` computes effective (mean layers) 1D data from 3D data.
* :func:`.split_block` Splits block data into layersed data
* :func:`.layerd_data` Splits optical data into layered data.

Utilities and sample data
-------------------------

* :func:`.nematic_droplet_data` builds sample data.
* :func:`.nematic_droplet_director` builds sample director.
* :func:`.cholesteric_droplet_data` builds sample data.
* :func:`.cholesteric_director` builds sample director.
* :func:`.expand` expands director data.
* :func:`.sphere_mask` builds a masking array.
* :func:`.rot90_director` rotates director by 90 degrees
* :func:`.rotate_director` rotates data by any angle
"""

import numpy as np
import numba
import sys

from dtmm.conf import FDTYPE, CDTYPE, NFDTYPE, NCDTYPE, NUMBA_CACHE,\
NF32DTYPE,NF64DTYPE,NC128DTYPE,NC64DTYPE, DTMMConfig, deprecation
from dtmm.rotation import rotation_matrix_x,rotation_matrix_y,rotation_matrix_z, rotate_vector, rotation_angles, rotation_matrix, rotate_diagonal_tensor
from dtmm.wave import betaphi, k0
from dtmm.fft import fft2, ifft2
from dtmm.linalg import tensor_eig

def read_director(file, shape, dtype = FDTYPE,  sep = "", endian = sys.byteorder, order = "zyxn", nvec = "xyz"):
    """Reads raw director data from a binary or text file. 
    
    A convinient way to read director data from file. 
    
    Parameters
    ----------
    file : str or file
        Open file object or filename.
    shape : sequence of ints
        Shape of the data array, e.g., ``(50, 24, 34, 3)``
    dtype : data-type
        Data type of the raw data. It is used to determine the size of the items 
        in the file.
    sep : str
        Separator between items if file is a text file.
        Empty ("") separator means the file should be treated as binary.
        Spaces (" ") in the separator match zero or more whitespace characters.
        A separator consisting only of spaces must match at least one
        whitespace.
    endian : str, optional
        Endianess of the data in file, e.g. 'little' or 'big'. If endian is 
        specified and it is different than sys.endian, data is byteswapped. 
        By default no byteswapping is done. 
    order : str, optional
        Data order. It can be any permutation of 'xyzn'. Defaults to 'zyxn'. It
        describes what are the meaning of axes in data.
    nvec : str, optional
        Order of the director data coordinates. Any permutation of 'x', 'y' and 
        'z', e.g. 'yxz', 'zxy' ... 
    """
    try:
        i,j,k,c = shape
    except:
        raise TypeError("shape must be director data shape (z,y,x,n)")
    data = read_raw(file, shape, dtype, sep = sep, endian = endian)
    return raw2director(data, order, nvec)

def read_tensor(file, shape, dtype = FDTYPE,  sep = "", endian = sys.byteorder, order = "zyxn"):
    """Reads raw tensor data from a binary or text file. 
    
    A convinient way to read tensor data from file.

    
    Parameters
    ----------
    file : str or file
        Open file object or filename.
    shape : sequence of ints
        Shape of the data array, e.g., ``(50, 24, 34, 6)``
    dtype : data-type
        Data type of the raw data. It is used to determine the size of the items 
        in the file.
    sep : str
        Separator between items if file is a text file.
        Empty ("") separator means the file should be treated as binary.
        Spaces (" ") in the separator match zero or more whitespace characters.
        A separator consisting only of spaces must match at least one
        whitespace.
    endian : str, optional
        Endianess of the data in file, e.g. 'little' or 'big'. If endian is 
        specified and it is different than sys.endian, data is byteswapped. 
        By default no byteswapping is done. 
    order : str, optional
        Data order. It can be any permutation of 'xyzn'. Defaults to 'zyxn'. It
        describes what are the meaning of axes in data.
    """
    try:
        i,j,k,c = shape
    except:
        raise TypeError("shape must be 3D tensor data shape (z,y,x,n)")
    data = read_raw(file, shape, dtype, sep = sep, endian = endian)
    return raw2director(data, order) #no swapping of Q tensor elements, so we can use raw2director

def rotate_director(rmat, data, method = "linear",  fill_value = (0.,0.,0.), norm = True, out = None):
    """
    Rotate a director field around the center of the compute box by a specified
    rotation matrix. This rotation is lossy, as datapoints are interpolated.
    The shape of the output remains the same.
    
    Parameters
    ----------
    rmat : array_like
        A 3x3 rotation matrix.
    data: array_like
        Array specifying director field with ndim = 4
    method : str
        Interpolation method "linear" or "nearest"
    fill_value : numbers, optional
        If provided, the values (length 3 vector) to use for points outside of the
        interpolation domain. Defaults to (0.,0.,0.).
    norm : bool,
        Whether to normalize the length of the director to 1. after rotation 
        (interpolation) is performed. Because of interpolation error, the length 
        of the director changes slightly, and this options adds a constant 
        length constraint to reduce the error.
    out : ndarray, optional
        Output array.
        
    Returns
    -------
    y : ndarray
        A rotated director field
        
    See Also
    --------   
    data.rot90_director : a lossless rotation by 90 degrees.
        
    """
    
    from scipy.interpolate import RegularGridInterpolator
    verbose_level = DTMMConfig.verbose
    if verbose_level >0:
        print("Rotating director.")   
        
    out = np.empty_like(data)
    nz,ny,nx,nv = data.shape
    shape = (nz,ny,nx)
    az, ay, ax = [np.arange(-l / 2. + .5, l / 2. + .5) for l in shape]
    
    fillx, filly, fillz = fill_value
    xdir = RegularGridInterpolator((az, ay, ax), data[...,0], 
              fill_value = fillx,bounds_error = False, method = method)
    ydir = RegularGridInterpolator((az, ay, ax), data[...,1], 
              fill_value = filly,bounds_error = False, method = method)
    zdir = RegularGridInterpolator((az, ay, ax), data[...,2], 
              fill_value = fillz,bounds_error = False, method = method)
    zz,yy,xx = np.meshgrid(az,ay,ax, indexing = "ij", copy = False, sparse = True)

    out[...,0] = xx
    out[...,1] = yy
    out[...,2] = zz

    out = rotate_vector(rmat.T,out, out) #rotate coordinates

    #out2 = out.copy()
    #out2[...,0] = out[...,2]
    #out2[...,2] = out[...,0]
    
    out2 = out[...,::-1] #reverse direction instead of copying
    
    #interpolate new director field
    xnew = xdir(out2) 
    ynew = ydir(out2)
    znew = zdir(out2)
    
    out[...,0] = xnew 
    out[...,1] = ynew
    out[...,2] = znew
    
    out = rotate_vector(rmat,out, out) #rotate vector in each voxel
    
    if norm == True:
        s = director2order(out)
        mask = (s == 0.)
        s[mask] = 1.
        return np.divide(out, s[...,None], out)
    return out

def rot90_director(data,axis = "+x", out = None):
    """
    Rotate a director field by 90 degrees around the specified axis.
    
    Parameters
    ----------
    data: array_like
        Array specifying director field with ndim = 4.
    axis: str
        Axis around which to perform rotation. Can be in the form of
        '[s][n]X' where the optional parameter 's' can be "+" or "-" decribing 
        the sign of rotation. [n] is an integer describing number of rotations 
        to perform, and 'X' is one of 'x', 'y' 'z', and defines rotation axis.
    out : ndarray, optional
        Output array.
    
    Returns
    -------
    y : ndarray
        A rotated director field
        
    See Also
    --------   
    data.rotate_director : a general rotation for arbitrary angle.        
    """
    nz,ny,nx,nv = data.shape

    axis_name = axis[-1]
    try:
        k = int(axis[:-1])
    except ValueError:
        k = int(axis[:-1]+"1")
    angle = np.pi/2*k
    if axis_name == "x":
        r = rotation_matrix_x(angle)
        axes = (1,0)
    elif axis_name == "y":
        r = rotation_matrix_y(angle)
        axes = (0,2)
    elif axis_name == "z":
        r = rotation_matrix_z(angle)
        axes = (2,1)
    else:
        raise ValueError("Unknown axis type {}".format( axis_name))
    data_rot = np.rot90(data,k = k, axes = axes)#first rotate data points
    return rotate_vector(r,data_rot,out)#rotate vector in each voxel
    
    
def director2data(director, mask = None, no = 1.5, ne = 1.6, nhost = None,scale_factor = 1.,
                  thickness = None):
    """Builds optical block data from director data. Director length is treated as
    an order parameter. Order parameter of S=1 means that refractive indices
    `no` and `ne` are set as the material parameters. With S!=1, a 
    :func:`uniaxial_order` is used to calculate actual material parameters.
    
    Parameters
    ----------
    director : ndarray
        A 4D array describing the director
    mask : ndarray, optional
        If provided, this mask must be a 3D bolean mask that define voxels where
        nematic is present. This mask is used to define the nematic part of the sample. 
        Volume not defined by the mask is treated as a host material. If mask is 
        not provided, all data points are treated as a director.
    no : float
        Ordinary refractive index
    ne : float
        Extraordinary refractive index 
    nhost : float
        Host refracitve index (if mask is provided)
    scale_factor : float
        The order parameter S obtained from the director length is scaled by this factor. 
        Optical anisotropy is then `epsa = S/scale_factor * (epse - epso)`.
    thickness : ndarray
        Thickness of layers (in pixels). If not provided, this defaults to ones.
        
    Returns
    -------
    optical_data : tuple
        A valid optical data tuple.
        
    """
    material = np.empty(shape = director.shape, dtype = FDTYPE)
    material[...] = refind2eps([no,no,ne])[None,...] 
    material = uniaxial_order(director2order(director)/scale_factor, material, out = material)
    
    if mask is not None:
        material[np.logical_not(mask),:] = refind2eps([nhost,nhost,nhost])[None,...] 
        
    if thickness is None:
        thickness = np.ones(shape = (material.shape[0],))
    return (thickness, material, director2angles(director))

def Q2data(tensor, mask = None, no = 1.5, ne = 1.6, nhost = None,scale_factor = 1., 
           biaxial = False, thickness = None):
    """Builds optical block data from Q tensor data. 
    
    Parameters
    ----------
    tensor : (...,6) or (...,3,3) array
        Q tensor with elements Q[0,0], Q[1,1], Q[2,2], Q[0,1], Q[0,2], Q[1,2]
    mask : ndarray, optional
        If provided, this mask must be a 3D bolean mask that define voxels where
        nematic is present. This mask is used to define the nematic part of the sample. 
        Volume not defined by the mask is treated as a host material. If mask is 
        not provided, all data points are treated as a director.
    no : float
        Ordinary refractive index (when biaxial = False)
    ne : float
        Extraordinary refractive index (when biaxial = False)
    nhost : float
        Host refracitve index (if mask is provided)
    scale_factor : float
        The order parameter S obtained from the Q tensor is scaled by this factor. 
        Optical anisotropy is then `epsa = S/scale_factor *(epse - epso)`.
    biaxial : bool
        Describes whether data is treated as biaxial or converted to uniaxial (default).
        If biaxial, no describes the mean value of n1 and n2 refractive indices
        eigenavalues and ne = n3.
    thickness : ndarray, optional
        Thickness of layers (in pixels). If not provided, this defaults to ones.
    
    Returns
    -------
    optical_data : list
        A valid optical data list.
    """
    tensor = np.asarray(tensor)
    eps = Q2eps(tensor, no = no, ne = ne,scale_factor = scale_factor)
    
    shape = eps.shape[:-1]
    if eps.shape[-1] == 6:
        shape = shape + (3,) 
        
    material, epsa = eps2epsva(eps)
            
    if not bool(biaxial):
        material = uniaxial_order(1, material, out = material)
    
    if mask is not None:
        material[np.logical_not(mask),:] = refind2eps([nhost,nhost,nhost])[None,...] 
        
    if thickness is None:
        thickness = np.ones(shape = (material.shape[0],))
    return  (thickness, material, epsa)


def director2Q(director, order = 1.):
    """Computes Q tensor form the uniaxial director. 
    
    Parameters
    ----------
    director : (...,3) array
        Director vector. The length of the vector is the square of the order 
        parameter.
    order : float, optional
        In case director is normalized, this describes the order parameter.
        
    Returns
    -------
    Q : (...,6) array
        Q tensor with elements Q[0,0], Q[1,1], Q[2,2], Q[0,1], Q[0,2], Q[1,2]
    """
    S = director2order(director)
    n = director
    
    out = np.empty(S.shape + (6,), S.dtype)
    
    d = S/3*order

    out[...,0] = order*n[...,0]**2 - d
    out[...,1] = order*n[...,1]**2 - d
    out[...,2] = order*n[...,2]**2 - d
    out[...,3] = order*n[...,0]*n[...,1]
    out[...,4] = order*n[...,0]*n[...,2]
    out[...,5] = order*n[...,1]*n[...,2]
    
    return out

               
def Q2director(tensor, qlength = False):
    """Computes the director from tensor data 
    
    Parameters
    ----------
    tensor : (...,6) or (...,3,3) array
        Tensor data.
    qlength : bool
        Specifies whether the length of the director is set to the S parameter
        or not. If not, director is scalled to unity length.
        
    Returns
    -------
    Q : (...,6) array
        Q tensor with elements Q[0,0], Q[1,1], Q[2,2], Q[0,1], Q[0,2], Q[1,2]
    """
    qeig, r = tensor_eig(tensor) #sorted eigenvalues.. qeig[2] is for the main axis (director)
    S = qeig[...,2] * 3/2. #the S parameter of the uniaxial q tensor
    if qlength:
        return S[...,None] * r[...,2]
    else:
        return r[...,2]
            
def Q2eps(tensor, no = 1.5, ne = 1.6,scale_factor = 1., out = None):
    """Converts Q tensor to epsilon tensor
    
    Parameters
    ----------
    tensor : (...,6) or (...,3,3) ndarray
        A 4D array describing the Q tensor. If provided as a matrix, rhe elemnents are
        Q[0,0], Q[1,1], Q[2,2], Q[0,1], Q[0,2], Q[1,2]
    no : float
        Ordinary refractive index when S = 1/scale_factor.
    ne : float
        Extraordinary refractive index when S = 1/scale_factor.
    scale_factor : float
        The order parameter S obtained from the Q tensor is scaled by this factor. 
        Optical anisotropy is then `epsa = S/scale_factor * (epse - epso)`.
    out : ndarray, optional
        Output array 
    
    Returns
    -------
    eps : ndarray
        Calculated epsilon tensor.
    """

    qtensor = np.asarray(tensor)

    if qtensor.shape[-1] == 6:
        qdiag = qtensor[...,0:3]
        qoff = qtensor[...,3:]
    elif qtensor.shape[-2:] == (3,3):
        mask_off = np.array(((False,True,True),(False,False,True),(False,False,False)))
        mask_diag = np.array(((True,False,False),(False,True,False),(False,False,True)))

        qdiag = qtensor[...,mask_diag]
        qoff = qtensor[...,mask_off]
    else:
        raise ValueError("input tensor must be an array of shape (...,6)")
    epse = ne**2
    epso = no**2
    if out is None:
        out = np.empty_like(qtensor)
    #: scaled anisotropy
    epsa = (epse-epso) / scale_factor
    #: mean epsilon
    epsm = (epso*2 + epse)/3.
    
    if qtensor.shape[-1] == 6:
        out[...,0:3] = epsa * qdiag + epsm
        out[...,3:] = epsa * qoff
    else:
        out[...,mask_diag] = epsa * qdiag + epsm
        out[...,mask_off] = epsa * qoff
        out[...,1,0] = out[...,0,1]
        out[...,2,0] = out[...,0,2]
        out[...,2,1] = out[...,1,2]
    return out
  
def eps2epsva(eps):
    """Computes epsilon eigenvalues (epsv) and rotation angles (epsa) from
    epsilon tensor of shape (...,6) or represented as a (...,3,3)
    
    Parameters
    ----------
    eps : (...,3,3) or (...,6) array
       Epsilon tensor. If provided as a (3,3) matrix, the elements are
       eps[0,0], eps[1,1], eps[2,2], eps[0,1], eps[0,2], eps[1,2]

    Returns
    -------
    epsv, epsa : ndarray, ndarray
        Eigenvalues and Euler angles arrays.
    """
    epsv, r = tensor_eig(eps)
    # r is in general complex for complex eps. But, if a complex tensor is a rotated diagonal,
    # the eigenvectors should be real. Test it here.
    atol = 1e-8 if CDTYPE == "complex128" else 1e-5
    rtol = 1e-5 if CDTYPE == "complex128" else 1e-3
    if not np.allclose(r,r.real, atol = atol, rtol = rtol):
        import warnings
        warnings.warn("Input tensor is not normal because eigevectors are not real. Results are unpredictive!", stacklevel = 2)
        
    return epsv, rotation_angles(r.real)

def epsva2eps(epsv,epsa):
    """Computes epsilon from eigenvalues (epsv) and rotation angles (epsa) 
    
    Parameters
    ----------
    epsv : (...,3) array
       Epsilon eigenvalues array.
    epsa : (...,3) array
       Euler angles array.

    Returns
    -------
    eps : ndarray
        Epsilon tensor arrays of shape (...,6).  The elements are
       eps[0,0], eps[1,1], eps[2,2], eps[0,1], eps[0,2], eps[1,2]
    """
    r = rotation_matrix(epsa)
    return rotate_diagonal_tensor(r,epsv)

def validate_optical_layer(data, shape = None, wavelength = None, broadcast = False, copy = False):
    """Convenience function. See validate_optical_block for details.
    
    Calls validate_optical_block with single_layer = True"""
    return validate_optical_block(data, shape = shape, wavelength = wavelength, \
                    broadcast = broadcast, copy = copy, single_layer = True)

        
def validate_optical_block(data, shape = None, wavelength = None, broadcast = False, copy = False, homogeneous = None, single_layer = False):
    """Validates optical block.
    
    This function inspects validity of the optical block data, and makes proper data
    conversions to match the optical block format. In case data is not valid and 
    it cannot be converted to a valid block data it raises an exception (ValueError). 
    
    Parameters
    ----------
    data : tuple or list of tuples
        An optical data: either a single optical block tuple, or a list of block tuples.
    shape : (int,int)
        If defined input epsilon tensor shape (eps.shape[:-1]) is checked to 
        be broadcastable with the given shape.
    wavelength : float
        Wavelength in nanometers at which to compute epsilon, in case epsv is 
        a callable function.
    broadcast : bool, optional
        Ehether to actually perform broadcasting of arrays. If set to False, only
        tests whether arrays can broadcast to a common shape, but no broadcasting
        is made. 
    copy: bool, optional
        Whether to copy data. If not set, (broadcasted) view of the input arrays
        is returned.
    single_layer : bool
        If set to True, input data has to be a single layer data. Validated data
        is not converted to optical block of length 1.
        
    Returns
    -------
    data : tuple
        Validated optical block tuple. 
    """
    if homogeneous is not None:
        import warnings
        warnings.warn("homogeneous argument is not used any more and it will be removed in future versions", DeprecationWarning)
    
    thickness, material, angles = data
    
    #for dispersive data, copy the coefficients, or evaluate if wavelength is provided.
    dispersive_material = None
    if is_callable(material):
        if isinstance(material, EpsilonDispersive) and wavelength is None:
            dispersive_material = material
            material = dispersive_material.coefficients
        else:
        #if material is callable, obtain eps values by calling the function
            if wavelength is None:
                raise ValueError("Epsilon is a callable. You must provide wavelength!")
            material = material(wavelength)   
    
    thickness = np.asarray(thickness, dtype = FDTYPE)
    material = np.asarray(material)
    if np.issubdtype(material.dtype, np.complexfloating):
        material = np.asarray(material, dtype = CDTYPE)
    else:
        material = np.asarray(material, dtype = FDTYPE)
    angles = np.asarray(angles, dtype = FDTYPE)
    
    if thickness.ndim == 0 and not single_layer:
        thickness = thickness[None] #make it 1D
        material = material[None,...]
        angles = angles[None,...]
    elif thickness.ndim != 1 and not single_layer:
        raise ValueError("Thickess dimension should be 1.")
    
    n = None if single_layer else len(thickness)
    layer_dim = 0 if n is None else 1
    
    if shape is None:
        angles_broadcast_shape = (1,1,3) if n is None else (n,1,1,3)
    else:
        height,width = shape
        angles_broadcast_shape = (height,width,3) if n is None else (n, height, width,3)
    if dispersive_material is None:
        #regular data, angles and epsilon data have same shape
        material_broadcast_shape = angles_broadcast_shape
        if material.ndim == 1 + layer_dim:
            material = material[...,None,None,:]
        elif material.ndim == 2 + layer_dim:
            material = material[...,None,:,:]
        elif material.ndim != 3 + layer_dim:
            raise ValueError("Invalid material dimensions.")
    else:
        #coefficients-based data has one extra dimension
        material_broadcast_shape = angles_broadcast_shape + (1,)
        if material.ndim == 2 + layer_dim:
            material = material[...,None,None,:,:]
        elif material.ndim == 3 + layer_dim:
            material = material[...,None,:,:,:]
        elif material.ndim != 4 + layer_dim:
            raise ValueError("Invalid material coefficients dimensions.")        

    if angles.ndim == 1 + layer_dim:
        angles = angles[...,None,None,:]
    elif angles.ndim == 2 + layer_dim:
        angles = angles[...,None,:,:]
    elif angles.ndim != 3 +layer_dim:
        raise ValueError("Invalid angles dimensions.")
    
    material_shape = np.broadcast_shapes(material.shape, material_broadcast_shape)
    
    if broadcast:
        material = np.broadcast_to(material, material_shape)
    
    angles_shape = np.broadcast_shapes(angles.shape, angles_broadcast_shape)
    if broadcast:
        angles = np.broadcast_to(angles, angles_shape)
    if copy:
        thickness, material, angles = thickness.copy(), material.copy(), angles.copy()
    #convert coefficients back to dispersive epsilon
    if dispersive_material is not None:
        material = dispersive_material.__class__(material)
    
    return thickness, material, angles

def validate_optical_data(data, shape = None, wavelength = None, broadcast = False, copy = False, homogeneous = None):
    """Validates optical data.
    
    This function inspects validity of the optical data, and makes proper data
    conversions to match the optical data format. In case data is not valid and 
    it cannot be converted to a valid optical data it raises an exception (ValueError). 
    
    Parameters
    ----------
    data : tuple or list of tuples
        An optical data: either a single optical block tuple, or a list of block tuples.
    shape : (int,int)
        Each block's epsilon tensor shape (eps.shape[:-1]) is checked to 
        be broadcastable with the given shape.
    wavelength : float
        Wavelength in nanometers at which to compute epsilon, in case epsv is 
        a callable function.
    broadcast : bool, optional
        Ehether to actually perform broadcasting of arrays. If set to False, only
        tests whether arrays can broadcast to a common shape, but no broadcasting
        is made. 
    copy: bool, optional
        Whether to copy data. If not set, (broadcasted) view of the input arrays
        is returned.
        
    Returns
    -------
    data : list of tuples
        Validated optical data list. 
    """
    if isinstance(data, list):
        if shape is None:
            raise ValueError("For heterogeneous data, you must provide the `shape` argument.")
        return [validate_optical_block(d, shape, wavelength, broadcast, copy) for d in data]
    else:
        import warnings
        warnings.warn("A single-block optical data must be a list of length 1. Converting optical data to list. In the future, exception will be raised.", DeprecationWarning)
        return validate_optical_data([data], shape, wavelength, broadcast, copy)

def evaluate_optical_block(optical_block, wavelength = 550):
    """In case of dispersive material. This function evaluates and returns optical block at a given wavelength"""
    d, epsv, epsa = optical_block
    if is_callable(epsv):
        epsv = epsv(wavelength)
    return d, epsv, epsa
    
def evaluate_optical_data(optical_data, wavelength = 550):
    """In case of dispersive material. This function evaluates and return optical data at a given wavelength"""
    return [evaluate_optical_block(block,wavelength) for block in optical_data]
    

def raw2director(data, order = "zyxn", nvec = "xyz"):
    """Converts raw data to director array.
    
    Parameters
    ----------
    data : array
        Data array
    order : str, optional
        Data order. It can be any permutation of 'xyzn'. Defaults to 'zyxn'. It
        describes what are the meaning of axes in data.
    nvec : str, optional
        Order of the director data coordinates. Any permutation of 'x', 'y' and 
        'z', e.g. 'yxz', 'zxy'. Defaults to 'xyz'  
        
    Returns
    -------
    director : array
        A new array or same array (if no trasposing and data copying was made)
        
    Example
    -------
    
    >>> a = np.random.randn(10,11,12,3)
    >>> director = raw2director(a, "xyzn")
    """
    if order != "zyxn":
        #if not in zxyn order, then we must transpose data
        try:
            axes = (order.find(c) for c in "zyxn")
            axes = tuple((i for i in axes if i != -1))
            data = np.transpose(data, axes)
        except:
            raise ValueError("Invalid value for 'order'. Must be a permutation of 'xyzn' characters")
        
    if nvec != "xyz":
        index = {"x" : 0, "y": 1, "z" : 2}
        out = np.empty_like(data)
        for i,idn in enumerate(nvec):
            j = index.pop(idn)
            out[...,j] = data[...,i]
        return out
    else:
        return data    

def read_raw(file, shape, dtype, sep = "", endian = sys.byteorder):
    """Reads raw data from a binary or text file.
    
    Parameters
    ----------
    file : str or file
        Open file object or filename.
    shape : sequence of ints
        Shape of the data array, e.g., ``(50, 24, 34, 3)``
    dtype : data-type
        Data type of the raw data. It is used to determine the size of the items 
        in the file.
    sep : str
        Separator between items if file is a text file.
        Empty ("") separator means the file should be treated as binary.
        Spaces (" ") in the separator match zero or more whitespace characters.
        A separator consisting only of spaces must match at least one
        whitespace.
    endian : str, optional
        Endianess of the data in file, e.g. 'little' or 'big'. If endian is 
        specified and it is different than sys.endian, data is byteswapped. 
        By default no byteswapping is done.
    """  
    dtype = np.dtype(dtype)
    count = np.multiply.reduce(shape) * dtype.itemsize
    a = np.fromfile(file, dtype, count, sep)
    if endian == sys.byteorder:
        return a.reshape(shape)  
    elif endian not in ("little", "big"):
        raise ValueError("Endian should be either 'little' or 'big'")
    else:
        return a.reshape(shape).byteswap(True)
       
def _r3(shape):
    """Returns r vector array of given shape."""
    az, ay, ax = [np.arange(-l / 2. + .5, l / 2. + .5) for l in shape]
    zz,yy,xx = np.meshgrid(az,ay,ax, indexing = "ij")
    return xx, yy, zz

    
def sphere_mask(shape, radius, offset = (0,0,0)):
    """Returns a bool mask array that defines a sphere.
    
    The resulting bool array will have ones (True) insede the sphere
    and zeros (False) outside of the sphere that is centered in the compute
    box center.
    
    Parameters
    ----------
    shape : (int,int,int)
        A tuple of (nlayers, height, width) defining the bounding box of the sphere.
    radius: int
        Radius of the sphere in pixels.
    offset: (int, int, int), optional
        Offset of the sphere from the center of the bounding box. The coordinates
        are (x,y,z).
        
    Returns
    -------
    out : array
        Bool array defining the sphere.
    """
    xx, yy, zz = _r3(shape)
    r = ((xx-offset[0])**2 + (yy-offset[1])**2 + (zz--offset[2])**2) ** 0.5 
    mask = (r <= radius)
    return mask   

def nematic_droplet_director(shape, radius, profile = "r", retmask = False):
    """Returns nematic director data of a nematic droplet with a given radius.
    
    Parameters
    ----------
    shape : tuple
        (nz,nx,ny) shape of the output data box. First dimension is the 
        number of layers, second and third are the x and y dimensions of the box.
    radius : float
        radius of the droplet.
    profile : str, optional
        Director profile type. It can be a radial profile "r", or homeotropic 
        profile with director orientation specified with the parameter "x", "y",
        or "z", or as a director tuple e.g. (np.sin(0.2),0,np.cos(0.2)). Note that
        director length  defines order parameter (S=1 for this example).
    retmask : bool, optional
        Whether to output mask data as well
        
    Returns
    -------
    out : array or tuple of arrays 
        A director data array, or tuple of director mask and director data arrays.
    """
    
    nz, ny, nx = shape
    out = np.zeros(shape = (nz,ny,nx,3), dtype = FDTYPE)
    xx, yy, zz = _r3(shape)
    
    r = (xx**2 + yy**2 + zz**2) ** 0.5 
    mask = (r <= radius)
    m = np.logical_and(mask,r != 0.)
    rm = r[m]
    if profile == "r":
        out[...,0][m] = xx[m]/rm
        out[...,1][m] = yy[m]/rm
        out[...,2][m] = zz[m]/rm
    elif isinstance(profile, str):
        index = {"x": 0,"y": 1,"z": 2}
        try:
            i = index[profile]
            out[...,i][m] = 1.
        except KeyError:
            raise ValueError("Unsupported profile type!")
    else:
        try:
            x,y,z = profile
            out[...,0][m] = x
            out[...,1][m] = y
            out[...,2][m] = z
        except:
            raise ValueError("Unsupported profile type!")
            
    if retmask == True:
        return mask, out
    else: 
        return out
    
def cholesteric_director(shape, pitch, hand = "left"):
    """Returns a cholesteric director data.
    
    Parameters
    ----------
    shape : tuple
        (nz,nx,ny) shape of the output data box. First dimension is the 
        number of layers, second and third are the x and y dimensions of the box.
    pitch : float
        Cholesteric pitch in pixel units.
    hand : str, optional
        Handedness of the pitch; either 'left' (default) or 'right'

    Returns
    -------
    out : ndarray
        A director data array
    """
    nz, ny, nx = shape
    
    if hand == 'left':
        phi = -2*np.pi/pitch*np.arange(nz)
    elif hand == "right":
        phi = 2*np.pi/pitch*np.arange(nz)
    else:
        raise ValueError("Unknown handedness '{}'".format(hand))
    out = np.zeros(shape = (nz,ny,nx,3), dtype = FDTYPE)

    for i in range(nz):
        out[i,...,0] = np.cos(phi[i])
        out[i,...,1] = np.sin(phi[i])
    return out    

def nematic_droplet_data(shape, radius, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5):
    """Returns nematic droplet optical block data.
    
    This function returns a thickness,  material_eps, angles, info tuple 
    of a nematic droplet, suitable for light propagation calculation tests.
    
    Parameters
    ----------
    shape : tuple
        (nz,nx,ny) shape of the stack. First dimension is the number of layers, 
        second and third are the x and y dimensions of the compute box.
    radius : float
        radius of the droplet.
    profile : str, optional
        Director profile type. It can be a radial profile "r", or homeotropic 
        profile with director orientation specified with the parameter "x", 
        "y", or "z".
    no : float, optional
        Ordinary refractive index of the material (1.5 by default)
    ne : float, optional
        Extraordinary refractive index (1.6 by default)
    nhost : float, optional
        Host material refractive index (1.5 by default)
        
    Returns
    -------
    out : tuple of length 3
        A (thickness, material_eps, angles) tuple of three arrays
    """
    mask, director = nematic_droplet_director(shape, radius, profile = profile, retmask = True)
    return director2data(director, mask = mask, no = no, ne = ne, nhost = nhost)

def cholesteric_droplet_data(shape, radius, pitch, hand = "left", no = 1.5, ne = 1.6, nhost = 1.5):
    """Returns cholesteric droplet optical block.
    
    This function returns a thickness,  material_eps, angles, info tuple 
    of a cholesteric droplet, suitable for light propagation calculation tests.
    
    Parameters
    ----------
    shape : tuple
        (nz,nx,ny) shape of the stack. First dimension is the number of layers, 
        second and third are the x and y dimensions of the compute box.
    radius : float
        radius of the droplet.
    pitch : float
        Cholesteric pitch in pixel units.
    hand : str, optional
        Handedness of the pitch; either 'left' (default) or 'right'
    no : float, optional
        Ordinary refractive index of the material (1.5 by default)
    ne : float, optional
        Extraordinary refractive index (1.6 by default)
    nhost : float, optional
        Host material refractive index (1.5 by default)
        
    Returns
    -------
    out : tuple of length 3
        A (thickness, material_eps, angles) tuple of three arrays
    """
    director = cholesteric_director(shape, pitch, hand = hand)
    mask = sphere_mask(shape, radius)
    return director2data(director, mask = mask, no = no, ne = ne, nhost = nhost)    

@numba.guvectorize([(NF32DTYPE[:],NF32DTYPE[:]),(NF64DTYPE[:],NFDTYPE[:])], "(n)->()", cache = NUMBA_CACHE)
def director2order(data, out):
    """Converts director data to order parameter (square root of the length of the director)"""
    c = data.shape[0]
    if c != 3:
        raise TypeError("invalid shape")
    x = data[0]
    y = data[1]
    z = data[2]
    s = np.sqrt(x**2+y**2+z**2)
    out[0] = s**0.5

@numba.guvectorize([(NF32DTYPE[:],NF32DTYPE[:]),(NF64DTYPE[:],NFDTYPE[:])], "(n)->(n)", cache = NUMBA_CACHE)
def director2angles(data, out):
    """Converts director data to angles (yaw, theta phi)"""
    c = data.shape[0]
    if c != 3:
        raise TypeError("invalid shape")

    x = data[0]
    y = data[1]
    z = data[2]
    phi = np.arctan2(y,x)
    theta = np.arctan2(np.sqrt(x**2+y**2),z)
    #s = np.sqrt(x**2+y**2+z**2)
    out[0] = 0. #yaw = 0.
    out[1] = theta
    out[2] = phi

@numba.guvectorize([(NF32DTYPE[:],NF32DTYPE[:]),(NF64DTYPE[:],NFDTYPE[:])], "(n)->(n)", cache = NUMBA_CACHE)
def angles2director(data, out):
    """Converts angles data (yaw,theta,phi) to director (nx,ny,nz)"""
    c = data.shape[0]
    if c != 3:
        raise TypeError("invalid shape")

    s = 1.
    theta = data[1]
    phi = data[2]

    ct = np.cos(theta)
    st = np.sin(theta)
    cf = np.cos(phi)
    sf = np.sin(phi)
    out[0] = s*cf*st
    out[1] = s*sf*st
    out[2] = s*ct

def expand(data, shape, xoff = None, yoff = None, zoff = None, fill_value = 0.):
    """Creates a new scalar or vector field data with an expanded volume. 
    Missing data points are filled with fill_value. Output data shape
    must be larger than the original data.
    
    Parameters
    ----------
    data : array_like
       Input vector or scalar field data
    shape : array_like
       A scalar or length 3 vector that defines the volume of the output data
    xoff : int, optional
       Data offset value in the x direction. If provided, original data is 
       copied to new data starting at this offset value. If not provided, data 
       is copied symmetrically (default).
    yoff, int, optional
       Data offset value in the x direction. 
    zoff, int, optional
       Data offset value in the z direction.     
    fill_value: array_like
       A length 3 vector of default values for the border volume data points.
       
    Returns
    -------
    y : array_like
       Expanded ouput data
    """
    data = np.asarray(data)
    nz,nx,ny = shape
    if nz >= data.shape[0] and ny >= data.shape[1] and nx >= data.shape[2]:
        out = np.empty(shape = shape + data.shape[3:], dtype = data.dtype)
        out[...,:] = fill_value
        if xoff is None:
            xoff = (shape[1] - data.shape[1])//2
        if yoff is None:
            yoff = (shape[2] - data.shape[2])//2
        if zoff is None:
            zoff = (shape[0] - data.shape[0])//2
    
        out[zoff:data.shape[0]+zoff,yoff:data.shape[1]+yoff,xoff:data.shape[2]+xoff,...] = data
        return out 
    else:
        raise ValueError("Requested shape {} is not larger than original data's shape".format(shape))

_REFIND_DECL = [(NF32DTYPE[:],NF32DTYPE[:]), (NF64DTYPE[:],NFDTYPE[:]),(NC64DTYPE[:],NC64DTYPE[:]), (NC128DTYPE[:],NCDTYPE[:])]

@numba.njit(_REFIND_DECL, cache = NUMBA_CACHE)  
def _refind2eps(refind, out):
    out[0] = refind[0]**2
    out[1] = refind[1]**2
    out[2] = refind[2]**2

_REFIND_DECL = [NF32DTYPE(NF32DTYPE), NFDTYPE(NF64DTYPE),NC64DTYPE(NC64DTYPE), NCDTYPE(NC128DTYPE)]

@numba.vectorize(_REFIND_DECL, cache = NUMBA_CACHE)  
def refind2eps(refind):
    """Converts refractive index to epsilon"""
    return refind**2

_SELLMEIER_DECL = [(NF32DTYPE[:],NF32DTYPE[:],NF32DTYPE[:]), (NF64DTYPE[:],NF64DTYPE[:],NFDTYPE[:]),(NC64DTYPE[:],NF32DTYPE[:],NC64DTYPE[:]), (NC128DTYPE[:],NF64DTYPE[:],NCDTYPE[:])]

@numba.guvectorize(_SELLMEIER_DECL, "(n),()->()", cache = NUMBA_CACHE)
def sellmeier2eps(coeff, wavelength, out):
    r"""Converts Sellmeier coefficents to epsilon
    
    Sellmeier formula is:
    
    eps = A + \sum_i B_i * w**2 / (w**2 - C_i)
    
    where A = coeff[0], B_1 = coeff[1], C_1 = coeff[2], ...
    and w is wavelength in microns. 
    
    Input coefficients array must be of odd length. How many elements (n) are there
    in the teries expansion of the Sellmeier formula depends on the length of
    the input coefficients n = (len(coeff)-1)//2
    """
    #number of sums in sellmeier formula
    n = (len(coeff)-1)//2
    out[0] = coeff[0] # A term
    for i in range(n):
        out[0] += (coeff[1 + i*2] * wavelength[0]**2)/(wavelength[0]**2 - coeff[2 + i*2])    

@numba.guvectorize(_SELLMEIER_DECL, "(n),()->()", cache = NUMBA_CACHE)
def cauchy2eps(coeff, wavelength, out):
    r"""Converts Cauchy coefficents to epsilon
    
    Cauchy formula is
    
    n = A + \sum_i B_i / (w**(2*i)
    
    where sumation is from i = 1 to imax = len(coeff)-1
    A = coeff[0], B_1 = coeff[1], B_2 = coeff[2], ...
    and w is wavelength in microns. 
    """
    #number of sums in sellmeier formula
    n = (len(coeff))
    out[0] = coeff[0] # A term
    for i in range(1,n):
        out[0] += (coeff[i] / wavelength[0]**(2*i))  
    out[0] = out[0]**2

def is_optical_block_dispersive(optical_block):
    """inspectes whether optical  block is dispersive or not"""
    d,epsv,epsa = optical_block
    return is_callable(epsv)   
    
def is_optical_data_dispersive(optical_data):
    """inspectes whether optical data is dispersive or not"""
    for block in optical_data:
        if is_optical_block_dispersive(block):
            return True
    #none of the blocks is dispersive, so data is not dispersive
    return False
    
class EpsilonDispersive(object):
    """Base class for all dispersive material classes.
    """
    def __init__(self, values = None, shape = None, n = None, broadcast = False, copy = False):
        """
        Parameters
        ----------
        values : array
            Coefficents array
        shape : tuple of ints
            Requested shape of the material.
        n : int
            Total number of coefficents.
        broadcast : bool, optional
            Whether to broadcast to the desired output shape. If not sethe coefficients may 
            still broadcast, so that coefficients get a minimum shape of (3,n)
        copy : bool, optional
            Whether to copy coefficients or not.
        """
        if values is not None:
            self.coefficients = np.asarray(values,FDTYPE)
        else:
            if shape is None or n is None:
                raise ValueError("Coefficients were not set, so you need to provide n and shape parameters.")

        if n is not None:
            n = int(n)
            if n < 0:
                raise ValueError("n must be >= 0")
        else:
            n = 1
            
        if shape is None:
            shape = ()
            
        cshape = shape + (3,n)
        
        if values is None:
            self.coefficients = np.empty(cshape,FDTYPE)
            self.coefficients[...,0] = 1.
            self.coefficients[...,1:] = 0.
        else:
            max_shape = np.broadcast_shapes(self.coefficients.shape, cshape)
            min_shape = np.broadcast_shapes(self.coefficients.shape, (3,n))
            
            #force broadcasting so that we have a shape of at least (3,n)
            self.coefficients = np.broadcast_to(self.coefficients, min_shape)            
            if broadcast:
                self.coefficients = np.broadcast_to(self.coefficients, max_shape)
            if copy:
                self.coefficients = self.coefficients.copy()
        
        self.dtype = FDTYPE
        
    @property    
    def shape(self):
        return self.coefficients.shape[:-1]
        
    def __getitem__(self,index):
        return self.__class__(self.coefficients[index])
                
    def __call__(self):
        import warnings
        warnings.warn("Subclass must implement __call__ method", UserWarning)
            
class EpsilonCauchy(EpsilonDispersive):
    """A callable epsilon tensor described with Cauchy coefficients
    
n = c[0] + c[1]/w**2 + c[2]/w**2 + ...
eps = n**2
    
where are c is the coefficient vector and w is wavelength in microns.
"""
      
    def __call__(self, wavelength):
        return cauchy2eps(self.coefficients,wavelength/1000)    
    
class EpsilonSellmeier(object):
    """A callable epsilon tensor described with Sellmeier coefficients.
    
eps = c[0] + w**2 * c[1] / (w**2 - c[2]) + w**2 * c[3] / (w**2 - c[4]) + ...
    
where are c is the coefficient vector and w is wavelength in microns.
"""
            
    def __call__(self, wavelength):
        return sellmeier2eps(self.coefficients,wavelength/1000)       

def is_callable(func):
    """Determines if func is a callable function or not"""
    return True if hasattr(func, "__call__") else False

_EPS_DECL = [(NF32DTYPE,NF32DTYPE[:],NF32DTYPE[:]), (NF64DTYPE, NF64DTYPE[:],NFDTYPE[:]),
             (NF32DTYPE,NC64DTYPE[:],NC64DTYPE[:]), (NF64DTYPE, NC128DTYPE[:],NCDTYPE[:])
             ]

@numba.njit(_EPS_DECL, cache = NUMBA_CACHE)
def _uniaxial_order(order, eps, out):
    if order >= 0.:
        m = (eps[0] + eps[1] + eps[2])/3.
        delta = eps[2] - (eps[0] + eps[1])/2.
        if order == 0.:
            eps1 = m
            eps3 = m
        else:
            eps1 = m - 1./3. *order * delta
            eps3 = m + 2./3. * order * delta
        eps2 = eps1
    else:
        eps1 = eps[0]
        eps2 = eps[1]
        eps3 = eps[2]

    out[0] = eps1
    out[1] = eps2
    out[2] = eps3

    
_EPS_DECL_VEC = [(NF32DTYPE[:],NF32DTYPE[:],NF32DTYPE[:]), (NF64DTYPE[:], NF64DTYPE[:],NFDTYPE[:]),
             (NF32DTYPE[:],NC64DTYPE[:],NC64DTYPE[:]), (NF64DTYPE[:], NC128DTYPE[:],NCDTYPE[:])
             ]
@numba.guvectorize(_EPS_DECL_VEC ,"(),(n)->(n)", cache = NUMBA_CACHE)
def uniaxial_order(order, eig, out):
    """
    uniaxial_order(order, eps)
    
    Calculates uniaxial (or isotropic) eigen tensor from a diagonal biaxial eigen tensor.
    
    Parameters
    ----------
    order : float
        The order parameter. 1.: uniaxial, 0.: isotropic. If order is negative
        no change is made (biaxial case). 
    eig : array
        Array of shape (...,3), the eigenvalues. The eigenvalue [2] is treated 
        as the extraordinary axis for the uniaxial order. 
    out :ndarray, optional
        Output array.
        
    Returns
    -------
    out : ndarray
        Effective eigenvalues based on the provided symmetry (order) argument    
    
    Examples
    --------
    >>> np.allclose(uniaxial_order(0,[1,2,3.]) , (2,2,2)) 
    True
    >>> np.allclose(uniaxial_order(1,[1,2,3.]), (1.5,1.5,3)) 
    True
    >>> np.allclose(uniaxial_order(-1,[1,2,3.]), (1,2,3))  #negative, so no change
    True
    >>> np.allclose(uniaxial_order(0.5,[1,2,3.]), (1.75,1.75,2.5)) #scale accordingly
    True
    """
    assert eig.shape[0] in (3,)
    _uniaxial_order(order[0], eig, out)
       
def eig_symmetry(order, eig, out = None):
    """Takes the ordered diagonal values of the tensor and converts it to 
    uniaxial or isotropic tensor, or keeps it as biaxial.

    Broadcasting rules apply.
    
    Parameters
    ----------
    order : int or array 
        Integer describing the symmetry 0 : isotropic, 1 : uniaxial, 2 : biaxial.
        If specified as an array it mast be broadcastable. See :func:`uniaxial_order`.
    eig : array
        Array of shape (...,3), the eigenvalues. The eigenvalue [2] is treated 
        as the extraordinary axis for the uniaxial order. 
    out :ndarray, optional
        Output array.
        
    Returns
    -------
    out : ndarray
        Effective eigenvalues based on the provided symmetry (order) argument
        
    See Also
    --------
    :func:`.uniaxial_order` for scalled order adjustment.
    """
    order = np.asarray(order, int)
    mask = order > 1
    #if order is negative... it is not applied in uniaxial_order function... so it remains biaxial
    order[mask] = -1
    return uniaxial_order(order ,eig, out)  
    

def filter_block(optical_block, wavelength, pixelsize, betamax = 1, symmetry = "isotropic"):
    d, epsv, epsa = optical_block
    k = k0(wavelength, pixelsize) 
    epsv, epsa = filter_epsva(epsv, epsa, k, betamax )
    if symmetry == "uniaxial":
        uniaxial_order(1., epsv, epsv)
    elif symmetry == "isotropic":
        uniaxial_order(0., epsv, epsv)
    elif symmetry != "biaxial":
        raise ValueError("Unknown symmetry.")    
    return d, epsv, epsa

def filter_data(optical_data, wavelength, pixelsize, betamax = 1, symmetry = "isotropic"):
    #for legacy optical data format
    if isinstance(optical_data, tuple):
        optical_data = [optical_data]
    return [filter_block(optical_block, wavelength, pixelsize, betamax, symmetry) for optical_block in optical_data]
         
def filter_eps(eps, k, betamax = 1):
    eps = np.moveaxis(eps,-1,-3)
    feps = fft2(eps)
    beta, phi = betaphi(feps.shape[-2:],k)
    mask = beta > betamax
    feps[...,mask] = 0.
    eps = ifft2(feps)
    eps = np.moveaxis(eps,-3,-1)
    return eps

def filter_epsva(epsv, epsa, k, betamax = 1):
    eps = epsva2eps(epsv,epsa)
    eps = filter_eps(eps, k, betamax)
    return eps2epsva(eps)

_symmetry_key_maps = {"isotropic" : 0, "uniaxial" : 1,  "biaxial" : 2}

def _symmetry_arg_to_int(arg):
    if isinstance(arg, str):
        try:
            return _symmetry_key_maps[arg]
        except KeyError:
            raise ValueError("Invalid symmetry argument.")
    elif arg in (0,1,2,-1):
        return arg
    else:
        raise ValueError("Invalid symmetry argument.")
        
def _parse_symmetry_argument(arg):
    if isinstance(arg, str):
        return _symmetry_arg_to_int(arg)
    else:
        try:
            return tuple((_parse_symmetry_argument(a) for a in arg))
        except TypeError:
            return _symmetry_arg_to_int(arg)

        
def effective_block(optical_block, symmetry = 0, broadcast = True):
    """Builds effective block from the optical_block.
    
    The material epsilon is averaged over the layers.
    
    Parameters
    ----------
    optical_block : tuple
        A valid optical block tuple.

    symmetry : str, int or array
        Either 'isotropic' or 0,  'uniaxial' or 1 or 'biaxial' or 2 .
        Defines the symmetry of the effective layer. When set to 'isotropic', 
        the averaging is done so that the effective layer tensor is isotropic. 
        When set to 'uniaxial' the  effective layer tensor is an uniaxial medium. 
        If it is an array, it defines the symmetry of each individual layers
        independetly.
    broadcast : bool, optional
        If set to True, output is assured to be broadcastable with optical_block
        elements.
        
    Returns
    -------
    out : tuple
        A valid optical block tuple of the effective layers.
    """
    d, epsv,epsa = optical_block
    #Whic axes are used for averaging averaging
    axis = list(range(len(epsv.shape)-1))
    
    if symmetry in ("isotropic",0):
        #no need to convert from eigenframe to laboratory frame to do averaging, just average out the
        epsv = eig_symmetry(0, epsv).mean(tuple(axis))
        epsa = np.zeros_like(epsv)
    else:
        order = np.asarray(_parse_symmetry_argument(symmetry),int)
        
        if order.ndim != 0:
            if len(order) != len(d):
                raise ValueError("Shape of the symmetry argument is incompatible with block size.")
            #do not average over thickness axis, so pop it out.
            axis.pop(0) 
            
        axis = tuple(axis) #must be a tuple for np.mean
        
        #convert to laboratory frame, and perform averaging
        eps0 = epsva2eps(epsv,epsa)
        eps = eps0.mean(axis)
        
        #convert back to eigenframe
        epsv, epsa = eps2epsva(eps)
        eig_symmetry(order, epsv, out = epsv)
     
    if epsv.ndim == 1 and broadcast == True:
        #must be same lengths as d, so repeat the matereial for each layer
        epsv = np.asarray((epsv,)*len(d)).copy()#make a copy, to have a contiguous layer
        epsa = np.asarray((epsa,)*len(d)).copy()

    return d, epsv, epsa

def effective_data(optical_data, symmetry = 0):
    """Builds effective data from the optical_data.
    
    The material epsilon is averaged over the layers.
    
    Parameters
    ----------
    optical_data : list
        A valid optical data list .

    symmetry : str, int, or list
        Either 'isotropic' or 0,  'uniaxial' or 1 or 'biaxial' or 2 .
        Defines the symmetry of the effective layer. When set to 'isotropic', 
        the averaging is done so that the effective layer tensor is isotropic. 
        When set to 'uniaxial' the  effective layer tensor is an uniaxial medium. 
        If it is a list, it defines the symmetry of each individual block of data.
        
    Returns
    -------
    out : list
        A valid optical data of the effective layer blocks.
    """
    #for legacy format, convert to new-style format
    if isinstance(optical_data, tuple):
        optical_data = [optical_data]
        deprecation("effective_data: optical_data is not a list. This will raise exception in the future.")
    if isinstance(symmetry, tuple):
        optical_data = [optical_data]
        deprecation("effective_data: optical_data is not a list. This will raise exception in the future.")
        
    if isinstance(symmetry,int) or isinstance(symmetry,str):
        symmetry = [symmetry] * len(optical_data)
        
    if len(symmetry) != len(optical_data):
        raise Exception("symmetry and optical data length mismatch.")

    return [effective_block(optical_block, sym) for optical_block, sym in zip(optical_data,symmetry)]
    
def tensor2matrix(tensor, out = None):
    """Converts  matrix to tensor
    
    Parameters
    ----------
    tensor : (...,6) array
       Input 3x3 array
    out : (...,3,3) array, optional
       Output array.
       
    Returns
    -------
    matrix : ndarray
        Matrix of shape (...,3,3).
    """
    tensor = np.asarray(tensor)
    if tensor.shape[-1:] != (6,):
        raise ValueError("Not a valid tensor.")
    if out is None:
        out = np.empty(shape = tensor.shape[:-1] + (3,3), dtype = tensor.dtype)
    for i in range(3):
        out[...,i,i] = tensor[i]
    out[...,0,1] = tensor[...,3]
    out[...,0,2] = tensor[...,4]
    out[...,1,2] = tensor[...,5]
    out[...,1,0] = tensor[...,3]
    out[...,2,0] = tensor[...,4]
    out[...,2,1] = tensor[...,5]   
    return out
    
def matrix2tensor(matrix, out = None):
    """Converts  matrix to tensor
    
    Parameters
    ----------
    matrix : (...,3,3) array
       Input 3x3 array
    matrix : (...,6) array, optional
       Output array.
       
    Returns
    -------
    tensor : ndarray
        Tensor of shape (...,6).
    """
    matrix = np.asarray(matrix)
    if matrix.shape[-2:] != (3,3):
        raise ValueError("Not a valid matrix.")
    if out is None:
        out = np.empty(shape = matrix.shape[:-2] + (6,), dtype = matrix.dtype)
    for i in range(3):
        out[...,i] = matrix[...,i,i]
    out[...,3] = matrix[...,0,1]
    out[...,4] = matrix[...,0,2]
    out[...,5] = matrix[...,1,2]
    return out

def split_block(block_data):
    """Splits optical block data into a list of layers"""
    block_data = validate_optical_block(block_data)
    return [(d,epsv,epsa) for d,epsv,epsa in zip(*block_data)]

def layered_data(optical_data):
    """Converts a list of blocks to a list of layers."""
    if not isinstance(optical_data, list):
        raise ValueError("Optical data must be a list")
    out = []
    for block in optical_data:
        out = out + split_block(block)
    return out

def merge_blocks(optical_data, shape = None, wavelength = None):
    """Merges blocks of optical data into a single block.""" 
    optical_data = validate_optical_data(optical_data,shape = shape, wavelength = wavelength)
    thickness = np.vstack(tuple((d[0] for d in optical_data)))
    epsv = np.vstack(tuple((d[1] for d in optical_data)))
    epsa = np.vstack(tuple((d[2] for d in optical_data)))
    return thickness, epsv, epsa

def material_shape(epsv, epsa):
    """Determines material 2D - crossection shape from the epsv -eigenvalues 
    and epsa - eigenangles arrays"""
    x,y = epsv.shape[-3:-1]
    xa,ya = epsa.shape[-3:-1]
    return (max(x,xa), max(y,ya))

def optical_block_shape(optical_block):
    """Determines optical block 2D - crossection shape from the epsv -eigenvalues 
    and epsa - eigenangles arrays"""
    d, epsv, epsa = validate_optical_block(optical_block)
    return material_shape(epsv, epsa)
    
def optical_data_shape(optical_data):
    """Determines optical data 2D - crossection shape from the epsv -eigenvalues 
    and epsa - eigenangles arrays"""
    common_shape = (1,1)
    for optical_block in optical_data:
        data_shape = optical_block_shape(optical_block)
        common_shape = tuple((max(x,y) for (x,y) in zip(common_shape, data_shape)))        
    return common_shape

def shape2dim(shape):
    """Converts material 2D shape to material dimension""" 
    if shape == (1,1):
        return 1
    elif shape[0] == 1 or shape[1] == 1:
        return 2
    else:
        return 3

def material_dim(epsv, epsa):
    """Returns material dimension"""
    shape = material_shape(epsv, epsa)
    return shape2dim(shape)

MAGIC = b"dtms" #legth 4 magic number for file ID

_VERSION_X01 = b"\x01"
VERSION = b"\x02"


#IOs fucntions
#-------------

def save_stack(file, optical_data):
    """Saves optical data to a binary file in ``.dtms`` format.
    
    Parameters
    ----------
    file : file, str
        File or filename to which the data is saved.  If file is a file-object,
        then the filename is unchanged.  If file is a string, a ``.dtms``
        extension will be appended to the file name if it does not already
        have one.
    optical_data: optical data tuple
        A valid optical data
    """    
    own_fid = False
    #validatedata format. Blocks are not tested for broadcasting (shape = (1,1))
    optical_data = validate_optical_data(optical_data, shape = (1,1))

    try:
        if isinstance(file, str):
            if not file.endswith('.dtms'):
                file = file + '.dtms'
            f = open(file, "wb")
            own_fid = True
        else:
            f = file
        f.write(MAGIC)
        f.write(VERSION)
        
        for data in optical_data:
            d,epsv,epsa = data
            np.save(f,d, allow_pickle = False)
            if isinstance(epsv, EpsilonCauchy):
                epsv = np.asarray(("cauchy",epsv.coefficients),dtype = object)
            elif isinstance(epsv, EpsilonSellmeier):
                epsv = np.asarray(("sellmeier",epsv.coefficients),dtype = object)
            np.save(f,epsv, allow_pickle = True)
            np.save(f,epsa, allow_pickle = False)
    finally:
        if own_fid == True:
            f.close()

def load_stack(file):
    """Load optical data from a file.
    
    Parameters
    ----------
    file : file, str
        The file to read.
    """
    own_fid = False
    try:
        if isinstance(file, str):
            f = open(file, "rb")
            own_fid = True
        else:
            f = file
        magic = f.read(len(MAGIC))
        if magic == MAGIC:
            version = f.read(1)
            if ord(version) > ord(VERSION):
                raise OSError("This file was created with a more recent version of dtmm. Please upgrade your dtmm package!")
            elif ord(version) == ord(_VERSION_X01):
                d = np.load(f)
                epsv = np.load(f)
                if f.peek(1) == b"":
                    #no more data to read.. epsa is not present
                    epsa =  None
                else:
                    epsa = np.load(f)
                return [d, epsv, epsa]
            else:
                out = []
                while f.peek(1) != b"":
                    d = np.load(f)
                    epsv = np.load(f, allow_pickle = True)
                    if epsv.dtype == object:
                        name, coefficients = epsv
                        if name == "cauchy":
                            epsv = EpsilonCauchy(coefficients)
                        elif name == "sellmeier":
                            epsv = EpsilonSellmeier(coefficients)
                    epsa = np.load(f) 
                    out.append((d,epsv,epsa))
                return out                
        else:
            raise OSError("Failed to interpret file {}".format(file))
    finally:
        if own_fid == True:
            f.close()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
