# -*- coding: utf-8 -*-
"""
Color conversion functions and utilities.

CMF creation function
---------------------

* :func:`.load_cmf` : loads specter cmf data from file or pre-defined table
* :func:`.load_tcmf` : loads transmission cmf data from file or pre-defined table
* :func:`.cmf2tcmf` : converts cmf to transmission cmf.
* :func:`.srf2cmf` : converts spectral respone data to cmf
* :func:`.load_specter` : load specter from file or from data.
* :func:`.normalize_specter` : for specter normalization.

Color conversion
----------------

* :func:`.specter2color` : converts specter data to color RGB or gray image.
* :func:`.apply_gamma` : applies gamma curve to linear data
* :func:`.apply_srgb_gamma` : applies sRGB gamma curve to linear data
* :func:`.xyz2rgb` : Converts XYZ data to RGB
* :func:`.xyz2gray` : Converts XYZ data to YYY (gray)
* :func:`.spec2xyz` : Converts specter to XYZ

"""

from __future__ import absolute_import, print_function, division

from dtmm.conf import FDTYPE, NUMBA_TARGET, NFDTYPE, NUMBA_CACHE, DATAPATH, CMF

import numpy as np
import numba
import os

#DATAPATH = os.path.join(os.path.dirname(__file__), "data")


# D65 standard light 5nm specter
D65PATH = os.path.join(DATAPATH, "D65.dat" )
#: color matrix for sRGB color space in D65 reference white
XYZ2RGBD65 = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                       [-0.9692660,  1.8760108,  0.0415560],
                       [ 0.0556434, -0.2040259,  1.0572252]])
RGB2XYZ = np.linalg.inv(XYZ2RGBD65)
#Srgb tranfer function constants
SRGBIGAMMA = 1/2.4
SRGBSLOPE = 12.92
SRGBLINPOINT = 0.0031308
SRGBA = 0.055

@numba.vectorize([NFDTYPE(NFDTYPE, NFDTYPE)], nopython = True, target = NUMBA_TARGET, cache = NUMBA_CACHE) 
def apply_gamma(value, gamma):
    """apply_gamma(value, gamma)
    
Applies standard gamma function (transfer function) to the given (linear) data.
    
Parameters
----------
value : float
    Input value
gamma : float
    Gamma factor"""
    if value > 1.:
        return 1.
    if value < 0:
        return 0.
    else:
        return value**(1./gamma)

@numba.vectorize([NFDTYPE(NFDTYPE)], nopython = True, target = NUMBA_TARGET, cache = NUMBA_CACHE) 
def apply_srgb_gamma(value):
    """apply_srgb_gamma(value)

Applies sRGB gamma function (transfer function) to the given (linear) data."""
    if value < SRGBLINPOINT:
        if value < 0.:
            return 0.
        else:
            return SRGBSLOPE * value
    else:
        if value > 1.:
            return 1.
        else:
            return (1+SRGBA)*value**SRGBIGAMMA-SRGBA

@numba.guvectorize([(NFDTYPE[:], NFDTYPE[:])], '(n)->(n)', target = NUMBA_TARGET, cache = NUMBA_CACHE)
def xyz2srgb(xyz, rgb):
    """xyz2srgb(xyz)
    
Converts XYZ value to RGB value based on the sRGB working space with 
D65 reference white.
"""
    assert len(xyz) >= 3
    xyz0 = xyz[0]
    xyz1 = xyz[1]
    xyz2 = xyz[2]
    for k in range(3):
        rgb[k] = XYZ2RGBD65[k,0] * xyz0 +  XYZ2RGBD65[k,1]* xyz1 +  XYZ2RGBD65[k,2]* xyz2
        
@numba.guvectorize([(NFDTYPE[:], NFDTYPE[:])], '(n)->(n)', target = NUMBA_TARGET, cache = NUMBA_CACHE)
def xyz2gray(xyz, gray):
    """xyz2gray(xyz)
    
Converts XYZ value to Gray color"""
    assert len(xyz) >= 3
    y = xyz[1]
    for k in range(3):
        gray[k] = y
                
@numba.guvectorize([(NFDTYPE[:],NFDTYPE[:,:],NFDTYPE[:])], '(n),(n,m)->(m)', target = NUMBA_TARGET, cache = NUMBA_CACHE)
def spec2xyz(spec,cmf,xyz):
    """spec2xyz(spec,cmf)
    
Converts specter array to xyz value.
    
Parameters
----------
spec : array_like
    Input specter data
cmf : array_like
    Color matching function
    
Returns
-------
xyz : ndarray
    Computed xyz value."""  
    
    for j in range(cmf.shape[1]):
        xyz[j] = 0.
        for i in range(cmf.shape[0]):
            xyz[j] = xyz[j] + cmf[i,j]*spec[i]          
            
def specter2color(spec, cmf, norm = False, gamma = True, gray = False, out = None):
    """Converts specter data to RGB data (color or gray).
    
    Specter shape must be [...,k], where wavelengths are in the last axis. cmf 
    must be a valid color matchin function array of size [k,3].
    
    Parameters
    ----------
    spec : array
        Specter data of shape [..., n] where each data element in the array has
        n wavelength values
    cmf : array
        A color matching function (array of shape [n,3]) that converts the specter data 
        to a XYZ color.
    norm : bool or float, optional
        If set to False, no data normalization is performed (default). If True,
        internally, xyz data is normalized in the range [0,1.], so that no clipping occurs.
        If it is a float, data is normalized to this value.
    gamma : bool or float, optional
        If gamma is True srgb gamma function is applied (default). If float is
        provided, standard gamma factor is applied with a given gamma value. If False,
        no gamma correction is performed.
    gray : bool, optional
        Whether gray output is calculated (color by default)
    out : array, optional
        Output array of shape (...,3)
        
    Returns
    -------
    rgb : ndarray
        A computed RGB value.
        
    Notes
    -----
    Numpy broadcasting rules apply to spec and cmf.
        
    Example
    -------
    >>> cmf = load_tcmf()
    >>> specter2color([1]*81, cmf)#should be close to 1,1,1
    ... # doctest: +NORMALIZE_WHITESPACE
    array([0.99994901, 1.        , 0.99998533])
    """
    #if isinstance(spec, list):
    #    spec = np.add.reduce(spec)
    
    cmf = np.asarray(cmf)
    if cmf.shape[-1] != 3:
        raise ValueError("Grayscale cmf! Cannot convert to color.")
      
    out = spec2xyz(spec,cmf, out)
        
    if norm is True:
        #normalize to max in any of the XYZ channels.. so that no clipping occurs.
        out = np.divide(out,out.max(),out)
        
    elif norm != 0:
        out = np.divide(out,norm,out)
 
    if gray == True:
        out = xyz2gray(out,out) 
    else:
        out = xyz2srgb(out,out)
    if gamma is True:
        apply_srgb_gamma(out,out)
    elif gamma is not False:
        apply_gamma(out,gamma,out)
    
    return out

def srf2cmf(srf, out = None):
    """Converts spectral response function (Y) to color matching function (XYZ).
    
    Parameters
    ----------
    srff : array_like
        Spectral response function of shape (...,n)
    out : ndarray, optional
        Output array.
        
    Returns
    -------
    cmf: ndarray
        A color matching function array of shape (...,n,3)
    """
    if out is None:
        out = np.empty(shape = srf.shape + (3,), dtype = srf.dtype)
    
    out[...,0] = RGB2XYZ[0,0] * srf +  RGB2XYZ[0,1] * srf +  RGB2XYZ[0,2] * srf
    out[...,1] = srf
    out[...,2] = RGB2XYZ[2,0] * srf +  RGB2XYZ[2,1] * srf +  RGB2XYZ[2,2] * srf
    return out
    
def normalize_specter(spec, cmf, out = None):
    """Normalizes specter based on the color matching function. (cmf array) so that
    calculated Y value is 1.
    
    Parameters
    ----------
    spec : array_like
        Input illuminant specter data of shape (...,n).
    cmf : array_like
        Color matching function of shape (...,n,3).
    out : ndarray, optional
        Output array.
        
    Returns
    -------
    normalized_spec : ndarray
        A normalized version of the input specter
        
    Notes
    -----
    Numpy broadcasting rules apply to spec and cmf.
    """
    cmf = np.asarray(cmf)
    if cmf.shape[-1] == 3:
        #cmf is color matching function
        xyz = spec2xyz(spec,cmf)
        norm = xyz[...,1] #Y value is ligtness.. normalize it to this
    else:
        raise ValueError("Incompatible cmf shape")
    return np.divide(spec,norm,out)

def load_tcmf(wavelengths = None, illuminant = "D65", cmf = CMF, norm = True, retx = False, 
              single_wavelength = False):
    """Loads transmission color matching function.
    
    This functions loads a CIE XYZ color matching function and transforms it
    to a transmission color matching function for a given illuminant. Resulting 
    CMF matrix will transform unity into white color.
    
    Parameters
    ----------
    wavelengths : array_like, optional
        Wavelengths at which data is computed. If not specified (default), original
        data from the 5nm tabulated data is returned.
    illuminant : str, optional
        Name of the standard illuminant or path to illuminant data.
    cmf : str, optional
        Name or path to the cmf function. Can be 'CIE1931' for CIE 1931 2-deg 
        5nm tabulated data, 'CIE1964' for CIE1964 10-deg 5nm tabulatd data, or
        'CIE2006-2' or 'CIE2006-10' for a proposed CIE 2006 2- or 10-deg 5nm 
        tabulated data. 
    norm : int, optional
        By default cmf is normalized so that unity transmission value over the
        full spectral range of the illuminant is converted to XYZ color with Y=1.
        You can disable this by setting norm = 0. If you set norm = 2, then the
        cmf is normalized for the interpolated spectra at given wavelengths, 
        and not to the full bandwidth of the spectra (norm = 1).
    retx : bool, optional
        Should the selected wavelengths be returned as well.
    single_wavelength : bool, optional
        If specified, color matching function for single wavelengths specter is
        calculated by interpolation. By default, specter is assumed to be a 
        piece-wise linear function and continuous between the specified 
        wavelengts, and data is integrated instead.
    
    Returns
    -------
    cmf : array
        Color matching function array of shape [n,3] or a tuple of (x,cmf) 
        if retx is specified.
    
    
    Example
    -------
    >>> cmf = load_tcmf()
    >>> specter2color([1]*81, cmf) #should be close to 1,1,1
    ... # doctest: +NORMALIZE_WHITESPACE
    array([0.99994901, 1.        , 0.99998533])
    """
    if wavelengths is not None and len(wavelengths) == 1:
        single_wavelength = True

    if single_wavelength == True:
        x, cmf = load_cmf(wavelengths, single_wavelength = True,retx = True, cmf = cmf)
        spec = load_specter(wavelengths = x, illuminant = illuminant)
        cmf = cmf2tcmf(cmf, spec, norm = bool(norm))  
    else:
        x,cmf = load_cmf(retx = True, cmf = cmf)
        spec = load_specter(wavelengths = x, illuminant = illuminant)
        cmf = cmf2tcmf(cmf, spec, norm = bool(norm)) 
        if wavelengths is not None:
             cmf = integrate_data(wavelengths, x,cmf)
             x = wavelengths
             if norm == 2:
                 cmf = cmf2tcmf(cmf, [1.]*len(wavelengths), norm = norm)  
    if retx == True:
        return x, cmf
    else:
        return cmf

def cmf2tcmf(cmf, spec, norm = True, out = None):
    """Converts CMF table to specter-normalized transmission CMF table
    
    Parameters
    ----------
    cmf : array_like
        Color matchinf function array
    spec : array_like
        Illuminant specter array
    norm : bool, optional
       Whether to normalize illuminant specter before constructing the CMF.
    out : ndarray, optional
       Output array
    
    Returns
    -------
    out : ndarray
        A transmission color matching function array.
       
    Notes
    -----
    Numpy broadcasting rules apply to spec and cmf.
    """
    cmf = np.asarray(cmf,FDTYPE)
    spec = np.asarray(spec,FDTYPE)
    if norm == True:
        spec = normalize_specter(spec, cmf)
    return np.multiply(spec[:,np.newaxis],cmf, out = out) 
    
def load_specter(wavelengths = None, illuminant = "D65", retx = False):
    """Loads illuminant specter data from file.
    
    Parameters
    ----------
    wavelengths : array_like, optional
        Wavelengths at which data is interpolated
    illuminant : str, or array, optional
        Name of the standard illuminant or filename. If specified as array, it must
        be an array of shape (n,2). The first column decribes wavelength and the
        second is the intensity.
    retx : bool, optional
        Should the selected wavelengths be returned as well.
        
    Returns
    -------
    specter : array
        Specter array of shape [num] or a tuple of (x,specter) 
        if retx is specified
        
    Example
    -------
    
    #D65 evaluated at three wavelengths
    >>> spec = load_specter((450,500,550), "D65") 
    
    #illuminant with triangular specter evaluated at three wavelengths
    >>> spec = load_specter([450,475,500,], illuminant = [[400,0],[500,1],[600,0]]) 
    
    """   
    if isinstance(illuminant, str):    
        try:
            # predefined data in a file
            data = np.loadtxt(os.path.join(DATAPATH, illuminant + ".dat"))
        except:
            data = np.loadtxt(illuminant)
    else:
        data = np.asarray(illuminant)
        if data.ndim != 2 and data.shape[-1] != 2:
            raise ValueError("Not a valid illuminant data")
        
    if wavelengths is not None:
        data = interpolate_data(wavelengths, data[:,0], data[:,1:])
        data = np.ascontiguousarray(data[:,0], dtype = FDTYPE)
    else:
        wavelengths = np.ascontiguousarray(data[:,0], dtype = FDTYPE)
        data = np.ascontiguousarray(data[:,1], dtype = FDTYPE)
        
    if retx == True:
        return wavelengths, data
    else:
        return data
    
def load_cmf(wavelengths = None, cmf = CMF, retx = False, single_wavelength = False):
    """Load XYZ Color Matching function.
    
    This function loads tabulated data and re-calculates xyz array on a 
    given range of wavelength values.
    
    See also load_tcmf.
    
    Parameters
    ----------
    wavelengths : array_like, optional
        A 1D array of wavelengths at which data is computed. If not specified 
        (default), original data from the 5nm tabulated data is returned.
    cmf : str, optional
        Name or path to the cmf function. Can be 'CIE1931' for CIE 1931 2-deg 
        5nm tabulated data, 'CIE1964' for CIE1964 10-deg 5nm tabulated data, or
        'CIE2006-2' or 'CIE2006-10' for a proposed CIE 2006 2- or 10-deg 5nm 
        tabulated data. For grayscale cameras, there is a 'CMOS' spectral 
        response data. You can also provide 'FLAT' for flat (unity) response function.
    retx : bool, optional
        Should the selected wavelengths be returned as well.
    single_wavelength : bool, optional
        If specified, color matching function for single wavelengths specter is
        calculated by interpolation. By default, specter is assumed to be a 
        piece-wise linear function and continuous between the specified 
        wavelengts, and data is integrated instead.
   
    Returns
    -------
    cmf : array
        Color matching function array of shape [n,3] or a tuple of (x,cmf) 
        if retx is specified.
    """
    try:
        if cmf == "FLAT":
            if wavelengths is None:
                wavelengths = np.arange(380,781,5)
            data = np.zeros((len(wavelengths),3))
            data[:,1] = 100. #100% QE
            if retx == True:
                return wavelengths, data
            else:
                return data       
        
        if cmf.startswith("CIE"):
            data = np.loadtxt(os.path.join(DATAPATH, cmf + "XYZ.dat"))
        else:
            data = np.loadtxt(os.path.join(DATAPATH, cmf + "Y.dat"))
    except:
        data = np.loadtxt(cmf)
    
    if data.shape[-1] == 4:
        x, data = np.ascontiguousarray(data[:,0], dtype = FDTYPE),  np.ascontiguousarray(data[:,1:], dtype = FDTYPE)
    elif data.shape[-1] == 2:
        x, data = np.ascontiguousarray(data[:,0], dtype = FDTYPE),  np.ascontiguousarray(data[:,1], dtype = FDTYPE)
    else:
        raise ValueError("Not a valid cmf data!")
        
    if wavelengths is not None:
        wavelengths = np.asarray(wavelengths, dtype = FDTYPE)
        if wavelengths.ndim != 1:
            raise ValueError("Wavelengths has to be 1D array")
        if len(wavelengths) == 1:
            single_wavelength = True
    if single_wavelength == True and wavelengths is not None:
        data = interpolate_data(wavelengths, x, data)
        x = wavelengths
    elif wavelengths is not None:
        data = integrate_data(wavelengths, x,data)
        x = wavelengths

    if data.ndim == 1:
        #convert spectral response to cmf
        data = srf2cmf(data)
    if retx == True:
        return x, data
    else:
        return data

def interpolate_data(x, x0, data):
    """Interpolates data
    
    Parameters
    ----------
    x : array_like 
        The x-coordinates at which to evaluate the interpolated values.
    x0 : array_like
        The x-coordinates of the data points, must be increasing.
    data : ndarray
        A 1D or 2D array of datapoints to interpolate.
        
    Returns
    -------
    y : ndarray
        The interpolated values.    
    """
    data = np.asarray(data, dtype = FDTYPE)
    x0 = np.asarray(x0)
    x = np.asarray(x)
    if data.ndim in (1,2) and x0.ndim == 1 and x.ndim == 1: 
        if data.ndim == 2:
            out = np.zeros(shape = x.shape + data.shape[1:], dtype = data.dtype)  
            rows, cols = data.shape  
            for i in range(cols):
                #f = interpolate.interp1d(x0, data[:,i],fill_value = 0, kind="linear")
                #out[...,i] = f(x)
                out[...,i] = np.interp(x, x0, data[:,i],left = 0., right = 0.)
            return out 
        else:
            return np.interp(x, x0, data, left = 0., right = 0.)
    else:
        raise ValueError("Invalid dimensions of input data.")
              
 
def integrate_data(x,x0,cmf):
    """Integrates data.
    
    This function takes the original data and computes new data at specified x
    coordinates by a weighted integration of the original data. For each new 
    x value, it multiplies the data with a triangular kernel and integrates.
    The width of the kernel is computed from the spacings in x. 
    
    Parameters
    ----------
    x : array_like 
        The x-coordinates at which to compute the integrated data.
    x0 : array_like
        The x-coordinates of the data points, must be increasing.
    data : ndarray
        A 1D or 2D array of datapoints to integrate.
        
    Returns
    -------
    y : ndarray
        The integrated values.    
    """
    
    cmf = np.asarray(cmf)
    x0 = np.asarray(x0)
    xout = np.asarray(x)
    ndim = cmf.ndim
    if ndim in (1,2) and x0.ndim == 1 and xout.ndim == 1:    
        dxs = x0[1:]-x0[0:-1]
        dx = dxs[0]
        if not np.all(dxs == dx):
            raise ValueError("x0 must have equal spacings")
        out = np.zeros(shape = (len(x),)+cmf.shape[1:], dtype = cmf.dtype) 
        n = len(x)
        for i in range(n):
            if i == 0:
                x,y = _rxn(xout,i,dx,ndim)    
                data = (interpolate_data(x,x0,cmf)*y).sum(0)
            elif i == n-1:
                x,y  = _lxn(xout,i,dx,ndim)
                data = (interpolate_data(x,x0,cmf)*y).sum(0) 
            else:
                x,y  = _rxn(xout,i,dx,ndim)
                tmp = (interpolate_data(x,x0,cmf)*y)
                tmp[0] = tmp[0]/2 #first element is counted two times...
                data = tmp.sum(0)
                x,y  = _lxn(xout,i,dx,ndim)
                tmp = (interpolate_data(x,x0,cmf)*y)
                tmp[0] = tmp[0]/2 #first element is counted two times...
                data += tmp.sum(0)
            out[i,...] = data
        return out 

            
    else:
        raise ValueError("Invalid dimensions of input data.")
 
def _rxn(x,i,dx,ndim):
    xlow, xhigh = x[i], x[i+1]
    dx = (xhigh - xlow)/dx
    if dx < 1:
        import warnings
        warnings.warn("The resolution of the integrand is too low.", stacklevel = 2)
    #n = int(round(dx))+1
    n = int(dx-1)+1
    dx = dx/n
    xout = np.linspace(xlow,xhigh,n+1)
    yout = np.linspace(1.,0.,n+1)*dx
    if ndim == 2:
        return xout,yout[:,None]
    else:
        return xout,yout

def _lxn(x,i,dx, ndim):
    xlow, xhigh = x[i-1], x[i]
    dx = (xhigh - xlow)/dx
    if dx < 1:
        import warnings
        warnings.warn("The resolution of the integrand is too low.", stacklevel = 2)
    #n = int(round(dx))+1
    n = int(dx-1)+1
    dx = dx/n
    xout = np.linspace(xhigh,xlow,n+1)
    yout = np.linspace(1.,0.,n+1)*dx
    if ndim == 2:
        return xout,yout[:,None]
    else:
        return xout,yout

   
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    import matplotlib.pyplot as plt
    num = 81
    for num in (81,41,21,11,5):
        wavelengths = np.linspace(380,780,num)
        x,xyz = load_tcmf(wavelengths, norm = True,retx = True, single_wavelength = True)
        
        plt.plot(x,xyz[:,0],label = num) #X color
        x,xyz = load_tcmf(wavelengths, norm = True,retx = True, single_wavelength = False)
        
        plt.plot(x,xyz[:,0],"o") #X color
    plt.legend()
    plt.show()

    
