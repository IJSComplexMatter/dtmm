# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:44:09 2017

@author: andrej

Color conversion functions and utilities...

"""

from __future__ import absolute_import, print_function, division

from dtmm.conf import FDTYPE, NUMBA_TARGET, NFDTYPE, NCDTYPE

import numpy as np
import numba
import os

#DATAPATH = os.path.join(os.path.dirname(__file__), "data")
DATAPATH = os.path.dirname(__file__)

#2-deg XYZ 5nm CMFs 
CMFPATH = os.path.join(DATAPATH, "CIE1931XYZ.dat" )
# D65 standard light 5nm specter
D65PATH = os.path.join(DATAPATH, "D65.dat" )
# color matrix for sRGB color space in D65 reference white
XYZ2RGBD65 = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                       [-0.9692660,  1.8760108,  0.0415560],
                       [ 0.0556434, -0.2040259,  1.0572252]])
#Srgb tranfer function constants
SRGBIGAMMA = 1/2.4
SRGBSLOPE = 12.92
SRGBLINPOINT = 0.0031308
SRGBA = 0.055


@numba.vectorize([NFDTYPE(NFDTYPE, NFDTYPE)], nopython = True, target = NUMBA_TARGET) 
def apply_gamma(value, gamma):
    """Applies standard gamma function (transfer function) to the given (linear) data"""
    if value > 1.:
        return 1.
    if value < 0:
        return 0.
    else:
        return value**(1./gamma)

@numba.vectorize([NFDTYPE(NFDTYPE)], nopython = True, target = NUMBA_TARGET) 
def apply_srgb_gamma(value):
    """Applies sRGB gamma function (transfer function)  to the given (linear) data"""
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

@numba.guvectorize([(NFDTYPE[:], NFDTYPE[:])], '(n)->(n)', target = NUMBA_TARGET)
def xyz2srgb(xyz, rgb):
    """Converts XYZ value to RGB value based on the sRGB working space with D65 reference white."""
    xyz0 = xyz[0]
    xyz1 = xyz[1]
    xyz2 = xyz[2]
    for k in range(3):
        rgb[k] = XYZ2RGBD65[k,0] * xyz0 +  XYZ2RGBD65[k,1]* xyz1 +  XYZ2RGBD65[k,2]* xyz2
        
@numba.guvectorize([(NFDTYPE[:], NFDTYPE[:])], '(n)->(n)', target = NUMBA_TARGET)
def xyz2gray(xyz, gray):
    """Converts XYZ value to Gray color"""
    y = xyz[1]
    for k in range(3):
        gray[k] = y
        
@numba.jit([NFDTYPE(NFDTYPE,NFDTYPE[:])], nopython = True)            
def interpolate1d(x, data):
    nmax= data.shape[0]-1
    assert nmax >= 0
    f = x*nmax
    i = int(f)
    if i >= nmax:
        return data[nmax]
    if i < 0:
        return data[0]
    k = f-i
    return (1.-k)*data[i] + (k)*data[i+1]
        
@numba.guvectorize([(NFDTYPE[:],NFDTYPE[:,:],NFDTYPE[:])], '(k),(n,m)->(m)', target = NUMBA_TARGET)
def spec2xyz(spec,cmf,xyz):
    """Converts specter array to xyz value, given a color matching function (cmf array)"""            
    if spec.shape[0] == cmf.shape[0]:
        for j in range(cmf.shape[1]):
            xyz[j] = 0.
            for i in range(cmf.shape[0]):
                xyz[j] = xyz[j] + cmf[i,j]*spec[i]
    else:
        nmax = cmf.shape[0]-1
        assert nmax > 0
        for j in range(cmf.shape[1]):
            xyz[j] = 0.
            for i in range(cmf.shape[0]):
                x = i/nmax
                xyz[j] = xyz[j] + cmf[i,j]*interpolate1d(x, spec)
            
def specter2color(spec, cmf, norm = False, gamma = True, gray = False, out = None):
    """Converts specter data to color (RGB or gray).
    
    Specter shape must be [...,k], where wavelengths are in the last axis. cmf 
    must be a valid color matchin function array of size [n,3] If n != k, 
    specter data is interpolated to match the data in cmf function.
    
    Parameters
    ----------
    spec : array
        Specter data of shape [..., n] where each data element in the array has
        n wavelength values
    cmf : array
        A color matching function (array of shape [n,3]) that converts the specter data 
        to a XYZ color.
    norm : bool, optional
        If set to False, no data normalization is performed (default). If True,
        internally, xyz data is normalized in the range [0,1.], so that no clipping occurs.
    gamma : bool or float, optional
        If gamma is True srgb gamma function is applied (default). If float is
        provided, standard gamma factor is applied with a given gamma value. If False,
        no gamma correction is performed.
    gray : bool, optional
        Whether gray output is calculated (color by default)
    out : array, optional
        Output array
        
        
    Example
    -------
    >>> cmf = load_tcmf()
    >>> specter2color([1], cmf)#should be close to 1,1,1
    array([ 0.99994901,  1.        ,  0.99998533])
    """
    #if isinstance(spec, list):
    #    spec = np.add.reduce(spec)
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
  
def normalize_specter(spec, cmf, out = None):
    """Normalizes specter based on the color matching function (cmf array) so that
    calculated Y value is 1."""
    xyz = spec2xyz(spec,cmf, out)
    norm = xyz[...,1] #Y value is ligtness.. normalize it to this
    return np.divide(spec,norm,out)

def load_tcmf(wavelengths = None, illuminant = "D65", norm = True, retx = False, single_wavelength = False):
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
        Name of the standard illuminant
    norm : bool, optional
        By default cmf is normalized, so that unity transmission value is converted
        to XYZ color with Y=1.
    retx : bool, optional
        Should the selected wavelengths be returned as well.
    single_wavelength : bool, optional
        If specified, color matching function for single wavelengths specter is
        calculated by interpolatiin. By default, specter is assumed to be 
        continuous between the specified wavelengts, and data is integrated instead.
        
    Returns
    -------
    cmf : array
        Color matching function array of shape [n,3] or a tuple of (x,cmf) 
        if retx is specified.
    
    
    Example
    -------
    >>> cmf = load_tcmf()
    >>> specter2color([1], cmf) #should be close to 1,1,1
    array([ 0.99994901,  1.        ,  0.99998533])
    """

    if single_wavelength == True:
        x, cmf = load_cmf(wavelengths, single_wavelength = True,retx = True)
        spec = load_specter(wavelengths = x, illuminant = illuminant)
        if norm == True:
            spec = normalize_specter(spec, cmf)
        cmf = spec[:,np.newaxis]*cmf    
    else:
        x,cmf = load_cmf(retx = True)
        spec = load_specter(wavelengths = x, illuminant = illuminant)
        if norm == True:
            spec = normalize_specter(spec, cmf)
        cmf = spec[:,np.newaxis]*cmf
        if wavelengths is not None:
            cmf = integrate_data(wavelengths, x,cmf)
            x = wavelengths
    if retx == True:
        return x, cmf
    else:
        return cmf

        
def load_specter(wavelengths = None, illuminant = "D65", retx = False):
    """Loads illuminant specter data.
    
    Parameters
    ----------
    wavelengths : array_like, optional
        Wavelengths at which data is interpolated
    illuminant : str, optional
        Name of the standard illuminant
    retx : bool, optional
        Should the selected wavelengths be returned as well.
        
    Returns
    -------
    specter : array
        Specter array of shape [num] or a tuple of (x,specter) 
        if retx is specified
        """
    if illuminant == "D65":   
        data = np.loadtxt(D65PATH)
    else:
        raise ValueError("Unsupported illuminant name")
        
    if wavelengths is not None:
        data = interpolate_data(wavelengths, data[:,0], data[:,1:])
        data = np.ascontiguousarray(data[:,0])
    else:
        wavelengths = np.ascontiguousarray(data[:,0])
        data = np.ascontiguousarray(data[:,1])
        
    if retx == True:
        return wavelengths, data
    else:
        return data


def load_cmf(wavelengths = None,  single_wavelength = False,retx = False):
    """Load XYZ Color Matching function as an array.
    
    This function loads 5nm tabulated data and re-calculates xyz array on a given range of
    wavelength values. 
    
    See also load_tcmf.
    
    Parameters
    ----------
    wavelengths : array_like, optional
        Wavelengths at which data is computed. If not specified (default), original
        data from the 5nm tabulated data is returned.
    retx : bool, optional
        Should the selected wavelengths be returned as well.
    single_wavelength : bool, optional
        If specified, color matching function for single wavelengths specter is
        calculated by interpolatiin. By default, specter is assumed to be 
        continuous between the specified wavelengts, and data is integrated instead.
        
    Returns
    -------
    cmf : array
        Color matching function array of shape [n,3] or a tuple of (x,cmf) 
        if retx is specified.
    """
    data = np.loadtxt(CMFPATH)
    x, data = np.ascontiguousarray(data[:,0]),  np.ascontiguousarray(data[:,1:])
    if single_wavelength == True and wavelengths is not None:
        data = interpolate_data(wavelengths, x,data)
        x = wavelengths
    elif wavelengths is not None:
        data = integrate_data(wavelengths, x,data)
        x = wavelengths
    if retx == True:
        return x, data
    else:
        return data


def interpolate_data(x, x0, data):
    data = np.asarray(data)
    x0 = np.asarray(x0)
    x = np.asarray(x)
    out = np.zeros(shape = (len(x), data.shape[1]), dtype = data.dtype)   
    rows, cols = data.shape  
    for i in range(cols):
        out[:,i] = np.interp(x, x0, data[:,i],left = 0., right = 0.)
    return out           

def integrate_data(x,x0,cmf):
    cmf = np.asarray(cmf)
    x0 = np.asarray(x0)
    x = np.asarray(x)
    out = np.zeros(shape = (len(x), cmf.shape[1]), dtype = cmf.dtype) 
    n = len(x)
    triang = np.array([0.,1,0])
    for i in range(n):
        if i == 0:
            data = np.interp(x0,x[0:i+2],triang[1:],left = 0.)
        elif i == n-1:
            data = np.interp(x0,x[i-1:i+1],triang[:-1],right = 0.)
        else:
            data = np.interp(x0,x[i-1:i+2],triang)      
        out[i,:] = (cmf*data[:,np.newaxis]).sum(0)
    return out
            

__all__ = ["specter2color", "load_tcmf", "load_cmf", "load_specter"]
        
#def _convolve_data(data, width):
#    data = np.asarray(data)
#    width = int(width)
#    rows,cols = data.shape
#    
#    #first convert data to 1nm spacing through interpolation
#    lmin = int(min(data[:,0]))
#    lmax = int(max(data[:,0]))
#    lambdas = range(lmin, lmax+1)
#    out = _interpolate_data(data, lambdas)
#    #perform convolution with a rectangular filter of a given width
#    v = np.array([1./width]*width)
#    for i in range(1,cols):
#        out[:,i] = np.convolve(out[:,i],v, mode = "same")   
#    return out                 
   
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    import matplotlib.pyplot as plt
    num = 81
    for num in (81,40,20,10,5):
        wavelengths = np.linspace(380,780,num)
        x,xyz = load_tcmf(wavelengths, norm = True,retx = True, single_wavelength = True)
        
        plt.plot(x,xyz[:,0],label = num) #X color
        x,xyz = load_tcmf(wavelengths, norm = True,retx = True, single_wavelength = False)
        
        plt.plot(x,xyz[:,0],"o") #X color
    plt.legend()
    plt.show()

    
