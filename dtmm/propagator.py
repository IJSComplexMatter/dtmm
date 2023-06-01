#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 08:43:42 2023

@author: andrej
"""
import numpy as np
from dtmm.wave import k0, eigenmask
from dtmm.conf import get_default_config_option, DTMMConfig, CDTYPE

AVAILABLE_PROPAGATOR_METHODS = '4x4', '2x2'


class BeamReadOnlyProperties(object):
    _dim = 3
    @property
    def dim(self):
        return self._dim
    
    _units = 'nm'
    @property
    def units(self):
        return self._units   
    
    _resolution = 1    
    @property
    def resolution(self):
        return self._resolution
    
    _pixelsize = 100
    @property
    def pixelsize(self):
        return self._pixelsize   
    
    _method = "4x4"
    @property
    def method(self):
        return self._method
    
    _wavelengths = None
    @property
    def wavelengths(self):
        return self._wavelengths  
    
    _wavenumbers = None
    @property
    def wavenumbers(self):
        return self._wavenumbers   
    
    _shape = None
    @property    
    def shape(self):
        return self._shape
    
    #: material shape
    _material_shape = None
    @property    
    def material_shape(self):
        return self._material_shape   
    
    #: material dimension
    _material_dim = None
    @property  
    def material_dim(self):
        return self._material_dim   

    _dispersive= None
    @property  
    def dispersive(self):
        return self._dispersive 
    
    _nin = None
    @property
    def nin(self):
        return self._nin
    
    _nout = None
    @property
    def nout(self):
        return self._nout    
    

class BasePropagator(BeamReadOnlyProperties):
    """Base class for all beam propagators"""
    
    #: module that implements TMM
    #tmm = tmm3d

    # field array data
    _field_out = None
    _field_in = None
    _modes_in = None
    _modes_out = None
    
    _mask = None
    
    #: resize parameter for layer_met calculation
    _resize = 1
    
    def __init__(self, shape, wavelengths = [500], pixelsize = 100, resolution = 1,  mask = None, method = "4x4", units = 'nm', betamax = None):
        """
        Paramters
        ---------
        shape : (int,int)
            Cross-section shape of the field data.
        wavelengths : float or array
            A list of wavelengths (in nanometers) or a single wavelength for 
            which to create the solver.
        pixelsize : float
            Pixel size in (nm).
        resolution : float
            Approximate sub-layer thickness (in units of pixelsize) used in the 
            calculation. With `resolution` = 1, layers thicker than `pixelsize` will
            be split into severeal thinner layers. Exact number of layers used
            in the calculation is obtained from :func:`get_optimal_steps`.
        mask : ndarray, optional
            A fft mode mask array. If not set, :func:`.wave.eigenmask` is 
            used with `betamax` to create such mask.
        method : str
            Either '4x4' (default), '4x4_1' or '2x2'.
        betamax : float, optional
            If `mask` is not set, this value is used to create the mask. If not 
            set, default value is taken from the config.
        """
        if units not in ('um', 'nm'):
            raise ValueError("Invalid units. Should be 'um' or 'nm'")
        self._units = units
        
        x,y = shape
        if not (isinstance(x, int) and isinstance(y, int)):
            raise ValueError("Invalid field shape.")
        self._shape = x,y
        self._wavelengths = np.asarray(wavelengths)
        if self.wavelengths.ndim not in (0,1):
            raise ValueError("`wavelengths` must be a scalar or an 1D array.")
        self._pixelsize = float(pixelsize)
        self._wavenumbers = k0(self.wavelengths, pixelsize)
        self._resolution = int(resolution)
        
        method = str(method) 
        if method in AVAILABLE_PROPAGATOR_METHODS :
            self._method = method
        else:
            raise ValueError("Unsupported method {}".format(method))
        
        if mask is None:
            betamax = get_default_config_option("betamax", betamax)
            self._mask = eigenmask(shape, self.wavenumbers, betamax)
        else:
            self.mask = mask

    def print_propagator_info(self):
        """prints propagator info"""
        print(" $ dim : {}".format(self.dim))
        print(" $ shape : {}".format(self.shape))
        print(" # pixel size : {}".format(self.pixelsize))
        print(" # resolution : {}".format(self.resolution))
        print(" $ method : {}".format(self.method))
        print(" $ wavelengths : {}".format(self.wavelengths))



class FieldData():
    def __init(self, field = None, wavelengths = [550], pixelsize = 100, units = 'nm'):
        self._field = field
        self._wavelengths = wavelengths
        self._pixelsize = pixelsize
        self._units = units
    
    @property
    def field(self):
        return self._field
    
    @property
    def wavelengths(self):
        return self._wavelengths
    
    @property
    def pixelsize(self):
        return self._pixelsize
    
    @property
    def units(self):
        return self._units
    
    def __len__(self):
        # for legacy support, act like a length three tuple of field, wavelengths, pixelsize 
        return 3
    
    def __getitem__(self, index):
        # for legacy support, act like a length three tuple of field, wavelengths, pixelsize 
        return (self.field, self.wavelengths, self.pixelsize)[index]
    
    @property
    def Ex(self):
        return self.field[...,0,:,:]   
    
    @property
    def Hy(self):
        return self.field[...,1,:,:]       

    @property
    def Ey(self):
        return self.field[...,2,:,:]   
    
    @property
    def Hx(self):
        return self.field[...,3,:,:]     
    
    def diffract(self, z, units = 'pixel', copy = True):
        pass
    
    def project(self, copy = True):
        pass
    
    def propagate(self, block):
        pass
        
    
class FieldPropagator():
    def __init__(self, overwrite_x = False, betamax = None):
        pass
    
    def __next__(self):
        return FieldData()

class BeamPropagator4x4(BasePropagator):
    def propagate_field(self, field_in):
        pass
    