"""Field visualizer (polarizing miscroscope simulator)"""

from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.image import imsave
import scipy.ndimage as nd

#from dtmm.project import projection_matrix, project
from dtmm.color import load_tcmf, specter2color
from dtmm.diffract import diffract, field_diffraction_matrix
from dtmm.polarization import mode_polarizer, ray_polarizer, normal_polarizer
from dtmm.field import field2specter
from dtmm.wave import k0
from dtmm.data import refind2eps
from dtmm.conf import BETAMAX, CDTYPE

from dtmm.linalg import dotmf, dotmm
from dtmm.fft import fft2, ifft2

#: settable viewer parameters
VIEWER_PARAMETERS = ("focus","analyzer", "polarizer", "sample", "intensity")


def _redim(a, ndim=1):
    """Reshapes dimensions of input array by flattenig over first few dimensions. If
    ndim is larger than input array ndim, it adds new axes to input array.
    
    >>> a = np.zeros((4,5,6,7))
    >>> _redim(a,ndim = 3).shape
    (20,6,7)
    >>> _redim(a,ndim = 5).shape
    (1,4,5,6,7)
    
    """
    n = a.ndim - ndim 
    old_shape = a.shape
    if n < 0:
        new_shape = (1,)*abs(n) + old_shape
    else:
        new_shape = (np.multiply.reduce(old_shape[0:n+1]),) + old_shape[n+1:]
    return a.reshape(new_shape)


def bulk_viewer(field_data, cmf=None, window=None, **parameters):
    """
    Returns a FieldViewer object for optical microscope simulation
    
    Parameters
    ----------
    field_data : tuple[np.ndarray]
        Input field data
    cmf : str, ndarray or None, optional
        Color matching function (table). If provided as an array, it must match 
        input field wavelengths. If provided as a string, it must match one of 
        available CMF names or be a valid path to tabulated data. See load_tcmf.
    window : ndarray, optional
        Window function by which the calculated field is multiplied. This can 
        be used for removing artefact from the boundaries.
    parameters : kwargs, optional
        Extra parameters passed directly to the :meth:`FieldViewer.set_parameters`
        
    Returns
    -------
    out : BulkViewer
        A :class:`BulkViewer` viewer object 
    
    """    
    return field_viewer(field_data, cmf, bulk_data=True, window=window, **parameters)


def field_viewer(field_data, cmf=None, bulk_data=False, n=1., mode=None,
                 window=None, diffraction=True, polarization_mode="normal", betamax=BETAMAX, **parameters):
    """
    Returns a FieldViewer object for optical microscope simulation
    
    Parameters
    ----------
    field_data : tuple[np.ndarray]
        Input field data
    cmf : str, ndarray or None, optional
        Color matching function (table). If provided as an array, it must match 
        input field wavelengths. If provided as a string, it must match one of 
        available CMF names or be a valid path to tabulated data. See load_tcmf.
    bulk_data: bool
        # TODO: I don't know what this value is
    n : float, optional
        Refractive index of the output material.
    mode : [ 't' | 'r' | None], optional
        Viewer mode 't' for transmission mode, 'r' for reflection mode None for
        as is data (no projection calculation - default).
    window : ndarray, optional
        Window function by which the calculated field is multiplied. This can 
        be used for removing artefact from the boundaries.
    diffraction : bool, optional
        Specifies whether field is treated as diffractive field or not (if it
        was calculated by diffraction > 0 algorithm or not). If set to False
        refocusing is disabled.
    polarization_mode : str, optional
        Defines polarization mode. That is, how the polarization of the light is
        treated after passing the analyzer. By default, polarizer is applied
        in real space (`mode`) which is good for normal (or mostly normal) 
        incidence light. You can use `mode` instead of `normal` for more 
        accurate, but slower computation. Here polarizers are applied to 
        mode coefficients in fft space. 
    betamax : float
        Betamax parameter used in the diffraction calculation function. With this
        you can simulate finite NA of the microscope (NA = betamax).
    parameters : kwargs, optional
        Extra parameters passed directly to the :meth:`FieldViewer.set_parameters`
        
    Returns
    -------
    out : FieldViewer
        A :class:`FieldViewer` or :class:`BulkViewer` viewer object 
    
    """
    # Valid polarization modes
    valid_polarization_modes = ("mode", "normal")
    # Extract components out of field_data
    field, wavelengths, pixelsize = field_data
    # Convert wavelengths and pixel size to wave numbers
    wave_numbers = k0(wavelengths, pixelsize)
    
    if not diffraction and mode is not None:
        import warnings
        warnings.warn("Diffraction has been enabled because projection mode is set!")
        diffraction = True

    # Check that the provided polarization mode is a value one
    if polarization_mode not in valid_polarization_modes:
        raise ValueError("Unknown polarization mode, should be one of {}".format(repr(valid_polarization_modes)))

    # Ensure a color matching function will be used
    if cmf is None:
        cmf = load_tcmf(wavelengths)
    elif isinstance(cmf, str):
        cmf = load_tcmf(wavelengths, cmf=cmf)

    if not bulk_data:
        if field.ndim < 4:
            raise ValueError("Incompatible field shape")

        viewer = FieldViewer(field, wave_numbers, cmf, mode=mode, n=n,
                             window=window, diffraction=diffraction,
                             polarization=polarization_mode, betamax=betamax)
        
        viewer.set_parameters(**parameters)
    else:
        if field.ndim < 5:
            raise ValueError("Incompatible field shape")

        parameters.setdefault("focus", 0)
        viewer = BulkViewer(field, wave_numbers, cmf, mode=mode, n=n,
                            window=window, diffraction=diffraction,
                            polarization=polarization_mode, betamax=betamax)
        viewer.set_parameters(**parameters)        
    return viewer


def _float_or_none(value):
    """
    Helper function to convert the passed value to a float and return it, or return None.

    Parameters
    ----------
    value : SupportsFloat, _SupportsIndex, str, bytes, bytearray
        A value which can be represented as a float, or None.

    Returns
    -------
    value: float, optional
        The passed value represented as a float, or None if it does not exist.
    """
    return float(value) if value is not None else None
 

class FieldViewer(object): 
    """Base viewer"""  
    _updated_parameters = set()
    _focus = None
    _polarizer = None
    _sample = None
    _analyzer = None
    _intensity = 1.
    _parameters = VIEWER_PARAMETERS
    _fmin = 0
    _fmax = 100
    ofield = None
    gamma = True
    gray = False
    
    def __init__(self,field,ks,cmf, mode = None,n = 1., polarization = "normal",
                window = None, diffraction = True, betamax = BETAMAX):
        self.betamax = betamax
        self.diffraction = diffraction
        self.pmode = polarization
        self.mode = mode  
        self.epsv = refind2eps([n,n,n])
        self.epsa = np.array([0.,0.,0.])
        self.ks = ks
        self.ifield = field 
        self._ffield = None
        self.window = window
        self.cmf = cmf
        self.dmat = None
        
    @property
    def _default_fmin(self):
        return self.focus - 100
    
    @property
    def _default_fmax(self):
        return self.focus + 100    
    
    @property
    def focus(self):
        """Focus position, relative to the calculated field position."""
        return self._focus   
    
    @property
    def ffield(self):
        if self._ffield is None:
            self._ffield = fft2(self.ifield)
        return self._ffield

    @focus.setter     
    def focus(self, z):
        if self.diffraction == True or z is None:
            self._focus = _float_or_none(z)
            self._updated_parameters.add("focus")
        else:
            raise ValueError("Cannot set focus of a non-diffractive field.")

    @property
    def sample(self):
        """Sample rotation angle."""
        return self._sample
    
    @sample.setter    
    def sample(self, angle):
        self._sample = _float_or_none(angle)
        self._updated_parameters.add("sample")   

    @property
    def polarizer(self):
        """Polarizer rotation angle."""
        return self._polarizer
        
    @polarizer.setter 
    def polarizer(self, angle):
        if angle is not None and self.ifield.ndim >= 5 and self.ifield.shape[-5] != 2:
            raise ValueError("Cannot set polarizer. Incompatible field shape.")
        self._polarizer = _float_or_none(angle)
        self._updated_parameters.add("polarizer")
        
    @property
    def analyzer(self):
        """Analyzer angle"""
        return self._analyzer    
        
    @analyzer.setter   
    def analyzer(self, angle):
        self._analyzer = _float_or_none(angle)
        self._updated_parameters.add("analyzer")
        
    @property
    def intensity(self):
        """Input light intensity"""
        return self._intensity   
    
    @intensity.setter   
    def intensity(self, intensity):
        self._intensity = _float_or_none(intensity)
        self._updated_parameters.add("intensity")
        
    def set_parameters(self, **kwargs):
        """Sets viewer parameters. Any of the :attr:`.VIEWER_PARAMETERS`
        """
        for key, value in kwargs.items():
            if key in self._parameters:
                setattr(self, key, value) 
            else:
                raise TypeError("Unexpected keyword argument '{}'".format(key))
    def get_parameters(self):
        """Returns viewer parameters as dict"""
        return {name : getattr(self,name) for name in VIEWER_PARAMETERS}
        
    def plot(self, ax = None, show_sliders = True, **kwargs):
        """Plots field intensity profile. You can set any of the below listed
        arguments. Additionaly, you can set any argument that imshow of
        matplotlib uses (e.g. 'interpolation = "sinc"').
        
        Parameters
        ----------
        fmin : float, optional
            Minimimum value for focus setting.
        fmax : float, optional
            Maximum value for focus setting.             
        imin : float, optional
            Minimimum value for intensity setting.
        imax : float, optional
            Maximum value for intensity setting.     
        pmin : float, optional
            Minimimum value for polarizer angle.
        pmax : float, optional
            Maximum value for polarizer angle.    
        smin : float, optional
            Minimimum value for sample rotation angle.
        smax : float, optional
            Maximum value for sample rotation angle.  
        amin : float, optional
            Minimimum value for analyzer angle.
        amax : float, optional
            Maximum value for analyzer angle.  
        """

        self.fig = plt.figure() if ax is None else ax.figure
        self.ax = self.fig.add_subplot(111) if ax is None else ax
        
        plt.subplots_adjust(bottom=0.25)  
        self.calculate_image()
        
        if show_sliders:

            def update_sample(d):
                self.sample = d
                self.update_plot()
                
            def update_focus(d):
                self.focus = d
                self.update_plot()
            
            def update_intensity(d):
                self.intensity = d
                self.update_plot()
                
            def update_analyzer(d):
                self.analyzer = d
                self.update_plot()
    
            def update_polarizer(d):
                self.polarizer = d
                self.update_plot()
                
            axes = [[0.25, 0.14, 0.65, 0.03],
                    [0.25, 0.11, 0.65, 0.03],
                    [0.25, 0.08, 0.65, 0.03],
                    [0.25, 0.05, 0.65, 0.03],
                    [0.25, 0.02, 0.65, 0.03]]
            
            if self.intensity is not None:
                self.axintensity = plt.axes(axes.pop())
                self._sintensity = Slider(self.axintensity, "intensity",kwargs.pop("imin",0),kwargs.pop("imax",max(10,self.intensity)),valinit = self.intensity, valfmt='%.1f')
                self._ids5 = self._sintensity.on_changed(update_intensity)
            if self.polarizer is not None:
                self.axpolarizer = plt.axes(axes.pop())
                self._spolarizer = Slider(self.axpolarizer, "polarizer",kwargs.pop("pmin",0),kwargs.pop("pmax",90),valinit = self.polarizer, valfmt='%.1f')
                self._ids4 = self._spolarizer.on_changed(update_polarizer)    
            if self.sample is not None:
                self.axsample = plt.axes(axes.pop())
                self._ssample = Slider(self.axsample, "sample",kwargs.pop("smin",-180),kwargs.pop("smax",180),valinit = self.sample, valfmt='%.1f')
                self._ids3 = self._ssample.on_changed(update_sample)    
            if self.analyzer is not None:
                self.axanalyzer = plt.axes(axes.pop())
                self._sanalyzer = Slider(self.axanalyzer, "analyzer",kwargs.pop("amin",0),kwargs.pop("amax",90),valinit = self.analyzer, valfmt='%.1f')
                self._ids2 = self._sanalyzer.on_changed(update_analyzer)
            if self.focus is not None:    
                self.axfocus = plt.axes(axes.pop())
                self._sfocus = Slider(self.axfocus, "focus",kwargs.pop("fmin",self._default_fmin),kwargs.pop("fmax",self._default_fmax),valinit = self.focus, valfmt='%.1f')
                self._ids1 = self._sfocus.on_changed(update_focus)
            
        self.axim = self.ax.imshow(self.image, origin = kwargs.pop("origin","lower"), **kwargs)
        
        return self.ax.figure, self.ax

    def calculate_specter(self, recalc = False, **params):
        """Calculates field specter.
        
        Parameters
        ----------
        recalc : bool, optional
            If specified, it forces recalculation. Otherwise, result is calculated
            only if calculation parameters have changed.
        params: kwargs, optional
            Any additional keyword arguments that are passed dirrectly to 
            set_parameters method.
        """
        if self.pmode == "mode":
            return self._calculate_specter_mode(recalc = recalc, **params)
        else:
            return self._calculate_specter_normal(recalc = recalc, **params)
            
    def _calculate_specter_normal(self, recalc = False, **params):
        """Calculates field specter.
        
        Parameters
        ----------
        recalc : bool, optional
            If specified, it forces recalculation. Otherwise, result is calculated
            only if calculation parameters have changed.
        params: kwargs, optional
            Any additional keyword arguments that are passed dirrectly to 
            set_parameters method.
        """
        self.set_parameters(**params)
        if self.ofield is None:
            recalc = True #first time only trigger calculation 
        if recalc or "focus" in self._updated_parameters:
            if self.diffraction == True or self.mode is not None:
                #if mode is selected, we need to project the filed using diffraction
                d = 0 if self.focus is None else self.focus
                dmat = field_diffraction_matrix(self.ifield.shape[-2:], self.ks,  d = d, 
                                          epsv = self.epsv, epsa = self.epsa, 
                                          mode = self.mode, betamax = self.betamax)
                
                self.ofield = diffract(self.ifield,dmat,window = self.window,out = self.ofield)
            else:
                #no diffraction at all..
                if self.window is not None:
                    self.ofield = self.ifield * self.window
                else:
                    self.ofield = self.ifield.copy()
            recalc = True
        if recalc or "polarizer" in self._updated_parameters or "analyzer" in self._updated_parameters or "sample" in self._updated_parameters:
            sample = self.sample
            if sample is None:
                sample = 0.
            if self.polarizer is None:
                tmp = _redim(self.ofield, ndim = 5)
                out = np.empty_like(tmp[0])
            else:
                angle = -np.pi/180*(self.polarizer - sample)
                c,s = np.cos(angle),np.sin(angle)  
                tmp = _redim(self.ofield, ndim = 6)
                out = np.empty_like(tmp[0,0])
            if self.analyzer is not None:
                
                angle = -np.pi/180*(self.analyzer - sample)
                #pmat = linear_polarizer(angle)
                pmat = normal_polarizer((np.cos(angle),np.sin(angle)))
                #pmat = ray_polarizer((np.cos(angle),np.sin(angle)),epsv = self.epsv, epsa = self.epsa)
                 
            for i,data in enumerate(tmp):
                if self.polarizer is not None:
                    x = data[0]*c
                    y = np.multiply(data[1], s, out = out)
                    ffield = np.add(x,y, out = out)#numexpr.evaluate("x*c+y*s", out = out)
                else: 
                    ffield = data
                    
                if self.analyzer is not None:
                    #pfield = apply_jones_matrix(pmat, ffield, out = out)
                    pfield = dotmf(pmat, ffield, out = out)
                else:
                    pfield = ffield
                if i == 0:
                    self.specter = field2specter(pfield)  
                else:
                    self.specter += field2specter(pfield) 
            recalc = True
        
        if recalc or "intensity" in self._updated_parameters:
            self._updated_parameters.clear()
            self._updated_parameters.add("intensity") #trigger calculate_image call
        else:
             self._updated_parameters.clear()
        return self.specter
    
    def _has_parameter_updated(self, *params):
        for p in params:
            if p in self._updated_parameters:
                return True
        return False

    def _calculate_specter_mode(self, recalc = False, **params):
        self.set_parameters(**params)
        if self.ofield is None:
            recalc = True #first time only trigger calculation 
        if recalc or self._has_parameter_updated("sample", "polarizer"):
            sample = self.sample if self.sample is not None else 0.
            if self.polarizer is not None:
                if self.ffield is None:
                    self.ffield = fft2(self.ifield)
                angle = -np.pi/180*(self.polarizer - sample)            
                c,s = np.cos(angle),np.sin(angle)  
                
                self.data = _redim(self.ffield, ndim = 6)
                x = c*self.data[:,0]
                y = s*self.data[:,1]
                self.data = x+y
            else:
                self.data = _redim(self.ffield, ndim = 5)
                
        if recalc or self._has_parameter_updated("focus"):
            if self.diffraction == True or self.mode is not None:
                #if mode is selected, we need to project the field using diffraction
                d = 0 if self.focus is None else self.focus
                self.dmat = field_diffraction_matrix(self.ifield.shape[-2:], self.ks,  d = d, 
                                          epsv = self.epsv, epsa = self.epsa, 
                                          mode = self.mode, betamax = self.betamax)
            else:
                self.dmat = np.asarray(np.diag((1,1,1,1)), CDTYPE)      
        if recalc or self._has_parameter_updated("analyzer", "sample") :
            sample = self.sample if self.sample is not None else 0.
            if self.analyzer is not None:
                angle = -np.pi/180*(self.analyzer - sample)
                c,s = np.cos(angle),np.sin(angle) 
                self.pmat = mode_polarizer(self.ifield.shape[-2:], self.ks,  jones = (c,s),
                                          epsv = self.epsv, epsa = self.epsa, 
                                          betamax = self.betamax) 
            else:
                self.pmat = None

        if recalc or self._has_parameter_updated("analyzer", "sample", "polarizer", "focus", "intensity") :
            tmat = None
            if self.pmat is not None and self.dmat is not None:
                tmat = dotmm(self.pmat,self.dmat)
            if self.pmat is None and self.dmat is not None:
                tmat = self.dmat
            if self.pmat is not None and self.dmat is None:
                tmat = self.pmat
            if tmat is not None:
                self.ofield = dotmf(tmat, self.data, out = self.ofield)
            self.ofield = ifft2(self.ofield, out = self.ofield)
            
            for i,data in enumerate(self.ofield):
                if i == 0:
                    self.specter = field2specter(data)  
                else:
                    self.specter += field2specter(data) 
                    
            recalc = True

        
        if recalc or "intensity" in self._updated_parameters:
            self._updated_parameters.clear()
            self._updated_parameters.add("intensity") #trigger calculate_image call
        else:
            self._updated_parameters.clear()        
            
        return self.specter
      
    def calculate_image(self, recalc = False, **params):
        """Calculates RGB image.
        
        Parameters
        ----------
        recalc : bool, optional
            If specified, it forces recalculation. Otherwise, result is calculated
            only if calculation parameters have changed.
        params: keyword arguments
            Any additional keyword arguments that are passed dirrectly to 
            set_parameters method.
            
        """   
        specter = self.calculate_specter(recalc,**params)
        if recalc or "intensity" in self._updated_parameters:
            if self.intensity is not None:
                if self.intensity != 0.0:
                    if self.mode == "r":
                        norm = -1./self.intensity
                    else:
                        norm = 1./self.intensity
                else:
                    norm = 0.0

                self.image = specter2color(specter,self.cmf, norm = norm, gamma = self.gamma, gray = self.gray) 
            else:
                if self.mode == "r":
                    self.image = specter2color(specter,self.cmf, norm = -1., gamma = self.gamma, gray = self.gray) 
                else:
                    self.image = specter2color(specter,self.cmf, gamma = self.gamma, gray = self.gray) 
            if self.sample != 0 and self.sample is not None:
                self.image = nd.rotate(self.image, self.sample, reshape = False, order = 1) 
        self._updated_parameters.clear()
        return self.image
    
    def save_image(self, fname, origin = "lower", **kwargs):
        """Calculates and saves image to file using matplotlib.image.imsave.
        
        Parameters
        ----------
        fname : str
            Output filename or file object.
        origin : [ 'upper' | 'lower' ]
            Indicates whether the (0, 0) index of the array is in the upper left 
            or lower left corner of the axes. Defaults to 'lower' 
        kwargs : optional
            Any extra keyword argument that is supported by matplotlib.image.imsave
        """
        im = self.calculate_image()
        imsave(fname, im, origin = origin, **kwargs)

    def update_plot(self):
        """Triggers plot redraw"""
        self.calculate_image()
        self.axim.set_data(self.image)
        self.fig.canvas.draw_idle()  
   
    def show(self):
        """Shows plot"""
        plt.show()

                 
class BulkViewer(FieldViewer):
    @property
    def _default_fmin(self):
        return 0
    
    @property
    def _default_fmax(self):
        return len(self.ifield) -1   

    @property
    def ffield(self):
        if self._ffield is None:
            self._ffield = fft2(self.ifield[self.focus])
        return self._ffield

    @ffield.setter
    def ffield(self, value):
        self._ffield = value
    
    @property
    def focus(self):
        """Focus position, relative to the calculated field position."""
        return self._focus       
    
    @focus.setter     
    def focus(self, z):
        self._focus = int(z)
        self._updated_parameters.add("focus")
        
    def _calculate_specter_mode(self, recalc = False, **params):
        self.set_parameters(**params)

        if self.ofield is None:
            recalc = True #first time only trigger calculation 
            
        if recalc or self._has_parameter_updated("analyzer", "sample") :
            sample = self.sample if self.sample is not None else 0.
            if self.analyzer is not None:
                angle = -np.pi/180*(self.analyzer - sample)
                c,s = np.cos(angle),np.sin(angle) 
                self.pmat = mode_polarizer(self.ifield.shape[-2:], self.ks,  jones = (c,s),
                                          epsv = self.epsv, epsa = self.epsa, 
                                          betamax = self.betamax) 
                if self.pmode != "mode":
                    self.pmat = self.pmat[...,0:1,0:1,:,:]


        if recalc or self._has_parameter_updated("focus"):
            if self.mode is None:
                self.ffield = fft2(self.ifield[self.focus])
            else:
                self.dmat = field_diffraction_matrix(self.ifield.shape[-2:], self.ks,  d = 0, 
                                      epsv = self.epsv, epsa = self.epsa, 
                                      mode = self.mode, betamax = self.betamax)
                self.ffield = fft2(self.ifield[self.focus])

            recalc = True #trigger update of self.data

        if recalc or self._has_parameter_updated("sample", "polarizer"):
            sample = self.sample if self.sample is not None else 0.
            if self.polarizer is not None:
                angle = -np.pi/180*(self.polarizer - sample)            
                c,s = np.cos(angle),np.sin(angle)  
                
                self.data = _redim(self.ffield, ndim = 6)
                x = c*self.data[:,0]
                y = s*self.data[:,1]
                self.data = x+y
            else:
                self.data = _redim(self.ffield, ndim = 5)
     

        if recalc or self._has_parameter_updated("analyzer", "sample", "polarizer", "focus", "intensity") :
            if self.dmat is not None:
                pmat = dotmm(self.pmat,self.dmat)
            else:
                pmat  = self.pmat
            self.ofield = dotmf(pmat, self.data, out = self.ofield)
            self.ofield = ifft2(self.ofield, out = self.ofield)
            
            for i,data in enumerate(self.ofield):
                if i == 0:
                    self.specter = field2specter(data)  
                else:
                    self.specter += field2specter(data) 
            recalc = True

        
        if recalc or "intensity" in self._updated_parameters:
            self._updated_parameters.clear()
            self._updated_parameters.add("intensity") #trigger calculate_image call
        else:
            self._updated_parameters.clear()        
            
        return self.specter
    
        
    def _calculate_specter_normal(self, recalc = False, **params):
        self.set_parameters(**params)
        if self.ofield is None:
            recalc = True #first time only trigger calculation 
        if recalc or "focus" in self._updated_parameters:
            if self.mode is None:
                self.ofield = self.ifield[self.focus]
            else:
                dmat = field_diffraction_matrix(self.ifield.shape[-2:], self.ks,  d = 0, 
                                      epsv = self.epsv, epsa = self.epsa, 
                                      mode = self.mode, betamax = self.betamax)
                self.ofield = diffract(self.ifield[self.focus],dmat,window = self.window,out = self.ofield)
            recalc = True

        if recalc or "polarizer" in self._updated_parameters or "analyzer" in self._updated_parameters or "sample" in self._updated_parameters:
            sample = self.sample
            if sample is None:
                sample = 0.
            if self.polarizer is None:
                tmp = _redim(self.ofield, ndim = 5)
                out = np.empty_like(tmp[0])
            else:
                angle = -np.pi/180*(self.polarizer - sample)
                c,s = np.cos(angle),np.sin(angle)  
                tmp = _redim(self.ofield, ndim = 6)
                out = np.empty_like(tmp[0,0])
            if self.analyzer is not None:
                angle = -np.pi/180*(self.analyzer - sample)
                #pmat = linear_polarizer(angle)
                pmat = ray_polarizer((np.cos(angle),np.sin(angle)),epsv = self.epsv, epsa = self.epsa)
                
            
            for i,data in enumerate(tmp):
                if self.polarizer is not None:
                    x = data[0]*c
                    y = np.multiply(data[1], s, out = out)
                    ffield = np.add(x,y, out = out)#numexpr.evaluate("x*c+y*s", out = out)
                else: 
                    ffield = data
                    
                if self.analyzer is not None:
                    pfield = dotmf(pmat, ffield, out = out)
                else:
                    pfield = ffield
                if i == 0:
                    self.specter = field2specter(pfield)  
                else:
                    self.specter += field2specter(pfield) 
            recalc = True
        
        if recalc or "intensity" in self._updated_parameters:
            self._updated_parameters.clear()
            self._updated_parameters.add("intensity") #trigger calculate_image call
        else:
             self._updated_parameters.clear()
        return self.specter
    

    

    