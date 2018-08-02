"""Field visualizer (polarizing miscroscope simulator)"""

from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.image import imsave
import scipy.ndimage as nd

from dtmm.color import load_tcmf, specter2color
from dtmm.diffract import diffract, diffraction_matrix
from dtmm.field import field2specter
from dtmm.jones import linear_polarizer, apply_jones_matrix
from dtmm.wave import k0
from dtmm.data import refind2eps
from dtmm.conf import BETAMAX

#: settable viewer parameters
VIEWER_PARAMETERS = ("analyzer", "polarizer", "sample", "intensity", "focus")

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

def field_viewer(field_data, cmf = None, n = 1., mode = None,
                 window = None, betamax = BETAMAX, **parameters):
    """Returns a FieldViewer object for optical microsope simulation
    
    Parameters
    ----------
    field_data : field data tuple
        Input field data
    cmf : ndarray or None, optional
        Color matching function (table). If provided, it must match input field
        wavelengths.
    n : float, optional
        Refractive index of the output material.
    mode : [ 't' | 'r' | None], optional
        Viewer mode 't' for transmission mode, 'r' for reflection mode None for
        as is data (no projection calculation - default).
    window : ndarray, optional
        Window function by which the calculated field is multiplied. This can 
        be used for removing artefact from the boundaries.
    betamax : float
        Betamaz parameter used in the diffraction calculation function.
    parameters : kwargs, optional
        Extra parameters passed directly to the :meth:`FieldViewer.set_parameters`
        
    Returns
    -------
    out : viewer
        A :class:`FieldViewer` viewer object 
    
    """
    field, wavelengths, pixelsize = field_data
    wavenumbers = k0(wavelengths, pixelsize)
    if field.ndim < 4:
        raise ValueError("Incompatible field shape")
   
    if cmf is None:
        cmf = load_tcmf(wavelengths)
    viewer = FieldViewer(field, wavenumbers, mode = mode, n = n,
                       cmf = cmf, window = window, betamax = betamax)
    viewer.set_parameters(**parameters)
    return viewer

def _float_or_none(value):
    return float(value) if value is not None else None
            
class FieldViewer(object): 
    """Field viewer for optical polarizing microscope simulation."""      
    def __init__(self,field,ks, mode = "t",n = 1.,
                 cmf = None, window = None, betamax = BETAMAX):
        if field.ndim < 4:
            raise ValueError("Incompatible field shape")
        self.betamax = betamax
        self.mode = mode  
        self.epsv = refind2eps([n,n,n])
        self.epsa = np.array([0.,0.,0.])
        self.ks = ks
        self.ifield = field
        
        self.ofield = np.empty_like(field)
        
        self._updated_parameters = set()
        self._updated_parameters.add("recalc") #this triggers image calculation
        
        self.intensity = 1.
        self.polarizer = None
        self.sample = None
        self.analyzer = None
        self.focus = 0
        self.window = window
        
        if cmf is None:
            self.cmf = load_tcmf()
        else:
            self.cmf = cmf

        
    def plot(self, ax = None, **kwargs):
        """Plots field intensity profile.
        
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
        self.axim = self.ax.imshow(self.image, origin = "lower")

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
            self._sintensity = Slider(self.axintensity, "intensity",kwargs.get("imin",0),kwargs.get("imax",max(10,self.intensity)),valinit = self.intensity, valfmt='%.1f')
            self._ids5 = self._sintensity.on_changed(update_intensity)
        if self.polarizer is not None:
            self.axpolarizer = plt.axes(axes.pop())
            self._spolarizer = Slider(self.axpolarizer, "polarizer",kwargs.get("pmin",0),kwargs.get("pmax",90),valinit = self.polarizer, valfmt='%.1f')
            self._ids4 = self._spolarizer.on_changed(update_polarizer)    
        if self.sample is not None:
            self.axsample = plt.axes(axes.pop())
            self._ssample = Slider(self.axsample, "sample",kwargs.get("smin",-180),kwargs.get("smax",180),valinit = self.sample, valfmt='%.1f')
            self._ids3 = self._ssample.on_changed(update_sample)    
        if self.analyzer is not None:
            self.axanalyzer = plt.axes(axes.pop())
            self._sanalyzer = Slider(self.axanalyzer, "analyzer",kwargs.get("amin",0),kwargs.get("amax",90),valinit = self.analyzer, valfmt='%.1f')
            self._ids2 = self._sanalyzer.on_changed(update_analyzer)
        if self.focus is not None:    
            self.axfocus = plt.axes(axes.pop())
            self._sfocus = Slider(self.axfocus, "focus",kwargs.get("fmin",self.focus-100),kwargs.get("fmax",self.focus + 100),valinit = self.focus, valfmt='%.1f')
            self._ids1 = self._sfocus.on_changed(update_focus)

        return self.ax.figure, self.ax
    
    
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
    def focus(self):
        """Focus position, relative to the calculated field position."""
        return self._focus   

    @focus.setter     
    def focus(self, z):
        self._focus = _float_or_none(z)
        self._updated_parameters.add("focus")
        
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
            if key in VIEWER_PARAMETERS:
                setattr(self, key, value) 
            else:
                raise TypeError("Unexpected keyword argument '{}'".format(key))
    def get_parameters(self):
        """Returns viewer parameters as dict"""
        return {name : getattr(self,name) for name in VIEWER_PARAMETERS}
        
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
        self.set_parameters(**params)
        if recalc or "focus" in self._updated_parameters:
            d = 0 if self.focus is None else self.focus
            dmat = diffraction_matrix(self.ifield.shape[-2:], self.ks,  d = d, 
                                      epsv = self.epsv, epsa = self.epsa, 
                                      mode = self.mode, betamax = self.betamax)
            diffract(self.ifield,dmat,window = self.window,out = self.ofield)
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
                pmat = linear_polarizer(angle)
            
            for i,data in enumerate(tmp):
                if self.polarizer is not None:
                    x = data[0]*c
                    y = np.multiply(data[1], s, out = out)
                    ffield = np.add(x,y, out = out)#numexpr.evaluate("x*c+y*s", out = out)
                else: 
                    ffield = data
                    
                if self.analyzer is not None:
                    pfield = apply_jones_matrix(pmat, ffield, out = out)
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
    
    def calculate_image(self, gamma = True, gray = False, recalc = False, **params):
        """Calculates RGB image.
        
        Parameters
        ----------
        gamma : bool or float, optional
            Whether to apply standard sRGB gamma curve or not. If float, applies
            gamma curve with the provided gamma value.
        gray : bool
            Whether to convert RGB image to gray (intensity)
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
                if self.mode == "r":
                    norm = -1./self.intensity
                else:
                    norm = 1./self.intensity
                self.image = specter2color(specter,self.cmf, norm = norm, gamma = gamma, gray = gray) 
            else:
                if self.mode == "r":
                    self.image = specter2color(specter,self.cmf, norm = -1., gamma = gamma, gray = gray) 
                else:
                    self.image = specter2color(specter,self.cmf, gamma = gamma, gray = gray) 
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



    