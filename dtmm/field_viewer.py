"""Field visualizer (polarizing miscroscope simulator)"""

from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, AxesWidget, RadioButtons
from matplotlib.image import imsave
import scipy.ndimage as nd


#from dtmm.project import projection_matrix, project
from dtmm.color import load_tcmf, specter2color
from dtmm.diffract import diffract, field_diffraction_matrix
from dtmm.polarization import ray_jonesmat4x4, mode_jonesmat4x4, ray_polarizer
from dtmm.field import field2specter
from dtmm.wave import k0
from dtmm.data import refind2eps
from dtmm.conf import BETAMAX, CDTYPE
from dtmm.jones import jonesvec, quarter_waveplate, half_waveplate
from dtmm import jones

from dtmm.linalg import dotmf, dotmm, dotmv
from dtmm.fft import fft2, ifft2

#: settable viewer parameters
VIEWER_PARAMETERS = ("focus","analyzer", "polarizer", "sample", "intensity", "cols","rows","gamma","gray","aperture", "retarder")

SAMPLE_LABELS = ("-90 ","-45 "," 0  ","+45 ", "+90 ")
SAMPLE_NAMES = tuple((s.strip() for s in SAMPLE_LABELS))
POLARIZER_LABELS = (" H  "," V  ","LCP ","RCP ","none")
POLARIZER_NAMES = tuple((s.strip().lower() for s in POLARIZER_LABELS))
RETARDER_LABELS = ("$\lambda/4$", "$\lambda/2$","none")
RETARDER_NAMES = ("lambda/4", "lambda/2", "none")

class CustomRadioButtons(RadioButtons):

    def __init__(self, ax, labels, active=0, activecolor='blue', size=49,
                 orientation="horizontal", **kwargs):
        """
        Add radio buttons to an `~.axes.Axes`.
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The axes to add the buttons to.
        labels : list of str
            The button labels.
        active : int
            The index of the initially selected button.
        activecolor : color
            The color of the selected button.
        size : float
            Size of the radio buttons
        orientation : str
            The orientation of the buttons: 'vertical' (default), or 'horizontal'.
        Further parameters are passed on to `Legend`.
        """
        AxesWidget.__init__(self, ax)
        self.activecolor = activecolor
        axcolor = ax.get_facecolor()
        self.value_selected = None

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)

        circles = []
        for i, label in enumerate(labels):
            if i == active:
                self.value_selected = label
                facecolor = activecolor
            else:
                facecolor = axcolor
            p = ax.scatter([],[], s=size, marker="o", edgecolor='black',
                           facecolor=facecolor)
            circles.append(p)
        if orientation == "horizontal":
            kwargs.update(ncol=len(labels), mode="expand")
        kwargs.setdefault("frameon", False)    
        self.box = ax.legend(circles, labels, loc="center", **kwargs)
        self.labels = self.box.texts
        self.circles = self.box.legendHandles
        for c in self.circles:
            c.set_picker(5)
        self.cnt = 0
        self.observers = {}

        self.connect_event('pick_event', self._clicked)


    def _clicked(self, event):
        if (self.ignore(event) or event.mouseevent.button != 1 or
            event.mouseevent.inaxes != self.ax):
            return
        if event.artist in self.circles:
            self.set_active(self.circles.index(event.artist))



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
                 window=None, diffraction=True, polarization_mode="normal", betamax=BETAMAX, beta = None, **parameters):
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
        in real space (`normal`) which is good for normal (or mostly normal) 
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
                             polarization=polarization_mode, betamax=betamax, beta = beta)
        
        viewer.set_parameters(**parameters)
    else:
        if field.ndim < 5:
            raise ValueError("Incompatible field shape")

        parameters.setdefault("focus", 0)
        viewer = BulkViewer(field, wave_numbers, cmf, mode=mode, n=n,
                            window=window, diffraction=diffraction,
                            polarization=polarization_mode, betamax=betamax, beta = beta)
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
 
def _jonesmatrix_type(value):
    if isinstance(value, str):
        name = value.lower().strip()
        if name in (POLARIZER_NAMES + RETARDER_NAMES + ("x","y")+RETARDER_LABELS):
            return name, _jmat_from_name(name)
        else:
            raise ValueError("Not a valid jones matrix")
    else:
        try:
            angle = _float_or_none(value)
            return angle, _jmat_from_angle(angle)
        except TypeError:
            jmat = np.asarray(value,CDTYPE)
            if jmat.shape != (2,2):
                raise ValueError("Not a valid jones matrix")
            return jmat, jmat
        
def _jonesvector_type(value):
    if isinstance(value, str):
        name = value.lower().strip()
        if name in (POLARIZER_NAMES + ("x","y")):
            return name, _jvec_from_name(name)
        else:
            raise ValueError("Unknown jones vector!")
    else:
        try:
            angle = _float_or_none(value)
            return angle, _jvec_from_angle(angle)
        except TypeError:
            jvec = np.asarray(value,CDTYPE)
            if jvec.shape != (2,):
                raise ValueError("Not a valid jones vector")
            return jvec, jvec

def _jmat_from_name(name):
    if name in ("qplate","$\lambda/4$"):
        return jones.quarter_waveplate(np.pi/4)
    if name in ("hplate","$\lambda/2$"):
        return jones.half_waveplate(np.pi/4)
    if name == "none":
        return None
    return jones.polarizer(_jvec_from_name(name)) 

def _jvec_from_name(name):
    if name in ("h","x"):
        return jones.jonesvec((1,0))
    if name in ("v","y"):
        return jones.jonesvec((0,1))
    if name == "rcp":
        return jones.jonesvec((1,-1j))
    if name == "lcp":
        return jones.jonesvec((1,1j))
    if name == "none":
        return None
          
def _jmat_from_angle(angle):
    return jones.polarizer(_jvec_from_angle(angle)) if angle is not None else None 

def _jvec_from_angle(angle):
    return jonesvec((np.cos(np.radians(angle)),np.sin(np.radians(angle)))) if angle is not None else None 

def _is_unpolarized(field):
    """Determines if input field is unpolarized"""
    return True if len(field.shape) >= 5 and field.shape[-5] == 2 else False


class FieldViewer(object): 
    """Base viewer"""  
    _updated_parameters = set()
    _retarder = "none"
    _retarder_jmat = None
    _rows = 1
    _cols = 1
    _focus = 0
    _polarizer = None
    _polarizer_jvec = None
    _sample = None
    _sample_angle = None
    _analyzer = "none"
    _analyzer_jmat = None
    _intensity = 1.
    _parameters = VIEWER_PARAMETERS
    _fmin = 0
    _fmax = 100
    ofield = None
    gamma = True
    gray = False
    
    def __init__(self,field,ks,cmf, mode = None,n = 1., polarization = "normal",
                window = None, diffraction = True, betamax = BETAMAX, beta = None):
        self.betamax = betamax
        self.diffraction = diffraction
        self.pmode = polarization
        self.mode = mode  
        self.beta = np.asarray(beta)
        self.epsv = refind2eps([n,n,n])
        self.epsa = np.array([0.,0.,0.])
        self.ks = ks
        self.ifield = np.asarray(field)
        if _is_unpolarized(self.ifield):
            self.polarizer = "none"
            self.sample = "none"
            
        self._ffield = None
        self.window = window
        self.cmf = cmf
        self.dmat = None
        self._aperture = None if beta is None else max(beta)
        
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
        """Fourier transform of the field"""
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
    def cols(self):
        """Number of columns used (for periodic tructures)"""
        return self._cols
    
    @cols.setter    
    def cols(self, cols):
        self._cols = max(1,int(cols))
        self._updated_parameters.add("cols")   

    @property
    def rows(self):
        return self._rows
    
    @rows.setter    
    def rows(self, rows):
        """Number of rows used (for periodic tructures)"""
        self._rows = max(1,int(rows))
        self._updated_parameters.add("rows")   

    @property
    def sample(self):
        """Sample rotation angle"""
        return self._sample
    
    @property
    def sample_angle(self):
        """Sample rotation angle in degrees in float"""
        return self._sample_angle
    
    @sample.setter    
    def sample(self, angle):
        """Sample rotation angle in degrees, in float or as a string"""
        if angle is not None:
            if not _is_unpolarized(self.ifield):
                raise ValueError("Input field must be unpolarized to use sample parameter!")
            
        if isinstance(angle, str):
            labels = tuple((label.strip() for label in SAMPLE_LABELS))
            if angle.strip() not in labels:
                if angle in ("none","+0","-0"):
                    angle = "0"
                else:
                    raise ValueError("sample angle must be a float or any of {}".format(labels))
        self._sample_angle = _float_or_none(angle)
        self._sample = angle
        self._updated_parameters.add("sample")  
        

        
    @property
    def polarizer_jvec(self):
        return self._polarizer_jvec
    
    @polarizer_jvec.setter 
    def polarizer_jvec(self, value):
        self.polarizer = value
        
    @property
    def analyzer_jmat(self):
        return self._analyzer_jmat
    
    @analyzer_jmat.setter 
    def analyzer_jmat(self, value):
        self.analyzer = value
    
    @property    
    def aperture(self):
        """Illumination field aperture"""
        return self._aperture
    
    @aperture.setter    
    def aperture(self, value):
        self._aperture = _float_or_none(value)
        self._updated_parameters.add("aperture")


    @property
    def polarizer(self):
        """Polarizer angle. Can be 'h','v', 'lcp', 'rcp', 'none', angle float or a jones vector"""
        return self._polarizer
        
    @polarizer.setter 
    def polarizer(self, angle):
        if angle is not None:
            if not _is_unpolarized(self.ifield):
                raise ValueError("Input field must be unpolarized to use polarizer parameter!")
        if angle is not None and self.ifield.ndim >= 5 and self.ifield.shape[-5] != 2:
            raise ValueError("Cannot set polarizer. Incompatible field shape.")
        self._polarizer, self._polarizer_jvec = _jonesvector_type(angle)
        self._updated_parameters.add("polarizer")
        
    @property
    def analyzer(self):
        """Analyzer angle. Can be 'h','v', 'lcp', 'rcp', 'none', angle float or a jones vector"""
        return self._analyzer    
        
    @analyzer.setter   
    def analyzer(self, angle):
        self._analyzer, self._analyzer_jmat = _jonesmatrix_type(angle)
        self._updated_parameters.add("analyzer")
 
    @property    
    def retarder_jmat(self):
        return self._retarder_jmat
    
    @property    
    def retarder(self):
        return self._retarder
    
    @retarder.setter    
    def retarder(self, value):
        self._retarder, self._retarder_jmat = _jonesmatrix_type(value)
        self._updated_parameters.add("retarder")
       
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
        
    def plot(self, fig = None,ax = None, sliders = None, show_sliders = True, **kwargs):
        """Plots field intensity profile. You can set any of the below listed
        arguments. Additionaly, you can set any argument that imshow of
        matplotlib uses (e.g. 'interpolation = "sinc"').
        
        Parameters
        ----------
        fmin : float, optional
            Minimimum value for the focus setting.
        fmax : float, optional
            Maximum value for the focus setting.             
        imin : float, optional
            Minimimum value for then intensity setting.
        imax : float, optional
            Maximum value for then intensity setting.     
        pmin : float, optional
            Minimimum value for the polarizer angle.
        pmax : float, optional
            Maximum value for the polarizer angle.    
        smin : float, optional
            Minimimum value for the sample rotation angle.
        smax : float, optional
            Maximum value for the sample rotation angle.  
        amin : float, optional
            Minimimum value for the analyzer angle.
        amax : float, optional
            Maximum value for the analyzer angle.  
        namin : float, optional
            Minimimum value for the numerical aperture.
        namax : float, optional
            Maximum value for the numerical aperture.  
        """
        if fig is None:
            if ax is None:
                self.fig = plt.figure() 
            else:
                self.fig = ax.figure
        else:
            self.fig = fig
                
        self.ax = self.fig.add_subplot(111) if ax is None else ax
        
        plt.subplots_adjust(bottom=0.25)  
        self.calculate_image()
        
        self.sliders = {} if sliders is None else sliders
        
        if show_sliders:
            
            def update_slider_func(name):
                def update(d):
                    setattr(self,name,d)
                    self.update_plot()
                return update
                
            axes = [[0.25, 0.19, 0.65, 0.03],
                    [0.25, 0.16, 0.65, 0.03],
                    [0.25, 0.13, 0.65, 0.03],
                    [0.25, 0.10, 0.65, 0.03],
                    [0.25, 0.07, 0.65, 0.03],
                    [0.25, 0.04, 0.65, 0.03],
                    [0.25, 0.01, 0.65, 0.03],]
            
            def add_slider(name, axpos, names = None,labels = None, min_name = None, max_name = None, min_value = 0, max_value = 1, valfmt = '%.1f'):
                func = update_slider_func(name)
                if self.sliders.get(name) is None:
                    ax = self.fig.add_axes(axpos)
                    obj = getattr(self,name)
                    
                    if isinstance(obj, str):
                        if names is None:
                            names = tuple((label.strip().lower() for label in labels))
                        active = names.index(obj)
                        self.sliders[name] = CustomRadioButtons(ax, labels, active = active)
                        ax.set_ylabel(name,rotation="horizontal",ha = "right", va = "center")
                    else:
                        self.sliders[name] = Slider(ax, name,min(kwargs.pop(min_name,min_value),obj),max(kwargs.pop(max_name,max_value),obj),valinit = obj, valfmt=valfmt)
                try:
                    id = self.sliders[name].on_changed(func)
                except AttributeError:
                    id = self.sliders[name].on_clicked(func)
                    
                return id, ax

            if self.aperture is not None:   
                axpos = axes.pop() 
                self._ids6, self.axaperture = add_slider("aperture", axpos, labels = POLARIZER_LABELS, min_name = "namin", max_name = "namax", min_value = 0, max_value = max(self.beta))
                            
            if self.intensity is not None:
                axpos = axes.pop() 
                self._ids5, self.axintensity = add_slider("intensity", axpos, min_name = "imin", max_name = "imax", min_value = 0, max_value = self.intensity * 10)
 
            if self.polarizer is not None:
                axpos = axes.pop() 
                self._ids4, self.axpolarizer = add_slider("polarizer", axpos, labels = POLARIZER_LABELS, min_name = "pmin", max_name = "pmax", min_value = -90, max_value = 90)
 
            if self.sample is not None:
                axpos = axes.pop() 
                self._ids3, self.axsample = add_slider("sample", axpos,labels = SAMPLE_LABELS, min_name = "smin", max_name = "smax", min_value = -180, max_value = 180)
 
            if self.retarder is not None:
                axpos = axes.pop() 
                self._ids2, self.axretarder = add_slider("retarder", axpos, names = RETARDER_NAMES, labels = RETARDER_LABELS)
 
            if self.analyzer is not None:
                axpos = axes.pop() 
                self._ids1, self.axanalyzer = add_slider("analyzer", axpos, labels = POLARIZER_LABELS, min_name = "amin", max_name = "amax", min_value = -90, max_value = 90)
 
            if self.focus is not None:  
                axpos = axes.pop() 
                self._ids0, self.axfocus = add_slider("focus", axpos, min_name = "fmin", max_name = "fmax", min_value = -90, max_value = 90)
 
            
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
                self.dmat = field_diffraction_matrix(self.ifield.shape[-2:], self.ks,  d = d, 
                                          epsv = self.epsv, epsa = self.epsa, 
                                          mode = self.mode, betamax = self.betamax)
                recalc = True
        if recalc or "aperture" in self._updated_parameters:
            self.ofield = None #we have to create new memory for output field
            if self.aperture is not None:
                mask = self.beta <= self.aperture
                ifield = self.ifield[mask]
            else:
                ifield = self.ifield
            if self.diffraction == True or self.mode is not None: 
                self.ofield = diffract(ifield,self.dmat,window = self.window,out = self.ofield)
            else:
                #no diffraction at all..
                if self.window is not None:
                    self.ofield = ifield * self.window
                else:
                    self.ofield = ifield.copy()
            recalc = True

        if recalc or self._has_parameter_updated("analyzer", "polarizer","sample","retarder"):
            sample = self.sample_angle
            if sample is None:
                sample = 0.
            if self.polarizer_jvec is None:
                tmp = _redim(self.ofield, ndim = 5)
                out = np.empty_like(tmp[0])
            else:
                r = jones.rotation_matrix2(-np.radians(sample))
                c,s = dotmv(r,self.polarizer_jvec)
                tmp = _redim(self.ofield, ndim = 6)
                out = np.empty_like(tmp[0,0])
                
            if self.analyzer_jmat is not None:
                m = dotmm(self.analyzer_jmat,self.retarder_jmat) if self.retarder_jmat is not None else self.analyzer_jmat
                m = jones.rotated_matrix(m,np.radians(sample))
                pmat = ray_jonesmat4x4(m, epsv = self.epsv)
                
            for i,data in enumerate(tmp):
                if self.polarizer_jvec is not None:
                    x = data[0]*c
                    y = np.multiply(data[1], s, out = out)
                    ffield = np.add(x,y, out = out)#numexpr.evaluate("x*c+y*s", out = out)
                else: 
                    ffield = data
                    
                if self.analyzer_jmat is not None:
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
        if self._ffield is None or recalc:
            self._ffield = fft2(self.ifield)

        if recalc or "aperture" in self._updated_parameters:
            self.ofield = None #we have to create new memory for output field
            if self.aperture is not None:
                mask = self.beta <= self.aperture
                self.ffield_masked = self.ffield[mask]
                    
            else:
                self.ffield_masked = self.ffield
            recalc = True

        if recalc or self._has_parameter_updated("sample", "polarizer"):
            sample = self.sample_angle if self.sample_angle is not None else 0.
            if self.polarizer_jvec is not None:
                r = jones.rotation_matrix2(-np.radians(sample))
                c,s = dotmv(r,self.polarizer_jvec)                    
                
                self.data = _redim(self.ffield_masked, ndim = 6)
                x = c*self.data[:,0]
                y = s*self.data[:,1]
                self.data = x+y
            else:
                self.data = _redim(self.ffield_masked, ndim = 5)
                
        if recalc or self._has_parameter_updated("focus"):
            if self.diffraction == True or self.mode is not None:
                #if mode is selected, we need to project the field using diffraction
                d = 0 if self.focus is None else self.focus
                self.dmat = field_diffraction_matrix(self.ifield.shape[-2:], self.ks,  d = d, 
                                          epsv = self.epsv, epsa = self.epsa, 
                                          mode = self.mode, betamax = self.betamax)
            else:
                self.dmat = np.asarray(np.diag((1,1,1,1)), CDTYPE)      
        if recalc or self._has_parameter_updated("analyzer", "sample","retarder") :
            sample = self.sample_angle if self.sample_angle is not None else 0.
            if self.analyzer_jmat is not None:
                m = dotmm(self.analyzer_jmat,self.retarder_jmat) if self.retarder_jmat is not None else self.analyzer_jmat
                m = jones.rotated_matrix(m,np.radians(sample))
                self.pmat = mode_jonesmat4x4(self.ifield.shape[-2:], self.ks,  m,
                                          epsv = self.epsv, epsa = self.epsa, 
                                          betamax = self.betamax) 
            else:
                self.pmat = None

        if recalc or self._has_parameter_updated("analyzer", "sample", "polarizer", "focus", "intensity","retarder","aperture") :
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
            
            self.image = np.hstack(tuple((self.image for i in range (self.cols))))
            self.image = np.vstack(tuple((self.image for i in range (self.rows))))
            
            if self.sample_angle != 0 and self.sample_angle is not None:
                self.image = nd.rotate(self.image, -self.sample_angle, reshape = False, order = 1) 
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
        #for key, slider in self.sliders.items():
        #    slider.set_active(False)
        #    slider.set_val(getattr(self,key))
        #    slider.set_active(True)
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
                self._ffield = fft2(self.ifield[self.focus])
            else:
                self.dmat = field_diffraction_matrix(self.ifield.shape[-2:], self.ks,  d = 0, 
                                      epsv = self.epsv, epsa = self.epsa, 
                                      mode = self.mode, betamax = self.betamax)
                self._ffield = fft2(self.ifield[self.focus])

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
    

    

    