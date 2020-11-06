"""
Field viewer
============

Matplotlib-based field visualizer (polarizing miscroscope simulator) and pom
image calculation functions

High level functions
--------------------

* :func:`.pom_viewer` for polarizing optical microscope simulation.
* :func:`.field_viewer` for raw field_data visualization.
* :func:`.bulk_viewer` for raw bulk_data visualization. 
* :func:`.calculate_pom_field` calculates polarizing optical microscope field.

Classes
-------

* :class:`.FieldViewer` is the actual field viewer object.
* :class:`.BulkViewer` is the actual bulk viewer object.
* :class:`.POMViewer` is the actual microscope viewer object.

"""

from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, AxesWidget, RadioButtons
from matplotlib.image import imsave
import scipy.ndimage as nd


#from dtmm.project import projection_matrix, project
from dtmm.color import load_tcmf, specter2color
from dtmm.diffract import diffract, field_diffraction_matrix, E_cover_diffraction_matrix, E_diffraction_matrix, E_tr_matrix
from dtmm.jones4 import ray_jonesmat4x4, mode_jonesmat4x4, mode_jonesmat2x2, ray_jonesmat2x2
from dtmm.field import field2specter, field2jones, jones2field
from dtmm.wave import k0
from dtmm.data import refind2eps
from dtmm.conf import BETAMAX, CDTYPE, get_default_config_option, DTMMConfig
from dtmm.jones import jonesvec
from dtmm import jones

from dtmm.linalg import dotmf, dotmm, dotmv
from dtmm.fft import fft2, ifft2

#: settable viewer parameters
VIEWER_PARAMETERS = ("focus","analyzer", "polarizer", "sample", "intensity","aperture", "retarder")
IMAGE_PARAMETERS = ("cols","rows","gamma","gray","cms")


SAMPLE_LABELS = ("-90 ","-45 "," 0  ","+45 ", "+90 ")
SAMPLE_NAMES = tuple((s.strip() for s in SAMPLE_LABELS))
POLARIZER_LABELS = (" H  "," V  ","LCP ","RCP ","none")
POLARIZER_NAMES = tuple((s.strip().lower() for s in POLARIZER_LABELS))
RETARDER_LABELS = ("$\lambda/4$", "$\lambda/2$","none")
RETARDER_NAMES = ("lambda/4", "lambda/2", "none")

def calculate_pom_field(field, jvec = None, pmat = None, dmat = None, window = None, input_fft = False, output_fft = False, out = None):
    """Calculates polarizing optical microscope field from the input field.
    
    This function refocuses the field, applies polarizer and analayzers
    
    Parameters
    ----------
    field : ndarray
        Input array of shape (...,:,4,:,:) describing multiwavelength polarized 
        field array or an array of shape (...,2,:,4,:,:) describing unpolarized 
        (x nad y polarized) multiwavelength field arrays. Works also with jones 
        multiwavelengths fields of shapes (...,:,2,:,:) and (...,2,:,2,:,:).
    jvec : jonesvec, optional
        Normalized jones vector describing which polarization state of the input
        field to choose. Input field must be of unpolarized type if this parameter
        is specified. 
    pmat : ndarray, optional
       A 4x4 or 2x2 matrix describing the analyzer and retarder matrix. This matrix
       is applied in real space only if both input_fft and output_fft are False,
       otherwise, the matrix is applied in Fourier space.
    dmat : ndarray, optional
       A diffraction matrix.
    window : array, optional
        If specified, windowing is applied after field is diffracted.
    input_fft : bool
        If specified, it idicates that we are working with fft data. pmat must
        be computed with mode_jonesmat4x4.
    output_fft : bool
        If specified, output data is left in FFT space. No inverse Fourier transform
        is performed if this parameter is set to True.
    out : ndarray, optional
        Output array.
        
    Returns
    -------
    pom_field : ndarray
        Computed field od
        
    Examples
    --------
    >>> polarizer_jvec = jones4.jonesvec((1,0))
    >>> analyzer_jvec = jones4.jonesvec((0,1))
    >>> pmat = jones4.polarizer4x4(analyzer_jvec)
    >>> field_out = calculate_pom_field(field_in, polarizer_jvec, pmat)
    
    """
    if jvec is not None:
        if field.shape[-5] != 2:
            raise ValueError("Invalid field shape.")
        c,s = jvec
        x = field[...,0,:,:,:,:]*c
        y = np.multiply(field[...,1,:,:,:,:], s, out = out)
        field = np.add(x,y, out = out)#numexpr.evaluate("x*c+y*s", out = out)
    
    if input_fft == True or output_fft == True:
        #we are working on fft data, applying polarizer matrix in fft space
    
        if pmat is not None and dmat is not None:
            tmat = dotmm(pmat,dmat)
        elif pmat is None and dmat is not None:
            tmat = dmat
        elif pmat is not None and dmat is None and input_fft == True:
            tmat = pmat
        else:
            tmat = np.asarray(np.diag((1,1,1,1)), CDTYPE) 
            
        diffract(field,tmat,window = window, input_fft = input_fft, output_fft = output_fft, out = out)
    else:
        diffract(field,dmat,window = window, input_fft = input_fft, output_fft = output_fft,  out = out)
        #apply polarizer matrix in real space
        if pmat is not None:
            dotmf(pmat, field, out = out)
    return out

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


def bulk_viewer(field_data, **kwargs):
    """
    Returns a BulkViewer object for bulk field data visualization. See 
    :func:`.field_viewer` for parameters.
    
    Returns
    -------
    out : BulkViewer
        A :class:`BulkViewer` viewer object 
    
    """    
    return field_viewer(field_data, bulk_data=True, **kwargs)

def field_viewer(field_data, cmf=None, bulk_data=False, n=1., mode=None, is_polarized = None, 
                 window=None, diffraction=True, polarization_mode="normal", betamax=BETAMAX, beta = None, **parameters):
    """
    Returns a FieldViewer object for field data visualization.
    
    Parameters
    ----------
    field_data : tuple[np.ndarray]
        Input field data
    cmf : str, ndarray or None, optional
        Color matching function (table). If provided as an array, it must match 
        input field wavelengths. If provided as a string, it must match one of 
        available CMF names or be a valid path to tabulated data. See load_tcmf.
    bulk_data: bool
        Specifies whether data is to be treated as bulk data, e.g as returned by the
        :func:`.transfer.transfer_field` function with `ret_bulk = True`.
    n : float, optional
        Refractive index of the output material. Set this to the value used in
        the calculation of the field.
    mode : [ 't' | 'r' | None], optional
        Viewer mode 't' for transmission mode, 'r' for reflection mode None for
        as is data (no projection calculation - default).
    is_polarized : bool, optional
        If specified, it defines whether the field is polarize or not. For 
        non-polarized fields, the field must be of shape [...,2,:,4,:,:]. If 
        not provided, the polarization state is guessed from the shape of the
        input data. Setting this to False(and having non-polarized field)
        will allow setting the polarizer and sample rotation.
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
    
    if not diffraction and mode is not None:
        import warnings
        warnings.warn("Diffraction has been enabled because projection mode is set!")
        diffraction = True

    # Check that the provided polarization mode is a value one
    if polarization_mode not in valid_polarization_modes:
        raise ValueError("Unknown polarization mode, should be one of {}".format(repr(valid_polarization_modes)))
    if is_polarized is None:
        #try to get polarization state from the input data
        is_polarized = not (field.ndim >= 5 and field.shape[-5] == 2) 
    # Ensure a color matching function will be used
    if cmf is None:
        cmf = load_tcmf(wavelengths)
    elif isinstance(cmf, str):
        cmf = load_tcmf(wavelengths, cmf=cmf)

    if not bulk_data:
        if field.ndim < 4:
            raise ValueError("Incompatible field shape")

        viewer = FieldViewer(field.shape[-2:], wavelengths, pixelsize, propagation_mode = mode,
                             diffraction=diffraction, is_polarized = is_polarized,
                             refractive_index = n,
                             polarization_mode=polarization_mode, betamax=betamax, beta = beta)
        viewer.field = field
        viewer.image_parameters.cmf = cmf
        viewer.image_parameters.window = window
        viewer.set_parameters(**parameters)
    else:
        if field.ndim < 5:
            raise ValueError("Incompatible field shape")

        parameters.setdefault("focus", 0)
        viewer = BulkViewer(field.shape[-2:], wavelengths, pixelsize, propagation_mode = mode,
                             diffraction=diffraction, is_polarized = is_polarized,
                             refractive_index = n,
                             polarization_mode=polarization_mode, betamax=betamax, beta = beta)
        viewer.field = field
        viewer.image_parameters.cmf = cmf
        viewer.image_parameters.window = window
        viewer.set_parameters(**parameters)
        
    if DTMMConfig.verbose > 1:
        print("Loading field viewer")
        print("------------------------------------")
        viewer.viewer_options.print_info()
        viewer.image_parameters.print_info()
        print("-------------------------------------")
            

    return viewer

def pom_viewer(field_data, cmf=None, n = None, immersion = None, n_cover = None, d_cover = None, mode = +1, 
                 
                 is_polarized = None, window=None, NA = None, beta = None, **parameters):
    """
    Returns a FieldViewer object for optical microscope simulation.
    
    Parameters
    ----------
    field_data : tuple[np.ndarray]
        Input field data tuple.
    cmf : str, ndarray or None, optional
        Color matching function (table). If provided as an array, it must match 
        input field wavelengths. If provided as a string, it must match one of 
        available CMF names or be a valid path to tabulated data. See load_tcmf.
    n : float, optional
        Refractive index of the output material. If not set, it is set to 1 
        if immerion == False or n_cover if immersion == True.
    n_cover : float
        Refractive index of the cover medium. To simulate thick cover you should prepare 
        simulation results with nout = n_cover, (or nin = n_cover, if in reflection mode), 
        set n = 1 and d_cover > 0.
    d_cover : float
        Thickness ot the thick isotropic layer (cover glass).When d_cover != 0, t
        his simulates thick isotropic layer effect.
    immersion : bool
        Specified whether oil immersion objective is being used or not. Note that
        setting the 'n' parameter defines the immersion oil refractive index in
        this case.
    NA : float
        Numerical aperture of the objective. 
    mode : [ 't' | 'r' | +1 | -1 ]
        Viewer mode 't' or +1 for transmission mode, 'r' or -1 for reflection mode.
    is_polarized : bool, optional
        If specified, it defines whether the field is polarize or not. For 
        non-polarized fields, the field must be of shape [...,2,:,4,:,:]. If 
        not provided, the polarization state is guessed from the shape of the
        input data. Setting this to False(and having non-polarized field)
        will allow setting the polarizer and sample rotation.
    window : ndarray, optional
        Window function by which the calculated field is multiplied. This can 
        be used for removing artefact from the boundaries.
    parameters : kwargs, optional
        Extra parameters passed directly to the :meth:`FieldViewer.set_parameters`
        
    Returns
    -------
    out : POMViewer
        A :class:`POMViewer` viewer object 
        
    """
    # Extract components out of field_data
    field, wavelengths, pixelsize = field_data


    # Ensure a color matching function will be used
    if cmf is None:
        cmf = load_tcmf(wavelengths)
    elif isinstance(cmf, str):
        cmf = load_tcmf(wavelengths, cmf=cmf)

    if is_polarized is None:
        #try to get polarization state from the input data
        is_polarized = not (field.ndim >= 5 and field.shape[-5] == 2) 
        
    
    parameters.setdefault("focus", 0)
    viewer = POMViewer(field.shape[-2:], wavelengths, pixelsize, propagation_mode = mode,
                         is_polarized = is_polarized,
                         refractive_index = n, n_cover = n_cover, d_cover = d_cover,
                         NA=NA, immersion = immersion, beta = beta)
    
    viewer.image_parameters.cmf = cmf
    viewer.image_parameters.window = window
    viewer.set_parameters(**parameters)

    
    if DTMMConfig.verbose > 1:
        print("Loading POM viewer")
        print("------------------------------------")
        viewer.viewer_options.print_info()
        print("------------------------------------")
            
    viewer.field = field


    return viewer

def _as_field_array(field, options):
    field = np.asarray(field, CDTYPE)
    shape = options.shape
    nk = len(options.wavenumbers)
    if field.ndim >= 4 and field.shape[-3] == 4 and field.shape[-2:] == shape and field.shape[-4] == nk:
        return field
    else:
        raise ValueError("Invalid field data shape.")
        
def _as_jones_array(field, options):
    field = np.asarray(field, CDTYPE)
    shape = options.shape
    nk = len(options.wavenumbers)
    if field.ndim >= 4 and field.shape[-3] == 2 and field.shape[-2:] == shape and field.shape[-4] == nk:
        return field
    else:
        raise ValueError("Invalid jones data shape.")
    
def _float_or_none(value):
    """
    Helper function to convert the passed value to a float and return it, or return None.
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

class BaseViewerOptions(object):
    #: whether input field is polarized or non-polarized
    is_polarized = True
    #: list of beta values for field aperture simulations (for multi-ray data input)
    beta = None
    #: field propagation medium refractive index. 
    refractive_index = 1.
    #: propagation mode (+1 or "t" for transmission or -1 or "r" for reflection mode)
    propagation_mode = +1
    #: pixel size
    pixel_size = None
    #: simulated wavelengths
    wavelengths = []
    
    @property
    def wavenumbers(self):
        """simulated wavenumbers"""
        return k0(self.wavelengths, self.pixel_size)

    @property
    def epsv(self):
        """epsilon eigenvalues of the field propagation medium"""
        return refind2eps((self.refractive_index,)*3)
 

class FieldViewerOptions(BaseViewerOptions):
    """These parameters are set at initialization of the FieldViewer object,
    and should not be changed afterwards"""
    #: polarization mode, either "normal" or "mode"
    polarization_mode = "normal"
    #: whether to simulate diffraction. For bulk data visualization this is set to False
    diffraction = True
    #: whether to try to preserve memory while doing calculations.
    preserve_memory = False
    #: viewer betamax value
    betamax = BETAMAX
    
    def print_info(self):
        print(" $ polarization mode: {}".format(self.polarization_mode))  
        print(" $ polarized input: {}".format(self.is_polarized))  
        print(" $ refractive index: {}".format(self.refractive_index))     
        print(" $ propagation mode: {}".format(self.propagation_mode))   
        print(" $ max beta: {}".format(self.betamax))         
        
 
class POMViewerOptions(BaseViewerOptions):
    """These parameters are set at initialization of the FieldViewer object,
    and should not be changed afterwards"""
    #: cover glass refractive index
    _n_cover = None
    #: cover glass thickness
    _d_cover = None
    #: whether we use immersion objective or not.
    _immersion = None
    #: NA of the objective
    _NA = None
    
    def print_info(self):
        print(" $ polarization mode: {}".format(self.polarization_mode))  
        print(" $ polarized input: {}".format(self.is_polarized))  
        print(" $ cover refractive index: {}".format(self.n_cover)) 
        print(" $ cover thickness: {} [mm]".format(self.d_cover)) 
        print(" $ oil immersion: {}".format(self.immersion))
        print(" $ medium refractive index: {}".format(self.refractive_index))
        print(" $ propagation mode: {}".format(self.propagation_mode))   
        print(" $ objective NA: {}".format(self.NA))    
    
    @property
    def preserve_memory(self):
        return False
    
    @property
    def betamax(self):
        return self.NA
    
    @property
    def diffraction(self):
        return True
    
    @property
    def polarization_mode(self):
        return "mode"
   
    @property
    def refractive_index(self):
        if self._refractive_index is None:
            return self.n_cover if self.immersion else 1.
        else:
            return self._refractive_index
        
    @refractive_index.setter
    def refractive_index(self, value):
        self._refractive_index = value
       
    @property
    def n_cover(self):
        return get_default_config_option("n_cover", self._n_cover)
        
    @n_cover.setter    
    def n_cover(self, value):
        self._n_cover = value   

    @property
    def d_cover(self):
        return get_default_config_option("d_cover", self._d_cover)
        
    @d_cover.setter    
    def d_cover(self, value):
        self._d_cover = value   

    @property
    def immersion(self):
        return get_default_config_option("immersion", self._immersion)
        
    @immersion.setter    
    def immersion(self, value):
        self._immersion = value   
        
    @property    
    def NA(self):
        return get_default_config_option("NA", self._NA)

    @NA.setter    
    def NA(self, value):
        self._NA = value       
        
 
class ImageParameters(object):
    """Image parameters are storred here"""
    _wavelengths = None
    #: whether to convert RGB to gray
    _gray = None
    #: gamma value
    _gamma = None
    
    _cmf = None
    #: number of rows for periodic structure multiplication
    cols = 1
    #: number of rows for periodic structure multiplication
    rows = 1
    #: window function applied to the calculated image
    window = None
    
    def print_info(self):
        cmf_name = self.cmf if isinstance(self.cmf, str) else "custom"
        print(" $ color matchin function: {}".format(cmf_name)) 
        print(" $ output gray: {}".format(self.gray)) 
        print(" $ gamma: {}".format(self.gamma))
    
    @property 
    def gray(self):
        return get_default_config_option("gray", self._gray)
    
    @gray.setter
    def gray(self, value):
        self._gray = value
    
    @property 
    def gamma(self):
        return get_default_config_option("gamma", self._gamma)
    
    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    @property      
    def cmf(self):
        """Color matching function data"""
        out = self._cmf 
        #if it does not exist, create one.
        if out is None:
            out = load_tcmf(self._wavelengths) 
            self._cmf = out
        return out 
    
    @cmf.setter        
    def cmf(self, cmf):
        if len(cmf) != len(self._wavelengths):
            raise ValueError("Incompatible cmf!")
        self._cmf = cmf
    
class FieldViewer(object): 
    """Field viewer. See :func:`.field_viewer`"""  
    _field = None
    _ffield = None
    _retarder = "none"
    _retarder_jmat = None
    _focus = 0
    _polarizer = None
    _polarizer_jvec = None
    _sample = None
    _sample_angle = None
    _analyzer = "none"
    _analyzer_jmat = None
    _intensity = 1.
    _parameters = VIEWER_PARAMETERS
    _image_parameters = IMAGE_PARAMETERS
    _fmin = 0
    _fmax = 100
    _jvec = None
    _pmat = None
    _dmat = None
    _ofield = None
    _specter = None
    
        
    def __init__(self,shape, wavelengths, pixelsize, **kwargs):
        self.viewer_options = FieldViewerOptions()
        self.image_parameters = ImageParameters()
        
        self.viewer_options.wavelengths = wavelengths
        self.viewer_options.pixel_size = pixelsize 
        self.viewer_options.shape = shape 
        self.image_parameters._wavelengths = wavelengths
        
        for key, value in kwargs.items():
            setattr(self.viewer_options, key, value)
            
        self._aperture = None if self.viewer_options.beta is None else max(self.viewer_options.beta)
        
        if not self.viewer_options.is_polarized:
            self.polarizer = "none"
            self.sample = "none"  
            
    def print_info(self):
        print(" $ intensity: {}".format(self.intensity)) 
        print(" $ polarizer: {}".format(self.polarizer)) 
        print(" $ sample rotation: {}".format(self.sample)) 
        print(" $ retarder: {}".format(self.retarder)) 
        print(" $ analyzer: {}".format(self.analyzer)) 
        print(" $ focus: {}".format(self.focus))

    def _clear_all_field_data(self):
        self._field = None
        self._ffield = None  
        self._specter = None
        
    @property
    def field(self):
        if self._field is None:
            self._field = ifft2(self._ffield)
        return self._field

    @field.setter
    def field(self, value):
        field = _as_field_array(value, self.viewer_options)
        self._clear_all_field_data()
        self._field = field
        
    @property
    def ffield(self):
        """Fourier transform of the field"""
        if self._ffield is None:
            self._ffield = fft2(self._field)
        return self._ffield
    
    @ffield.setter
    def ffield(self, value):
        """Fourier transform of the field"""
        ffield = _as_field_array(value, self.viewer_options)
        self._clear_all_field_data()
        self._ffield = ffield
        
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
    def masked_field(self):
        if self.aperture is not None:
            mask = self.viewer_options.beta <= self.aperture
            return self.field[mask]
        else:
            return self.field
    
    @property
    def masked_ffield(self):
        """Fourier transform of the field"""
        if self.aperture is not None:
            mask = self.viewer_options.beta <= self.aperture
            return self.ffield[mask]
        else:
            return self.ffield

    @focus.setter     
    def focus(self, z):
        if self.viewer_options.diffraction == True or z is None:
            self._dmat = None
            self._specter = None
            self._focus = _float_or_none(z)
        else:
            raise ValueError("Cannot set focus of a non-diffractive field.")

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
            if self.viewer_options.is_polarized:
                raise ValueError("Input field must be unpolarized to use sample parameter!")
            
        if isinstance(angle, str):
            labels = tuple((label.strip() for label in SAMPLE_LABELS))
            if angle.strip() not in labels:
                if angle in ("none","+0","-0"):
                    angle = "0"
                else:
                    raise ValueError("sample angle must be a float or any of {}".format(labels))
        self._pmat = None #force recalculation of pmat
        self._jvec = None
        self._specter = None
        
        self._sample_angle = _float_or_none(angle)
        self._sample = angle
           
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
        self._specter = None

    @property
    def polarizer(self):
        """Polarizer angle. Can be 'h','v', 'lcp', 'rcp', 'none', angle float or a jones vector"""
        return self._polarizer
        
    @polarizer.setter 
    def polarizer(self, angle):
        if angle is not None:
            if self.viewer_options.is_polarized:
                raise ValueError("Input field must be unpolarized to use polarizer parameter!")

        self._polarizer, self._polarizer_jvec = _jonesvector_type(angle)
        self._jvec = None
        self._specter = None
        
    @property
    def analyzer(self):
        """Analyzer angle. Can be 'h','v', 'lcp', 'rcp', 'none', angle float or a jones vector"""
        return self._analyzer    
        
    @analyzer.setter   
    def analyzer(self, angle):
        self._analyzer, self._analyzer_jmat = _jonesmatrix_type(angle)
        self._pmat = None
        self._specter = None
 
    @property    
    def retarder_jmat(self):
        return self._retarder_jmat
    
    @property    
    def retarder(self):
        return self._retarder
    
    @retarder.setter    
    def retarder(self, value):
        self._retarder, self._retarder_jmat = _jonesmatrix_type(value)
        self._pmat = None #force recalculation of pmat
        self._specter = None
       
    @property
    def intensity(self):
        """Input light intensity"""
        return self._intensity   
    
    @intensity.setter   
    def intensity(self, intensity):
        self._intensity = _float_or_none(intensity)
    
    @property
    def input_jones(self):
        """Input field jones vector"""
        if self._jvec is None:
            sample = self.sample_angle
            if sample is None:
                sample = 0.
            if self.polarizer_jvec is None:
                self._jvec = None
            else:
                r = jones.rotation_matrix2(-np.radians(sample))
                self._jvec = dotmv(r,self.polarizer_jvec)
        return self._jvec
    
    @property
    def diffraction_matrix(self):
        """Diffraction matrix for diffraction calculation"""
        if self._dmat is None:
            vp = self.viewer_options
            if vp.diffraction or vp.propagation_mode is not None:
                #if mode is selected, we need to project the filed using diffraction
                d = 0 if self.focus is None else self.focus
                epsv = vp.epsv
                self._dmat = field_diffraction_matrix(vp.shape, vp.wavenumbers, d = d, 
                                          epsv = epsv, mode = vp.propagation_mode, betamax = vp.betamax) 
            else:
                self._dmat = None
        return self._dmat
    
    @property
    def output_matrix(self):
        """4x4 jones output matrix"""
        sample = self.sample_angle
        if sample is None:
            sample = 0.
        if self._pmat is None:
            if self.analyzer_jmat is not None:
                m = dotmm(self.analyzer_jmat,self.retarder_jmat) if self.retarder_jmat is not None else self.analyzer_jmat
                m = jones.rotated_matrix(m,np.radians(sample))       
                vp = self.viewer_options
                epsv = vp.epsv
                if vp.polarization_mode== "mode":                 
                    self._pmat = mode_jonesmat4x4(vp.shape, vp.wavenumbers, m, epsv = epsv)
                else:
                    self._pmat = ray_jonesmat4x4(m, epsv = epsv)
            else:
                self._pmat = None
        return self._pmat
        
    def set_parameters(self, **kwargs):
        """Sets viewer parameters. Any of the :attr:`.VIEWER_PARAMETERS`
        """
        for key, value in kwargs.items():
            if key in self._parameters:
                setattr(self, key, value) 
            elif key in self._image_parameters:
                setattr(self.image_parameters, key, value) 
            else:
                raise TypeError("Unexpected keyword argument '{}'".format(key))

    def get_parameters(self):
        """Returns viewer parameters as dict"""
        return {name : getattr(self,name) for name in VIEWER_PARAMETERS}, {name : getattr(self.image_parameters,name) for name in IMAGE_PARAMETERS}
        
    def plot(self, fig = None,ax = None, sliders = None, show_sliders = None, show_scalebar = None, show_ticks = None, **kwargs):
        """Plots field intensity profile. You can set any of the below listed
        arguments. Additionaly, you can set any argument that imshow of
        matplotlib uses (e.g. 'interpolation = "sinc"').
        
        Parameters
        ----------
        show_slider : bool, optional
            Specifies whether to show sliders or not.
        show_scalebar : bool, optional
            Specifies whether to show scalebar or not.
        show_ticks : bool, optional
            Specifies whether to show ticks in imshow or not.
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
        show_sliders = get_default_config_option("show_sliders",show_sliders)
        show_ticks = get_default_config_option("show_ticks",show_ticks)
        show_scalebar = get_default_config_option("show_scalebar",show_scalebar)
        
        if fig is None:
            if ax is None:
                self.fig = plt.figure() 
            else:
                self.fig = ax.figure
        else:
            self.fig = fig
                
        self.ax = self.fig.add_subplot(111) if ax is None else ax
        
        plt.subplots_adjust(bottom=0.27)  
        image = self.calculate_image()
        
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
                self._ids6, self.axaperture = add_slider("aperture", axpos, labels = POLARIZER_LABELS, min_name = "namin", max_name = "namax", min_value = 0, max_value = self.viewer_options.beta.max())
                            
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
                self._ids0, self.axfocus = add_slider("focus", axpos, min_name = "fmin", max_name = "fmax", min_value = self._default_fmin, max_value = self._default_fmax)
 
            
        self.axim = self.ax.imshow(image, origin = kwargs.pop("origin","lower"), **kwargs)
        
        if show_scalebar == True:
            if self.viewer_options.pixel_size is None:
                raise ValueError("You must provide pixel_size to show scale bar.")
            try:
                from matplotlib_scalebar.scalebar import ScaleBar
            except ImportError:
                raise ValueError("You must have matplotlib_scalebar installed to use this feature.")
            
            scalebar = ScaleBar(self.viewer_options.pixel_size , "nm")
            self.ax.add_artist(scalebar)
            
        show_ticks = False if show_scalebar and show_ticks is None else show_ticks 
        
        if show_ticks == False:
            self.ax.set_xticklabels([])
            self.ax.set_xticks([])
            self.ax.set_yticklabels([])
            self.ax.set_yticks([])            
        
        return self.ax.figure, self.ax
                    
    def _calculate_diffraction(self):  
        if self._dmat is None:
            self._ofield = None #we have to create new memory for output field  
            vp = self.viewer_options
            modal = vp.polarization_mode == "mode"
            diffractive = vp.diffraction == True
            preserve_memory = vp.preserve_memory
            if preserve_memory or modal or self.diffraction_matrix is None:
                # we calculate diffraction later or not at all
                self._ofield = self.masked_ffield if modal else self.masked_field 
            else:
                if diffractive or vp.propagation_mode is not None: 
                    self._ofield = diffract(self.masked_field,self.diffraction_matrix,window = self.image_parameters.window,out = self._ofield)
                else:
                    #no diffraction at all..
                    if self.image_parameters.window is not None:
                        self._ofield = self.masked_field * self.image_parameters.window
                    else:
                        self._ofield = self.masked_field.copy()
                    
    def _field2specter(self,field):
        return field2specter(field)
    
    def _calculate_pom_field(self, data, jvec, pmat, dmat, window = None, input_fft = False, out = None):
        return calculate_pom_field(data,jvec,pmat ,dmat,window = window,input_fft = input_fft, out = out)
        
    def _calculate_specter(self):
        if self._specter is None:
            vp = self.viewer_options
            modal = vp.polarization_mode == "mode"
            preserve_memory = vp.preserve_memory
            jvec = self.input_jones
            
            input_fft = True if modal else False 
            
            field = self._ofield
            
            if jvec is None:
                tmp = _redim(field, ndim = 5)
                out = np.empty_like(tmp[0])
          
            else:
                tmp = _redim(field, ndim = 6)
                out = np.empty_like(tmp[0,0])
    
            self._specter = 0.
            
            #we may heve calculated diffraction before, set to None if we had
            dmat = self.diffraction_matrix if preserve_memory or modal else None
            
            for i,data in enumerate(tmp): 
                field = self._calculate_pom_field(data,jvec,self.output_matrix,dmat,window = self.image_parameters.window,input_fft = input_fft, out = out)
                self._specter += self._field2specter(field)
                   
    def calculate_field(self, recalc = False, **params):
        self.set_parameters(**params)
        vp = self.viewer_options
        modal = vp.polarization_mode == "mode"
        preserve_memory = vp.preserve_memory

        self._calculate_diffraction()
            
        jvec = self.input_jones
        
        input_fft = True if modal else False 
        
        field = self._ofield
        
        #we may heve calculated diffraction before, set to None if we had
        dmat = self.diffraction_matrix if preserve_memory or modal else None

        return calculate_pom_field(field,jvec,self.output_matrix,dmat,window = self.image_parameters.window,input_fft = input_fft)
                    
    def calculate_specter(self, **params):
        """Calculates field specter.
        
        Parameters
        ----------
        params: kwargs, optional
            Any additional keyword arguments that are passed dirrectly to 
            set_parameters method.
        """        
        self.set_parameters(**params)
        if DTMMConfig.verbose > 1:
            print("Calculating specter.")
            print("------------------------------------")            
            self.print_info()
            print("------------------------------------")
                
        
        self._calculate_diffraction()
        self._calculate_specter()
        return self._specter
 
    def calculate_image(self, **params):
        """Calculates RGB image.
        
        Parameters
        ----------
        params: keyword arguments
            Any additional keyword arguments that are passed dirrectly to 
            set_parameters method.
            
        """   
        specter = self.calculate_specter(**params)
        
        if DTMMConfig.verbose > 1:
            print("Calculating image.")
            print("------------------------------------")
            self.image_parameters.print_info()
            print("------------------------------------")
        
        vp = self.viewer_options
        ip =  self.image_parameters
        
        cmf = ip.cmf
    
        if self.intensity is not None:
            if self.intensity != 0.0:
                if vp.propagation_mode in (-1,"r"):
                    #poynting is negative, make it positive
                    norm = -1./self.intensity
                else:
                    norm = 1./self.intensity
            else:
                norm = 0.0

            image = specter2color(specter,cmf, norm = norm, gray = ip.gray, gamma = ip.gamma) 
        else:
            if vp.propagation_mode in (-1,"r"):
                image = specter2color(specter,cmf, norm = -1., gray = ip.gray, gamma = ip.gamma) 
            else:
                image = specter2color(specter,cmf, gray = ip.gray, gamma = ip.gamma) 
        
        image = np.hstack(tuple((image for i in range (ip.cols))))
        image = np.vstack(tuple((image for i in range (ip.rows))))
        
        if self.sample_angle != 0 and self.sample_angle is not None:
            image = nd.rotate(image, -self.sample_angle, reshape = False, order = 1) 

        return image
 
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
        image = self.calculate_image()
        self.axim.set_data(image)
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
        return len(self.field) -1   
    
    @property
    def focus(self):
        """Focus position"""
        return self._focus       
    
    @focus.setter     
    def focus(self, z):
        #focos must be integer here, index of the layer
        i = int(z)
        #check is ok, raise IndexError else
        self.field[i]
        self._focus = i
        
    @property
    def masked_field(self):
        if self.aperture is not None:
            mask = self.viewer_options.beta <= self.aperture
            return self.field[self.focus,mask]
        else:
            return self.field[self.focus]
    
    @property
    def masked_ffield(self):
        """Fourier transform of the field"""
        if self.aperture is not None:
            mask = self.viewer_options.beta <= self.aperture
            return self.ffield[self.focus,mask]
        else:
            return self.ffield[self.focus]
        
class POMViewer(FieldViewer):
    """Similar to FieldViewer, with the following differences:
        
    Computation is done on jones field, instead of full field data. Therefore,
    you need to provide the mode parameter to select the propagation mode.
    
    """
    _fjones = None
    _jones = None
    _cmat = None
    
    def __init__(self,shape, wavelengths, pixelsize, **kwargs):
        self.viewer_options = POMViewerOptions()
        self.image_parameters = ImageParameters()
        
        self.viewer_options.wavelengths = wavelengths
        self.viewer_options.pixel_size = pixelsize 
        self.viewer_options.shape = shape 
        self.image_parameters._wavelengths = wavelengths
        
        for key, value in kwargs.items():
            setattr(self.viewer_options, key, value)
            
        self._aperture = None if self.viewer_options.beta is None else max(self.viewer_options.beta)
        
        if not self.viewer_options.is_polarized:
            self.polarizer = "none"
            self.sample = "none"  
        
    def _clear_all_field_data(self):
        self._jones = None
        self._fjones = None
        self._field = None
        self._ffield = None      
         
    @property
    def jones(self):
        if self._jones is None:   
            vp = self.viewer_options
            epsv = vp.epsv
            self._jones = field2jones(self._field, vp.wavenumbers, epsv = epsv, mode = vp.propagation_mode, output_fft = False, betamax = vp.betamax)
        return self._jones   
    
    @jones.setter
    def jones(self,value):
        self._clear_all_field_data()
        self._jones = value
        self._fjones = None
        self._field = None
        self._ffield = None
        
    
    @property
    def fjones(self):
        if self._fjones is None:
            vp = self.viewer_options
            epsv = vp.epsv
            if self._jones is None:
                self._fjones = field2jones(self._field, vp.wavenumbers, epsv = epsv, mode = vp.propagation_mode, output_fft = True, betamax = vp.betamax)
            else:
                self._fjones = fft2(self._jones)
        return self._fjones
    
    @fjones.setter
    def fjones(self, value):
        self._fjones = value
        self._field = None
        self._ffield = None
        self._jones = None

    @property
    def masked_fjones(self):
        """Fourier transform of the field"""
        if self.aperture is not None:
            mask = self.viewer_options.beta <= self.aperture
            return self.fjones[mask]
        else:
            return self.fjones
        
    def _calculate_diffraction(self):
        #diffraction is calculated during 
        self._ofield = self.masked_fjones

    def _field2specter(self,field):
        vp = self.viewer_options
        epsv = vp.epsv
        return field2specter(jones2field(field, vp.wavenumbers, epsv = epsv, mode = vp.propagation_mode, input_fft = True, betamax = vp.betamax))

    def _calculate_pom_field(self, data, jvec, pmat, dmat, window = None, input_fft = False, out = None):
        return calculate_pom_field(data,jvec,pmat ,dmat,window = window,input_fft = input_fft, output_fft = True, out = out)
      
    @property
    def output_matrix(self):
        """2x2 jones output matrix"""
        sample = self.sample_angle
        if sample is None:
            sample = 0.
        if self._pmat is None:
            vp = self.viewer_options
            if self.analyzer_jmat is not None:
                m = dotmm(self.analyzer_jmat,self.retarder_jmat) if self.retarder_jmat is not None else self.analyzer_jmat
                m = jones.rotated_matrix(m,np.radians(sample))
                epsv = vp.epsv
                self._pmat = mode_jonesmat2x2(vp.shape, vp.wavenumbers, m, mode = vp.propagation_mode, epsv = epsv, betamax = vp.betamax)

            else:
                self._pmat = None
        return self._pmat
    
    @property
    def cover_matrix(self):
        if self._cmat is None:
            vp = self.viewer_options
            # vp.d_cover is in mm, calculat d_cover in pixel units
            d_cover =vp.d_cover / vp.pixel_size * 1000000 
            cmat = E_cover_diffraction_matrix(vp.shape, vp.wavenumbers, 
                                      n = vp.refractive_index, d_cover = d_cover,n_cover = vp.n_cover,
                                      mode = vp.propagation_mode, betamax = vp.betamax) 
            # Fresnel reflection matrix
            eps = vp.epsv
            eps_cover = refind2eps((vp.n_cover,)*3)
            tmat,rmat = E_tr_matrix(vp.shape, vp.wavenumbers, epsv_in = eps_cover,
                        epsv_out = eps, mode = vp.propagation_mode, betamax = vp.betamax)
            #self._cmat = cmat
            self._cmat = dotmm(tmat,cmat)
            
        return self._cmat
    
    @property
    def diffraction_matrix(self):
        """Diffraction matrix for diffraction calculation"""
        if self._dmat is None:
            vp = self.viewer_options
            #if mode is selected, we need to project the filed using diffraction
            d = 0 if self.focus is None else self.focus
            epsv = vp.epsv
            #diffraction matrix
            dmat = E_diffraction_matrix(vp.shape, vp.wavenumbers,  d = d, 
                                      epsv = epsv, 
                                      mode = vp.propagation_mode, betamax = vp.betamax) 
            self._dmat = dmat

            cmat = self.cover_matrix  
            self._dmat = dotmm(self._dmat, cmat) 
        return self._dmat   

    
__all__ = ["calculate_pom_field", "field_viewer", "bulk_viewer", "FieldViewer", "BulkViewer"]
    