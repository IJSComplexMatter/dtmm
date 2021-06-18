"""
TMM Matrix solvers for 3d, 2d and 1d data.

These objects can be used for a simplified treatment of the transfer matrices
for 3d field data on 1d, 2d (and 3d) optical data. 

* :class:`MatrixBlockSolver3D` for matrix-based 3d simulations on optical blocks.
* :class:`MatrixDataSolver3D` for matrix-based 3d simulations on optical datas.

"""
from dtmm.conf import get_default_config_option, DTMMConfig
from dtmm.data import validate_optical_data, material_shape, validate_optical_block, validate_optical_layer, is_callable, shape2dim, optical_data_shape
from dtmm import tmm3d, tmm2d
from dtmm.field import field2modes, modes2field, field2modes1, modes2field1
from dtmm.wave import k0, eigenmask, eigenmask1
from dtmm.print_tools import print_progress
import numpy as np

#: available solver methods
AVAILABLE_MATRIX_SOLVER_METHODS = ("4x4", "4x4_1", "2x2")

def _dispersive_stack_matrix3d(optical_block, wavelengths, wavenumbers, mask, shape, method):
    return tuple((tmm3d.stack_mat3d(k,*validate_optical_block(optical_block, shape = shape, wavelength = w),\
                        mask = m, method = method) \
                         for w,k,m in  zip(wavelengths, wavenumbers,mask)))
        
def _iterate_dispersive_layers3d(optical_block, wavelengths, wavenumbers, mask, shape, method, reverse = False):
    n = len(optical_block[0])
    if reverse:
        optical_block = map(reversed, optical_block)
    for i, (d, epsv, epsa) in enumerate(zip(*optical_block)):
        print_progress(i+1, n)
        layer = tuple((tmm3d.layer_mat3d(k, *validate_optical_layer((d,epsv,epsa), wavelength = w, shape = shape), \
                mask = m, method = method) \
             for w,k,m in zip(wavelengths, wavenumbers, mask)))
        yield layer    
 
class BaseMatrixSolver2D(object): 
    """Base class for all 2D Matrix-based solvers."""
    
    tmm_system_mat = tmm2d.system_mat2d
    tmm_reflection_mat = tmm2d.reflection_mat2d
    tmm_stack_mat =  tmm2d.stack_mat2d
    tmm_reflect = tmm2d.reflect2d
    tmm_f_iso = tmm2d.f_iso2d
        
    # field array
    _field_out = None
    _field_in = None
    modes_in = None
    modes_out = None
    mask = None
    
    # transfer matrices
    field_matrix_in = None
    field_matrix_out = None
    stack_matrix = None
    refl_matrix = None
    trans_matrix = None
    
    #material shape and dimension
    material_shape = None
    material_dim = None

    def __init__(self, shape, betay = 0, wavelengths = [500], pixelsize = 100,  mask = None, method = "4x4", betamax = None):
        """
        Paramters
        ---------
        shape : int
            Cross-section shape of the field data.
        wavelengths : float or array
            A list of wavelengths (in nanometers) or a single wavelength for 
            which to create the solver.
        pixelsize : float
            Pixel size in (nm).
        mask : ndarray, optional
            A fft mode mask array. If not set, :func:`.wave.eigenmask` is 
            used with `betamax` to create such mask.
        method : str
            Either '4x4' (default), '4x4_1' or '2x2'.
        betamax : float, optional
            If `mask` is not set, this value is used to create the mask. If not 
            set, default value is taken from the config.
        """
        self.shape = int(shape),
        self.betay = np.asarray(betay)
        if self.betay.ndim == 1:
            if not len(betay) == len(wavelengths):
                raise ValueError("`betay` must have same length as `wavelengths`.")
        elif self.betay.ndim != 0:
            raise ValueError("`betay` must be a scalar or a 1D array")
        self.wavelengths = np.asarray(wavelengths)
        if self.wavelengths.ndim not in (0,1):
            raise ValueError("`wavelengths` must be a scalar or an 1D array.")
        self.pixelsize = float(pixelsize)
        self.wavenumbers = k0(self.wavelengths, pixelsize)
        
        method = str(method) 
        if method in AVAILABLE_MATRIX_SOLVER_METHODS :
            self.method = method
        else:
            raise ValueError("Unsupported method {}".format(method))
        
        if mask is None:
            betamax = get_default_config_option("betamax", betamax)
            self.mask = eigenmask(shape, self.wavenumbers, betamax)
        else:
            self.set_mask(mask)
            
        self._field_dim = 2 if self.wavelengths.ndim == 0 else 3
            
    def set_mask(self, mask):
        """Sets fft mask.
        
        Parameters
        ----------
        mask : ndarray
            A boolean mask, describing fft modes that are used in the solver.
        """
        mask = np.asarray(mask)
        mask_shape = self.wavelengths.shape + self.shape
        
        if mask.shape != mask_shape:
            raise ValueError("Mask shape must be of shape {}".format(mask_shape))
        if not mask.dtype == bool:
            raise ValueError("Not a bool mask.")
            
        self.mask = mask
        self.clear_matrices()
        self.clear_data()
    
    def clear_matrices(self):
        self.stack_matrix = None
        self.refl_matrix = None
        self.field_matrix_in = None
        self.field_matrix_out = None
        self.trans_matrix = None
        self.layer_matrices = []
        
    def clear_data(self):
        self.modes_in = None
        self.modes_out = None

    def _validate_field(self, field):
        field = np.asarray(field)
        field_shape = self.wavelengths.shape + (4,) + self.shape
            
        if field.shape[-self._field_dim:] != field_shape:
            raise ValueError("Field shape not comaptible with solver's requirements. Must be (at least) of shape {}".format(field_shape))
        return field
        
    @property
    def field_in(self):
        """Input field array"""
        return self._field_in
    
    @field_in.setter
    def field_in(self, field):
        self._field_in =  self._validate_field(field)       
        mask, self.modes_in = field2modes1(self._field_in,self.wavenumbers, betay = self.betay, mask = self.mask)    

    def _get_field_data(self, field, copy):
        if copy:
            return field.copy(), self.wavelengths.copy(), self.pixelsize
        else:
            return field, self.wavelengths, self.pixelsize
        
    def get_field_data_in(self, copy = True):
        """Returns input field data tuple.
        
        If copy is set to True (default), a copy of the field_out array is made. 
        """
        return self._get_field_data(self.field_in, copy)

    @property
    def field_out(self):
        """Output field array"""
        return self._field_out
    
    @field_out.setter
    def field_out(self, field):
        self._field_out = self._validate_field(field)   
        mask, self.modes_out = field2modes1(self._field_out, self.wavenumbers, betay = self.betay, mask = self.mask)    

    def get_field_data_out(self, copy = True):
        """Returns output field data tuple.
        
        If copy is set to True (default), a copy of the field_out array is made. 
        """
        return self._get_field_data(self.field_out, copy)
    
    def calculate_field_matrix(self,nin = 1., nout = 1.):
        """Calculates field matrices. 
        
        This must be called after you set material data.
        """
        if self.material_shape is None:
            raise ValueError("You must first set material data")
        self.nin = float(nin)
        self.nout = float(nout)
        #now build up matrices field matrices       
        self.field_matrix_in = self.tmm_f_iso(self.mask, self.wavenumbers, n = self.nin, shape = self.material_shape, betay = self.betay)
        self.field_matrix_out  = self.tmm_f_iso(self.mask, self.wavenumbers, n = self.nout, shape = self.material_shape, betay = self.betay)

    def calculate_reflectance_matrix(self):
        """Calculates reflectance matrix.
        
        Available in "4x4","4x4_1" methods only. This must be called after you 
        have calculated the stack and field matrices.
        """
        if self.method not in ("4x4","4x4_1"):
            raise ValueError("reflectance matrix is available in 4x4 and 4x4_1 methods only")
        if self.stack_matrix is None:
            raise ValueError("You must first calculate stack matrix")
        if self.field_matrix_in is None:
            raise ValueError("You must first calculate field matrix")
        
        mat = self.tmm_system_mat(self.stack_matrix, self.field_matrix_in, self.field_matrix_out)
        mat = self.tmm_reflection_mat(mat)
        self.refl_matrix = mat

      
    def calculate_transmittance_matrix(self):
        """Calculates transmittance matrix.
        
        Available in "2x2" method only. This must be called after you have
        calculated the stack matrix.
        """
        if self.method != "2x2":
            raise ValueError("transmittance matrix is available in 2x2 method only")
        if self.stack_matrix is None:
            raise ValueError("You must first calculate stack matrix")
        
        self.trans_matrix  = self.tmm_transmission_mat(self.stack_matrix) 
        
    def _group_modes(self,modes):
        return modes
        
    def transfer_field(self, field_in = None, field_out = None):
        """Transfers field.
        
        This must be called after you have calculated the transmittance/reflectance 
        matrix.
        
        Parameters
        ----------
        field_in : ndarray, optional
            If set, the field_in attribute will be set to this value prior to
            transfering the field.
        field_out : ndarray, optional
            If set, the field_out attribute will be set to this value prior to
            transfering the field.            
        """
        if self.refl_matrix is None and self.trans_matrix is None:
            raise ValueError("You must first create reflectance/transmittance matrix")
        if field_in is not None:
            self.field_in = field_in
        if field_out is not None:
            self.field_out = field_out       
            
        if self.modes_in is None:
            raise ValueError("You must first set `field_in` data or set the `field_in` argument.")

        grouped_modes_in = self._group_modes(self.modes_in)
        grouped_modes_out = None if self.modes_out is None else self._group_modes(self.modes_out)
    
        # transfer field
        if self.method.startswith("4x4"):
            grouped_modes_out = self.tmm_reflect(grouped_modes_in, rmat = self.refl_matrix, fmatin = self.field_matrix_in, fmatout = self.field_matrix_out, fvecout = grouped_modes_out)
        else:
            grouped_modes_out = self.tmm_transmit(grouped_modes_in, tmat = self.trans_matrix, fmatin = self.field_matrix_in, fmatout = self.field_matrix_out, fvecout = grouped_modes_out)
        
        # now ungroup and convert modes to field array
        self.modes_out = self._ungroup_modes(grouped_modes_out)
        self.modes_in = self._ungroup_modes(grouped_modes_in)
        
        self._field_out = modes2field(self.mask, self.modes_out, out = self._field_out)
    
        if self.method.startswith("4x4"):
            #we need to update input field because of relfections.
            self.modes_in = self._ungroup_modes(grouped_modes_in)
    
class BaseMatrixSolver3D(object): 
    """Base class for all Matrix-based solvers."""

    # field array
    _field_out = None
    _field_in = None
    modes_in = None
    modes_out = None
    mask = None
    
    # transfer matrices
    field_matrix_in = None
    field_matrix_out = None
    stack_matrix = None
    refl_matrix = None
    trans_matrix = None
    
    #material shape and dimension
    material_shape = None
    material_dim = None

    def __init__(self, shape, wavelengths = [500], pixelsize = 100,  mask = None, method = "4x4", betamax = None):
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
        mask : ndarray, optional
            A fft mode mask array. If not set, :func:`.wave.eigenmask` is 
            used with `betamax` to create such mask.
        method : str
            Either '4x4' (default), '4x4_1' or '2x2'.
        betamax : float, optional
            If `mask` is not set, this value is used to create the mask. If not 
            set, default value is taken from the config.
        """
        x,y = shape
        if not (isinstance(x, int) and isinstance(y, int)):
            raise ValueError("Invalid field shape.")
        self.shape = x,y
        self.wavelengths = np.asarray(wavelengths)
        if self.wavelengths.ndim not in (0,1):
            raise ValueError("`wavelengths` must be a scalar or an 1D array.")
        self.pixelsize = float(pixelsize)
        self.wavenumbers = k0(self.wavelengths, pixelsize)
        
        method = str(method) 
        if method in AVAILABLE_MATRIX_SOLVER_METHODS :
            self.method = method
        else:
            raise ValueError("Unsupported method {}".format(method))
        
        if mask is None:
            betamax = get_default_config_option("betamax", betamax)
            self.mask = eigenmask(shape, self.wavenumbers, betamax)
        else:
            self.set_mask(mask)
            
    def set_mask(self, mask):
        """Sets fft mask.
        
        Parameters
        ----------
        mask : ndarray
            A boolean mask, describing fft modes that are used in the solver.
        """
        mask = np.asarray(mask)
        mask_shape = self.wavelengths.shape + self.shape
        
        if mask.shape != mask_shape:
            raise ValueError("Mask shape must be of shape {}".format(mask_shape))
        if not mask.dtype == bool:
            raise ValueError("Not a bool mask.")
            
        self.mask = mask
        self.clear_matrices()
        self.clear_data()
    
    def clear_matrices(self):
        self.stack_matrix = None
        self.refl_matrix = None
        self.field_matrix_in = None
        self.field_matrix_out = None
        self.trans_matrix = None
        self.layer_matrices = []
        
    def clear_data(self):
        self.modes_in = None
        self.modes_out = None

    def _validate_field(self, field):
        field = np.asarray(field)
        min_dim = 3 if self.wavelengths.ndim == 0 else 4
        field_shape = self.wavelengths.shape + (4,) + self.shape
            
        if field.shape[-min_dim:] != field_shape:
            raise ValueError("Field shape not comaptible with solver's requirements. Must be (at least) of shape {}".format(field_shape))
        return field
        
    @property
    def field_in(self):
        """Input field array"""
        return self._field_in
    
    @field_in.setter
    def field_in(self, field):
        self._field_in =  self._validate_field(field)       
        mask, self.modes_in = field2modes(self._field_in,self.wavenumbers, mask = self.mask)    

    def _get_field_data(self, field, copy):
        if copy:
            return field.copy(), self.wavelengths.copy(), self.pixelsize
        else:
            return field, self.wavelengths, self.pixelsize
        
    def get_field_data_in(self, copy = True):
        """Returns input field data tuple.
        
        If copy is set to True (default), a copy of the field_out array is made. 
        """
        return self._get_field_data(self.field_in, copy)

    @property
    def field_out(self):
        """Output field array"""
        return self._field_out
    
    @field_out.setter
    def field_out(self, field):
        self._field_out = self._validate_field(field)   
        mask, self.modes_out = field2modes(self._field_out, self.wavenumbers, mask = self.mask)    

    def get_field_data_out(self, copy = True):
        """Returns output field data tuple.
        
        If copy is set to True (default), a copy of the field_out array is made. 
        """
        return self._get_field_data(self.field_out, copy)
    
    def calculate_field_matrix(self,nin = 1., nout = 1.):
        """Calculates field matrices. 
        
        This must be called after you set material data.
        """
        if self.material_shape is None:
            raise ValueError("You must first set material data")
        self.nin = float(nin)
        self.nout = float(nout)
        #now build up matrices field matrices       
        self.field_matrix_in = tmm3d.f_iso3d(self.mask, self.wavenumbers, n = self.nin, shape = self.material_shape)
        self.field_matrix_out  = tmm3d.f_iso3d(self.mask, self.wavenumbers, n = self.nout, shape = self.material_shape)

    def calculate_reflectance_matrix(self):
        """Calculates reflectance matrix.
        
        Available in "4x4","4x4_1" methods only. This must be called after you 
        have calculated the stack and field matrices.
        """
        if self.method not in ("4x4","4x4_1"):
            raise ValueError("reflectance matrix is available in 4x4 and 4x4_1 methods only")
        if self.stack_matrix is None:
            raise ValueError("You must first calculate stack matrix")
        if self.field_matrix_in is None:
            raise ValueError("You must first calculate field matrix")
        
        mat = tmm3d.system_mat3d(self.stack_matrix, self.field_matrix_in, self.field_matrix_out)
        mat = tmm3d.reflection_mat3d(mat)
        self.refl_matrix = mat

      
    def calculate_transmittance_matrix(self):
        """Calculates transmittance matrix.
        
        Available in "2x2" method only. This must be called after you have
        calculated the stack matrix.
        """
        if self.method != "2x2":
            raise ValueError("transmittance matrix is available in 2x2 method only")
        if self.stack_matrix is None:
            raise ValueError("You must first calculate stack matrix")
        
        self.trans_matrix  = tmm3d.transmission_mat3d(self.stack_matrix) 
       
    def transfer_field(self, field_in = None, field_out = None):
        """Transfers field.
        
        This must be called after you have calculated the transmittance/reflectance 
        matrix.
        
        Parameters
        ----------
        field_in : ndarray, optional
            If set, the field_in attribute will be set to this value prior to
            transfering the field.
        field_out : ndarray, optional
            If set, the field_out attribute will be set to this value prior to
            transfering the field.            
        """
        if self.refl_matrix is None and self.trans_matrix is None:
            raise ValueError("You must first create reflectance/transmittance matrix")
        if field_in is not None:
            self.field_in = field_in
        if field_out is not None:
            self.field_out = field_out       
            
        if self.modes_in is None:
            raise ValueError("You must first set `field_in` data or set the `field_in` argument.")

        m = tmm3d.mode_masks(self.mask, shape = self.material_shape)
        grouped_modes_in = tmm3d.group_modes(self.modes_in, m)
        grouped_modes_out = None if self.modes_out is None else tmm3d.group_modes(self.modes_out, m)
    
        # transfer field
        if self.method.startswith("4x4"):
            grouped_modes_out = tmm3d.reflect3d(grouped_modes_in, rmat = self.refl_matrix, fmatin = self.field_matrix_in, fmatout = self.field_matrix_out, fvecout = grouped_modes_out)
        else:
            grouped_modes_out = tmm3d.transmit3d(grouped_modes_in, tmat = self.trans_matrix, fmatin = self.field_matrix_in, fmatout = self.field_matrix_out, fvecout = grouped_modes_out)
        
        # now ungroup and convert modes to field array
        self.modes_out = tmm3d.ungroup_modes(grouped_modes_out, m)
        self._field_out = modes2field(self.mask, self.modes_out, out = self._field_out)
    
        if self.method.startswith("4x4"):
            #we need to update input field because of relfections.
            self.modes_in = tmm3d.ungroup_modes(grouped_modes_in, m)
            modes2field(self.mask, self.modes_in, out = self._field_in) 


class MatrixBlockSolver3D(BaseMatrixSolver3D):
    """TMM matrix solver for 1d,2d or 3d optical block data using 3d field data.
    
    Examples
    --------
    
    # initialize with a problem shape, wavelengths and pixelsize
    >>> solver = MatrixBlockSolver3D(shape = (96,96), wavelengths = [500,550], pixelsize = 100)
    
    # set optical data first. Data must be compatible with the solver shape.
    >>> solver.set_optical_block(optical_block)
    
    # main transfer matrix calculation procedure.
    >>> solver.calculate_stack_matrix()
    
    # set input and putput field matrices. These can be set without re-computing 
    # the stack matrix.
    >>> solver.calculate_field_matrix(nin = 1.5, nout = 1.5)
    
    # each time you change input/ouptut field matrices you must recompute 
    # the reflectance matrix.
    >>> solver.calculate_reflectance_matrix()
    
    # now transfer some input, optionally set also output field...
    >>> solver.transfer_field(field_in = field_in)
    # you can do this many times with different input fields..
    >>> solver.transfer_field(field_in = field_in)   
    
    # obtain results
    >>> field_data_out = solver.get_field_data_out()
    >>> field_data_in = solver.get_field_data_in()
    """
        
    #: material data, thickness
    d = None
    #: material data, eps values
    epsv = None
    #: material data, eps angles
    epsa = None
    #: specifes number of layers
    nlayers = None
    #: a lsit of layer transfer matrices
    layer_matrices = []
    #: specifies whether material is dispersive
    dispersive = None
    
    def set_optical_block(self, data):
        """Sets optical block data."""
        if self.wavelengths.ndim == 0:
            # we can evaluate epsv in case it is dispersive.
            wavelength = self.wavelengths
        else:
            # we cannot evaluate, so we set wavelength to None
            wavelength = None
            
         #validate and possibly, evaluate at given wavelength
        self.d, self.epsv, self.epsa = validate_optical_block(data, shape = self.shape, wavelength = wavelength)
        
        self.dispersive = is_callable(self.epsv)
        
        self.clear_matrices()
        self.material_shape = material_shape(self.epsv, self.epsa)
        self.material_dim = shape2dim(self.material_shape)
        self.nlayers = len(self.d)
        
    def calculate_layer_matrices(self):
        """Calculates layer matrices. Results are storred in the `layers_matrices` 
        attribute.
        """
        if len(self.layer_matrices) == 0:
            if self.dispersive == True:
                self.layer_matrices = [layer for layer in _iterate_dispersive_layers3d((self.d, self.epsv, self.epsa), self.wavelengths, self.wavenumbers, self.mask, self.shape, self.method)]
            else:
                self.layer_matrices = [tmm3d.layer_mat3d(self.wavenumbers, self.d[i], self.epsv[i], self.epsa[i], mask = self.mask, method = self.method) for i in range(self.nlayers)]
   
    def calculate_stack_matrix(self, keep_layer_matrices = False):
        """Calculates the block stack matrix. Results are storred in the 
        `stack_matrix` attribute.
        
        Parameters
        ----------
        keep_layer_matrices : bool
            If set to True, it will store layer_matrices in the `layer_matrices` 
            attribute.
        """
        if self.epsv is None:
            raise ValueError("You must first set material data")
        if keep_layer_matrices == True:
            self.calculate_layer_matrices()
            if self.stack_matrix is None:
                self.stack_matrix = tmm3d.multiply_mat3d(self.layer_matrices)           
        if self.stack_matrix is None:
            if self.dispersive:
                layer_matrices = _iterate_dispersive_layers3d((self.d, self.epsv, self.epsa), self.wavelengths, self.wavenumbers, self.mask, self.shape, self.method)  
                self.stack_matrix = tmm3d.multiply_mat3d(layer_matrices) 
                #self.stack_matrix = _dispersive_stack_matrix3d((self.d, self.epsv, self.epsa), self.wavelengths, self.wavenumbers, self.mask, self.shape, self.method)
            else:
                self.stack_matrix = tmm3d.stack_mat3d(self.wavenumbers,self.d,self.epsv,self.epsa, method = self.method, mask = self.mask)
    
    def propagate_field(self, field = None):
        """Propagates field over the stack. Returns an iterator. Each next value
        of the iterator is a result of field propagation over each next layer 
        in the optical block.
        
        For "4x4" methods, it propagates from the ouptut layer to the input layer.
        For "2x2" method, it propagates from the input layer to the output layer.
        
        Parameters
        ----------
        field : ndarray, optional
            Field array that is propagated. If not set, `field_out` attribute 
            is used for "4x4" methods and `field_in` attribute is used for
            the "2x2" method.
        
        Yields
        ------
        field : ndarray
            Field array.
        """
        if field is not None:
            field = self._validate_field(field)
            mask, modes = field2modes(field,self.wavenumbers,mask = self.mask)
        else:
            modes = self.modes_out if self.method.startswith("4x4") else self.modes_in
        for modesi in self._propagate_modes(modes):
            yield modes2field(self.mask, modesi)
        
    def _propagate_modes(self, modes = None):
        if self.epsv is None:
            raise ValueError("You must first set material data.")
        verbose_level = DTMMConfig.verbose
        if verbose_level > 1:        
            print("Propagating modes... ")
        direction = -1 if self.method.startswith("4x4") else +1
        
        if modes is None:
            if direction == +1:
                modes = self.modes_in 
            elif direction == -1:
                modes = self.modes_out
        layer_matrices = self.layer_matrices
        
        if layer_matrices != []:
            if direction == -1 :
                layer_matrices = reversed(layer_matrices)
        else:
            indices = range(len(self.d))
            if direction == -1:
                indices = reversed(indices)
            if self.dispersive:
                reverse = True if direction == -1 else False
                layer_matrices = _iterate_dispersive_layers3d((self.d, self.epsv, self.epsa), self.wavelengths, self.wavenumbers, self.mask, self.shape, self.method, reverse = reverse)
            else:
                layer_matrices = (tmm3d.layer_mat3d(self.wavenumbers, self.d[i], self.epsv[i], self.epsa[i], mask = self.mask, method = self.method) for i in indices)

        m = tmm3d.mode_masks(self.mask, shape = self.material_shape)
        grouped_modes = tmm3d.group_modes(modes, m)
        
        for i,layer_matrix in enumerate(layer_matrices): 
            print_progress(i,self.nlayers) 
            grouped_modes = tmm3d.dotmv3d(layer_matrix, grouped_modes)
            yield tmm3d.ungroup_modes(grouped_modes, m)
            
    # def calculate_bulk_modes(self, modes = None):
    #     out = [modes for modes in self.propagate_modes(modes)]
    #     if self.method.startswith("4x4"):
    #         out.reverse()
    #     return out
    
    def calculate_bulk_field(self, field = None, out = None):
        """
        Propagates field and calculates bulk field data.
        
        Parameters
        ----------
        field :  ndarray, optional
            Field array that is propagated. If not set, `field_out` attribute 
            is used for "4x4" methods and `field_in` attribute is used for
            the "2x2" method.
        out : ndarray, optional
            Output array for storring the bulk data.
        
        """
        if field is None:
            field = self.field_out if self.method.startswith("4x4") else self.field_in
            modes = self.modes_out if self.method.startswith("4x4") else self.modes_in
        else:
            field = self._validate_field(field)
            _, modes = field2modes(field,self.wavenumbers, mask = self.mask)
        
        out_shape = (len(self.d) + 1,) + field.shape
        out_dtype = field.dtype
        
        if out is None:
            out = np.empty(out_shape, out_dtype)
        else:
            if out.shape != out_shape:
                raise ValueError("Invalid `out` shape.")
            if out.dtype != out_dtype:
                raise ValueError("Invalid `out`dtype.")
                
        if self.method.startswith("4x4"):
            out[-1] = field
            _out = list(reversed(out))
        else:
            out[0] = field
            _out = out
            
        for m,o in zip(self._propagate_modes(modes), _out[1:]):
            modes2field(self.mask, m, out = o)
        return out
            
class MatrixDataSolver3D(BaseMatrixSolver3D):
    """TMM matrix solver for 1d,2d or 3d optical data using 3d field data.
    
    Examples
    --------
    
    # initialize with a problem shape, wavelengths and pixelsize
    >>> solver = MatrixBlockSolver3D(shape = (96,96), wavelengths = [500,550], pixelsize = 100)
    
    # set optical data first. Data must be compatible with the solver shape.
    >>> solver.set_optical_data(optical_data)
    
    # main transfer matrix calculation procedure.
    >>> solver.calculate_stack_matrix()
    
    # set input and putput field matrices. These can be set without re-computing 
    # the stack matrix.
    >>> solver.calculate_field_matrix(nin = 1.5, nout = 1.5)
    
    # each time you change input/ouptut field matrices you must recompute 
    # the reflectance matrix.
    >>> solver.calculate_reflectance_matrix()
    
    # now transfer some input, optionally set also output field...
    >>> solver.transfer_field(field_in = field_in)
    # you can do this many times with different input fields..
    >>> solver.transfer_field(field_in = field_in)   
    
    # obtain results
    >>> field_data_out = solver.get_field_data_out()
    >>> field_data_in = solver.get_field_data_in()
    """
    #: optical data list
    optical_data = None
    #: a list of block solvers
    block_solvers = []
    #: how many blocks we have
    nblocks = None
    #: describes whether and which blocks are dispersive or not
    dispersive = []
    
    def set_optical_data(self, data):
        self.optical_data = validate_optical_data(data, shape = self.shape)
        self.clear_matrices()
        self.material_shape = optical_data_shape(self.optical_data)
        self.material_dim = shape2dim(self.material_shape)    
        
        self.block_solvers = [MatrixBlockSolver3D(self.shape, wavelengths = self.wavelengths, pixelsize = self.pixelsize, mask = self.mask) for i in range(len(self.optical_data))]
        for block_solver, block_data in zip(self.block_solvers, self.optical_data):
            block_solver.set_optical_block(block_data)
        self.nblocks = len(self.block_solvers)
        self.dispersive = [block_solver.dispersive for block_solver in self.block_solvers]
        
    def calculate_stack_matrix(self, keep_layer_matrices = False, keep_stack_matrices = False):
        if self.optical_data is None:
            raise ValueError("You must first set optical data")
        verbose_level = DTMMConfig.verbose
        if verbose_level > 0:    
            print("Computing optical data stack matrices.")
        
        def iterate_stack_matrices():
            for i,block_solver in enumerate(self.block_solvers):
                if verbose_level > 1:    
                    print("Block {}/{}".format(i+1,self.nblocks))
                block_solver.calculate_stack_matrix(keep_layer_matrices)
                yield block_solver.stack_matrix
                if keep_stack_matrices == False:
                    block_solver.stack_matrix = None
                    
        if self.stack_matrix is None:
                
            data_shapes = tuple((material_shape(epsv,epsa) for (d,epsv,epsa) in self.optical_data))
            reverse = False if self.method.startswith("4x4") else True
            out = tmm3d.multiply_mat3d(iterate_stack_matrices(), mask = self.mask, data_shapes = data_shapes, reverse = reverse)
            
            self.stack_matrix = out

    def calculate_bulk_field(self, field = None):
        verbose_level = DTMMConfig.verbose
        if verbose_level > 0:    
            print("Computing optical data bulk field.")
        if field is None:
            field = field = self.field_out if self.method.startswith("4x4") else self.field_in
        out = []
        block_solvers = self.block_solvers
        if self.method.startswith("4x4") :
            block_solvers = reversed(block_solvers)
        for i,block_solver in enumerate(block_solvers):
            if verbose_level > 1:    
                print("Block {}/{}".format(i+1,self.nblocks))
            bulk = block_solver.calculate_bulk_field(field)
            field = bulk[0] if self.method.startswith("4x4") else bulk[-1]
            out.append(bulk)
        if self.method.startswith("4x4"):
            out.reverse()
        return out
                     
def transfer3d(field_data_in, optical_data, nin = 1., nout = 1., method = "4x4", betamax = None, split_wavelengths = False, field_out = None):
    betamax = get_default_config_option("betamax", betamax)
    
    field_in, ws, p = field_data_in
    
    if split_wavelengths == True:
    
        if field_out is None:
            field_out = np.zeros_like(field_in) 

        out_field = [field_out[...,i,:,:,:] for i in range(len(ws))]
        in_field = (field_in[...,i,:,:,:] for i in range(len(ws)))
        for i,(w,fin,fout) in enumerate(zip(ws,in_field,out_field)):
            # not sure why i need to copy, but this makes it work
            # maybe some issue with non-contiguous data handling with numba?
            # TODO: inspect this issue
            fin = fin.copy()
            fout, w,p = transfer3d((fin,w,p), optical_data, nin = nin, nout = nout, method = method, betamax = betamax) 
            field_out[...,i,:,:,:] = fout
            field_in[...,i,:,:,:] = fin

        return field_out, ws, p
    
    solver = MatrixDataSolver3D(shape = field_in[0].shape[-2:], wavelengths = ws, pixelsize = p, betamax = betamax)
    #set optical data first. Data must be compatible with the solver shape.
    solver.set_optical_data(optical_data)
    # main transfer matrix calculation procedure.
    solver.calculate_stack_matrix()
    solver.calculate_field_matrix(nin = nin, nout = nout)
    # the reflectance matrix.
    
    solver.calculate_reflectance_matrix()

    solver.transfer_field(field_in = field_in)

    return solver.get_field_data_out(copy = False)


    
    
        