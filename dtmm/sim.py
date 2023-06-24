"""
TMM Matrix solvers for 3d, 2d and 1d data.

These objects can be used for a simplified treatment of the transfer matrices
for 3d field data on 1d, 2d (and 3d) optical data, or for 2d field data on
1d or 2d optical data. 

User functions
--------------

* :func:`matrix_data_solver`for matrix-based 3d or 2d simulations on optical data (a list of optical blocks).
* :func:`matrix_block_solver`for matrix-based 3d or 2d simulations on an optical block.

Classes
-------

* :class:`MatrixBlockSolver3D` for matrix-based 3d simulations on an optical block.
* :class:`MatrixDataSolver3D` for matrix-based 3d simulations on a list of blocks.
* :class:`MatrixBlockSolver2D` for matrix-based 2d simulations on an optical block.
* :class:`MatrixDataSolver2D` for matrix-based 2d simulations on a list of blocks.

Examples
--------

# initialize with a problem shape, wavelengths and pixelsize, optionally set resolution to a higher value (1 by default).
>>> solver = matrix_block_solver(shape = (96,96), wavelengths = [500,550], 
                            pixelsize = 100, method = "4x4", resolution = 10)

# set optical data first. Data must be compatible with the solver shape.
# for our case, optical data can be of 1D shape (1,1), 2D (96,1) or (1,96) or 3D shape (96,96)
>>> solver.set_optical_block(optical_block)

# main transfer matrix calculation procedure.
>>> solver.calculate_stack_matrix()

# set input and output field matrices. These can be set without re-computing 
# the stack matrix. So, you can set different input/output coupling material
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
from dtmm.conf import get_default_config_option, DTMMConfig, CDTYPE
from dtmm.data import validate_optical_data, material_shape, validate_optical_block, validate_optical_layer, is_callable, shape2dim, optical_data_shape
from dtmm import tmm3d, tmm2d, tmm1d
from dtmm.field import field2modes, modes2field, field2modes1, modes2field1
from dtmm.wave import k0, eigenmask, eigenmask1
from dtmm.print_tools import print_progress
import numpy as np

#: available solver methods
AVAILABLE_MATRIX_SOLVER_METHODS = ("4x4", "4x4_1", "2x2")

def matrix_data_solver(shape, wavelengths  = [500], pixelsize = 100, resolution = 1, mask = None, method = "4x4", betamax = None, **kwargs):
    if isinstance(shape,int) or len(shape) == 1:
        return MatrixDataSolver2D(shape,wavelengths, pixelsize, resolution, mask = mask, method = method, betamax = betamax)
    elif len(shape) == 2:
        return MatrixDataSolver3D(shape,wavelengths, pixelsize, resolution, mask = mask, method = method, betamax = betamax, **kwargs)
    else:
        raise ValueError("Invalid solver shape.")      
def matrix_block_solver(shape, wavelengths  = [500], pixelsize = 100, resolution = 1, mask = None, method = "4x4", betamax = None, **kwargs):
    if isinstance(shape,int) or len(shape) == 1:
        return MatrixBlockSolver2D(shape,wavelengths, pixelsize, resolution, mask = mask, method = method,betamax = betamax)
    elif len(shape) == 2:
        return MatrixBlockSolver3D(shape,wavelengths, pixelsize, resolution, mask = mask, method = method, betamax = betamax, **kwargs)
    else:
        raise ValueError("Invalid solver shape.")

def get_optimal_steps(d, resolution = 1):
    """Given the thickness (d) array, and resolution, it returns optimal number
    of steps for the layer characteristic matrix calculation.
    
    Parameters
    ----------
    d : float or array
        An array of layer thicknesses, or a single thickness value.
    resolution : float
        Approximate resolution for the propagation step in pixel units. 
        
    Returns
    -------
    steps : int or array
        Number of steps for a single layer or an array of number of steps needed
        in the calculation of the layer charactteristic matrix.
    """
    d = np.asarray(d)
    out = np.asarray(np.log2(d*resolution).round(),int).clip(min = 0)      
    return 2**out   

class MatrixReadOnlyProperties(object):
    _dim = 3
    @property
    def dim(self):
        return self._dim
    
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
    
    #: input field matrix
    _field_matrix_in = None
    @property    
    def field_matrix_in(self):
        return self._field_matrix_in   
    
    #: output field matrix
    _field_matrix_out = None
    @property    
    def field_matrix_out(self):
        return self._field_matrix_out   
    
    #: stack matrix
    _stack_matrix = None
    @property    
    def stack_matrix(self):
        return self._stack_matrix   
    
    #: reflectance matrix (for 4x4 approach)
    _refl_matrix = None
    @property    
    def refl_matrix(self):
        return self._refl_matrix   
    
    #: transmittance matrix (for 2x2 approach)
    _trans_matrix = None   
    @property    
    def trans_matrix(self):
        return self._trans_matrix   
    
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
    
class BaseMatrixSolver(MatrixReadOnlyProperties): 
    """Base class for all matrix-based solvers."""
    
    #: module that implements TMM
    tmm = tmm3d

    # field array data
    _field_out = None
    _field_in = None
    _modes_in = None
    _modes_out = None
    
    _mask = None
    
    #: resize parameter for layer_met calculation
    _resize = 1
    
    def __init__(self, shape, wavelengths = [500], pixelsize = 100, resolution = 1,  mask = None, method = "4x4", betamax = None):
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
        if method in AVAILABLE_MATRIX_SOLVER_METHODS :
            self._method = method
        else:
            raise ValueError("Unsupported method {}".format(method))
        
        if mask is None:
            betamax = get_default_config_option("betamax", betamax)
            self._mask = eigenmask(shape, self.wavenumbers, betamax)
        else:
            self.mask = mask
            
    @property       
    def stack_memory_size(self):
        """Specifies memory requirement for stack matrix."""
        size = 2 if self.method.startswith("2x2") else 4
        dt = np.dtype(CDTYPE)
        nmodes = np.asarray(self.nmodes)
        B = (nmodes**2 * size**2 * dt.itemsize).sum()
        kB = B/1024
        MB = kB/1024
        GB = MB/1024
        return {"B": B, "kB" : kB, "MB" : MB, "GB" : GB}
          
    def print_solver_info(self):
        """prints solver info"""
        print(" $ dim : {}".format(self.dim))
        print(" $ shape : {}".format(self.shape))
        print(" # pixel size : {}".format(self.pixelsize))
        print(" # resolution : {}".format(self.resolution))
        print(" $ method : {}".format(self.method))
        print(" $ wavelengths : {}".format(self.wavelengths))
        print(" $ n modes : {}".format(self.nmodes))
        
    def print_data_info(self):
        pass
    
    def print_memory_info(self):
        """prints memory requirement"""
        size_dict = self.stack_memory_size
        size_out = size_dict["B"]
        key_out = "B"
        
        for key in ("kB", "MB", "GB"):
            if size_dict[key] > 1:
                size_out = size_dict[key]
                key_out = key
                
        print(" $ stack size ({}) : {}".format(key_out, size_out))
                
        
    def print_info(self):
        """prints all info"""
        print("-----------------------")
        print("Solver: ")
        self.print_solver_info()
        print("-----------------------")
        print("Memory: ")
        self.print_memory_info()
        print("-----------------------")
        print("Data:")
        self.print_data_info()

    @property            
    def field_dim(self):
        """Required minimum field dim."""
        return self.dim if self.wavelengths.ndim == 0 else self.dim + 1
    
    @property 
    def field_shape(self):
        """Required field shape."""
        return self.wavelengths.shape + (4,) + self.shape
    
    @property
    def mask(self):
        return self._mask
    
    @mask.setter      
    def mask(self, mask):
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
            
        self._mask = mask
        self.clear_matrices()
        self.clear_data()
    
    def clear_matrices(self):
        self._stack_matrix = None
        self._refl_matrix = None
        self._field_matrix_in = None
        self._field_matrix_out = None
        self._trans_matrix = None
        self.layer_matrices = []
        
    def clear_data(self):
        self._modes_in = None
        self._modes_out = None
        self._field_in = None
        self._field_out = None

    def _validate_field(self, field):
        field = np.asarray(field)
            
        if field.shape[-self.field_dim:] != self.field_shape:
            raise ValueError("Field shape not comaptible with solver's requirements. Must be (at least) of shape {}".format(self.field_shape))
        return field
    
    def _validate_modes(self, modes):
        _, modes = self.tmm._validate_modes(self.mask, modes)
        return modes
    
    def _field2modes(self,field):
        _, modes = field2modes(field, self.wavenumbers, mask = self.mask)  
        return modes
    
    def _modes2field(self, modes, out = None):
        field = modes2field(self.mask, modes, out = out)  
        return field     
            
    @property
    def field_in(self):
        """Input field array"""
        if self._field_in is not None:
            return self._field_in
        elif self._modes_in is not None:
            self._field_in = self._modes2field(self._modes_in)
            return self._field_in
        
    @field_in.setter
    def field_in(self, field):
        self._field_in =  self._validate_field(field)   
        # convert field to modes, we must set private attribute not to owerwrite _field_in
        self._modes_in = self._field2modes(self._field_in)    
        
    @property
    def field_out(self):
        """Output field array"""
        if self._field_out is not None:
            return self._field_out
        elif self._modes_out is not None:
            self._field_out = self._modes2field(self._modes_out)
            return self._field_out
    
    @field_out.setter
    def field_out(self, field):
        self._field_out = self._validate_field(field)   
        # convert field to modes, we must set private attribute not to owerwrite _field_out
        self._modes_out = self._field2modes(self._field_out)    
        
    @property
    def modes_in(self):
        return self._modes_in

    @modes_in.setter
    def modes_in(self, modes):
        self._modes_in = self._validate_modes(modes)
        self._field_in = None
        
    @property
    def modes_out(self):
        return self._modes_out

    @modes_out.setter
    def modes_out(self, modes):
        self._modes_out = self._validate_modes(modes)
        self._field_out = None

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

    def get_field_data_out(self, copy = True):
        """Returns output field data tuple.
        
        If copy is set to True (default), a copy of the field_out array is made. 
        """
        return self._get_field_data(self.field_out, copy)
    
    def _get_f_iso(self, n):
        return self.tmm._f_iso(self.mask, self.wavenumbers, n = n, shape = self.material_shape)
    
    def _get_layer_mat(self,d,epsv,epsa, nsteps = 1):
        if self.dispersive == True:
            return self.tmm._dispersive_layer_mat(self.wavenumbers,d,epsv,epsa,mask = self.mask, method = self.method,nsteps = nsteps,resize = self._resize, wavelength = self.wavelengths)
        else:
            return self.tmm._layer_mat(self.wavenumbers,d,epsv,epsa,mask = self.mask, method = self.method,nsteps = nsteps, resize = self._resize)

    def _get_stack_matrix(self,d,epsv,epsa,nsteps = 1):
        return self.tmm._stack_mat(self.wavenumbers,d,epsv,epsa, method = self.method, mask = self.mask, nsteps = nsteps, resize = self._resize)
        
    def calculate_field_matrix(self,nin = 1., nout = 1.):
        """Calculates field matrices. 
        
        This must be called after you set material data.
        """
        if self.material_shape is None:
            raise ValueError("You must first set material data")
        self._nin = float(nin)
        self._nout = float(nout)
        #now build up matrices field matrices       
        self._field_matrix_in = self._get_f_iso(self.nin)
        self._field_matrix_out = self._get_f_iso(self.nout)
        
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
        mat = self.tmm._system_mat(self.stack_matrix, self.field_matrix_in, self.field_matrix_out)
        mat = self.tmm._reflection_mat(mat)
        self._refl_matrix = mat
      
    def calculate_transmittance_matrix(self):
        """Calculates transmittance matrix.
        
        Available in "2x2" method only. This must be called after you have
        calculated the stack matrix.
        """
        if self.method != "2x2":
            raise ValueError("transmittance matrix is available in 2x2 method only")
        if self.stack_matrix is None:
            raise ValueError("You must first calculate stack matrix")
        
        self._trans_matrix  = self.tmm._transmission_mat(self.stack_matrix) 
       
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
        self.transfer_modes()
        return self.field_out

    def transfer_modes(self, modes_in = None, modes_out = None):
        """Transfers modes.
        
        This must be called after you have calculated the transmittance/reflectance 
        matrix.
        
        Parameters
        ----------
        modes_in : mode, optional
            If set, the mode_in attribute will be set to this value prior to
            transfering the field.
        modes_out : mode, optional
            If set, the mode_out attribute will be set to this value prior to
            transfering the field.            
        """

        if self.refl_matrix is None and self.trans_matrix is None:
            raise ValueError("You must first create reflectance/transmittance matrix")
        if modes_in is not None:
            self.modes_in = modes_in
        if modes_out is not None:
            self.modes_out = modes_out       
            
        if self.modes_in is None:
            raise ValueError("You must first set `field_in` data or set the `field_in` argument.")

        grouped_modes_in = self.grouped_modes_in
        grouped_modes_out = self.grouped_modes_out

    
        # transfer field
        if self.method.startswith("4x4"):
            grouped_modes_out = self.tmm._reflect(grouped_modes_in, rmat = self.refl_matrix, fmatin = self.field_matrix_in, fmatout = self.field_matrix_out, fvecout = grouped_modes_out, gvec = self.scattered_modes)
        else:
            grouped_modes_out = self.tmm._transmit(grouped_modes_in, tmat = self.trans_matrix, fmatin = self.field_matrix_in, fmatout = self.field_matrix_out, fvecout = grouped_modes_out)
        
        # now ungroup and convert modes to field array. This also sets the modes_out 
        self.grouped_modes_out = grouped_modes_out
        
        # in case we have field_out array, we update that as well
        if self._field_out is not None:
            self._modes2field(self.modes_out, out = self._field_out) 
    
        if self.method.startswith("4x4"):
            #we need to update input field because of relfections. This also sets the modes_in
            self.grouped_modes_in = grouped_modes_in
            # in case we have field_in array, we update that as well
            if self._field_in is not None:
                self._modes2field(self.modes_in, out = self._field_in) 
        return self.modes_out

    def _group_modes(self, modes):
        if modes is not None:
            m = self.tmm.mode_masks(self.mask, shape = self.material_shape)
            return self.tmm.group_modes(modes, m)
    
    def _ungroup_modes(self, modes):
        if modes is not None:
            m = self.tmm.mode_masks(self.mask, shape = self.material_shape)
            return self.tmm.ungroup_modes(modes, m)

    @property
    def grouped_modes_in(self):
        """Grouped input modes"""
        return self._group_modes(self.modes_in) 

    @grouped_modes_in.setter
    def grouped_modes_in(self, modes):
        self._modes_in = self._ungroup_modes(modes)

    @property
    def grouped_modes_out(self):
        """Grouped output modes"""
        return self._group_modes(self.modes_out) 

    @grouped_modes_out.setter
    def grouped_modes_out(self, modes):
        self._modes_out = self._ungroup_modes(modes)
    
    @property
    def nmodes(self):
        """Number of coupled modes per wavelength"""
        if self.wavelengths.ndim == 0:
            if self.mask is not None:
                return self.mask.sum()
        else:
            if self.mask is not None:
                return tuple((m.sum() for m in self.mask))  

                
    _scattered_modes = None
    _scattered_field = None  
    
    @property
    def grouped_scattered_modes(self):
        """Grouped input modes"""
        return self._group_modes(self.scattered_modes) 

    @grouped_scattered_modes.setter
    def grouped_scattered_modes(self, modes):
        self._scattered_modes = self._ungroup_modes(modes)
    
    @property
    def scattered_field(self):
        """Input field array"""
        if self._scattered_field is not None:
            return self._scattered_field
        elif self._scattered_modes is not None:
            self._scattered_field = self._modes2field(self._scattered_modes)
            return self._scattered_field
        
    @scattered_field.setter
    def scattered_field(self, field):
        self._scattered_field =  self._validate_field(field)   
        # convert field to modes, we must set private attribute not to owerwrite _scattered_field
        self._scattered_modes = self._field2modes(self._scattered_field)    
        

    @property
    def scattered_modes(self):
        return self._scattered_modes

    @scattered_modes.setter
    def scattered_modes(self, modes):
        self._scattered_modes = self._validate_modes(modes)
        self._scattered_field = None

        
class MatrixBlockSolver3D(BaseMatrixSolver):
    """TMM matrix solver for 1d,2d or 3d optical block data using 3d field data.
    
    Examples
    --------
    
    # initialize with a problem shape, wavelengths and pixelsize
    >>> solver = MatrixBlockSolver3D(shape = (96,96), wavelengths = [500,550], pixelsize = 100)
    
    # set optical data first. Data must be compatible with the solver shape.
    >>> solver.set_optical_block(optical_block)
    
    # main transfer matrix calculation procedure.
    >>> solver.calculate_stack_matrix()
    
    # set input and output field matrices. These can be set without re-computing 
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
        
    #: a lsit of layer transfer matrices
    layer_matrices = []
    #: for each layer it specifies number of sub-steps used in the layer characteristic matrix calculation
    nsteps = []
    
    def print_data_info(self):
        print(" $ n layers : {}".format(self.nlayers))
        print(" $ n steps : {}".format(self.nsteps))
        print(" $ material dim : {}".format(self.material_dim))
        print(" $ material shape : {}".format(self.material_shape))
        print(" $ dispersive : {}".format(self.dispersive))

    def set_optical_block(self, data, resize = 1):
        """Sets optical block data."""
        if self.wavelengths.ndim == 0:
            # we can evaluate epsv in case it is dispersive.
            wavelength = self.wavelengths
        else:
            # we cannot evaluate, so we set wavelength to None
            wavelength = None
            
        #validate and possibly, evaluate at given wavelength
        self._d, self._epsv, self._epsa = validate_optical_block(data, shape = self.shape, wavelength = wavelength, dim = self.dim)
        self._dispersive = is_callable(self.epsv)
        
        self.clear_matrices()
        self._material_shape = material_shape(self.epsv, self.epsa, dim = self.dim)
        self._material_dim = shape2dim(self.material_shape, dim = self.dim)

        self.nsteps = get_optimal_steps(self.d, resolution = self.resolution)
        self._resize = resize
        
    @property
    def d(self):
        return self._d
    
    @property
    def epsv(self):
        return self._epsv
    
    @property
    def epsa(self):
        return self._epsa
    
    @property
    def nlayers(self):
        return len(self.d) if self.d is not None else None
       
    def iterate_layer_matrices(self, reverse = False):
        args = (self.d,self.epsv,self.epsa,self.nsteps)
        args = map(reversed, args) if reverse == True else args 
        for d,epsv,epsa,nsteps in zip(*args):
            yield self._get_layer_mat(d,epsv,epsa,nsteps)
            
    def calculate_layer_matrices(self):
        """Calculates layer matrices. Results are storred in the `layers_matrices` 
        attribute.
        """
        if len(self.layer_matrices) == 0:
            self.layer_matrices = [mat for mat in self.iterate_layer_matrices()]
            
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
                self._stack_matrix = self.tmm._multiply_mat(self.layer_matrices)           
        if self.stack_matrix is None:
            if self.dispersive:
                self._stack_matrix = self.tmm._multiply_mat(self.iterate_layer_matrices()) 
            else:
                self._stack_matrix = self._get_stack_matrix(self.d, self.epsv, self.epsa, nsteps = self.nsteps)
                
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
            modes = self._field2modes(field)
        else:
            modes = self.modes_out if self.method.startswith("4x4") else self.modes_in
        for modesi in self.propagate_modes(modes):
            yield self._modes2field(modesi)
        
    def propagate_modes(self, modes = None, invert = False):
        if self.epsv is None:
            raise ValueError("You must first set material data.")
        verbose_level = DTMMConfig.verbose
        if verbose_level > 1:        
            print("Propagating modes... ")
        direction = -1 if self.method.startswith("4x4") else +1
        if invert == True:
            direction = direction * (-1)
        
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
            layer_matrices = self.iterate_layer_matrices(reverse = (direction == -1))

        grouped_modes = self._group_modes(modes)
        
        for i,layer_matrix in enumerate(layer_matrices): 
            print_progress(i,self.nlayers) 
            grouped_modes = self.tmm._dotmv(layer_matrix, grouped_modes)
            yield self._ungroup_modes(grouped_modes)
            
    def get_bulk_field(self, out = None):
        """
        Calculates bulk field data by propagating field over the layers.
        
        Parameters
        ----------
        out : ndarray, optional
            Output array for storring the bulk data.
            
        Returns
        -------
        out : ndarray
            Bulk field array.
        """
        # for 4x4 methods we must propagate backwards, so we porpagate output field
        field = self.field_out if self.method.startswith("4x4") else self.field_in
        modes = self.modes_out if self.method.startswith("4x4") else self.modes_in
            
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
            #for 4x4 methods we must propagate backwards, so reverse output
            out[-1] = field
            out_list = list(reversed(out))
        else:
            out[0] = field
            out_list = out
            
        for m,o in zip(self.propagate_modes(modes), out_list[1:]):
            self._modes2field( m, out = o)
        return out
    
from dtmm.data import effective_block
from dtmm.transfer import transfer_4x4, transfer_field
    
from dtmm.propagate_4x4 import propagate_4x4_effective_4


#(field, wavenumbers, layer, effective_layer, beta = 0, phi=0,
#                    nsteps = 1, diffraction = True, 
#                    betamax = BETAMAX,out = None):

class ScatteringBlockSolver3D(MatrixBlockSolver3D):
    scattering_matrices = []
    ref = None
    
    def _group_modes(self, modes):
        #no mode grouping for 1d
        return modes

    def _ungroup_modes(self, modes):
        #no mode grouping for 1d
        return modes
    
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
                self._stack_matrix = self.tmm._multiply_mat(self.layer_matrices)           
        if self.stack_matrix is None:
            if self.dispersive:
                self._stack_matrix = self.tmm._multiply_mat(self.iterate_layer_matrices()) 
            else:
                self._stack_matrix = self._get_stack_matrix(self.d, self.epsv_eff, self.epsa_eff, nsteps = self.nsteps)
    
    def set_optical_block(self, data, resize = 1):
        """Sets optical block data."""
        super().set_optical_block(data, resize)
        print('yeaf')
        _, self._epsv_eff, self._epsa_eff = effective_block(data,1)
        self._epsv_eff  = self._epsv_eff[...,None,None,:] 
        self._epsa_eff  = self._epsa_eff[...,None,None,:] 

    
    @property
    def epsv_eff(self):
        return self._epsv_eff
    
    @property
    def epsa_eff(self):
        return self._epsa_eff   
    
    def iterate_layer_matrices(self, reverse = False):
        args = (self.d,self.epsv_eff,self.epsa_eff,self.nsteps)
        args = map(reversed, args) if reverse == True else args 
        
        for d,epsv,epsa,nsteps in zip(*args):
            yield self._get_layer_mat(d,epsv,epsa,nsteps)
            
    def calculate_scattering_matrices(self):
        if len(self.scattering_matrices) == 0:
            self.scattering_matrices = [mat for mat in self.iterate_scattering_matrices()]
            
    def iterate_scattering_matrices(self, reverse = False):
        args = (self.d,self.epsv, self.epsv_eff,self.epsa, self.epsa_eff,self.nsteps)
        args = map(reversed, args) if reverse == True else args 
        for d,epsv,epsv_eff,epsa,epsa_eff,nsteps in zip(*args):
            S = self._get_scattering_mat(d,epsv,epsv_eff,epsa,epsa_eff,nsteps)
            yield S
            
    def scatter_modes(self):
        modes = self.modes_out if self.method.startswith("4x4") else self.modes_in
        
        scattering_matrices = self.scattering_matrices if len(self.scattering_matrices) !=0 else self.iterate_scattering_matrices()

        for i, (mat, modes) in enumerate(zip(scattering_matrices, self.propagate_modes(modes))):
            grouped_modes = self._group_modes(modes)
            print_progress(i,self.nlayers) 
            yield self._ungroup_modes(grouped_modes)
            
    def scatter_field(self, refin = None):
        field_out, _ , _ = transfer_4x4((self.field_in, self.wavelengths, self.pixelsize), [(self.d,self.epsv,self.epsa)], 
                                                   eff_data = [[1]*len(self.d)],
                                                   #eff_data = [(self.d[i], self.epsv[i,...,0,0,:], self.epsa[i,...,0,0,:]) for i in range(len(self.d))],
                                                   betamax = 0.9,
                                                   reflection = 4,
                                                   nin = 1.5, nout = 1.5,
                                                   invert = False,
                                                   project = False)
        
        for mode_out in self.propagate_modes(invert = True):
            pass
        
        #self.modes_out = mode_out
        
        print('mean',field_out.mean())
        
        ps = list((range(self.nlayers)))
        
        
        
        # for p in range(self.nlayers): 
        #     print_progress(p,self.nlayers) 

            
        #     layer = (self.d[p], self.epsv[p], self.epsa[p])
        #     effective_layer =(-self.d[p], self.epsv_eff[p], self.epsa_eff[p])
            
        #     field = propagate_4x4_effective_4(field, self.wavenumbers, layer, effective_layer, beta = 0, phi=0,
        #                         nsteps = 1, diffraction = True, 
        #                         betamax = 0.9, out = None)
        
        
        scattered_modes = self.tmm._add(self.modes_out, self._field2modes(-field_out))
        
        if self.scattered_modes is not None:
            self.scattered_modes = self.tmm._add(self.scattered_modes, scattered_modes)
        else:
        
            self.scattered_modes = self.tmm._dotmv(self.stack_matrix, scattered_modes)
            
     

    
    def scatter_field2(self):
        for field in self.propagate_scattered_field():
            pass
        
        self.scattered_field = field #+ self.field_in
    
    def propagate_scattered_field(self, field = None):
        if field is not None:
            field = self._validate_field(field)
            modes = self._field2modes(field)
        else:
            modes = self.modes_out if self.method.startswith("4x4") else self.modes_in
        for modesi in self.propagate_scattered_modes(modes):
            yield self._modes2field(modesi)
    
    
    def propagate_scattered_modes(self, transmitted_modes = None):
        if self.epsv is None:
            raise ValueError("You must first set material data.")
        verbose_level = DTMMConfig.verbose
        if verbose_level > 1:        
            print("Propagating modes... ")
        direction = -1 if self.method.startswith("4x4") else +1
        
        if transmitted_modes is None:
            if direction == +1:
                transmitted_modes = self.modes_in 
            elif direction == -1:
                transmitted_modes = self.modes_out
        layer_matrices = self.layer_matrices
        
        if layer_matrices != []:
            if direction == -1 :
                layer_matrices = reversed(layer_matrices)
        else:
            layer_matrices = self.iterate_layer_matrices(reverse = (direction == -1))

        grouped_transmitted_modes = self._group_modes(transmitted_modes)
        grouped_scattered_modes = None
        
        ps = list(reversed(range(self.nlayers)))
        
        
        for i,layer_matrix in enumerate(layer_matrices): 
            print_progress(i,self.nlayers) 
            p = ps[i]
            
            ref = self._modes2field(transmitted_modes)
            
            layer = (-self.d[p], self.epsv[p], self.epsa[p])
            effective_layer =(-self.d[p], self.epsv_eff[p], self.epsa_eff[p])
            
            ref = propagate_4x4_effective_4(ref, self.wavenumbers, layer, effective_layer, beta = 0, phi=0,
                                nsteps = 1, diffraction = True, 
                                betamax = 0.9, out = None)
            
            grouped_transmitted_modes = self.tmm._dotmv(layer_matrix, grouped_transmitted_modes)
            
            if grouped_scattered_modes is not None:
                grouped_scattered_modes = self.tmm._dotmv(layer_matrix, grouped_scattered_modes)
            
            transmitted_modes = self._ungroup_modes(grouped_transmitted_modes)
            transmitted_field = self._modes2field(transmitted_modes)
            
            
            scattered_field = ref - transmitted_field
            scattered_modes = self._field2modes(scattered_field )
            if grouped_scattered_modes is not None:
                grouped_scattered_modes = self.tmm._add(grouped_scattered_modes, self._group_modes(scattered_modes))
            else:
                grouped_scattered_modes = self._group_modes(scattered_modes)
            scattered_modes = self._ungroup_modes(grouped_scattered_modes)
            
            yield scattered_modes
            
            
    
    
class MatrixDataSolver3D(BaseMatrixSolver):
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
    
    def print_data_info(self):
        print(" $ n layers : {}".format(self.nlayers))
        print(" $ material dim : {}".format(self.material_dim))
        print(" $ material shape : {}".format(self.material_shape))
        print(" $ dispersive : {}".format(self.dispersive))
    
    @property
    def nblocks(self):
        return len(self.block_solvers) if self.block_solvers is not None else None
    
    @property
    def nlayers(self):
        return [b.nlayers for b in self.block_solvers]
        
    def get_block_solver(self, i = None, resize = 1):
        """Returns a new instance of block solver based on the settings of the data solver"""
        block_solver =  MatrixBlockSolver3D(self.shape, wavelengths = self.wavelengths, method = self.method, pixelsize = self.pixelsize, mask = self.mask, resolution = self.resolution)
        if i is not None:
            block_solver.set_optical_block(self.optical_data[i], resize = resize)
        return block_solver
    
    def set_optical_data(self, data, resize = 1):
        
        self.optical_data = validate_optical_data(data, shape = self.shape, dim = self.dim)
        self.clear_matrices()
        self._material_shape = optical_data_shape(self.optical_data,dim = self.dim)
        self._material_dim = shape2dim(self.material_shape, dim = self.dim)    
        
        self.block_solvers = [self.get_block_solver(i, resize = resize) for i in range(len(self.optical_data))]
        self._dispersive = [block_solver.dispersive for block_solver in self.block_solvers]


    def iterate_stack_matrices(self, keep_layer_matrices = False, keep_stack_matrices = False):
        verbose_level = DTMMConfig.verbose
        for i,block_solver in enumerate(self.block_solvers):
            if verbose_level > 1:    
                print("Block {}/{}".format(i+1,self.nblocks))
            block_solver.calculate_stack_matrix(keep_layer_matrices)
            yield block_solver.stack_matrix
            if keep_stack_matrices == False:
                block_solver._stack_matrix = None

        
    def calculate_stack_matrix(self, keep_layer_matrices = False, keep_stack_matrices = False):
        if self.optical_data is None:
            raise ValueError("You must first set optical data")
        verbose_level = DTMMConfig.verbose
        if verbose_level > 0:    
            print("Computing optical data stack matrices.")
        
                    
        if self.stack_matrix is None:
                
            reverse = False if self.method.startswith("4x4") else True
            if self.dim == 3:
                data_shapes = tuple((material_shape(epsv,epsa, dim = self.dim) for (d,epsv,epsa) in self.optical_data))
                out = self.tmm._multiply_mat(self.iterate_stack_matrices( keep_layer_matrices = keep_layer_matrices , keep_stack_matrices = keep_stack_matrices), mask = self.mask, data_shapes = data_shapes, reverse = reverse)
            else:
                # no need to upscale matrices when using 2D solver
                out = self.tmm._multiply_mat(self.iterate_stack_matrices( keep_layer_matrices = keep_layer_matrices , keep_stack_matrices = keep_stack_matrices),  reverse = reverse)
                
            self._stack_matrix = out

    def get_bulk_field(self):
        verbose_level = DTMMConfig.verbose
        if verbose_level > 0:    
            print("Computing optical data bulk field.")
        field = self.field_out if self.method.startswith("4x4") else self.field_in
        out = []
        block_solvers = self.block_solvers
        if self.method.startswith("4x4") :
            block_solvers = reversed(block_solvers)
        for i,block_solver in enumerate(block_solvers):
            if verbose_level > 1:    
                print("Block {}/{}".format(i+1,self.nblocks))
            bulk = block_solver.get_bulk_field(field)
            field = bulk[0] if self.method.startswith("4x4") else bulk[-1]
            out.append(bulk)
        if self.method.startswith("4x4"):
            out.reverse()
        return out
    
 
class MatrixBlockSolver2D(MatrixBlockSolver3D):
    tmm = tmm2d
    dim = 2

    def __init__(self, n, wavelengths = [500], pixelsize = 100, resolution = 1,  
                 mask = None, method = "4x4", betay = 0., swap_axes = False, betamax = None):
        """
        Paramters
        ---------
        shape : int
            Size of the field data in pixels.
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
        betay : float, optional
            The betay value (0 by default) of the input field. If swap_axes it 
            defines the betax value. In this case the modal decomposition is in the
            y plane instead of the x plane.
        swap_axes : bool, optional
            Whether to swap axes betay->betax. Set to True if your data is a vertical
            (y) grating nad not a horizontal (x).
        betamax : float, optional
            If `mask` is not set, this value is used to create the mask. If not 
            set, default value is taken from the config.
        """
        if isinstance(n,int):
            self._shape = n,
        else:
            n, = n
            self._shape = int(n),

        self._wavelengths = np.asarray(wavelengths)
        if self.wavelengths.ndim not in (0,1):
            raise ValueError("`wavelengths` must be a scalar or an 1D array.")
        self._pixelsize = float(pixelsize)
        self._wavenumbers = k0(self.wavelengths, pixelsize)
        self._resolution = resolution
        
        self._betay = float(betay)
        self._swap_axes = bool(swap_axes)

        method = str(method) 
        if method in AVAILABLE_MATRIX_SOLVER_METHODS :
            self._method = method
        else:
            raise ValueError("Unsupported method {}".format(method))
        
        if mask is None:
            betamax = get_default_config_option("betamax", betamax)
            self._mask = eigenmask1(n, self.wavenumbers, betay = self.betay, betamax = betamax)
        else:
            self.mask = mask
            
    @property       
    def betay(self):
        return self._betay
    
    @property   
    def swap_axes(self):
        return self._swap_axes
        
    def print_solver_info(self):
        """prints solver info"""
        super().print_solver_info()
        print(" $ betay : {}".format(self.betay))
        print(" $ swap axes: {}".format(self.swap_axes))
            
    def _group_modes(self, modes):
        #no mode grouping for 2d
        return modes

    def _ungroup_modes(self, modes):
        #no mode grouping for 2d
        return modes
    
    def _field2modes(self,field):
        _, modes = field2modes1(field, self.wavenumbers, betay = self.betay, mask = self.mask)  
        return modes
    
    def _modes2field(self, modes, out = None):
        field = modes2field1(self.mask, modes, out = out)  
        return field      

    def _get_f_iso(self, n):
        return self.tmm._f_iso(self.mask, self.wavenumbers, n = n, shape = self.material_shape,swap_axes = self.swap_axes)

    def _get_layer_mat(self,d,epsv,epsa,nsteps = 1):
        if self.dispersive == True:
            return self.tmm._dispersive_layer_mat(self.wavenumbers,d,epsv,epsa,mask = self.mask, method = self.method, betay = self.betay, swap_axes  = self.swap_axes, nsteps = nsteps,wavelength = self.wavelengths)
        else:
            return self.tmm._layer_mat(self.wavenumbers,d,epsv,epsa,mask = self.mask, method = self.method, betay = self.betay, swap_axes  = self.swap_axes, nsteps = nsteps)

    def _get_stack_mat(self,d,epsv,epsa,nsteps = 1):
        return self.tmm._stack_mat(self.wavenumbers,d,epsv,epsa, method = self.method, mask = self.mask, betay = self.betay, nsteps = nsteps, swap_axes = self.swap_axes)

class MatrixBlockSolver1D(MatrixBlockSolver3D):
    tmm = tmm1d
    dim = 1

    def __init__(self, wavelengths = [500], pixelsize = 100, resolution = 1,  
                 method = "4x4", betax = 0, betay = 0.,):
        """
        Paramters
        ---------
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
        betay : float, optional
            The betay value (0 by default) of the input field. If swap_axes it 
            defines the betax value. In this case the modal decomposition is in the
            y plane instead of the x plane.
        swap_axes : bool, optional
            Whether to swap axes betay->betax. Set to True if your data is a vertical
            (y) grating nad not a horizontal (x).
        betamax : float, optional
            If `mask` is not set, this value is used to create the mask. If not 
            set, default value is taken from the config.
        """
        #1D case, no shape
        self._shape = ()
        self._wavelengths = np.asarray(wavelengths)
        if self.wavelengths.ndim not in (0,1):
            raise ValueError("`wavelengths` must be a scalar or an 1D array.")
        self._pixelsize = float(pixelsize)
        self._wavenumbers = k0(self.wavelengths, pixelsize)
        self._resolution = resolution
        
        self._betay = float(betay)
        self._betax = float(betax)

        method = str(method) 
        if method in AVAILABLE_MATRIX_SOLVER_METHODS :
            self._method = method
        else:
            raise ValueError("Unsupported method {}".format(method))
        
    @property       
    def betay(self):
        return self._betay

    @property       
    def betax(self):
        return self._betax
       
    def print_solver_info(self):
        """prints solver info"""
        super().print_solver_info()
        print(" $ betay : {}".format(self.betay))
        print(" $ betax: {}".format(self.betax))
            
    def _group_modes(self, modes):
        #no mode grouping for 2d
        return modes

    def _ungroup_modes(self, modes):
        #no mode grouping for 2d
        return modes
    
    def _validate_modes(self, modes):
        #no modes in 1D
        return modes
    
    def _field2modes(self,field):
        # no modes in 1D
        return field
    
    def _modes2field(self, modes):
        # no modes in 1D
        return modes     

    def _get_f_iso(self, n):
        return self.tmm._f_iso(self.wavenumbers, n = n)

    def _get_layer_mat(self,d,epsv,epsa,nsteps = 1):
        if self.dispersive == True:
            return self.tmm._dispersive_layer_mat(self.wavenumbers,d,epsv,epsa,betax = self.betax, method = self.method, betay = self.betay, nsteps = nsteps,wavelength = self.wavelengths)
        else:
            return self.tmm._layer_mat(self.wavenumbers,d,epsv,epsa,betax = self.betax, method = self.method, betay = self.betay, nsteps = nsteps)

    def _get_stack_mat(self,d,epsv,epsa,nsteps = 1):
        return self.tmm._stack_mat(self.wavenumbers,d,epsv,epsa, method = self.method, betax = self.betax, betay = self.betay, nsteps = nsteps)
    
    
class MatrixDataSolver2D(MatrixDataSolver3D,MatrixBlockSolver2D):
    """TMM matrix solver for 1d, or 2d optical data using 2d field data.
    """
    def get_block_solver(self, i = None, resize = 1):
        """Returns a new instance of block solver based on the settings of the data solver"""
        block_solver =  MatrixBlockSolver2D(self.shape, wavelengths = self.wavelengths, pixelsize = self.pixelsize, method = self.method, mask = self.mask,betay = self.betay, swap_axes = self.swap_axes,  resolution = self.resolution)
        if i is not None:
            block_solver.set_optical_block(self.optical_data[i], resize = resize)
        return block_solver     
 
class MatrixDataSolver1D(MatrixBlockSolver1D,MatrixDataSolver3D):
    """TMM matrix solver for 1d, or 2d optical data using 2d field data.
    """
    def get_block_solver(self, i = None, resize = None):
        """Returns a new instance of block solver based on the settings of the data solver"""
        block_solver =  MatrixBlockSolver1D(wavelengths = self.wavelengths, pixelsize = self.pixelsize, method = self.method, betax = self.betax,betay = self.betay, resolution = self.resolution)
        if i is not None:
            block_solver.set_optical_block(self.optical_data[i])
        return block_solver             
 
def transfer3d(field_data_in, optical_data, nin = 1., nout = 1., resolution = 1, method = "4x4", betamax = None, split_wavelengths = False, field_out = None):
    """Transfers 3d field data using 3d matrix solver."""
    betamax = get_default_config_option("betamax", betamax)
    
    field_in, ws, p = field_data_in
    
    if field_out is not None:
        if field_out.shape != field_in.shape or field_in.dtype != field_out.dtype:
            raise ValueError("Input/output fields shape/dtype mismatch.")
    
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
            fout, w,p = transfer3d((fin,w,p), optical_data, nin = nin, nout = nout, resolution = resolution, method = method, betamax = betamax) 
            field_out[...,i,:,:,:] = fout
            field_in[...,i,:,:,:] = fin

        return field_out, ws, p
    
    solver = MatrixDataSolver3D(shape = field_in[0].shape[-2:], wavelengths = ws, pixelsize = p,resolution = resolution,  method = method, betamax = betamax)
    #set optical data first. Data must be compatible with the solver shape.
    solver.set_optical_data(optical_data)
    # main transfer matrix calculation procedure.
    solver.calculate_stack_matrix()
    solver.calculate_field_matrix(nin = nin, nout = nout)
    # the reflectance matrix.
    
    solver.calculate_reflectance_matrix()

    solver.transfer_field(field_in = field_in, field_out = field_out)

    return solver.get_field_data_out(copy = False)

def transfer2d(field_data_in, optical_data, nin = 1., nout = 1., resolution = 1, betay = 0., method = "4x4", betamax = None, split_wavelengths = False, field_out = None):
    """Transfers 2d field data using 2d matrix solver"""
    
    betamax = get_default_config_option("betamax", betamax)
    
    field_in, ws, p = field_data_in
    
    if field_out is not None:
        if field_out.shape != field_in.shape or field_in.dtype != field_out.dtype:
            raise ValueError("Input/output fields shape/dtype mismatch.")
            
    if split_wavelengths == True:
    
        if field_out is None:
            field_out = np.zeros_like(field_in) 

        out_field = [field_out[...,i,:,:] for i in range(len(ws))]
        in_field = (field_in[...,i,:,:] for i in range(len(ws)))
        for i,(w,fin,fout) in enumerate(zip(ws,in_field,out_field)):
            # not sure why i need to copy, but this makes it work
            # maybe some issue with non-contiguous data handling with numba?
            # TODO: inspect this issue
            fin = fin.copy()
            fout, w,p = transfer2d((fin,w,p), optical_data, nin = nin, nout = nout, resolution = resolution, betay = betay, method = method, betamax = betamax) 
            field_out[...,i,:,:] = fout
            field_in[...,i,:,:] = fin

        return field_out, ws, p
    
    solver = MatrixDataSolver2D(n = field_in[0].shape[-1], wavelengths = ws, pixelsize = p, betamax = betamax, betay = betay, resolution = resolution)
    #set optical data first. Data must be compatible with the solver shape.
    solver.set_optical_data(optical_data)
    # main transfer matrix calculation procedure.
    solver.calculate_stack_matrix()
    solver.calculate_field_matrix(nin = nin, nout = nout)
    # the reflectance matrix.
    
    solver.calculate_reflectance_matrix()
    solver.transfer_field(field_out = field_out, field_in = field_in)

    return solver.get_field_data_out(copy = False)
       

def transfer1d(field_data_in, optical_data, nin = 1., nout = 1., resolution = 1, betax = 0, betay = 0., method = "4x4", split_wavelengths = False, field_out = None):
    """Transfers 2d field data using 2d matrix solver"""
    
    field_in, ws, p = field_data_in
    
    if field_out is not None:
        if field_out.shape != field_in.shape or field_in.dtype != field_out.dtype:
            raise ValueError("Input/output fields shape/dtype mismatch.")
            
    if split_wavelengths == True:
    
        if field_out is None:
            field_out = np.zeros_like(field_in) 

        out_field = [field_out[...,i,:,:] for i in range(len(ws))]
        in_field = (field_in[...,i,:,:] for i in range(len(ws)))
        for i,(w,fin,fout) in enumerate(zip(ws,in_field,out_field)):
            # not sure why i need to copy, but this makes it work
            # maybe some issue with non-contiguous data handling with numba?
            # TODO: inspect this issue
            fin = fin.copy()
            fout, w,p = transfer1d((fin,w,p), optical_data, nin = nin, nout = nout, resolution = resolution, betay = betay, method = method, betax = betax) 
            field_out[...,i,:,:] = fout
            field_in[...,i,:,:] = fin

        return field_out, ws, p
    
    solver = MatrixDataSolver1D( wavelengths = ws, pixelsize = p, betax = betax, betay = betay, resolution = resolution)
    #set optical data first. Data must be compatible with the solver shape.
    solver.set_optical_data(optical_data)
    # main transfer matrix calculation procedure.
    solver.calculate_stack_matrix()
    solver.calculate_field_matrix(nin = nin, nout = nout)
    # the reflectance matrix.
    
    solver.calculate_reflectance_matrix()
    solver.transfer_field(field_out = field_out, field_in = field_in)

    return solver.get_field_data_out(copy = False)