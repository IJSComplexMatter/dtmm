"""Diffractive transfer matrix method
"""

__version__ = "0.6.1"

import dtmm.conf
import numpy as np
import time
from .window import *
from .wave import *
from .linalg import * 
from .field import *
from .data import expand, rot90_director, rotate_director, cholesteric_droplet_data,load_stack, save_stack, read_raw, sphere_mask, director2data, validate_optical_data, angles2director, director2angles, read_director, refind2eps, nematic_droplet_data, nematic_droplet_director
from .color import *
from .tmm import *
from .field_viewer import field_viewer, pom_viewer
from .diffract import *
from .data_viewer import *
from .transfer import *
from .jones import jonesvec
from .rotation import rotation_matrix,rotation_matrix_x,rotation_matrix_y,rotation_matrix_z
