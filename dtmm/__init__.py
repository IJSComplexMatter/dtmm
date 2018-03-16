__version__ = "0.0.1.dev"


from .linalg import * 
from .wave import *
from .window import *
from .dirdata import read_director, refind2eps, director2stack, nematic_droplet
from .color import *
from .field import *
from .fviewer import field_viewer
from .diffract import propagate