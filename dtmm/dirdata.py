# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:44:09 2017

@author: andrej
"""

from __future__ import absolute_import, print_function, division

import numpy as np
#from mayavi import mlab

#from dir_to_stack import dirtostack

import numba

from dtmm.conf import  NF32DTYPE, F32DTYPE, FDTYPE

def read_director(file, shape, dtype = "float32", nvec = "xyz"):
    """Reads raw director data from a binary file. 
    
    Shape must be provided. It defines shape of the compute box. 
    """
    try:
        i,j,k = shape
        shape = i,j,k,3
    except:
        raise TypeError("shape must be director data shape (x,y,z)")
    data = read_data(file, shape, dtype)

    if nvec != "xyz":
        index = {"x" : 0, "y": 1, "z" : 2}
        out = np.empty_like(data)
        for i,idn in enumerate(nvec):
            j = index[idn]
            out[...,j] = data[...,i]
        return out
    else:
        return data
    

def read_data(file, shape, dtype):
    """Reads raw data from a binary file.
    
    A valid shape and array dtype must be provided.
    """  
    dtype = np.dtype(dtype)
    count = np.multiply.reduce(shape) * dtype.itemsize
    a = np.fromfile(file, dtype, count)
    return a.reshape(shape)  

def plot_director(data):
    """Plots director data."""
    x,y,z,c = data.shape
    
    # Make the grid
    x, y, z = np.meshgrid(np.arange(0., x, 1.),
                          np.arange(0., y, 1.),
                          np.arange(0., z, 1.))
    
    # Set the direction data for the arrows
    u = data[...,0]
    v = data[...,1]
    w = data[...,2]
    
    src = mlab.pipeline.vector_field(u, v, w)
    return mlab.pipeline.vector_cut_plane(src, mask_points=2, scale_factor=3)
    #return mlab.pipeline.vectors(src, mask_points=20, scale_factor=3.)
    
    #return mlab.quiver3d(x, y, z, u, v, w, scale_factor = 0.5)    

#def stackview(data, height = 0, theta = 0., phi = 0., ):
#    fill = np.zeros(shape = (4,), dtype = FDTYPE)
#    id = fill.view(UDTYPE)
#    id[0] = 0
#    return _stackview(data, height, theta, phi, fill)
#    
#@numba.njit(["float32[:,:,:,:](float32[:,:,:,:],int32,float32,float32,float32[:])"])    
#def _stackview(data, height, theta, phi, fill):
#    nx,ny,nz,c = data.shape
#    #if out is None:
#    out = np.empty(shape = (nx,ny,nz,4),dtype = F32DTYPE)    
#    if c != 4:
#        raise TypeError("invalid shape")    
#    x = np.cos(phi) * np.sin(theta)   
#    y = np.sin(phi) * np.sin(theta) 
#    z = np.cos(theta)
#    for i in range(nx):
#        for j in range(ny):
#            for k in range(nz):
#                kk = k - height
#                ii = int(0.5 + i + x * kk/z)
#                jj = int(0.5 + j + y * kk/z)
#                if ii >= 0 and jj >= 0 and ii < nx and jj < ny:
#                    out[i,j,k] = data[ii,jj,k]
#                else:
#                    out[i,j,k] = fill
#    return out  
   
def refind(n1 = 1, n3 = None, n2 = None):
    """Returns material array (eps) from a list of input refractive indices"""
    if n3 is None:
        if n2 is None:
            n3 = n1
            n2 = n1
        else:
            raise ValueError("Both n2, and n3 must be set")
    if n2 is None:
        n2 = n1
    return np.array([n1,n2,n3])
    
def _r3(shape):
    """Returns r vector array of given shape."""
    nz,nx,ny = [l//2 for l in shape]
    az, ax, ay = [np.arange(-l / 2. + .5, l / 2. + .5) for l in shape]
    zz, xx, yy = np.meshgrid(az, ax, ay, indexing = "ij")
    return xx, yy, zz
    #r = ((xx/(nx*scale))**2 + (yy/(ny*scale))**2 + (zz/(nz*scale))**2) ** 0.5 
    #return r
    
def nematic_droplet_director(shape, radius, profile = "r", retmask = False):
    """Returns nematic director data of a nematic droplet with a given radius.
    
    Parameters
    ----------
    shape : tuple
        (nz,nx,ny) shape of the output data box. First dimension is the 
        number of layers, second and third are the x and y dimensions of the box.
    radius : float
        radius of the droplet.
    profile : str, optional
        Director profile type. It can be a radial profile "r", or homeotropic 
        profile with director orientation specified with the parameter "x", "y",
        or "z".
    director : bool, optional
        Whether to output mask data as well
        
    Returns
    -------
    out : array or tuple of arrays 
        A director data array, or tuple of director mask and director data arrays.
    """
    
    nz, nx, ny = shape
    out = np.zeros(shape = (nz,nx,ny,3), dtype = F32DTYPE)
    xx, yy, zz = _r3(shape)
    
    r = (xx**2 + yy**2 + zz**2) ** 0.5 
    mask = (r <= radius)
    m = np.logical_and(mask,r != 0.)
    rm = r[m]
    if profile == "r":
        out[...,0][m] = xx[m]/rm
        out[...,1][m] = yy[m]/rm
        out[...,2][m] = zz[m]/rm
    else:
        index = {"x": 0,"y": 1,"z": 2}
        try:
            i = index[profile]
            out[...,i][m] = 1.
        except KeyError:
            raise ValueError("Unsupported profile type!")
    if retmask == True:
        return mask, out
    else: 
        return out

def nematic_droplet_data(shape, radius, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5):
    """Returns nematic droplet data.
    
    This function returns a stack, mask and material arrays of a 
    nematic droplet, suitable for light propagation calculation tests.
    
    Parameters
    ----------
    shape : tuple
        (nz,nx,ny) shape of the stack. First dimension is the 
        number of layers, second and third are the x and y dimensions of the box.
    radius : float
        radius of the droplet.
    profile : str, optional
        Director profile type. It can be a radial profile "r", or homeotropic 
        profile with director orientation specified with the parameter "x", "y",
        or "z".
    no : float, optional
        Ordinary refractive index of the material (1.5 by default)
    ne : float, optional
        Extraordinary refractive index (1.6 by default)
    nhost : float, optional
        Host material refractive index (1.5 by default)
        
    Returns
    -------
    out : tuple  
        A (stack, mask, material) tuple of arrays.
    """
    mask, director = nematic_droplet_director(shape, radius, profile = profile, retmask = True)
    material = refind2eps([refind(nhost), refind(no,ne)])
    return director2stack(director), mask, material
    
    
@numba.njit([NF32DTYPE[:,:,:,:](NF32DTYPE[:,:,:,:])])
def director2stack(data):
    """Converts director data (nx,ny,nz) to stack data (order,theta phi).
    Director data should be a vector with size 0 <= order <= 1."""
    nz,nx,ny,c = data.shape
    out = np.empty(shape = (nz,nx,ny,3),dtype = F32DTYPE)

    if c != 3:
        raise TypeError("invalid shape")
    for i in range(nz):
        for j in range(nx):
            for k in range(ny):
                vec = data[i,j,k]
                x = vec[0]
                y = vec[1]
                z = vec[2]
                phi = np.arctan2(y,x)
                theta = np.arctan2(np.sqrt(x**2+y**2),z)
                tmp = out[i,j,k]
                #id = tmp.view(UDTYPE)
                
                #id[0] = 0
                tmp[0] = np.sqrt(x**2+y**2+z**2)
                tmp[1] = theta
                tmp[2] = phi
    return out

def add_isotropic_border(data, shape, xoff = None, yoff = None, zoff = None):
    """Adds isotropic border area to director or stack data"""
    nz,nx,ny = shape
    out = np.zeros(shape = shape + data.shape[3:], dtype = data.dtype)
    if xoff is None:
        xoff = (shape[1] - data.shape[1])//2
    if yoff is None:
        yoff = (shape[2] - data.shape[2])//2
    if zoff is None:
        zoff = (shape[0] - data.shape[0])//2

    out[:,xoff:data.shape[1]+xoff,yoff:data.shape[2]+yoff,:] = data
    return out 

@numba.njit(["complex64[:](complex64[:],complex64[:])","complex128[:](complex128[:],complex128[:])"])  
def _refind2eps(refind, out):
    out[0] = refind[0]**2
    out[1] = refind[1]**2
    out[2] = refind[2]**2
    return out

@numba.guvectorize(["(complex64[:],complex64[:])","(complex128[:],complex128[:])"],"(n)->(n)")  
def refind2eps(refind, out):
    """Converts three eigen (complex) refractive indices to three eigen dielectric tensor elements"""
    assert refind.shape[0] == 3
    _refind2eps(refind, out)
    
    

@numba.njit(["complex64[:](float32,complex64[:],complex64[:])","complex128[:](float64,complex128[:],complex128[:])"])
def _uniaxial_order(order, eps, out):
    m = (eps[0] + eps[1] + eps[2])/3.
    delta = eps[2] - (eps[0] + eps[1])/2.
    if order == 0.:
        eps1 = m
        eps3 = m
    else:
        eps1 = m - 1./3. *order * delta
        eps3 = m + 2./3. * order * delta

    out[0] = eps1
    out[1] = eps1
    out[2] = eps3
    
    return out

@numba.guvectorize(["(float32[:],complex64[:],complex64[:])","(float64[:],complex128[:],complex128[:])"],"(),(n)->(n)")
def uniaxial_order(order, eps, out):
    """
    uniaxial_order(order, eps)
    
    Calculates uniaxial dielectric tensor of a material with a given orientational order parameter
    from a diagonal dielectric (eps) tensor of the same material with perfect order (order = 1)
    
    >>> uniaxial_order(0,[1,2,3.])
    array([ 2.+0.j,  2.+0.j,  2.+0.j])
    >>> uniaxial_order(1,[1,2,3.])
    array([ 1.5+0.j,  1.5+0.j,  3.0+0.j])
    """
    assert eps.shape[0] == 3
    _uniaxial_order(order[0], eps, out)
    
@numba.guvectorize(["(complex64[:],float32[:],complex64[:])","(complex128[:],float64[:],complex128[:])"],"(n),()->(n)")
def eps2ueps(eps, order, out):
    """
    eps2ueps(eps, order)
    
    Calculates uniaxial dielectric tensor of a material with a given orientational order parameter
    from a diagonal dielectric (eps) tensor of the same material with perfect order (order = 1)
    
    >>> eps2ueps([1,2,3.],0)
    array([ 2.+0.j,  2.+0.j,  2.+0.j])
    >>> eps2ueps([1,2,3.],1)
    array([ 1.5+0.j,  1.5+0.j,  3.0+0.j])
    """
    assert eps.shape[0] == 3
    _uniaxial_order(order[0], eps, out)
    
@numba.guvectorize(["(complex64[:],complex64[:])","(complex128[:],complex128[:])"],"(n)->(n)")
def eps2ieps(eps, out):
    """
    eps2ieps(eps)
    
    Calculates isotropic dielectric tensor of a material with a given orientational order parameter order=0
    from a diagonal dielectric (eps) tensor of the same material with perfect order (order = 1)

    """
    assert eps.shape[0] == 3
    _uniaxial_order(0., eps, out)
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
