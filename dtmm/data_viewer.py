#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 10:32:48 2018

@author: andrej
"""

from __future__ import absolute_import, print_function, division

from dtmm.data import angles2director, validate_optical_data
from mpl_toolkits.mplot3d import Axes3D #needed in order to have projection 3d
import matplotlib.pyplot as plt
import numpy as np

def _r3(shape, center = False):
    """Returns r vector array of given shape."""
    nz,ny,nx = [l//2 for l in shape]
    if center == False:
        az, ay, ax = [np.arange(l) for l in shape]
    else:
        az, ay, ax = [np.arange(-l / 2. + .5, l / 2. + .5) for l in shape]
    zz, yy, xx = np.meshgrid(az, ax, ay, indexing = "ij")
    return xx, yy, zz

def _add_mask(mask,data, lim):
    if lim is not None:
        low, high = lim
        m = np.logical_and(data>= low, data<high)
        mask = np.logical_and(m, mask)
    return mask

def _mask(x,y,z, xlim = None, ylim = None, zlim = None):       
    mask = np.ones(x.shape, dtype = bool)
    mask = _add_mask(mask,x, xlim)
    mask = _add_mask(mask,y, ylim)
    mask = _add_mask(mask,z, zlim)
    return mask

#def plot_id(data, id = 0, center = False, xlim = None, 
#            ylim = None, zlim = None, fig = None, ax = None,):
#    """Plots material id of the optical data.
#    
#    Parameters
#    ----------
#    
#    deta : optical_data or material_id 
#        A valid optical data tuple, or material_id array
#    id : int or array_like of ints
#        material id or multiple material ids to plot. 
#    """
#    if isinstance(data, tuple):
#        d,material_id,eps, angles = data
#    else:
#        material_id = data
#    if material_id.ndim == 2:
#        material_id = material_id[None,...]
#    assert material_id.ndim == 3
#    index = np.asarray(id)
#    if index.ndim == 0:
#        index = index[None]
#    material_id = np.asarray(material_id,dtype = "uint32")
#    if fig is None:
#        fig = plt.figure()
#    if ax is  None:
#        ax = fig.add_subplot(111, projection='3d')
#        ax.set_aspect('equal')
#    xx, yy, zz = _r3(material_id.shape, center)
#    for i in index:
#        mask = (material_id == i)
#        mask = _add_mask(mask,xx, xlim)
#        mask = _add_mask(mask,yy, ylim)
#        mask = _add_mask(mask,zz, zlim)
#        xs = xx[mask]
#        ys = yy[mask]
#        zs = zz[mask]
#        ax.scatter(xs, ys, zs,depthshade = False, s = 1, marker = ".", label = i)
#        
#    ax.set_xlabel('X')
#    ax.set_ylabel('Y')
#    ax.set_zlabel('Z')
#    plt.legend()
#    return fig, ax


def unique_eps(material, nmax = 5):
    n = np.multiply.reduce(material.shape[0:-1])
    mat = material.reshape((n,material.shape[-1]))
    eps = mat[0]
    out = [eps]
    for i in range(nmax-1):
        mask = (mat[...,0] != eps[0]) | (mat[...,1] != eps[1]) | (mat[...,2] != eps[2])
        mat = mat[mask,:]
        try:
            eps = mat[0]
            out.append(eps)
        except IndexError:
            #no more uniqe eps values
            break
    return out
        

def plot_material(data, eps = None, center = False, xlim = None, 
            ylim = None, zlim = None, ax = None):
    """Plots material (in color) of the optical data.
    
    Parameters
    ----------
    
    data : optical_data or material (eps)
        A valid optical data tuple, eps array
    eps : array or None, optional
        Specifies which eps values to plot. If not given, all data is plotted.
    center : bool, optional
        Whether to view coordinates from the center of the box.
    xlim : (low, high) or None, optional
        A tuple describing the coordinate limits in the x direction (width).
    ylim : (low, high) or None, optional
        A tuple describing the coordinate limits in the y direction (height).
    zlim : (low, high) or None, optional
        A tuple describing the coordinate limits in the z direction (layer index).
    ax : matplotlib.axes or None, optional
        If specified, plot to ax.
    """
    if isinstance(data, tuple):
        d, material, angles = validate_optical_data(data)
    else:
        material = data
    if material.ndim == 3:
        material = material[None,...]
    material = np.asarray(material,dtype = "float32")
    
    assert material.ndim == 4
    
    if eps is None:
        eps = unique_eps(material)
    else:
        eps = np.asarray(eps)
        if eps.ndim == 1:
            eps = eps[None,:]
    
    if ax is  None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        try:
            #this does not work in recent versions of matplotlib
            ax.set_aspect('equal')
        except NotImplementedError:
            pass
    xx, yy, zz = _r3(material.shape[0:-1], center)
    
    #mmin, mmax  = material.min(), material.max()
    #if mmax != mmin:
    #    colors = (material-mmin)/(mmax-mmin)
    #else:
    #    colors = material-mmin
    
    for e in eps:
        mask = (material[...,0] == e[0]) & (material[...,1] == e[1]) & (material[...,2] == e[2])
        mask = _add_mask(mask,xx, xlim)
        mask = _add_mask(mask,yy, ylim)
        mask = _add_mask(mask,zz, zlim)
        xs = xx[mask]
        ys = yy[mask]
        zs = zz[mask]
        #cs = colors[mask,:]
        #ax.scatter(xs, ys, zs,depthshade = False, s = 1, marker = ".", c = cs)
        ax.scatter(xs, ys, zs,depthshade = False, s = 1, marker = ".", label = e)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()

    return ax.figure, ax

def plot_director(director, fig = None, center = False, xlim = None, 
                  ylim = None, zlim = None, ax = None):
    """Plots director data.
    
    Parameters
    ----------
    
    director : optical_data or material (eps)
        A valid optical data tuple, eps array
    eps : array or None, optional
        Specifies which eps values to plot. If not given, all data is plotted.
    center : bool, optional
        Whether to view coordinates from the center of the box.
    xlim : (low, high) or None, optional
        A tuple describing the coordinate limits in the x direction (width).
    ylim : (low, high) or None, optional
        A tuple describing the coordinate limits in the y direction (height).
    zlim : (low, high) or None, optional
        A tuple describing the coordinate limits in the z direction (layer index).
    ax : matplotlib.axes or None, optional
        If specified, plot to ax.
    """
    director = np.asarray(director)
    if director.ndim == 3:
        director = director[None,...]
    assert director.ndim == 4
    
    if ax is  None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            #for some reason nev versions of matplotlib wont allow you to set aspect
            pass
    # Make the grid
    xx, yy, zz = _r3(director.shape[0:3], center)
    mask = _mask(xx,yy,zz,xlim,ylim,zlim)
    
    # Direction data for the arrows
    u = director[...,0][mask]
    v = director[...,1][mask]
    w = director[...,2][mask]

    ax.quiver(xx[mask], yy[mask], zz[mask], u, v, w, length = 1., arrow_length_ratio = 0.3 ,pivot = "middle")#, length=0.1, normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')    
    return ax.figure, ax

def plot_angles(data, **kwargs):
    """Plots eps angles for optical data or angles data.
    
    Parameters
    ----------
    
    data : optical_data or material (eps)
        A valid optical data tuple, eps array
    eps : array or None, optional
        Specifies which eps values to plot. If not given, all data is plotted.
    center : bool, optional
        Whether to view coordinates from the center of the box.
    xlim : (low, high) or None, optional
        A tuple describing the coordinate limits in the x direction (width).
    ylim : (low, high) or None, optional
        A tuple describing the coordinate limits in the y direction (height).
    zlim : (low, high) or None, optional
        A tuple describing the coordinate limits in the z direction (layer index).
    ax : matplotlib.axes or None, optional
        If specified, plot to ax.
    """
    if isinstance(data, tuple):
        d,eps, angles = validate_optical_data(data)
    else:
        angles = data
    director = angles2director(angles)
    return plot_director(director, **kwargs)

__all__ = ["plot_material", "plot_angles", "plot_director"]
    
if __name__ == "__main__":
    import dtmm
    import dtmm
    data = dtmm.nematic_droplet_data((60, 128, 128), 
                    radius = 30, profile = "r", no = 1.5, ne = 1.7, nhost = 1.5)
    thickness, material_eps, angles = data

    fig = plot_material(material_eps, center = True, xlim = (-40,-20),ylim = (-10,10), zlim = (-10,10))
    fig = dtmm.plot_angles(angles, center = True, xlim = (-3,3), ylim = (-3,3), zlim = (-3,3))