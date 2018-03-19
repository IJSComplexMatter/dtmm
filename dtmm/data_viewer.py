#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 10:32:48 2018

@author: andrej
"""

from __future__ import absolute_import, print_function, division

import dtmm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def _r3(shape, center = False):
    """Returns r vector array of given shape."""
    nz,nx,ny = [l//2 for l in shape]
    if center == False:
        az, ax, ay = [np.arange(l) for l in shape]
    else:
        az, ax, ay = [np.arange(-l / 2. + .5, l / 2. + .5) for l in shape]
    zz, xx, yy = np.meshgrid(az, ax, ay, indexing = "ij")
    return xx, yy, zz

def plot_id(material_id, id = 0, fig = None, ax = None):
    index = np.asarray(id)
    if index.ndim == 0:
        index = index[None]
    material_id = np.asarray(material_id,dtype = "uint32")
    if fig is None:
        fig = plt.figure()
    if ax is  None:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')
    xx, yy, zz = _r3(material_id.shape)
    for i in index:
        mask = (material_id == i)
        xs = xx[mask]
        ys = yy[mask]
        zs = zz[mask]
        ax.scatter(xs, ys, zs,depthshade = False, s = 0.1, marker = ".")
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig

def plot_director(director, fig = None, ax = None):
    director = np.asarray(director)
    
    if fig is None:
        fig = plt.figure()
    if ax is  None:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')
    # Make the grid
    xx, yy, zz = _r3(director.shape[0:3])
    
    # Direction data for the arrows
    u = director[...,0]
    v = director[...,1]
    w = director[...,2]

    ax.quiver(xx, yy, zz, u, v, w,normalize = True, arrow_length_ratio = 0.3,length = 0.9)#, length=0.1, normalize=True)
    
    return fig

def plot_angles(angles, fig = None, ax = None):
    director = dtmm.angles2director(angles)
    return plot_director(director, fig = None, ax = None)
    

director = dtmm.nematic_droplet_director((60, 128, 128), 
       radius = 30, profile = "r")
    
optical_data = dtmm.nematic_droplet_data((60, 128, 128), 
       radius = 30, profile = "r", no = 1.5, ne = 1.6, nhost = 1.5)


t,id,e,a = optical_data

fig = plot_angles(a[26:-26,58:-58,58:-58])
fig.show()
#plot_material_id(id, 1)


