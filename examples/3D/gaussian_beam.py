#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we create a gaussian beam and diffract it. intensity profile is fitted
with gaussian profile to determine waist parameter as a function of z coordinate
obtained data is compared with paraxial model of a gaussian beam.
"""

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import dtmm

def _r(shape):
    ay, ax = [np.arange(-l / 2. + .5, l / 2. + .5) for l in shape]
    xx, yy = np.meshgrid(ax, ay, indexing = "xy")
    r = (xx**2 + yy**2) ** 0.5    
    return r

w = 600 #wavelength
n = 1.5 #refractive index
shape = (256,256)
w0 = 24

r = _r(shape).flatten() #radial coordinate for fitting

def gauss(r,w,a):
    return a*np.exp(-2*(r/w)**2)

def beam_waist(z,w0,k):
    z0 = w0**2*k/2.
    return w0 * (1+(z/z0)**2)**0.5

window = dtmm.window.gaussian_beam(shape,w0,dtmm.k0(600,50), n = n, z = 0)
field,w,p = dtmm.illumination_data(shape, [w],
                                           pixelsize = 50, n = n, window = window, jones = (1,0)) 

ks = dtmm.k0(w, p)
epsv = dtmm.refind2eps([n]*3)

power = dtmm.field2intensity(field)
plt.subplot(121)
plt.imshow(power[0])
plt.title("z=0")

waist = []
z = np.linspace(-512,512,13)

for d in z:
    dmat = dtmm.diffract.field_diffraction_matrix(shape, ks, d = d, epsv = epsv, betamax = 1.)
    out = dtmm.diffract.diffract(field,dmat)
    power = dtmm.field2intensity(out)
    y = power.flatten()
    popt,pcov = opt.curve_fit(gauss,r,y)
    waist.append(popt[0])

plt.subplot(122)
plt.imshow(power[0])
plt.title("z={}".format(d))
 

plt.figure() 
plt.plot(z,waist,"o", label = "sim")
z= np.linspace(-512,512,100)
plt.plot(z,beam_waist(z,w0,ks[0]*n), label = "model")
plt.legend()
    

