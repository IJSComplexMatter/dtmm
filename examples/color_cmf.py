#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plots tristimulus values of CIE 1931 XYZ color sapce """

import dtmm.color as dc
import matplotlib.pyplot as plt

wavelengths, cmf = dc.load_cmf(retx = True)

plt.plot(wavelengths,cmf[...,0], "r-",label = "X")
plt.plot(wavelengths,cmf[...,1], "g-",label = "Y")
plt.plot(wavelengths,cmf[...,2], "b-",label = "Z")

plt.xlabel("Wavelength [nm]")

plt.legend()
plt.show()
