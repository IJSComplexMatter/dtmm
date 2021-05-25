#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plots D65 tabulated data"""

import dtmm.color as dc
import matplotlib.pyplot as plt

wavelengths, tcmf = dc.load_tcmf(retx = True)

plt.plot(wavelengths,tcmf[...,0], "r-",label = "X")
plt.plot(wavelengths,tcmf[...,1], "g-",label = "Y")
plt.plot(wavelengths,tcmf[...,2], "b-",label = "Z")

plt.xlabel("Wavelength [nm]")

plt.legend()
plt.show()
