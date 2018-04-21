#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plots D65 tabulated data"""

import dtmm.color as dc
import matplotlib.pyplot as plt

wavelengths, specter = dc.load_specter(retx = True)

plt.plot(wavelengths, specter, label = "D65")

plt.xlabel("Wavelength [nm]")

plt.legend()
plt.show()
