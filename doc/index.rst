.. dtmm documentation master file, created by
   sphinx-quickstart on Thu Mar 15 12:34:43 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dtmm's documentation!
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. plot::

   import matplotlib.pyplot as plt
   import numpy as np
   x = np.random.randn(1000)
   plt.hist( x, 20)
   plt.grid()
   plt.title(r'Normal: $\mu=%.2f, \sigma=%.2f$'%(x.mean(), x.std()))
   plt.show()


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
