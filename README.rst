dtmm: Diffractive Transfer Matrix Method
========================================

.. image:: https://img.shields.io/pypi/pyversions/dtmm
    :target: https://pypi.org/project/dtmm/
    :alt: Python version

.. image:: https://travis-ci.com/IJSComplexMatter/dtmm.svg?branch=master
    :target: https://travis-ci.com/IJSComplexMatter/dtmm
    :alt: Build Status
    
.. image:: https://readthedocs.org/projects/dtmm/badge/?version=latest
    :target: https://dtmm.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://zenodo.org/badge/125330690.svg
   :target: https://zenodo.org/badge/latestdoi/125330690

``dtmm`` is a simple-to-use light (electro-magnetic field) transmission and reflection calculation engine and visualizer. It can be used for calculation of transmission or reflection properties of layered homogeneous or inhomogeneous materials, such as liquid-crystal with homogeneous or inhomogeneous director profile. *DTMM* stands for Diffractive Transfer Matrix Method and is an adapted Berreman 4x4 transfer matrix method. Details of the method are given in *... some future paper*.

Requisites
----------

* numba >=0.35.0
* numpy
* scipy
* matplotlib


Optional:

* mkl_fft


Installation
------------

Install with pip::

    $ pip install dtmm
    
or clone the latest development version and run::

    $ python setup.py install

Documentation
-------------

You can find the online manual at:

http://dtmm.readthedocs.io

but of course, you can always access docstrings from the console
(i.e. ``help(dtmm.transfer_field)``).

Also, you may want to look at the examples/ directory for some examples
of use.

License
-------

``dtmm`` is released under MIT license so you can use it freely. Please cite the package if you use it. See the provided DOI badge.



