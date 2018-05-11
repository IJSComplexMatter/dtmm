dtmm: Diffractive Transfer Matrix Method
========================================

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

No official release exists yet, so you have to clone or download the latest development code and run::

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

``dtmm`` will be released under MIT license so you will be able to use it freely, however, I will ask you to cite the *... some future paper*. In the mean time you can play with the current development version freely, but i kindly ask you not to redistribute the code, or use or publish data based on the results of the package. **Please wait until the package is officially released!**



