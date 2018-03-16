Installation
============

The code is hosted at GitHub. You have to clone the repository or download the code from `repository`_ and run::

    python setup.py install

which should install the package, provided that all requirements are met.

Requirements
------------

Prior to installing, you should have a working python 2.7 or 3.x environment consisting of:

* numba
* numpy
* scipy
* numexpr
* matplotlib

The package relies heavily on `numba`_, which is sometimes difficult to install by hand, so I suggest you install `anaconda`_, `canopy`_ or any other python distribution that comes shipped with the above packages.

For faster calculation, one should also install `mkl_fft`_ that is readily available in `anaconda`_.

Installing in Canopy
--------------------

After you have downloaded the package, open the canopy code editor and start the Package manager (tools -> Package Manager). Install the above listed packages and then start the
canopy terminal (tools -> Canopy Terminal) and `cd` to the downloaded source code and run::

    python setup.py install

Installing in Anaconda
----------------------

After you have downloaded the package, open the terminal (command prompt) `cd` to the downloaded source code and run::

    conda install numba scipy matplotlib numba numexpr
    python setup.py install

optionally, for faster computation, you can install `mkl_fft` through the `intell` channel::

    conda install -c intel mkl_fft


.. _repository: https://github.com/IJSComplexMatter/dtmm
.. _numba: http://numba.pydata.org
.. _anaconda: https://www.anaconda.com
.. _canopy: https://www.enthought.com/product/canopy/
.. _mkl_fft: https://github.com/IntelPython/mkl_fft