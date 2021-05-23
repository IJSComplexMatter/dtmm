Installation
============

The code is hosted on GitHub. You can install latest stable release with::

    $ pip install dtmm

Or you can clone or download the latest development code from the `repository`_ and run::

    $ python setup.py install

which should install the package, provided that all requirements are met.

Requirements
------------

Prior to installing, you should have a working python 3.x environment consisting of:

* numba >=0.45.0
* numpy
* matplotlib

To install these, it is best to go with one of the python distributions, e.g. `anaconda`_, or any other python distribution that is shipped with the above packages. 

For faster FFT calculation, you should also install `mkl_fft`_ that is readily available in `anaconda`_. If mkl_fft is not available in your system, you can also try installing `pyfftw`_. See the :ref:`optimization` for details.

Installing in Anaconda
----------------------

Open the terminal (command prompt) and run::

    $ conda install numba scipy matplotlib numba
    $ pip install ddmm

Optionally, for faster FFT computation, you can install `mkl_fft`::

    $ conda install mkl_fft

.. _repository: https://github.com/IJSComplexMatter/dtmm
.. _numba: http://numba.pydata.org
.. _anaconda: https://www.anaconda.com
.. _mkl_fft: https://github.com/IntelPython/mkl_fft
.. _pyfftw: https://github.com/pyFFTW/pyFFTW
