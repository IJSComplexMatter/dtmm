.. _method:

The Method
==========

``dtmm`` implements a few different algorithms/implementations. The choice of the method used depends on the dimensionality of the system. For 3D, there is an iterative algorithm beam propagation implementation, and for 2D and 1D data you can use the non-iterative algorithm, a 4x4 or 2x2 transfer-matrix-based formulation.

Non-iterative 1D & 2D
---------------------

For 1D simulations, the package provides the standard 4x4 transfer matrix method and the 2x2 extended Jones method. For the 4x4 approach, please refer to the literature on the 4x4 transfer matrix approach, or Berreman calculus. The 2x2 approach was implemented as a backend for the 3D simulations and is a scattering matrix formulation done with E-field only (no H field). Therefore, it handles transmission only. For reflection calculations you must use the 4x4 approach.

For 2D simulations the package also provides the 2D transfer matrix formalism, so you can  build the NxNx4x4 (or NxNx2x2) transfer matrices for the N propagating modes. This is computationally expensive, but it is manageable in 2D. With this approach you can compute the reflections from reflection gratings, for instance. 

The same concept is available for 3D, but here it is impractical to use the transfer matrix approach, except when dimensions are small or for very high-reflecting data and you are advised to use the iterative algorithm developed for 3D data, as explained below.

Iterative 3D
------------

For 3D simulations, a vectorial split-step beam propagation method is used. The algorithm splits the material into layers and performs propagation through the layers layer-by-layer in a split-step fashion. Each layer is treated as a combination of a first thin birefringent film, a homogeneous thick medium, and a second thin birefringent film. The two thin layers are there to simulate the phase difference that the input wave acquires as it passes or reflects from the layer because of the birefringence of the material. The homogeneous thick layer simulates the phase shift that the waves acquire as they travel through the layer, and therefore captures the diffraction properties of the light propagation. The algorithm computes Fresnel reflection coefficients at the interfaces, which allows one to track the reflected waves. The phase shift and reflection/transmission coefficients are calculated using Berreman 4x4 calculus (or the extended Jones formalism - user selectable). The diffraction propagation step is done in Fourier space; the field is Fourier transformed and propagated forward or backward (depending if we treat transmitted waves or reflected waves) through the layer thickness. To calculate reflected fields, multiple passes of the input field are performed. User provides the number of passes.

The algorithm allows one to tune the diffraction/reflection calculation accuracy. In the simplified scheme, the input field is treated as a single beam with a well-defined propagation direction. This scheme is good for thin material, where the diffraction effects are small. In a more advanced (and more accurate) scheme, the propagation/reflection is done using the mode-grouping technique. After each pass through the layer, the algorithm performs mode decomposition. Then, it combines the computed modes into a user-defined number of beams with different wave vector orientations by a grouping of modes. In other words, the EM field is assumed to be a sum of `n` beams, where `n` is the user-defined parameter. In samples that induce high-frequency reflection/transmission modes, this approach improves the calculation of reflection coefficients and accurately diffracts high-frequency modes, which is important for thick cells. 

.. _accuracy:

Accuracy and efficiency
+++++++++++++++++++++++

The algorithm converges to the exact solution in samples with homogeneous layers, and is a reasonable approximation when used in samples with inhomogeneous layers if the lateral (within the layer) variations of the dielectric tensor are slow (compared to the wavelength) and if the birefringence is weak. In thin disordered systems with rapidly varying dielectric tensor, the algorithm also works reasonably well, because the light "sees" only the mean refractive index, and the diffraction effects become less important. The algorithm fails in sub-wavelength periodic structures, metamaterials ... 

The iterative approach is very efficient, especially for transmittance/reflectance calculations in weakly reflecting systems. For instance, the algorithm can be more than 100x faster than an equivalent calculation done with FDTD algorithm. Although the code is written in python, the code's computationally expensive parts are optimized by `numba`_, and it can run in parallel on multi-core CPUs.

Reflection calculation speed and accuracy depends on the required number of light passes, so it can become slower to compute in system with high reflectance. Also, for highly reflecting samples (eg. cholesteric reflection mirror with reflectance >> 50%) the algorithm may not converge, but the algorithm allows you to determine whether it has converged or not.

.. _`numba`: http://numba.pydata.org