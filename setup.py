    
from setuptools import setup, find_packages
from dtmm import __version__

long_description = """
dtmm is an electro-magnetic field transmission and reflection calculation engine and visualizer. It can be used for calculation of transmission or reflection properties of layered homogeneous or inhomogeneous materials, such as confined liquid-crystals with homogeneous or inhomogeneous director profile. DTMM stands for Diffractive Transfer Matrix Method and is an adapted Berreman 4x4 transfer matrix method and an adapted 2x2 extended Jones method.
"""

packages = find_packages()

setup(name = 'dtmm',
      version = __version__,
      description = 'Diffractive Transfer Matrix Method',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author = 'Andrej Petelin',
      author_email = 'andrej.petelin@gmail.com',
      url="https://github.com/IJSComplexMatter/dtmm",
      packages = packages,
      #include_package_data=True
      package_data={
        # If any package contains *.dat, or *.ini include them:
        '': ['*.dat',"*.ini"]},
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
      )