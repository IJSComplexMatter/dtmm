#import distribute_setup
#distribute_setup.use_setuptools()

# Setuptools Must be installed from Canopy and not by ez_setup...

#import ez_setup
#ez_setup.use_setuptools() 

from setuptools import setup, find_packages

packages = find_packages()

setup(name = 'dtmm',
      version = "0.0.1.dev0",
      description = 'Diffractive Transfer Matrix Method',
      author = 'Andrej Petelin',
      author_email = 'andrej.petelin@gmail.com',
      packages = packages,
      #include_package_data=True
    package_data={
        # If any package contains *.dat, include them:
        '': ['*.dat']}
      )