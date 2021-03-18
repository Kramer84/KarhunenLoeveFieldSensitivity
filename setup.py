import sys
import os
from setuptools import setup


# load the version from the file
#with open("VERSION", 'r') as fic:
#    version = fic.read()

with open('README.md') as f:
    readme = f.read()

# set the parameter of the setup
setup(name='klfs',
      version=0.1,
      description='openTURNS module for sensitivity analysis on models with stochastic fields',
      author='Kristof Simady',
      author_email='ksimady@sigma-clermont.fr',
      url = "https://github.com/Kramer84/klfs/",
      license = 'MIT',
      # define packages which can be imported
      packages=['klfs'],
      package_dir={
      'klfs': 'KarhunenLoeveFieldSensitivity'},
      long_description=readme,
      # set the executable scripts
      # List of dependancies
      install_requires= ['numpy','openturns','matplotlib','joblib'],
      classifiers=["Development Status :: 2 - Pre-Alpha",
                   "Intended Audience :: Science/Research",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python",
                   "Topic :: Scientific/Engineering",
                   "License :: OSI Approved :: MIT License",
                   "Natural Language :: English"]
      )
