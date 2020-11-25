import sys
import os
from setuptools import setup
from distutils.version import LooseVersion
import openturns as ot
import otmorris as otm

ot_version_require = '1.16rc1'
if LooseVersion(ot.__version__) != ot_version_require:
    raise Exception('Version of openturns must be : {}, found {}.'.format(ot_version_require, ot.__version__))

otm_version_require = '0.9'
if LooseVersion(otm.__version__) != otm_version_require:
    raise Exception('Version of otmorris must be : {}, found {}.'.format(otm_version_require, otm.__version__))

# load the version from the file
with open("VERSION", 'r') as fic:
    version = fic.read()


# set the parameter of the setup
setup(name='pythontools',
      version=version,
      description='Python tools module for Phimeca',
      author='Antoine Dumas',
      author_email='dumas@phimeca.com',
      # define packages which can be imported
      packages=['pythontools'],
      package_dir={"pythontools": "src"},
      data_files = [('.', ["VERSION"])],
      # List of dependancies
      install_requires= ['numpy',
                         'pytest',
                         'sphinx',
                         'numpydoc']
      )
