
.. image:: https://git.phimeca.com/phimeca/pythontools/badges/master/pipeline.svg
    :target: https://git.phimeca.com/phimeca/pythontools/commits/master

Python Tools
============

Module containing useful Python tools for Phimeca. Documentation is available at
http://phimeca.pages.phimeca.com/pythontools.

Installation and test
=====================

Developed in Python 3.8 with the following dependencies:

- openturns 1.15
- otmorris 0.8
- numpy
- argparse (in standard library)
- pytest

To build the documentation:

- sphinx
- numpydoc
- sphinx-argparse
- nbsphinx

You can use the environment.yml file to create a conda environment "pythontools" :

.. code-block:: bash

    conda env create -f environment.yml

Installation (admin, user or in dev mode): 

.. code-block:: bash

    python setup.py install
    python setup.py install --user
    python setup.py develop


Launch test (using unittest) (as simple user) :

.. code-block:: bash

    pytest test
    pytest --doctest-modules src # test la documentation


Building the documentation (as simple user) :

.. code-block:: bash

    make html
    make latexpdf