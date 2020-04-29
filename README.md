# stochastic_process_analysis

This repository contains codes used for sensitity analysis on models governed by stochastic processes and random input variables. 
They were done in the frame of an internship at PHIMECA in Clermont-Ferrand.

## Getting Started

To use the codes, please install the virtual environment provided in the yaml file. A throughout example can be found in the 
"Demo Analyse de sensibilité poutre" notebook, but for the moment only in french (Sorry).

### Prerequisites

Works with python 3.6, but should work with other versions of python through the intensive usage of numpy.

Important packages used:
 - openTURNS
 - NumPy
 - Numba (not really used, but acceleration could be intersting)
 - anastruct (for the random beam example)

### Installing

The environment installation is simple, just have conda install and paste that line in your terminal
```
conda env create -f  sensitivityEnv.yml
```

## Built With

* [openTURNS](https://github.com/openturns/openturns) - Codes and methods for efficient sensitivity analysis
* [anaStruct](https://github.com/ritchie46/anaStruct) - 2D efficient finite element analysis in Python
* [Numba](https://numba.pydata.org/)                  - Easy code acceleration 
* [LHSMDU](https://github.com/sahilm89/lhsmdu)        - SciPy/NumPy implementation of the LHSMDU paper (Latin Hypercube Sampling with multidimensional uniformity)

## Contributing

As this project is done in the frame of an internship with the company PHIMECA in Clermont-Ferrand, France, i should be the only one commiting for now.

## Authors

* **Kristof S.** - *Initial work* 

## Acknowledgments

* A lot of thanks to Ritchie Vink and the superb anastruct library : https://www.ritchievink.com/
* Also a lot of thanks to PHIMECA and the team working on openTURNS, for their really efficient sensitivity analysis library
