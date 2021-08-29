# KLFS : KarhunenLoeveFieldSensitivity

This repository contains codes used for sensitity analysis of models where input uncertainties are modeled by stochastic processes and scalar distributions. 
These codes are meant to be used in parallel with the [openTURNS](http://openturns.github.io) library.
They were done in the frame of an internship at [PHIMECA](http://www.phimeca.com/) in Clermont-Ferrand.

## Aim of the project

[Sensitivity analysis](https://en.wikipedia.org/wiki/Sensitivity_analysis) is the study of how the uncertainty in the output of a mathematical model or system (numerical or otherwise) can be divided and allocated to different sources of uncertainty in its inputs. A related practice is uncertainty analysis, which has a greater focus on uncertainty quantification and propagation of uncertainty; ideally, uncertainty and sensitivity analysis should be run in tandem. 

Sensitivity analysis itself is, although being almost 30 years old, a rather new discipline with it's limitations. In fact, sensitivity analysis is usually carried out on scalar uncertainties and not so often on more complex random structures, as random fields. The aim of this project is to develop a set of usable codes, that will allow to carry out sensitivity analysis on models where the input uncertainty is not only scalar, but can also occur under the form of random fields. 

The methodology was based on different research papers, with the principal idea gotten from a 2017 paper called **"Time-variant global reliability sensitivity analysis of structures with both input random variables and stochastic processes"** from **P. Wei, Y. Wang & C. Tang**. [Paper can be found here.](https://link.springer.com/article/10.1007/s00158-016-1598-8)

## Getting Started

To use the codes, the only pre-requisite is openTURNS, and python 3.x.

<!---

#### Creating stochastic processes:

```python
import spsa
import matplotlib.pyplot as plt

stochasticProcess = spsa.StochasticProcessConstructor() 
stochasticProcess.setDimension(2)
stochasticProcess.setGrid([[0,100,100],[0,100,100],])
stochasticProcess.setCovarianceModel({'Model':'MaternModel','scale':[25,25],'amplitude':[5],'nu':3.})
realization = stochasticProcess.getRealization(True) #True to get it as a reshaped numpy array and not a openturns object
plt.imshow(realization)
```

#### Sensitivity analysis with stochastic fields:
This is in the case where you have a function that takes as an input fields and random variables, but also random variables alone or only fields.
The general case would be to have a function **F(X,Y) = U,V**, where **X** and **Y** would be a collection of fields and scalars, and the the outputs **U** and **V** would also be fields and scalars. 

```python
import spsa
import openturns as ot

RV0 = ot.Normal()        #Centered and reduced normal law
SP0 = stochasticProcess  #The stochastic process from above
outputs = {1:{'name':'U','position':0,'shape':(1,)},2:{'name':'V','position':1,'shape':(10,10)}} #We have to know the name, the position in the output tuple, as well as the dimension
size = 1000 #size of the sobol experiment
singleFunc = Fsingle #Function doing single evaluations
sampleFunc = Fsample #Function doing multiple evaluations

sensitivityAnalysis = spsa.StochasticProcessSensitivityAnalysis([RV0, SP0], outputs, sampleFunc, singleFunc, size)
sensitivityAnalysis.run(generationType = 1)
sensitivityAnalysis.getSensitivityAnalysisResults()
results = sensitivityAnalysis.SensitivityAnalysisResults
resutls[0].draw()
#This is a dummy example
```
-->


## Written with :

* [openTURNS](https://github.com/openturns/openturns) - An Open source initiative for the Treatment of Uncertainties, Risks'N Statistics.
* [anaStruct](https://github.com/ritchie46/anaStruct) - Analyse 2D Frames and trusses for slender structures. Determine the bending moments, shear forces, axial forces and displacements.

## Authors

* **Kristof S.** - *Initial work* 

## Acknowledgments

* A lot of thanks to Ritchie Vink and the superb anastruct library : https://www.ritchievink.com/
* Also a lot of thanks to PHIMECA and the team working on openTURNS, for their really efficient sensitivity analysis library
