### Organisation of the codes inside of the Karhunen Loeve Field Sensitivity Analysis module


##### _aggregatedKarhunenLoeveResults.py
	Class that makes the link between the non homogenous space of the collections of stochastic fields and random variables, and the unit vector space.
	This method uses the Karhunen-Loève decomposition to make that link

##### _karhunenLoeveGeneralizedFunctionWrapper.py
	Class to wrap the model we want to make our sensitivity analysis on, so we have a new model with a homogenous set of input variables.

##### _karhunenLoeveSobolIndicesExperiment.py
	Class to generatet the design of experiment for the sensitivity analysis of the wrapped model.


##### _sobolIndicesFactory.py
	Class to calculate the Sobol' indices of the design of experiment generated above

