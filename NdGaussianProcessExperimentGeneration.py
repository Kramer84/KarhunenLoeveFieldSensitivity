import lhsmdu 
import openturns 
import numpy 


'''Here we are going to generate samples for the Monte-Carlo experiement,
knowing that the variables that we are generating are a mix of random-variables
representing Physical variables and random-variables used to reconstruct stochastic
field. 
This has little implication of the Latin Hypercube sampling itself, but will change the 
way we shuffle to retrieve the conditional variances. 
'''