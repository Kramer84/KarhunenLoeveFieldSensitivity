import openturns 
import numpy 

'''Here we are going to rewrite a vectorized method to calculate the Sobol' Indices,
using the different methods at our disposal : 
-Jansen
-Saltelli
-Mauntz-Kucherenko
-Martinez
This rewriting is necessary, as the version in openTURNS only allow samples
in the form of a matrix having a size of N(2+d) rows. (With d being the dimension
of the function analysed.) In the case of function governed by stochastic processes
the size of the samlpes will allways be smaller than N(2+d), as we work whith the KL
decomposition, and that in that case, the increase of the dimension of the function is 
due to the fact that for one field (one dimension) we express it using multiple simple RVs,
but that we only need to calculate one sensitity per field. 
We will write a basic function that takes as an input the sample A, B as well as the mixed matrix,
(And also the output vector associated to each of these samples) and solely calculates one 
sensitivity index. This will allow us to scale the method to any type of input sample size.
'''


class NdGaussianProcessSensitivityIndicesBase(object):
	'''Basic methods to calculate unitary sensitivity indices
	'''
	@staticmethod
	def SaltelliIndices(vectA, vectB, vectBAi):
