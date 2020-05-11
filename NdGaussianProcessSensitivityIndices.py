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
    We first set the samples Y_A and Y_B and calculate the means and
    variances of those, so they don't have to be calculated again. 
    The notations are those of A. DUMAS in the paper :
    "Lois asymptotiques des estimateurs des indices de Sobol"


    This class can accept vectors (unidimensional outputs) as well
    as matrices (multidimensional outputs)
    '''
    def __init__(self):

    def setSamples(Y_A, Y_B):
        self.Y_A = numpy.squeeze(numpy.array(Y_A))
        self.Y_B = numpy.squeeze(numpy.array(Y_B))
        assert self.Y_A.shape == self.Y_B.shape, "samples have to have same shape"
        self.sampleShape = self.Y_A.shape
        self.N_samples   = self.sampleShape[0]
        #Centering of samples (hopefully rightly done)
        self.Y_Ac = self.Y_A-self.Y_A.mean(axis=0)
        self.Y_Bc = self.Y_B-self.Y_B.mean(axis=0)

    @staticmethod
    def checkSampleIntegretyAndCenter(Y_A, Y_B, Y_E):
        Y_A = numpy.squeeze(Y_A)
        Y_B = numpy.squeeze(Y_B)
        Y_E = numpy.squeeze(Y_E)
        Y_Ac = Y_A  - Y_A.mean(axis=0)
        Y_Bc = Y_B  - Y_B.mean(axis=0)
        Y_Ec = Y_E  - Y_E.mean(axis=0)
        assert Y_A.shape == Y_B.shape == Y_E.shape ,"samples have to have same shape"
        return Y_Ac, Y_Bc, Y_Ec

    @staticmethod
    def SaltelliIndices(Y_Ac, Y_Bc, Y_Ec):
        assert (Y_Ac.shape == Y_Bc.shape == Y_Ec.shape ), "samples have to have same shape"
        N = Y_Ac.shape[0]
        Ni = 1/N
        #Original version
        '''S = numpy.divide(numpy.substract(Ni*numpy.sum(numpy.multiply(Y_Bc,Y_Ec),axis=0),
                                                                 numpy.multiply(Ni*numpy.sum(Y_Bc,axis=0),
                                                                                Ni*numpy.sum(Y_Ac,axis=0))),
                                                 numpy.substract(Ni*numpy.sum(numpy.square(Y_Ac),axis=0),
                                                                 numpy.square(Ni*numpy.sum(Y_Ac,axis=0)))
                                                 )'''
        #Simplified indice as samples centered
        S = numpy.divide(Ni*numpy.sum(numpy.multiply(Y_Bc,Y_Ec),axis=0),
                                                     Ni*numpy.sum(numpy.square(Y_Ac),axis=0)
                                                     )

        return S