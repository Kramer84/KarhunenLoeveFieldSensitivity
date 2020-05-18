import openturns 
import numpy 
from joblib import Parallel, delayed, cpu_count

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
        pass

    @staticmethod
    def centerSobolExp(SobolExperiment, N):
        nSamps = int(SobolExperiment.shape[0]/N)
        inputListParallel = list()
        for i in range(nSamps):
            #Centering
            SobolExperiment[i*N:(i+1)*N,...] = SobolExperiment[i*N:(i+1)*N,...] - SobolExperiment[i*N:(i+1)*N,...].mean(axis=0)
        for p in range(nSamps-2):
            inputListParallel.append((SobolExperiment[:N,...], SobolExperiment[N:2*N,...], SobolExperiment[(2+p)*N:(3+p)*N,...]))
        return SobolExperiment, inputListParallel

    @staticmethod
    def getSobolIndices(SobolExperiment, N, method = 'Saltelli'):
        expShape = SobolExperiment.shape 
        nIndices = int(expShape[0]/N) - 2
        dim = expShape[1:]
        if dim == (): dim = 1
        print('There are',nIndices,'indices to get in',dim,'dimensions with',SobolExperiment[0].size,'elements')
        SobolExperiment, inputListParallel = NdGaussianProcessSensitivityIndicesBase.centerSobolExp(SobolExperiment, N)
        Y_Ac = SobolExperiment[0:N,...] 
        B_Ac = SobolExperiment[N:2*N,...]
        if method is 'Saltelli':
            SobolIndices = Parallel(
                            n_jobs = cpu_count())(
                            delayed(NdGaussianProcessSensitivityIndicesBase.SaltelliIndices)(
                            *inputListParallel[i]) for i in range(nIndices)
                            )
            SobolIndices, SobolIndicesTot = map(list,zip(*SobolIndices))
            SobolIndices = numpy.stack(SobolIndices)
            SobolIndicesTot = numpy.stack(SobolIndicesTot)
        return SobolIndices, SobolIndicesTot
        
    @staticmethod
    def SaltelliIndices(Y_Ac, Y_Bc, Y_Ec):
        assert (Y_Ac.shape == Y_Bc.shape == Y_Ec.shape ), "samples have to have same shape"
        N = Y_Ac.shape[0]
        Ni = 1./N
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
        S_tot = numpy.subtract(1., 
                    numpy.divide(Ni*numpy.sum(numpy.multiply(Y_Ac,Y_Ec),axis=0),
                                                     Ni*numpy.sum(numpy.square(Y_Ac)))
                                                     )
        return S, S_tot