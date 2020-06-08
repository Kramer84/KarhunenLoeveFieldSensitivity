import openturns 
import numpy 
from joblib import Parallel, delayed, cpu_count
from itertools import chain



import openturns as ot
import numpy as np
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
    @staticmethod
    def centerSobolExp(SobolExperiment, N):
        nSamps = int(SobolExperiment.shape[0]/N)
        inputListParallel = list()
        SobolExperiment0 = SobolExperiment
        psi_fo, psi_to = NdGaussianProcessSensitivityIndicesBase.SymbolicSaltelliIndices(N)
        for i in range(nSamps):
            #Centering
            SobolExperiment[i*N:(i+1)*N,...] = SobolExperiment[i*N:(i+1)*N,...] - SobolExperiment[i*N:(i+1)*N,...].mean(axis=0)
        for p in range(nSamps-2):
            inputListParallel.append((SobolExperiment[:N,...], SobolExperiment[N:2*N,...], SobolExperiment[(2+p)*N:(3+p)*N,...], psi_fo, psi_to))
        return SobolExperiment, inputListParallel

    @staticmethod
    def getSobolIndices(SobolExperiment, N, method = 'Saltelli'):
        expShape = SobolExperiment.shape 
        nIndices = int(expShape[0]/N) - 2
        dim = expShape[1:]
        SobolExperiment0 = SobolExperiment
        if dim == (): dim = 1
        print('There are',nIndices,'indices to get in',dim,'dimensions with',SobolExperiment[0].size,'elements')
        SobolExperiment, inputListParallel = NdGaussianProcessSensitivityIndicesBase.centerSobolExp(SobolExperiment, N)
        if method is 'Saltelli':
            '''SobolIndices = Parallel(
                                                                        n_jobs = cpu_count())(
                                                                        delayed(NdGaussianProcessSensitivityIndicesBase.SaltelliIndices)(
                                                                        *inputListParallel[i]) for i in range(nIndices)
                                                                        )'''
            SobolIndices       = [NdGaussianProcessSensitivityIndicesBase.SaltelliIndices(*inputListParallel[i]) for i in range(nIndices)]
            SobolIndices, SobolIndicesTot, VarSobolIndices, VarSobolIndicesTot = map(list,zip(*SobolIndices))
            SobolIndices       = numpy.stack(SobolIndices)
            SobolIndicesTot    = numpy.stack(SobolIndicesTot)
            VarSobolIndices    = numpy.stack(VarSobolIndices)
            VarSobolIndicesTot = numpy.stack(VarSobolIndicesTot)
        return SobolIndices, SobolIndicesTot,VarSobolIndices ,VarSobolIndicesTot
        
    @staticmethod
    def SaltelliIndices(Y_Ac, Y_Bc, Y_Ec, psi_fo, psi_to):
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
        varS, varS_tot = NdGaussianProcessSensitivityIndicesBase.computeVariance(Y_Ac, Y_Bc, Y_Ec, N, psi_fo, psi_to)
        return S, S_tot, varS, varS_tot

    @staticmethod
    def SymbolicSaltelliIndices(N):
        x, y = (ot.Description.BuildDefault(N, 'X'), 
                       ot.Description.BuildDefault(N, 'Y'))
        # in order X0, Y0, X1, Y1
        xy = list(x)
        for i, yy in enumerate(y):
            xy.insert(2*i+1, yy)
        # psi  = (x1 + x2 + ...) / (y1 + y2 + ...). 
        symbolic_num, symbolic_denom   = '',''

        symbolic_num,symbolic_denom  = ([item for sublist in zip(x,['+']*N) for item in sublist], 
                                               [item for sublist in zip(y,['+']*N) for item in sublist])
        (symbolic_num.pop(), symbolic_denom.pop())
        symbolic_num   = ''.join(symbolic_num)
        symbolic_denom   = ''.join(symbolic_denom)
        print('Type=',type(symbolic_num),type(symbolic_denom))
        psi_fo, psi_to = (ot.SymbolicFunction(xy, ['('+symbolic_num + ')/(' + symbolic_denom + ')']), 
                                 ot.SymbolicFunction(xy, ['1 - ' + '('+symbolic_num + ')/(' + symbolic_denom + ')']))
        return psi_fo, psi_to

    @staticmethod
    def computeVariance(YAc, YBc, YEc, N, psi_fo, psi_to):
        """
        Compute the variance of the estimator sample

        Parameters
        ----------
        outputDim : int
            Dimension of the output (1 if scalar), only flat arrays
        N : int
            The size of the sample.
        outputDesign : numpy.array
            The array containing the output of the model for the whole simulation
        psi_fo : symbolic function
            First order saltelli indices symbolic function
        psi_to : symbolic function
            Total order saltelli indices symbolic function
        """
        baseShape = YAc.shape
        YAc = numpy.atleast_2d(YAc).T
        YBc = numpy.atleast_2d(YBc).T
        YEc = numpy.atleast_2d(YEc).T
        flatDim   = numpy.prod(baseShape[1:])
        flatShape = (N, flatDim)
        print('flatDim is ',flatDim)
        YAc = numpy.reshape(YAc, flatShape)
        YBc = numpy.reshape(YBc, flatShape)
        YEc = numpy.reshape(YEc, flatShape)

        #some intermediary calculus
        #first order:
        X_fo = numpy.multiply(YBc,YEc)
        Y_fo = numpy.square(YAc)

        X_to = numpy.multiply(YAc,YEc)
        Y_to = numpy.square(YAc)  

        print('Prepared')
        varianceFO = NdGaussianProcessSensitivityIndicesBase.computeSobolVariance(X_fo, Y_fo, psi_fo, N)
        varianceTO = NdGaussianProcessSensitivityIndicesBase.computeSobolVariance(X_to, Y_to, psi_to, N)
        shape = baseShape[1:]
        if len(baseShape)<=1:
            shape=(1,)
        varianceFO = numpy.reshape(numpy.squeeze(numpy.array(varianceFO)),shape)
        varianceTO = numpy.reshape(numpy.squeeze(numpy.array(varianceTO)),shape)
        return varianceFO, varianceTO


    @staticmethod
    def computeSobolVariance(X, Y, psi, N):
        """
        Compute the variance of the estimators

        Parameters
        ----------
        U : sample
            The sample of yA, yB, yE or combination of them, defined according the
            sobol estimators
        psi : Function
            The function that computes the sobol estimates.
        N : int
            The size of the sample.
        """
        if len(X.shape) > 1 :
            rge = X.shape[1]
        else:
            rge = 1
        U        = np.concatenate(list(zip(X, Y)))
        covar    = numpy.stack([numpy.cov(U,rowvar=True) for i in range(rge)])
        print('covariance shape:',covar.shape)
        mean_psi = numpy.squeeze(numpy.asarray(psi.gradient(U.mean(axis=1)))) # * ot.Point(1, 1) # to transform into a Point
        print('mean_psi shape ', mean_psi.shape)
        print('U.shape = ', U.shape, 'N = ',N)
        #P        = numpy.cov(U,rowvar=False)*mean_psi
        P2       = numpy.squeeze(numpy.dot(covar,mean_psi))
        print('P2 shape',P2.shape)
        variance = numpy.dot(mean_psi, P2.T) / N
        print('variance is:',variance)
        return variance



class SobolIndicesClass(object):
    def __init__(self, SobolExperiment, N ,method = 'Saltelli'):
        self.method            = method
        self.N                 = N
        self.experiment        = SobolExperiment
        self.firstOrderIndices = None

    def getFirstOrderIndices(self):
        self.firstOrderIndices = NdGaussianProcessSensitivityIndicesBase.getSobolIndices(self.experiment, self.N, self,method)











def computeSobolVariance(U, psi, size):
    """
    Compute the variance of the estimators

    Parameters
    ----------
    U : sample
        The sample of yA, yB, yE or combination of them, defined according the
        sobol estimators
    psi : Function
        The function that computes the sobol estimates.
    size : int
        The size of the sample.
    """
    mean_psi = psi.gradient(U.computeMean()) * ot.Point(1, 1) # to transform into a Point
    variance = ot.dot(mean_psi, U.computeCovariance() * mean_psi) / size
    return variance




class SaltelliSensitivityAlgorithm(ot.SaltelliSensitivityAlgorithm):

    def __init__(self, inputDesign, outputDesign, N):
        super(SaltelliSensitivityAlgorithm, self).__init__(inputDesign,
                                                           outputDesign,
                                                           N)
        self.inputDesign  = inputDesign
        self.input_dim    = inputDesign.getDimension()
        self.output_dim   = outputDesign.getDimension()
        self.size         = N
        # centrage de l'échantillon de sortie
        self.outputDesign = outputDesign # - outputDesign.computeMean()[0]

    def computeVariance(self):

        x = ot.Description.BuildDefault(self.output_dim, 'X')
        y = ot.Description.BuildDefault(self.output_dim, 'Y')
        # in order X0, Y0, X1, Y1, ...
        xy = list(x)
        for i, yy in enumerate(y):
            xy.insert(2*i+1, yy)
        # psi  = (x1 + x2 + ...) / (y1 + y2 + ...). 
        symbolic_num   = ''
        symbolic_denom = ''
        for i in range(self.output_dim):
            symbolic_num   += x[i]
            symbolic_denom += y[i]
            if i<self.output_dim-1:
                symbolic_num   += '+'
                symbolic_denom += '+'
        psi_fo = ot.SymbolicFunction(xy, ['('+symbolic_num + ')/(' + symbolic_denom + ')'])
        psi_to = ot.SymbolicFunction(xy, ['1 - ' + '('+symbolic_num + ')/(' + symbolic_denom + ')'])

        varianceFO = ot.Point(self.input_dim)
        varianceTO = ot.Point(self.input_dim)
        for p in range(self.input_dim):
            U_fo = ot.Sample(self.size, 0)
            U_to = ot.Sample(self.size, 0)
            for q in range(self.output_dim):

                yA  = ot.Sample(self.outputDesign[:, q], 0, self.size)
                yB  = ot.Sample(self.outputDesign[:, q], self.size, 2 * self.size)
                yAc = (yA - yA.computeMean()[0])
                yBc = (yB - yB.computeMean()[0])
                yE  = ot.Sample(self.outputDesign[:, q], (2 + p) * self.size, (3 + p) * self.size)
                yEc = (yE - yE.computeMean()[0])

                ## first order
                U_fo.stack(np.array(yBc) * np.array(yEc))
                U_fo.stack(np.array(yAc)**2) # centré dans tous les cas ici

                ## total order
                U_to.stack(np.array(yAc) * np.array(yEc))
                U_to.stack(np.array(yAc)**2) # centré dans tous les cas ici

            varianceFO[p] = computeSobolVariance(U_fo, psi_fo, self.size)
            varianceTO[p] = computeSobolVariance(U_to, psi_to, self.size)

        return varianceFO, varianceTO

    def getFirstOrderAsymptoticDistribution(self):
        indicesFO = self.getAggregatedFirstOrderIndices()
        varianceFO, varianceTO = self.computeVariance()
        foDist = ot.DistributionCollection(self.input_dim)
        for p in range(self.input_dim):
                foDist[p] = ot.Normal(indicesFO[p], np.sqrt(varianceFO[p]))
        return foDist

    def getTotalOrderAsymptoticDistribution(self):
        indicesTO = self.getAggregatedTotalOrderIndices()
        varianceFO, varianceTO = self.computeVariance()
        toDist = ot.DistributionCollection(self.input_dim)
        for p in range(self.input_dim):
            toDist[p] = ot.Normal(indicesTO[p], np.sqrt(varianceTO[p]))
        return toDist

    def getAggregatedFirstOrderIndices(self):

        sumVariance = 0
        VarianceI = ot.Point(self.input_dim)
        for q in range(self.output_dim):
            yA = ot.Sample(self.outputDesign[:, q], 0, self.size)
            yAc = yA - yA.computeMean()[0]
            yB = ot.Sample(self.outputDesign[:, q], self.size, 2 * self.size)
            yBc = yB - yB.computeMean()[0]
            sumVariance += yA.computeVariance()[0]

            # FOindices = ot.Point(self.input_dim)
            for p in range(self.input_dim):
                yE = ot.Sample(self.outputDesign[:, q], (2 + p) * self.size, (3 + p) * self.size )
                yEc = yE - yE.computeMean()[0]

                x = np.array(yB) * np.array(yE)
                xc = np.array(yBc) * np.array(yEc)
                mean_yz = yB.computeMean()[0] * yA.computeMean()[0]
                yz = np.array(yB) * np.array(yA)
                # FOindices[p] = (np.mean(x) - np.mean(yA)**2) / yA.computeVariance()[0]
                # FOindices[p] = (np.mean(xc) - np.mean(yAc) * np.mean(yBc)) / yA.computeVariance()[0]
                VarianceI[p] += (np.mean(xc) - np.mean(yAc) * np.mean(yBc))

        FOindices = ot.Point(VarianceI / sumVariance)
        return FOindices

