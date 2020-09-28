__author__ = 'Kristof Attila S.'
__version__ = '0.1'
__date__  = '17.09.20'

__all__ = ['AggregatedKarhunenLoeveResults', 
           'KarhunenLoeveGeneralizedFunctionWrapper',
           'KarhunenLoeveSobolIndicesExperiment']

import openturns as ot 
import math 
import uuid
from collections import Iterable
from copy import copy,deepcopy
from joblib import Parallel, delayed





##############################################################################

class AggregatedKarhunenLoeveResults(ot.KarhunenLoeveResult):
    '''Function being a buffer between the processes and the sensitivity
    Analysis
    '''
    def __init__(self, KLResList):
        self._KLRL = atLeastList(KLResList) #KLRL : Karhunen Loeve Result List
        assert len(self._KLRL)>0
        self._N = len(self._KLRL)
        self._KLLift = [ot.KarhunenLoeveLifting(self._KLRL[i]) for i in range(self._N)]
        self._homogenMesh = all_same([self._KLRL[i].getMesh() for i in range(self._N)])
        self._homogenDim = (all_same([self._KLRL[i].getCovarianceModel().getOutputDimension() for i in range(self._N)])  \
                            and all_same([self._KLRL[i].getCovarianceModel().getInputDimension() for i in range(self._N)]))
        self._aggregFlag = (self._KLRL[0].getCovarianceModel().getOutputDimension() > self._KLRL[0].getMesh().getDimension() \
                            and self._N ==1 )
        #Cause when aggregated there is usage of multvariate covariance functions
        if self._aggregFlag : print('Process seems to be aggregated. ')
        self.threshold = max([self._KLRL[i].getThreshold() for i in range(self._N)])
        super(AggregatedKarhunenLoeveResults, self).__init__()
        #Now we gonna get the data we will usually need
        self._subNames = [self._KLRL[i].getName() for i in range(self._N)]
        self._checkSubNames()
        self._modesPerProcess = [self._KLRL[i].getEigenValues().getSize() for i in range(self._N)]
        self._modeDescription = self._getModeDescription()

    def __repr__(self):
        covarianceList = self.getCovarianceModel()
        eigValList = self.getEigenValues()
        meshList = self.getMesh()
        reprStr = '| '.join(['class = AggregatedKarhunenLoeveResults',
                             'name = {}'.format(self.getName()),
                            'Aggregation Order = {}'.format(str(self._N)),
                            'Threshold = {}'.format(str(self.threshold)),
                            *['Covariance Model {} = '.format(str(i))+covarianceList[i].__repr__() for i in range(self._N)],
                            *['Eigen Value {} = '.format(str(i))+eigValList[i].__repr__() for i in range(self._N)],
                            *['Mesh {} = '.format(str(i))+meshList[i].__repr__().partition('data=')[0] for i in range(self._N)]])
        return reprStr


    def _checkSubNames(self):
        '''Here we're gonna see if all the names are unique, so there can be 
        no confusion. We could also check ID's'''
        if len(set(self._subNames)) != len(self._subNames) :
            print('The process names are not unique. Adding uuid.')
            for i, process in enumerate(self._KLRL) :
                oldName = process.getName()
                newName = oldName + '_' + str(i) + '_' + str(uuid.uuid1())
                print('Old name was {}, new one is {}'.format(oldName, newName))
                process.setName(newName)
            self._subNames = [self._KLRL[i].getName() for i in range(self._N)]

    def _getModeDescription(self):
        modeDescription = list()
        for i, nMode in enumerate(self._modesPerProcess):
            for j in range(nMode):
                modeDescription.append(self._subNames[i]+'_'+str(j))
        return modeDescription

    def _checkCoefficients(self, coefficients):
        '''Function to check if the vector passed has the right number of 
        elements'''
        nModes = sum(self._modesPerProcess)
        if (isinstance(coefficients, ot.Point), len(coefficients) == nModes):
            return True
        elif (isinstance(coefficients, (ot.Sample, ot.SampleImplementation)) and len(coefficients[0]) == nModes):
            return True
        else : 
            print('The vector passed has not the right number of elements.')
            print('n° elems: {} != {}'.format(str(len(coefficients)), str(nModes)))
            return False


    def getClassName(self):
        '''returns list of class names
        '''
        classNames=[self._KLRL[i].getClassName() for i in range(self._N)]
        return list(set(classNames))

    def getCovarianceModel(self):
        '''
        '''
        return [self._KLRL[i].getCovarianceModel() for i in range(self._N)]

    def getEigenValues(self):
        '''
        '''
        return [self._KLRL[i].getEigenValues() for i in range(self._N)]

    def getId(self):
        '''
        '''
        return [self._KLRL[i].getId() for i in range(self._N)]

    def getImplementation(self):
        '''
        '''
        return [self._KLRL[i].getImplementation() for i in range(self._N)]

    def getMesh(self):
        '''
        '''
        return [self._KLRL[i].getMesh() for i in range(self._N)]

    def getModes(self):
        '''
        '''
        return [self._KLRL[i].getModes() for i in range(self._N)]

    def getModesAsProcessSample(self):
        '''
        '''
        return [self._KLRL[i].getModesAsProcessSample() for i in range(self._N)]

    def getProjectionMatrix(self):
        '''
        '''
        return [self._KLRL[i].getProjectionMatrix() for i in range(self._N)]

    def getScaledModes(self):
        '''
        '''
        return [self._KLRL[i].getScaledModes() for i in range(self._N)]

    def getScaledModesAsProcessSample(self):
        '''
        '''
        return [self._KLRL[i].getScaledModesAsProcessSample() for i in range(self._N)]

    def getThreshold(self):
        '''
        '''
        return self.threshold

    def lift(self, coefficients):
        '''lift a point into a function
        '''
        assert isinstance(coefficients, (ot.Point)), 'function only lifts points'
        valid = self._checkCoefficients(coefficients)
        modes = copy(self._modesPerProcess)
        if valid :
            return [self._KLRL[i].lift(coefficients[i*modes[i]:(i+1)*modes[i]]) for i in range(self._N)]
        else : 
            raise Exception('DimensionError : the vector of coefficient has the wrong shape')

    def liftAsProcessSample(self, coefficients):
        '''Function to lift a sample of coefficients into a ProcessSample
        '''
        assert isinstance(coefficients, (ot.Sample, ot.SampleImplementation))
        modes = copy(self._modesPerProcess)
        modes.insert(0,0)
        processes = [self._KLLift[i](coefficients[:,(modes[i]+sum(modes[:i])):(modes[i+1] + sum(modes[:i+1]))]) for i in range(self._N)]
        return processes

    def liftAsField(self, coefficients):
        '''function lifting a vector of coefficients into a field.
        A sample of multiple vectors can also be passed and are lifted into
        a list of process samples (as the processes can have different 
        dimensions)
        '''
        assert isinstance(coefficients, (ot.Point)), 'function only lifts points'
        valid = self._checkCoefficients(coefficients)
        modes = self._modesPerProcess
        if valid :
            return [self._KLRL[i].liftAsField(coefficients[i*modes[i]:(i+1)*modes[i]]) for i in range(self._N)]
        else : 
            raise Exception('DimensionError : the vector of coefficient has the wrong shape')

    def liftAsSample(self, coefficients):
        '''
        '''
        assert isinstance(coefficients, ot.Point)
        valid = self._checkCoefficients(coefficients)
        modes = self._modesPerProcess
        if valid :
            if self._aggregFlag :
                sample = self._KLRL[0].liftAsSample(coefficients)
                sample.setDescription(self._modeDescription)
                return sample
            else :
                return [self._KLRL[i].liftAsSample(coefficients[i*modes[i]:(i+1)*modes[i]]) for i in range(self._N)]
        else : 
            raise Exception('DimensionError : the vector of coefficient has the wrong shape')

    def project(self, args):
        '''Project a function or a field on the eigenmodes basis.

        Note
        ----
        Some precaution is needed as some ambiguity can arise while using 
        this function.
        If the karhunen loeve results structure has originated from the 
        usage of the Karhunen Loeve algorithm on an aggregated process, 
        then only pass functions or fields that have the same output dimension
        as the aggregation order.
        If the karhunen loeve results structure is a list of non homogenous 
        processes, then only pass lists of functions, samples or fields of the 
        same dimension. 
        '''
        args = atLeastList(args)
        nArgs = len(args)
        nProcess = self._N
        isAggreg = self._aggregFlag
        homogenMesh = self._homogenMesh
        homogenDim = self._homogenDim
        assert isinstance(args[0], (ot.Field, ot.Sample, ot.ProcessSample,
                                    ot.Function, ot.AggregatedFunction,
                                    ot.SampleImplementation))
        if isAggreg :
            assert nProcess==1, 'do not work with lists of aggregated processes'
            assert homogenMesh, 'if aggregated then the mesh is shared'
            assert homogenDim, 'if aggregated then the dimension is shared'
            inDim = self._KLRL[0].getCovarianceModel().getInputDimension()
            outDim = self._KLRL[0].getCovarianceModel().getOutputDimension()
            if isinstance(args[0], (ot.Function, ot.Field, 
                                    ot.ProcessSample, ot.AggregatedFunction)):
                try : fdi = args[0].getInputDimension()
                except : fdi = args[0].getMesh().getDimension()
                try : fdo = args[0].getOutputDimension()
                except : fdo = args[0].getDimension()

                if fdi == inDim and fdo == outDim :
                    if nArgs > 1 and not isinstance(args[0], ot.ProcessSample):
                        sample = ot.Sample([self._KLRL[0].project(args[i]) for i in range(nArgs)])
                        sample.setDescription(self._modeDescription)
                        return sample 
                    elif isinstance(args[0], ot.Field) : 
                        projection = self._KLRL[0].project(args[0])
                        projDescription = list(zip(self._modeDescription, projection))
                        projection = ot.PointWithDescription(projDescription)
                        return projection
                    elif isinstance(args[0], ot.ProcessSample):
                        projection = self._KLRL[0].project(args[0])
                        projection.setDescription(self._modeDescription)
                        return projection
                else :
                    raise Exception('InvalidDimensionException')

        else : 
            if isinstance(args[0], (ot.Field, ot.Function, ot.Sample)):
                assert nArgs==nProcess, 'Pass a list of same length then aggregation order'
                try:
                    projection = [list(self._KLRL[i].project(args[i])) for i in range(nProcess)]
                    projectionFlat = [item for sublist in projection for item in sublist]
                    output = ot.PointWithDescription(list(zip(self._modeDescription, projectionFlat)))
                    return output
                except Exception as e :
                    raise e

            elif isinstance(args[0], ot.ProcessSample):
                assert nArgs==nProcess, 'Pass a list of same length then aggregation order'
                try:
                    projectionSample = ot.Sample(0,sum(self._modesPerProcess))
                    sampleSize = args[0].getSize()
                    projList = [self._KLRL[idx].project(args[idx]) for idx in range(nProcess)]
                    for idx in range(sampleSize):
                        l = [list(projList[j][idx]) for j in range(nProcess)]
                        projectionSample.add(
                            [item for sublist in l for item in sublist])
                    projectionSample.setDescription(self._modeDescription)
                    return projectionSample
                except Exception as e:
                    raise e


    def getAggregationOrder(self):
        return self._N

    def getSizeModes(self):
        return sum(self._modesPerProcess)





##############################################################################

def all_same(items):
    #Checks if all items of a list are the same
    return all(x == items[0] for x in items)





##############################################################################

def atLeastList(elem):
    if isinstance(elem, Iterable) :
        return list(elem)
    else : 
        return [elem]





##############################################################################

class KarhunenLoeveGeneralizedFunctionWrapper(ot.OpenTURNSPythonPointToFieldFunction):
    '''Class allowing to rewrite any function taking as an input a list
    of fields, samples or ProcessSamples as only dependent of a vector.
    The function passed should return a list of fields.

    Note
    ----
    The usage of this wrapper implies having already done the karhunen loeve
    decomposition of the list of the Processes with wich we will feed the 
    function. The input dimension of this function is dependent of the 
    order of the Karhunen Loeve decomposition. 
    '''
    def __init__(self, AggregatedKarhunenLoeveResults=None, func=None, 
        func_sample=None, outputDim=None):
        self.func = func
        self.func_sample = func_sample
        self.AKLR = AggregatedKarhunenLoeveResults
        inputDim = self.AKLR.getSizeModes()
        super(KarhunenLoeveGeneralizedFunctionWrapper,self).__init__(inputDim,
                                                                  ot.Mesh(),
                                                                  outputDim)
        self.setInputDescription(self.AKLR._modeDescription)

    def _exec(self, X):
        assert len(X)==self.getInputDimension()
        inputFields = self.AKLR.liftAsField(X)
        #evaluating ...
        result = self.func(inputFields)
        return result

    def _exec_sample(self, X):
        assert len(X[0])==self.getInputDimension()
        inputProcessSamples = self.AKLR.liftAsProcessSample(X)
        try :
            result = self.func_sample(inputProcessSamples)
            return X
        except Exception as e:
            raise e





##############################################################################

class KarhunenLoeveSobolIndicesExperiment(object):
    def __init__(self, AggregatedKarhunenLoeveResults=None, size=None, second_order=False):
        self.AKLR = AggregatedKarhunenLoeveResults
        self.size = None

        self.composedDistribution = None
        self.inputVarNames = list()
        self.inputVarNamesKL = list()
        self._dataMixSamples = list()

        if size is not None:
            self.setSize(size)
        if AggregatedKarhunenLoeveResults is not None:
            self._setKaruhnenLoeveResults(AggregatedKarhunenLoeveResults)

        # here we come to the samples (our outputs)
        self._sample_A = None
        self._sample_B = None
        self._experimentSample = None

    def generate(self, **kwargs):
        '''generate final sample with A and b mixed
        '''
        assert (self.AKLR is not None) and \
               (self.size is not None), \
                    "Please intialise sample size and PythonFunction wrapper"
        self._generateSample(**kwargs)

        self._mixSamples()
        return self._experimentSample

    def setSize(self, N):
        '''set size of the samples 
        '''
        assert isinstance(N,int) and N>0, \
                "Sample size can only be positive integer"
        if self.size is None:
            self.size = N
        else:
            self.size = N
            self._sample_A = self._sample_B = self._experimentSample = None

    def _setKaruhnenLoeveResults(self, AKLR):
        '''set the wrapped function from the NdGaussianProcessSensitivity;
        '''
        self.AKLR = AKLR
        self.inputVarNames = self.AKLR._subNames
        self.inputVarNamesKL = self.AKLR._modeDescription
        self.composedDistribution = ot.ComposedDistribution([ot.Normal()]*self.AKLR.getSizeModes())
        self._dataMixSamples = self.AKLR._modesPerProcess


    def _mixSamples(self):
        '''Here we mix the samples together 
        '''
        n_vars = self.AKLR._N
        N = self.size
        self._experimentSample = deepcopy(self._sample_A)
        print(self._sample_A)
        [[self._experimentSample.add(self._sample_A[j]) for j in range(len(self._sample_A))] for _ in range(2+n_vars)]
        print(self._experimentSample)
        print(self._sample_B)
        self._experimentSample[:,N:2 * N] = self._sample_B
        jump = 2 * N
        jumpDim = 0
        for i in range(n_vars):
            self._experimentSample[jump + N*i:jump + N*(i+1), 
              jumpDim:jumpDim+self._dataMixSamples[i]] = \
                self._sample_B[:,jumpDim:jumpDim + self._dataMixSamples[i]]
            jumpDim += self._dataMixSamples[i]

    def _ramp(self, X):
        '''simple _ramp function
        '''
        if X >= 0 : return X
        else : return 0

    def _generateSample(self, **kwargs):
        '''Generation of two samples A and B using diverse methods
        '''
        distribution = self.composedDistribution
        if 'method' in kwargs :
            method = method
        else : 
            method = 'MonteCarlo'
        N2 = 2 * self.size
        if method == 'MonteCarlo':
            sample = distribution.getSample(N2)
        elif method == 'LHS':
            lhsExp = openturns.LHSExperiment(distribution,
                                             N2,
                                             False,  # alwaysShuffle
                                             True)  # randomShift
            sample = lhsExp.generate()
        elif method == 'QMC':
            restart = True
            if 'sequence' in kwargs:
                if kwargs['sequence'] == 'Faure':
                    seq = openturns.FaureSequenc
                if kwargs['sequence'] == 'Halton':
                    seq = openturns.HaltonSequence
                if kwargs['sequence'] == 'ReverseHalton':
                    seq = openturns.ReverseHaltonSequence
                if kwargs['sequence'] == 'Haselgrove':
                    seq = openturns.HaselgroveSequence
                if kwargs['sequence'] == 'Sobol':
                    seq = openturns.SobolSequence
            else:
                print('sequence undefined for low discrepancy experiment, setting default to SobolSequence')
                print("possible vals for 'sequence' argument:\n\
                    ['Faure','Halton','ReverseHalton','Haselgrove','Sobol']")
                seq = openturns.SobolSequence
            LDExperiment = openturns.LowDiscrepancyExperiment(seq(),
                                                              distribution,
                                                              N2,
                                                              True)
            LDExperiment.setRandomize(False)
            sample = LDExperiment.generate()
        sample = ot.Sample(sample)
        self._sample_A = sample[:self.size, :]
        self._sample_B = sample[self.size:, :]


'''
law = ot.Normal()
basis = ot.Basis([ot.SymbolicFunction(['x'],['1'])])
myMesh = ot.RegularGrid(0.0, 0.1, 10)
lawAsprocess = ot.FunctionalBasisProcess(law, basis, myMesh)
'''