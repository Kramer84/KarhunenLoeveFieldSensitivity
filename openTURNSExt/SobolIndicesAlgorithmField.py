__author__ = 'Kristof Attila S.'
__version__ = '0.1'
__date__  = '17.09.20'

__all__ = ['AggregatedKarhunenLoeveResults']

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
    def __init__(self, AggregatedKarhunenLoeveResults, func=None, 
        func_sample=None, outputMesh=None, outputDim=None):
        self.func = func
        self.func_sample = func_sample
        self.AKLR = AggregatedKarhunenLoeveResults
        inputDim = self.AKLR.getSizeModes()
        outputMesh = outputMesh
        outputDim = outputDim
        super(KarhunenLoeveGeneralizedFunctionWrapper,self).__init__(inputDim,
                                                                  outputMesh,
                                                                  outputDim)

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
            result = self.func_sample(X)
            return X
        except Exception as e:
            raise e


##############################################################################

class KarhunenLoeveSobolIndicesExperiment(object):
    def __init__(self, AggregatedKarhunenLoeveResults=None, size=None, second_order=False):
        self.AKLR = AggregatedKarhunenLoeveResults
        self.size = None
        self._genType = generationType

        self.composedDistribution = None
        self.inputVarNames = list()
        self.inputVarNamesKL = list()

        print('Generation types are:\n1 : Random (default)\n2 : LHS\n\
            3 : LowDiscrepancySequence\n4 : SimulatedAnnealingLHS')
        print('You choose', self.genTypes[self._genType], 'generation')
        if size is not None:
            self.setSize(size)
        if AKLR is not None:
            self._setKaruhnenLoeveResults(AKLR)
        if generationType is not None:
            self.setGenType(generationType)

        # here we come to the samples (our outputs)
        self.sample_A = None
        self.sample_B = None
        self.dataMixSamples = list()
        self.experimentSample = None

    def generate(self, **kwargs):
        '''generate final sample with A and b mixed
        '''
        assert (self.AKLR is not None) and \
               (self.size is not None), \
                    "Please intialise sample size and PythonFunction wrapper"
        self.generateSample(**kwargs)
        self.getDataFieldAndRV()
        self.getExperiment()
        return self.experimentSample

    def setSize(self, N):
        '''set size of the samples 
        '''
        assert (type(N) is int) and (N > 0), \
                                    "Sample size can only be positive integer"
        if self.size is None:
            self.size = N
        else:
            self.size = N
            self.sample_A = self.sample_B = self.experimentSample = None

    def _setKaruhnenLoeveResults(self, AKLR):
        '''set the wrapped function from the NdGaussianProcessSensitivity;
        '''
        self.AKLR = AKLR
        self.inputVarNames = self.AKLR._subNames
        self.inputVarNamesKL = self.AKLR._modeDescription
        self.composedDistribution = ot.ComposedDistribution([ot.Normal()]*self.AKLR.getSizeModes())
        self.getDataFieldAndRV()

    def setGenType(self, arg):
        '''set type of experiment generation
        '''
        arg = int(arg)
        if arg not in [1, 2, 3, 4]:
            print('Generation types are :\n1 : Random (default)\n2 : LHS\n\
                3 : LowDiscrepancySequence\n4 : SimulatedAnnealingLHS')
            print('Please pick one.')
            raise TypeError
        self._genType = arg

    def getDataFieldAndRV(self):
        '''Here we analyse the names of the variables, to know which columns
        belong to RVs or Fields
        '''
        n_vars = len(self.inputVarNames)
        n_vars_KL = len(self.inputVarNamesKL)
        self.dataMixSamples = list()
        for i in range(n_vars):
            k = 0
            timesInList = 0
            jump = self.ramp(sum(self.dataMixSamples) - i)
            while self.inputVarNamesKL[i + k + jump].startswith(
                                                       self.inputVarNames[i]):
                timesInList += 1
                k += 1
                if i + k + jump == n_vars_KL:
                    break
            self.dataMixSamples.append(timesInList)

    def getExperiment(self):
        '''Here we mix the samples together 
        '''
        n_vars = len(self.inputVarNames)
        N = self.size
        self.experimentSample = numpy.tile(self.sample_A, [2 + n_vars, 1])
        self.experimentSample[N:2 * N, ...] = self.sample_B
        jump = 2 * N
        jumpDim = 0
        for i in range(n_vars):
            self.experimentSample[jump + N*i:jump + N*(i+1), jumpDim:jumpDim + self.dataMixSamples[i]] = \
                self.sample_B[..., jumpDim:jumpDim + self.dataMixSamples[i]]
            jumpDim += self.dataMixSamples[i]

    def ramp(self, X):
        '''simple ramp function
        '''
        if X >= 0: return X
        else: return 0

    def generateSample(self, **kwargs):
        '''Generation of two samples A and B using diverse methods
        '''
        distribution = self.composedDistribution
        method = self._genType
        N2 = 2 * self.size
        if method == 1:
            sample = distribution.getSample(N2)
        elif (method == 2) or (method == 4):
            lhsExp = openturns.LHSExperiment(distribution,
                                             N2,
                                             False,  # alwaysShuffle
                                             True)  # randomShift
            if method == 2:
                sample = lhsExp.generate()
            if method == 4:
                lhsExp.setAlwaysShuffle(True)
                if 'SpaceFilling' in kwargs:
                    if kwargs['SpaceFilling'] == 'SpaceFillingC2':
                        spaceFill = openturns.SpaceFillingC2
                    if kwargs['SpaceFilling'] == 'SpaceFillingMinDist':
                        spaceFill = openturns.SpaceFillingMinDist
                    if kwargs['SpaceFilling'] == 'SpaceFillingPhiP':
                        spaceFill = openturns.SpaceFillingPhiP
                        if 'p' in kwargs:
                            if isinstance(kwargs['p'],int) or isinstance(kwargs['p'], float):
                                p = int(kwargs['p'])
                            else:
                                print(
                                    'Wrong type for p parameter in SpaceFillingPhiP algorithm, setting to default p = 50')
                                p = 50
                        else:
                            print(
                                'undefined parameter p in SpaceFillingPhiP algorithm, setting to default p = 50')
                            p = 50
                else:
                    print("undefined parameter 'SpaceFilling', setting to default 'SpaceFillingC2'")
                    spaceFill = openturns.SpaceFillingC2
                if 'TemperatureProfile' in kwargs:
                    if kwargs['TemperatureProfile'] == 'GeometricProfile':
                        geomProfile = openturns.GeometricProfile(
                                                             10.0, 0.95, 2000)  # Default value
                    if kwargs['TemperatureProfile'] == 'LinearProfile':
                        geomProfile = openturns.LinearProfile(10.0, 100)
                else:
                    print("undefined parameter 'TemperatureProfile', setting default GeometricProfile")
                    geomProfile = openturns.GeometricProfile(10.0, 0.95, 2000)
                optimalLHSAlgorithm = openturns.SimulatedAnnealingLHS(
                    lhsExp, geomProfile, spaceFill())
                sample = optimalLHSAlgorithm.generate()
        elif method == 3:
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
            sample = LDExperiment.generate()
        sample = numpy.array(sample)
        self.sample_A = sample[:self.size, :]
        self.sample_B = sample[self.size:, :]










'''
law = ot.Normal()
basis = ot.Basis([ot.SymbolicFunction(['x'],['1'])])
myMesh = ot.RegularGrid(0.0, 0.1, 10)
lawAsprocess = ot.FunctionalBasisProcess(law, basis, myMesh)
'''