__author__ = 'Kristof Attila S.'
__version__ = '0.1'
__date__  = '17.09.20'

__all__ = ['AggregatedKarhunenLoeveResults']

import openturns as ot 
import uuid
from collections import Iterable
from copy import copy, deepcopy

def all_same(items):
    #Checks if all items of a list are the same
    return all(x == items[0] for x in items)

def atLeastList(elem):
    if isinstance(elem, Iterable) :
        return list(elem)
    else : 
        return [elem]

def addConstantToProcessSample(processSample, Constant):
    for k in range(processSample.getSize()):
        processSample[k] += Constant



class AggregatedKarhunenLoeveResults(object):  ### ComposedKLResultsAndDistributions ########## 
    '''Function being a buffer between the processes and the sensitivity
    Analysis
    '''
    def __init__(self, composedKLResultsAndDistributions):
        self._cKLResDist = atLeastList(composedKLResultsAndDistributions) #KLRL : Karhunen Loeve Result List
        assert len(self._cKLResDist)>0
        self._N = len(self._cKLResDist)
        self.__name__ = 'Unnamed'
        self._KLDistSampleLifting = []

        #Flags
        self.__isProcess__ = [False]*self._N
        self.__has_distributions__ = False
        self.__unified_dimension__ = False
        self.__unified_mesh__ = False
        self.__isAggregated__ = False
        self.__means__ = [0]*self._N
        self.__liftWithMean__ = False

        for i in range(self._N):
            if isinstance(self._cKLResDist[i], ot.KarhunenLoeveResult):
                self._KLDistSampleLifting.append(ot.KarhunenLoeveLifting(self._cKLResDist[i]))
                self.__isProcess__[i] = True
            elif isinstance(self._cKLResDist[i], (ot.Distribution, ot.DistributionImplementation)):
                self.__has_distributions__ = True
                if self._cKLResDist[i].getMean()[0] != 0 :
                    print('The mean value of distribution at index {} of type {} is not 0.'.format(str(i), self._cKLResDist[i].getClassName()))
                    self.__means__[i] = self._cKLResDist[i].getMean()[0]
                    self._cKLResDist[i] -= self.__means__[i] 
                    print('Distribution recentered and mean added to list of means')
                    print('Set the "liftWithMean" flag to true if you want to include the mean.')
                self._KLDistSampleLifting.append(self._cKLResDist[i].getInverseIsoProbabilisticTransformation())

        if not self.__has_distributions__ :  # If the function has distributions it cant ne homogen 
            self.__unified_mesh__ = all_same([self._cKLResDist[i].getMesh() for i in range(self._N)])
            self.__unified_dimension__ = (   all_same([self._cKLResDist[i].getCovarianceModel().getOutputDimension() for i in range(self._N)])\
                                         and all_same([self._cKLResDist[i].getCovarianceModel().getInputDimension() for i in range(self._N)]))

        if self._N == 1 : 
            if hasattr(self._cKLResDist[0], 'getCovarianceModel') and hasattr(self._cKLResDist[0], 'getMesh'):
                #Cause when aggregated there is usage of multvariate covariance functions
                self.__isAggregated__ = self._cKLResDist[0].getCovarianceModel().getOutputDimension() > self._cKLResDist[0].getMesh().getDimension()
                print('Process seems to be aggregated. ')
            else : 
                print('There is no point in passing only one process that is not aggregated')
                raise TypeError

        self.threshold = max([self._cKLResDist[i].getThreshold() if hasattr(self._cKLResDist[i], 'getThreshold') else 1e-3 for i in range(self._N)])
        #Now we gonna get the data we will usually need
        self.__process_distribution_description__ = [self._cKLResDist[i].getName() for i in range(self._N)]
        self._checkSubNames()
        self.__mode_count__ = [self._cKLResDist[i].getEigenValues().getSize() if hasattr(self._cKLResDist[i], 'getEigenValues') else 1 for i in range(self._N)]
        self.__mode_description__ = self._getModeDescription()

    def __repr__(self):
        covarianceList = self.getCovarianceModel()
        eigValList = self.getEigenValues()
        meshList = self.getMesh()
        reprStr = '| '.join(['class = ComposedKarhunenLoeveResultsAndDistributions',
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
        if len(set(self.__process_distribution_description__)) != len(self.__process_distribution_description__) :
            print('The process names are not unique.')
            print('Using generic name. ')
            for i, process in enumerate(self._cKLResDist):
                oldName = process.getName()
                newName = 'X_'+str(i)
                print('Old name was {}, new one is {}'.format(oldName, newName))
                process.setName(newName)
            self.__process_distribution_description__ = [self._cKLResDist[i].getName() for i in range(self._N)]

    def _getModeDescription(self):
        modeDescription = list()
        for i, nMode in enumerate(self.__mode_count__):
            for j in range(nMode):
                modeDescription.append(self.__process_distribution_description__[i]+'_'+str(j))
        return modeDescription

    def _checkCoefficients(self, coefficients):
        '''Function to check if the vector passed has the right number of 
        elements'''
        nModes = sum(self.__mode_count__)
        if (isinstance(coefficients, ot.Point), len(coefficients) == nModes):
            return True
        elif (isinstance(coefficients, (ot.Sample, ot.SampleImplementation)) and len(coefficients[0]) == nModes):
            return True
        else : 
            print('The vector passed has not the right number of elements.')
            print('n° elems: {} != {}'.format(str(len(coefficients)), str(nModes)))
            return False

    # new method 
    def getMean(self, i = None):
        if i is not None:
            return self.__means__[i]
        else :
            return self.__means__

    # new method 
    def setMean(self, i, val ):
        self.__means__[i] = val

    # new method
    def setLiftWithMean(self, theBool):
        self.__liftWithMean__ = theBool

    def getClassName(self):
        '''returns list of class names
        '''
        classNames=[self._cKLResDist[i].__class__.__name__ for i in range(self._N) ]
        return list(set(classNames))

    def getCovarianceModel(self):
        '''
        '''
        return [self._cKLResDist[i].getCovarianceModel() if hasattr(self._cKLResDist[i], 'getCovarianceModel') else None for i in range(self._N) ]

    def getEigenValues(self):
        '''
        '''
        return [self._cKLResDist[i].getEigenValues() if hasattr(self._cKLResDist[i], 'getEigenValues') else None for i in range(self._N) ]

    def getId(self):
        '''
        '''
        return [self._cKLResDist[i].getId() for i in range(self._N) ]

    def getImplementation(self):
        '''
        '''
        return [self._cKLResDist[i].getImplementation() if hasattr(self._cKLResDist[i], 'getImplementation') else None for i in range(self._N) ]

    def getMesh(self):
        '''
        '''
        return [self._cKLResDist[i].getMesh() if hasattr(self._cKLResDist[i], 'getMesh') else None for i in range(self._N) ]

    def getModes(self):
        '''
        '''
        return [self._cKLResDist[i].getModes() if hasattr(self._cKLResDist[i], 'getModes') else None for i in range(self._N) ]

    def getModesAsProcessSample(self):
        '''
        '''
        return [self._cKLResDist[i].getModesAsProcessSample() if hasattr(self._cKLResDist[i], 'getModesAsProcessSample') else None for i in range(self._N) ]

    def getName(self):
        return self.__name__

    def getProjectionMatrix(self):
        '''
        '''
        return [self._cKLResDist[i].getProjectionMatrix() if hasattr(self._cKLResDist[i], 'getProjectionMatrix') else None for i in range(self._N) ]

    def getScaledModes(self):
        '''
        '''
        return [self._cKLResDist[i].getScaledModes() if hasattr(self._cKLResDist[i], 'getScaledModes') else None for i in range(self._N) ]

    def getScaledModesAsProcessSample(self):
        '''
        '''
        return [self._cKLResDist[i].getScaledModesAsProcessSample() if hasattr(self._cKLResDist[i], 'getScaledModes') else None for i in range(self._N) ]

    def getThreshold(self):
        '''
        '''
        return self.threshold

    def setName(self,name):
        self.__name__ = name

    def lift(self, coefficients):
        '''lift a point into a list of functions
        '''
        assert isinstance(coefficients, (ot.Point)), 'function only lifts points'
        valid = self._checkCoefficients(coefficients)
        modes = copy(self.__mode_count__)
        if valid :
            to_return = []
            for i in range(self._N):
                if self.__isProcess__[i] :
                    if not self.__liftWithMean__:
                        to_return.append(self._cKLResDist[i].lift(coefficients[i*modes[i]:(i+1)*modes[i]]))
                    else :
                        function = self._cKLResDist[i].lift(coefficients[i*modes[i]:(i+1)*modes[i]])
                        cst_func = ot.SymbolicFunction(ot.Description_BuildDefault(function.getInputDimension(),'X'),[str(self.__means__[i])])
                        out_func = ot.LinearCombinationFunction([function,cst_func],[1,1])
                        to_return.append(out_func)
                else :
                    if not self.__liftWithMean__:
                        # it is not a process so only scalar distribution, centered
                        const = self._KLDistSampleLifting[i](coefficients[i*modes[i]:(i+1)*modes[i]])
                        # make dummy class. 
                        class constFunc :
                            def __init__(self,x):
                                self.x = x
                            def __call__(self,*args):
                                return self.x
                        func = constFunc(const)
                        to_return.append(func)
                    else :
                        const = self._KLDistSampleLifting[i](coefficients[i*modes[i]:(i+1)*modes[i]])
                        # make dummy class. 
                        class constFunc :
                            def __init__(self,x):
                                self.x = x
                            def __call__(self,*args):
                                return self.x
                        func = constFunc(const+self.__means__[i])
                        to_return.append(func)
            return to_return
        else : 
            raise Exception('DimensionError : the vector of coefficient has the wrong shape')

    def liftAsProcessSample(self, coefficients):
        '''Function to lift a sample of coefficients into a collections of 
        process samples and points
        '''
        assert isinstance(coefficients, (ot.Sample, ot.SampleImplementation))
        modes = copy(self.__mode_count__)
        modes.insert(0,0)
        processes = []
        for i in range(self._N):
            if self.__isProcess__[i] :
                if not self.__liftWithMean__:
                    processes.append(self._KLDistSampleLifting[i](coefficients[:,(modes[i]+sum(modes[:i])):(modes[i+1] + sum(modes[:i+1]))]))
                else :
                    processSample = self._KLDistSampleLifting[i](coefficients[:,(modes[i]+sum(modes[:i])):(modes[i+1] + sum(modes[:i+1]))])
                    addConstantToProcessSample(processSample, self.__means__[i])
                    processes.append(processSample)
            else :
                if not self.__liftWithMean__:
                    processSample = ot.ProcessSample(ot.Mesh(), 0, 1)
                    val_sample = self._KLDistSampleLifting[i](coefficients[:,(modes[i]+sum(modes[:i])):(modes[i+1] + sum(modes[:i+1]))])
                    for j, value in enumerate(val_sample):
                        field = ot.Field(ot.Mesh(),1)
                        field.setValueAtIndex(0,value)
                        processSample.add(field)
                    processes.append(processSample)
                else : 
                    processSample = ot.ProcessSample(ot.Mesh(), 0, 1)
                    val_sample = self._KLDistSampleLifting[i](coefficients[:,(modes[i]+sum(modes[:i])):(modes[i+1] + sum(modes[:i+1]))])
                    mean = self.__means__[i]
                    for j, value in enumerate(val_sample):
                        field = ot.Field(ot.Mesh(),1)
                        field.setValueAtIndex(0,[value[0]+mean]) # adding mean
                        processSample.add(field)
                    processes.append(processSample)
        return processes

    def liftAsField(self, coefficients):
        '''function lifting a vector of coefficients into a field.
        '''
        assert isinstance(coefficients, (ot.Point)), 'function only lifts points'
        valid = self._checkCoefficients(coefficients)
        modes = self.__mode_count__
        if valid :
            to_return = []
            for i in range(self._N):
                if self.__isProcess__[i] :
                    field = self._cKLResDist[i].liftAsField(coefficients[i*modes[i]:(i+1)*modes[i]])
                    if not self.__liftWithMean__:
                        to_return.append(field)
                    else :
                        vals = field.getValues()
                        vals += self.__means__[i]
                        field.setValues(vals)
                        to_return.append(field)
                else :
                    value = self._KLDistSampleLifting[i](coefficients[i*modes[i]:(i+1)*modes[i]])
                    if not self.__liftWithMean__:
                        field = ot.Field(ot.Mesh(),1)
                        field.setValueAtIndex(0,value)
                        to_return.append(field)
                    else :
                        field = ot.Field(ot.Mesh(),1)
                        field.setValueAtIndex(0,value+self.__means__[i])
                        to_return.append(field)
            return to_return
        else : 
            raise Exception('DimensionError : the vector of coefficient has the wrong shape')

    def liftAsSample(self, coefficients):
        ''' function to lift into a list of samples a Point of coefficents
        '''
        assert isinstance(coefficients, ot.Point)
        valid = self._checkCoefficients(coefficients)
        modes = self.__mode_count__
        if valid :
            if self.__isAggregated__ :
                if not self.__liftWithMean__ :
                    sample = self._cKLResDist[0].liftAsSample(coefficients)
                    sample.setDescription(self.__mode_description__)
                    return sample
                else : 
                    raise NotImplementedError
            else :
                to_return = []
                for i in range(self._N):
                    if self.__isProcess__[i] :
                        if not self.__liftWithMean__ : 
                            sample = self._cKLResDist[i].liftAsSample(coefficients[i*modes[i]:(i+1)*modes[i]])
                            to_return.append(sample)
                        else : 
                            sample = self._cKLResDist[i].liftAsSample(coefficients[i*modes[i]:(i+1)*modes[i]])
                            sample += self.__means__[i]
                            to_return.append(sample)
                    else :
                        if not self.__liftWithMean__ : 
                            value = self._KLDistSampleLifting[i](coefficients[i*modes[i]:(i+1)*modes[i]])
                            sample = ot.Sample([value])
                            to_return.append(sample)
                        else :
                            value = self._KLDistSampleLifting[i](coefficients[i*modes[i]:(i+1)*modes[i]])
                            sample = ot.Sample([value+self.__means__[i]])
                            to_return.append(sample)                            
                return to_return
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
        isAggreg = self.__isAggregated__
        homogenMesh = self.__unified_mesh__
        homogenDim = self.__unified_dimension__
        assert isinstance(args[0], (ot.Field, ot.Sample, ot.ProcessSample,
                                    ot.Function, ot.AggregatedFunction,
                                    ot.SampleImplementation))
        if isAggreg :
            assert nProcess==1, 'do not work with lists of aggregated processes'
            assert homogenMesh, 'if aggregated then the mesh is shared'
            assert homogenDim, 'if aggregated then the dimension is shared'
            inDim = self._cKLResDist[0].getCovarianceModel().getInputDimension()
            outDim = self._cKLResDist[0].getCovarianceModel().getOutputDimension()
            if isinstance(args[0], (ot.Function, ot.Field, 
                                    ot.ProcessSample, ot.AggregatedFunction)):
                try : fdi = args[0].getInputDimension()
                except : fdi = args[0].getMesh().getDimension()
                try : fdo = args[0].getOutputDimension()
                except : fdo = args[0].getDimension()

                if fdi == inDim and fdo == outDim :
                    if nArgs > 1 and not isinstance(args[0], ot.ProcessSample):
                        sample = ot.Sample([self._cKLResDist[0].project(args[i]) for i in range(nArgs)])
                        sample.setDescription(self.__mode_description__)
                        return sample 
                    elif isinstance(args[0], ot.Field) : 
                        projection = self._cKLResDist[0].project(args[0])
                        projDescription = list(zip(self.__mode_description__, projection))
                        projection = ot.PointWithDescription(projDescription)
                        return projection
                    elif isinstance(args[0], ot.ProcessSample):
                        projection = self._cKLResDist[0].project(args[0])
                        projection.setDescription(self.__mode_description__)
                        return projection
                else :
                    raise Exception('InvalidDimensionException')

        else : 
            if isinstance(args[0], (ot.Field, ot.Function, ot.Sample)):
                assert nArgs==nProcess, 'Pass a list of same length then aggregation order'
                try:
                    projection = [list(self._cKLResDist[i].project(args[i])) for i in range(nProcess)]
                    projectionFlat = [item for sublist in projection for item in sublist]
                    output = ot.PointWithDescription(list(zip(self.__mode_description__, projectionFlat)))
                    return output
                except Exception as e :
                    raise e

            elif isinstance(args[0], ot.ProcessSample):
                assert nArgs==nProcess, 'Pass a list of same length then aggregation order'
                try:
                    projectionSample = ot.Sample(0,sum(self.__mode_count__))
                    sampleSize = args[0].getSize()
                    projList = [self._cKLResDist[idx].project(args[idx]) for idx in range(nProcess)]
                    for idx in range(sampleSize):
                        l = [list(projList[j][idx]) for j in range(nProcess)]
                        projectionSample.add(
                            [item for sublist in l for item in sublist])
                    projectionSample.setDescription(self.__mode_description__)
                    return projectionSample
                except Exception as e:
                    raise e


    def getAggregationOrder(self):
        return self._N

    def getSizeModes(self):
        return sum(self.__mode_count__)







'''

law = ot.Normal()
basis = ot.Basis([ot.SymbolicFunction(['x'],['1'])])
myMesh = ot.RegularGrid(0.0, 0.1, 10)
lawAsprocess = ot.FunctionalBasisProcess(law, basis, myMesh)

'''