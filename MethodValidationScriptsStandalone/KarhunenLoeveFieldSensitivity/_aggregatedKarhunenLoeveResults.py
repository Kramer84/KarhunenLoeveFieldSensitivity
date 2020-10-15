__author__ = 'Kristof Attila S.'
__version__ = '0.1'
__date__  = '17.09.20'

__all__ = ['AggregatedKarhunenLoeveResults']

import openturns as ot 
import uuid
from collections import Iterable
from copy import copy

def all_same(items):
    #Checks if all items of a list are the same
    return all(x == items[0] for x in items)

def atLeastList(elem):
    if isinstance(elem, Iterable) :
        return list(elem)
    else : 
        return [elem]

class AggregatedKarhunenLoeveResults(object):
    '''Function being a buffer between the processes and the sensitivity
    Analysis
    '''
    def __init__(self, KLResList):
        self._KLRL = atLeastList(KLResList) #KLRL : Karhunen Loeve Result List
        assert len(self._KLRL)>0
        self._N = len(self._KLRL)
        self.__name__ = 'Unnamed'
        self._KLLift = [ot.KarhunenLoeveLifting(self._KLRL[i]) for i in range(self._N)]
        self._homogenMesh = all_same([self._KLRL[i].getMesh() for i in range(self._N)])
        self._homogenDim = (all_same([self._KLRL[i].getCovarianceModel().getOutputDimension() for i in range(self._N)])  \
                            and all_same([self._KLRL[i].getCovarianceModel().getInputDimension() for i in range(self._N)]))
        self._aggregFlag = (self._KLRL[0].getCovarianceModel().getOutputDimension() > self._KLRL[0].getMesh().getDimension() \
                            and self._N ==1 )
        #Cause when aggregated there is usage of multvariate covariance functions
        if self._aggregFlag : print('Process seems to be aggregated. ')
        self.threshold = max([self._KLRL[i].getThreshold() for i in range(self._N)])
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
            print('The process names are not unique.')
            print('Using generic name. ')
            for i, process in enumerate(self._KLRL) :
                oldName = process.getName()
                newName = 'X_'+str(i)
                print('Old name was {}, new one is {}'.format(oldName, newName))
                process.setName(newName)
            self._subNames = [self._KLRL[i].getName() for i in range(self._N)]

    def _getModeDescription(self):
        modeDescription = list()
        for i, nMode in enumerate(self._modesPerProcess):
            for j in range(nMode):
                modeDescription.append(self._subNames[i]+str(j))
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

    def getName(self):
        return self.__name__

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

    def setName(self,name):
        self.__name__ = name

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







'''
law = ot.Normal()
basis = ot.Basis([ot.SymbolicFunction(['x'],['1'])])
myMesh = ot.RegularGrid(0.0, 0.1, 10)
lawAsprocess = ot.FunctionalBasisProcess(law, basis, myMesh)
'''