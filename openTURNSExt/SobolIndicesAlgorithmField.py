__author__ = 'Kristof Attila S.'
__version__ = '0.1'
__date__  = '17.09.20'

__all__ = ['AggregatedKarhunenLoeveResults']

import openturns as ot 
import math 
import uuid
from collections import Iterable

class SobolIndicesAlgorithmField(object):
    '''All the input parameters are defined on a same Mesh 

    We want to be able to do the sensitivirty analysis on any field function 
    defined in openturns
    '''
    def __init__(self, *args):
        self._inputMesh = None
        self._outputMesh = None

    def setInputMesh(self, Mesh):
        assert isinstance(Mesh,ot.Mesh)
        self._inputMesh = Mesh

    def setOutputMesh(self,Mesh):
        assert isinstance(Mesh,ot.Mesh)
        self._outputMesh = Mesh

    def setInputDimension(self,Dim):
        pass


def all_same(items):
    #Checks if all items of a list are the same
    return all(x == items[0] for x in items)


def atLeastList(elem):
    if isinstance(elem, Iterable) :
        return list(elem)
    else : 
        return [elem]



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

    def _fieldsToProcessSamples(self, listOfFields, _mesh = None):
        assert isinstance(listOfFields, Iterable)
        if isinstance(listOfFields[0],Iterable):
            assert all_same([len(listOfFields[i]) for i in range(len(listOfFields))])
            assert len(listOfFields) == self._N
            meshes = self.getMesh()

            if isinstance(listOfFields[0][0],ot.Field):
                processSamples = [ot.ProcessSample(mesh, 0, listOfFields[0][0].getOutputDimension()) for mesh in meshes]
                for i in range(self._N):
                    [processSamples[i].add(listOfFields[i][j]) for j in range(len(listOfFields[i]))]
                return processSamples

            elif isinstance(listOfFields[0][0],ot.Sample):
                processSamples = [ot.ProcessSample(mesh, 0, listOfFields[0][0].getDimension()) for mesh in meshes]
                for i in range(self._N):
                    [processSamples[i].add(ot.Field(meshes[i],listOfFields[i][j])) for j in range(len(listOfFields[i]))]
                return processSamples

        if isinstance(listOfFields[0],ot.Field):
            processSample = ot.ProcessSample(listOfFields[0].getMesh(), 0, listOfFields[0].getOutputDimension())
            [processSample.add(listOfFields[j]) for j in range(len(listOfFields))]
            return processSample

        elif isinstance(listOfFields[0],ot.Sample):
            assert _mesh is not None, 'please give mesh'
            processSample = ot.ProcessSample(_mesh, 0, listOfFields[0].getDimension())
            [processSample.add(ot.Field(_mesh, listOfFields[j])) for j in range(len(listOfFields))]
            return processSample

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
        '''
        '''
        valid = self._checkCoefficients(coefficients)
        modes = self._modesPerProcess
        if valid :
            return [self._KLRL[i].lift(coefficients[i*modes[i]:(i+1)*modes[i]]) for i in range(self._N)]
        else : 
            raise Exception('DimensionError : the vector of coefficient has the wrong shape')


    def liftDistributionToField(self, randomVector):
        '''transforms a randomVector or collection of random vectors
        into one or multiple gaussian fields
        '''
        # if only one realisation
        if isinstance(randomVector[0], float):
            field = self.KarhunenLoeveResult.liftAsField(randomVector)
            return field
        # if a collection of realisations
        elif isinstance(randomVector[0], list):
            dimension = len(randomVector)
            field_list = [self.KarhunenLoeveResult.liftAsField(
                            randomVector[i]) for i in range(dimension)]
            process_sample = openturns.ProcessSample(self.mesh, 0, dimension)
            [process_sample.add(
                field_list[i]) for i in range(len(field_list))]
            return process_sample

    def liftAsField(self, coefficients):
        '''function lifting a vector of coefficients into a field.
        A sample of multiple vectors can also be passed and are lifted into
        a list of process samples (as the processes can have different 
        dimensions)
        '''
        valid = self._checkCoefficients(coefficients)
        modes = self._modesPerProcess
        if valid :
            if isinstance(coefficients, ot.Point):
                return [self._KLRL[i].liftAsField(coefficients[i*modes[i]:(i+1)*modes[i]]) for i in range(self._N)]
            
            elif isinstance(coefficients, (ot.Sample, ot.SampleImplementation)):
                #In this case we want to return a list of process sample
                field_list = list()
                for j in range(coefficients.getSize()):
                    field_list.append([self._KLRL[i].liftAsField(coefficients[j][i*modes[i]:(i+1)*modes[i]]) for i in range(self._N)])
                field_list = list(zip(*field_list))
                processes = self._fieldsToProcessSamples(field_list)
                return processes

        else : 
            raise Exception('DimensionError : the vector of coefficient has the wrong shape')

    def liftAsSample(self, coefficients):
        '''
        '''
        valid = self._checkCoefficients(coefficients)
        modes = self._modesPerProcess
        if valid :
            print('Coeffs seem valid')
            if self._aggregFlag :
                print('Is aggregated!')
                print('Look at that! : \n',type(coefficients),'\n',coefficients)
                if isinstance(coefficients, Iterable):
                    if isinstance(coefficients[0], (ot.SampleImplementation, ot.Sample)):
                        return [self._KLLift[i](coefficients[i]) for i in range(self._N)]
                    print('Pass coefficients OAT not as a Sample')
                    raise NotImplementedError
                else:
                    sample = self._KLRL[0].liftAsSample(coefficients)
                    #sample.setDescription(self._modeDescription)
                return sample
            else :
                print('type coeffs',type(coefficients))
                if isinstance(coefficients, (ot.SampleImplementation, ot.Sample)):
                    sample_list = list()
                    for j in range(coefficients.getSize()):
                        sample_list.append([self._KLRL[i].liftAsSample(coefficients[j][i*modes[i]:(i+1)*modes[i]]) for i in range(self._N)])
                    sample_list = list(zip(*sample_list))
                    processes = sample_list
                    return processes

                elif isinstance(coefficients, ot.Point):
                    return [self._KLRL[i].liftAsSample(coefficients[i*modes[i]:(i+1)*modes[i]]) for i in range(self._N)]
                elif isinstance(coefficients, Iterable):
                    if len(coefficients)==self._N and isinstance(coefficients[0], ot.SampleImplementation):
                        return [self._KLLift[i](coefficients[i]) for i in range(self._N)]
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

        # in the case where the class was initialized with an other KLResult
        # structure, but that was extracted from an AggregatedProcess
        if isAggreg and homogenMesh and nProcess==1 :
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

        # in this case, we intialised the class with a list of processes
        # difficulty is here to handle the way the projections are made as 
        #the dimensions defer between 
        if nProcess>1 :
            if isAggreg :
                print('''
You can not pass multiple aggregated processes decomposition results to the 
constructor.''')
                raise Exception('NotImplementedError')
            elif not homogenMesh :
                if isinstance(args[0], ot.Function):
                    if homogenDim :
                        print('Dimensions are consistent between processes')
                        print('''
It is possible to project {} function on each of them'''.format(str(nArgs)))
                        result = list()
                        for arg in args:
                            resPerArg = list()
                            for KLresult in self._KLRL:
                                resPerArg.append(KLresult.project(arg))
                            result.append(resPerArg)
                        return result

                    elif nArgs!=nProcess:
                        print('''
Dimensions inconsistent between KLResult structures. You have to pass 
functions to project in a list of the same length than the order of KL Result 
structures, or a multiple of the KL structures. To pass samples pass list of 
lists.''')
                        raise Exception('InvalidDimensionException')
                    elif nArgs==nProcess:
                        projection = [self._KLRL[i].project(args[i]) for i in range(nProcess)]
                        projectionFlat = [item for sublist in projection for item in sublist]
                        output = ot.PointWithDescription(list(zip(self._modeDescription, projectionFlat)))
                        return output

                elif isinstance(args[0], ot.Field):
                    if nArgs == nProcess :
                        try:
                            projection = [list(self._KLRL[i].project(args[i])) for i in range(nProcess)]
                            projectionFlat = [item for sublist in projection for item in sublist]
                            output = ot.PointWithDescription(list(zip(self._modeDescription, projectionFlat)))
                            return output
                        except Exception as e :
                            print(
                              'Verify the order of your processes and fields')
                            raise Exception('InvalidArgumentException')
                    else : 
                        print(
                            'Pass as much fields than the number of process')
                        raise Exception('InvalidArgumentException') 

                elif isinstance(args[0], Iterable):
                    print('''
The usage of Iterables is ambiguous, as we dont know over which dimension to
iterate. The prefered way is to pass a comprehensive list where the first 
dimension is the number of samples and the second dimension must have the same
dimensions than the aggregated processes.
In the case where the first dimension has the same dimension than the 
aggregated process and the second dimension not, the second dimension will be 
considered as the sample dimension.
When the two dimension are equal to the dimension of the process, the first 
method is choosen.''')
                    isConsistent = all_same([len(args[i]) for i in range(nArgs)])
                    try:
                        if len(args[0]) == nProcess and isConsistent:
                            result = list()
                            for i in range(nArgs):
                                interm = list()
                                for j in range(nProcess):
                                    interm.append(
                                        self._KLRL[j].project(args[i][j]))
                                interm = ot.PointWithDescription(list(zip( self._modeDescription, interm)))
                                result.append(interm)
                            return result
                    except Exception as e:
                        print('projection failed, check dimensions')
                        raise e 

                    try : 
                        if len(args[0])!=nProcess and nArgs==nProcess and isConsistent:
                            result = list()
                            for j in range(len(args[0])):
                                interm = list()
                                for i in range(nArgs):
                                    interm.append(
                                        self._KLRL[i].project(args[i][j]))
                                interm = ot.PointWithDescription(list(zip( self._modeDescription, interm)))
                                result.append(interm)
                            return result
                    except Exception as e:
                        print('projection failed, check dimensions')
                        raise e 

                    if (len(args[0])!=nProcess and nArgs!=nProcess) or not isConsistent:
                        print('Shape broadcasting failed')
                        raise Exception('InvalidDimensionException')

                elif isinstance(args[0], ot.ProcessSample):
                    ###########
                    ###########
                                        ###########
                                                            ###########
                                                                                ###########
            elif homogenMesh :
                print('''
As all your processes are homogenous, you could have passed them as an
Aggregated process.''')
                print('''
The results will be returned as a list of projection coefficients anyway 
and not a Sample as it would be the case with an AggregatedProcess.''')
                if isinstance(args[0], (ot.Function, ot.Field)):
                    if len(args)==nProcess:
                        projection = [self._KLRL[i].project(args[i]) for i in range(nProcess)]
                        projectionFlat = [item for sublist in projection for item in sublist]
                        output = ot.PointWithDescription(list(zip(self._modeDescription, projectionFlat)))
                        return output
                    else : 
                        print('''
The length of the arguments is different from the number of processes in this 
object. As the processes all share the same mesh though, one field can be 
broadcasted to every process of the structure. We will consider that this 
is the desired behaviour and the function will consider each function/field
as a sample''')         
                        results = list()
                        for arg in args :
                            projection = [self._KLRL[i].project(arg) for i in range(nProcess)]
                            projectionFlat = [item for sublist in projection for item in sublist]
                            output = ot.PointWithDescription(list(zip(self._modeDescription, projectionFlat)))  
                            results.append(output)
                        return results

                if isinstance(args[0], ot.ProcessSample):
                    if len(args) == nProcess : 
                        dim = sum(self._modesPerProcess)
                        projectionSample = ot.Sample(0,dim)
                        sampleSize = args[0].getSize()
                        projList = [self._KLRL[idx].project(args[idx]) for idx in range(nProcess)]
                        for idx in range(sampleSize):
                            l = [list(projList[j][idx]) for j in range(nProcess)]
                            projectionSample.add(
                                [item for sublist in l for item in sublist])
                        projectionSample.setDescription(self._modeDescription)
                        return projectionSample

        if (not isAggreg or not homogenMesh) and nProcess==1:
            raise Exception('InvalidArgumentException')

    def getAggregationOrder(self):
        return self._N




#Distribution as process
'''
law = ot.Normal()
basis = ot.basis([ot.SymbolicFunction(['x'],['1']]))
myMesh = ot.RegularGrid(0.0, 0.1, 10)
lawAsprocess = ot.FunctionalBasisProcess(law, basis, myMesh)


'''