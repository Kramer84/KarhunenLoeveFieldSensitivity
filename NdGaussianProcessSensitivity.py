import openturns
import uuid 
import numpy
from   typing                                import Callable, List, Tuple, Optional, Any, Union
from   copy                                  import deepcopy
import NdGaussianProcessConstructor          as     ngpc
import NdGaussianProcessExperimentGeneration as     ngpeg
import NdGaussianProcessSensitivityIndices   as     ngpsi 
import atexit
import gc

class NdGaussianProcessSensitivityAnalysis(object):
    '''Custom class to do sensitivity analysis on complex models
        
        This class will guide through the process of doing sensitivity
        analysis on functions that take as an input random variables
        and Gaussian Processes, and output fields as well as scalars.

        To make the analysis possible, some informations are required in 
        advance, mainly the shape of the inputs, the laws or processes 
        that these inputs follow, as well as the nature of the outputs.
        This informations are added by passing a list of objects defined in
        an other class: NdGaussianProcessConstructor
        
        The function indicated can be any python function, but not the 
        openTurnsPythonFunction type, as this is done internaly. The
        function also needs to take as an input the full field, not the 
        modified function taking in decompositions like kahrunen loeve.

        Attributes:

            inputProcessesDistributions    : list
                list of NdGaussianProcessConstructor.NdGaussianProcessConstructor and scalar
                probabilistic openturns distributions

            outputVariables   : nested dict
                dictionary containg the parameters for the output processes
                    dictModel = {'outputField_1' :
                                 {
                                  'nameField'    : str,
                                  'position'     : int, #position in output
                                  'shape'        : list
                                 },
                                }

            funcSample         : pythonFunction
                function taking as an input samples of rvs (vectors) 
                and samples of fields (ndarrays) for multiprocessing)

            funcSolo           : pythonFunction
                function taking as a input rvs (scalars) and fields

            sampleSize         : int
                size of the sample for the sensitivity analysis
    '''
    def __init__(self, inputProcessesDistributions : Optional[list]     = None , ###  
                       outputVariables             : Optional[dict]     = None , ##  While being optional in the init method
                       funcSample                  : Optional[Callable] = None , ##  it is still necessary to set the variables 
                       funcSolo                    : Optional[Callable] = None , ##  through the .set* methods
                       sampleSize                  : Optional[int]      = None): ##

        self.inputProcessesDistributions = inputProcessesDistributions
        self.outputVariables             = outputVariables
        self.functionSample              = funcSample
        self.functionSolo                = funcSolo
        self.wrappedFunction = OpenturnsPythonFunctionWrapper(functionSample              = self.functionSample,
                                                              functionSolo                = self.functionSolo, 
                                                              inputProcessesDistributions = self.inputProcessesDistributions,
                                                              outputDict                  = self.outputVariables)
        self.sampleSize          = sampleSize
        self.sobolBatchSize      = None
        self.inputDesign         = None
        self.outputDesignList    = None 
        self.sensitivityResults  = None 

        self._errorWNans         = 0
        self._inputDesignNC      = None   #non corrected designs => the functions are expected to malfunction sometimes and to return nan values 
        self._outputDesignListNC = None   #non corrected designs  
        self._designsWErrors     = None

    def makeExperiment(self, **kwargs):
        self.prepareSobolIndicesExperiment(**kwargs)
        self.getOutputDesignAndPostprocess(**kwargs)

    def prepareSobolIndicesExperiment(self, gen_type = 1,  **kwargs):
        sobolExperiment         = ngpeg.NdGaussianProcessExperiment(self.sampleSize, self.wrappedFunction, gen_type)
        inputDesign             = sobolExperiment.generate(**kwargs)
        sobolBatchSize          = len(inputDesign)
        print('number of samples for sobol experiment = ', sobolBatchSize, '\n')
        self.sobolBatchSize     = sobolBatchSize
        self._inputDesignNC     = inputDesign
        print('input design is: ',inputDesign)

    def getOutputDesignAndPostprocess(self, **kwargs):
        assert self._inputDesignNC is not None, ""
        assert self.functionSample is not None or self.functionSolo is not None , ""
        if self.functionSample is not None : 
            inputDes         = deepcopy(self._inputDesignNC)
            outputDesign     = self.wrappedFunction(inputDes)
        else :
          ## We first have to implement the multiprocessing of the single evaluation model
            raise NotImplementedError
        n_outputs = len(self.outputVariables.keys())
        if n_outputs >1 :
            outputDesignList     = [numpy.array(outputDesign[i]) for i in range(n_outputs)]
        else : 
            outputDesignList     = [numpy.array(outputDesign)]
        self._outputDesignListNC = outputDesignList
        self._postProcessOutputDesign(**kwargs)

    def _postProcessOutputDesign(self, **kwargs):
        '''To check if there are nan values and replace them with new realizations
           We not only have to erase the value with the nan, but all the corresponding
           permutations. 
           
        '''
        outputList           = deepcopy(self._outputDesignListNC)
        ## We flatten all the realisation of each sample, to check if we have np.nans
        outputMatrix         = self.wrappedFunction.outputListToMatrix(outputList)
        inputArray           = numpy.array(deepcopy(self._inputDesignNC)) 
        composedDistribution = self.wrappedFunction.KLComposedDistribution
        N                    = self.sampleSize
        d_implicit           = int(inputArray.shape[0]/N)-2  #Any type of input shape being a multiple > 2 of N
        d_inputKL            = composedDistribution.getDimension() #dimension of the karhunen loeve composed distribution
        combinedMatrix       = numpy.hstack([inputArray, outputMatrix]).copy() # of size (N_samples, inputDim*outputDim)
        combinedMatrix0      = combinedMatrix.copy()
        whereNan             = numpy.argwhere(numpy.isnan(deepcopy(combinedMatrix)))[...,0]
        columnIdx            = numpy.atleast_1d(numpy.squeeze(numpy.unique(whereNan)))
        print('Columns where nan : ', columnIdx,'\n')
        n_nans               = len(columnIdx)
        self._errorWNans      += n_nans
        self._designsWErrors = inputArray[columnIdx, ...]
        if n_nans > 0:
            print('There were ',n_nans, ' errors (numpy.nan) while processing, trying to regenerate missing outputs \n')
            for i  in range(n_nans):
                idx2Change   = numpy.arange(d_implicit+2)*N + columnIdx[i]%N
                print('index to change: ',idx2Change)
                newCombinedMatrix        = self._regenerate_missing_vals_safe(**kwargs)
                for q in range(len(idx2Change)):
                    p                      = idx2Change[q]
                    combinedMatrix[p, ...] = newCombinedMatrix[q, ...]
            try :
                assert numpy.allclose(combinedMatrix0,combinedMatrix, equal_nan=True) == False, "after correction they should be some difference"
                assert numpy.isnan(combinedMatrix).any() == False, "should have no nan anymore"
            except : 
                whereNan             = numpy.argwhere(numpy.isnan(deepcopy(combinedMatrix)))[...,0]
                columnIdx            = numpy.atleast_1d(numpy.squeeze(numpy.unique(whereNan)))
                print('columns where still nan', columnIdx)

            print(' - Post replacement assertion passed - \n')
            inputArray  = combinedMatrix[..., :d_inputKL]    
            outputArray = combinedMatrix[..., d_inputKL:]
            inputSample = openturns.Sample(inputArray)
            inputSample.setDescription(self.wrappedFunction.getInputDescription())
            self.outputDesignList = self.wrappedFunction.matrixToOutputList(outputArray)
            self.inputDesign      = inputSample
        else :
            self.outputDesignList = self._outputDesignListNC
            self.inputDesign      = self._inputDesignNC
            print('\nNo errors while processing, the function has returned no np.nan.\n')

    def _regenerate_missing_vals_safe(self, **kwargs): 
        composedDistribution = self.wrappedFunction.KLComposedDistribution
        exit                 = 1
        tries                = 0
        while exit != 0:
            sobolReg      = ngpeg.NdGaussianProcessExperiment(1,self.wrappedFunction)
            inputDes      = sobolReg.generate(**kwargs)
            outputDes     = self.wrappedFunction(inputDes)
            inputDes      = numpy.array(inputDes).tolist()
            outputDesFlat = self.wrappedFunction.outputListToMatrix(outputDes)
            #should be 0 when there is no nan
            exit  = len(numpy.atleast_1d(numpy.squeeze(numpy.argwhere(numpy.isnan(outputDesFlat))[...,0])))
            tries += 1
            if tries == 50 :
                print('Error with function')
                raise FunctionError('The function used does only return nans, done over 50 loops')

        return numpy.hstack([inputDes, outputDesFlat])

    def getSensitivityAnalysisResults(self, methodOfChoice = 'Saltelli', returnStuff = False):
        '''get sobol indices for each element of the output
        As the output should be a list of fields and scalars, this step will
        return a list of scalar sobol indices and field sobol indices
        '''
        algoDict = {'Jansen'            : 1,
                    'Saltelli'          : 2,
                    'Mauntz-Kucherenko' : 3,
                    'Martinez'          : 4}
        assert methodOfChoice in algoDict, "argument has to be a string:\n['Jansen','Saltelli','Mauntz-Kucherenko','Martinez'] "
        size             = self.sampleSize
        inputDesign      = self.inputDesign
        dimensionInput   = int(len(inputDesign[0]))
        dimensionOutput  = len(self.outputVariables.keys())
        outputDesignList = self.outputDesignList
        sensitivityAnalysisList = list()
        n_tot  = size*(2+dimensionInput)
        for k in range(dimensionOutput):
            outputDesign        = numpy.array(outputDesignList[k])
            shapeDesign         = list(outputDesign.shape) 
            shapeNew            = int(numpy.prod(shapeDesign[1:]))
            print('new shape is : ', shapeNew)
            outputDesResh        = numpy.reshape(outputDesign, [n_tot, shapeNew])
            if shapeNew == 1 :
                outputDesResh        = numpy.squeeze(outputDesResh.flatten())
                print(outputDesResh)
                sensitivityAna = algoDict[methodOfChoice](inputDesign, numpy.expand_dims(outputDesResh,1), size)
                sensitivityAnalysisList.append(numpy.array([sensitivityAna]))
            elif shapeNew > 1 :            
                sensitivityIdList = [algoDict[methodOfChoice](inputDesign, numpy.expand_dims(outputDesResh[...,i], 1), size) for i in range(shapeNew)]
                sensitivityIdArr  = numpy.array(sensitivityIdList)
                endShape          = shapeDesign[1:] # we suppress the first dimension as we have fields of objects, not indices
                print('sobol field shape: ',endShape)
                sensitivityField  = numpy.reshape(sensitivityIdArr, endShape)
                print('Shape sensitivity field : ',sensitivityField.shape)
                sensitivityAnalysisList.append(sensitivityField)
            else :
                print('Unknown problem')
                raise TypeError
        self.sensitivityResults = senstivityAnalysisWrapper(sensitivityAnalysisList, self.wrappedFunction.getInputDescription()) 
        if returnStuff == True :
            return sensitivityAnalysisList

    def setFunctionSample(self, wrapFunction):
        '''Python function taking as an input random variables and fields

        Note
        ----
        the function's arguments order is the one defined in
        self.InputProcesses and self.InputRVs
        '''
        self.funcModel = wrapFunction

    def KarhunenLoeveSVDAlgorithm(self, ndarray : numpy.ndarray, process_sample = None, threshold = 0.0001, centeredFlag = False):
        '''Function to get Kahrunen Loeve decomposition from samples stored in array
        '''
        if process_sample is None :
            process_sample = self.getProcessSampleFromNumpyArray(ndarray)
        FL_SVD = openturns.KarhunenLoeveSVDAlgorithm(process_sample, threshold, centeredFlag)
        #FL_SVD.setNbModes(nbModes)
        FL_SVD.run()
        return FL_SVD

    def getProcessSampleFromNumpyArray(self, ndarray : numpy.array) -> openturns.ProcessSample :
        '''Function to get a field out of a numpy array representing a sample
        As it is a sample we will have a collection of fields
        ''' 
        arr_shape = list(ndarray[0,...].shape) #first build mesh of same size than array, given that each [i,:] is a sample
        dimension = len(arr_shape)
        if len(arr_shape)<5 : raise NotImplementedError # Maximum 4 dimensions
        # We build a unit grid
        grid_shape = [[0, arr_shape[i], arr_shape[i]] for i in range(dimension)]
        otMesh      = self.getMesh(grid_shape, dimension)
        field_list = [openturns.Field(otMesh,numpy.expand_dims(ndarray[i,...].flatten(order='C'), axis=1).tolist()) for i in range(numpy_array.shape[0])]
        process_sample = openturns.ProcessSample(otMesh, 0, dimension)
        [process_sample.add(field_list[i]) for i in range(len(field_list))]
        return process_sample

    def getMesh(self, grid_shape : List[Tuple[float,float,int],], dimension : Optional[int] = None) -> openturns.Mesh :
        '''Function to set the grid on which the process will be defined
        
        Arguments
        ---------
        grid_shape : list
            List containing the lower bound, the length and the number of elements
            for each dimension. Ex: [[x0 , Lx , Nx], **]
        '''
        n_intervals     = [grid_shape[i][2]-1         for i in range(dimension)]
        low_bounds      = [grid_shape[i][0]           for i in range(dimension)]
        lengths         = [grid_shape[i][1]-1         for i in range(dimension)]
        high_bounds     = [low_bounds[i] + lengths[i] for i in range(dimension)]
        mesherObj       = openturns.IntervalMesher(n_intervals)
        grid_interval   = openturns.Interval(low_bounds, high_bounds)
        mesh            = mesherObj.build(grid_interval)
        mesh.setName(str(dimension)+'D_Grid')
        return mesh






#######################################################################################
#######################################################################################
#######################################################################################
######
###
#                  Wrapper for the python functions, to translate between the KL
#                  coefficients and the stochastic fields 
###
#####
#######################################################################################
#######################################################################################
#######################################################################################

class OpenturnsPythonFunctionWrapper(openturns.OpenTURNSPythonFunction):
    '''Wrapper for python functions that take as an argument stochastic fields,
    as well as scalar probabilistic distributions. The wrapper creates an interface
    to a new function, that only takes in scalar values, by decomposing the 
    stochastic processes using Karhunen-Loeve

    Note
    ----
    This class is used internaly by the NdGaussianProcessSensitivityAnalysis class,
    so the order of decomposition of the Processes hs not to be known and all is 
    tracked internaly relatively robustly. 

    Arguments
    ---------
    functionSample : python function
        function taking as an input random fields and scalars, specialy optimized 
        for batch processing
    functionSolo : python function
        same function than above, but only working with one sample at once 
    inputProcessesDistributions : list 
        list of distributions and processes (processes are defined with the 
        NdGaussianProcessConstructor)
    outputDict : dict
        dictionary containing data about the outputs of the function. The 
        dimension as well as the output position has to be indicated

    Attributes
    ----------


    '''
    def __init__(self, functionSample          : Optional[Callable] = None, 
                       functionSolo            : Optional[Callable] = None,
                       inputProcessesDistributions : Optional[list] = None , 
                       outputDict                  : Optional[dict] = None):
        self.inputProcessesDistributions = inputProcessesDistributions
        self.outputDict             = outputDict
        self.PythonFunctionSample   = functionSample
        self.PythonFunction         = functionSolo
        self.NdGaussianProcessList  = list()
        self.inputVarNames          = list()
        self.outputVarNames         = list()
        self.getInputVariablesName()
        self.getOutputVariablesName()
        self.inputDim               = len(self.inputVarNames)
        self.outputDim              = len(self.outputVarNames)
        self.KLComposedDistribution = self.composeFromProcessAndDistribution(self.inputProcessesDistributions)
        self.inputVarNamesKL        = self.getKLDecompositionVarNames()
        self.inputDimKL             = len(self.inputVarNamesKL)
        super(OpenturnsPythonFunctionWrapper, self).__init__(self.inputDimKL, self.outputDim)
        self.setInputDescription(self.inputVarNamesKL)
        self.setOutputDescription(self.outputVarNames)

    def composeFromProcessAndDistribution(self, inputProcessesDistributions : list, ntemp : int = 1750):
        '''Using a list of ordered openturns distributions and custom defined Process
        with the NdGaussianProcessConstructor class we construct a vector of scalar 
        distributions, according to the distributions and the Karhunen-Loeve decomposition
        of the processes.

        Arguments
        ---------
        inputProcessesDistributions : list
            list of probabilistic distributions as well as Processes 

        Returns
        -------
        composedDistribution : openturns.ComposedDistribution
            random vector of various probabilistic distributions, in the order 
            of the input list and with Processes decomposed as random variables
        '''
        listNames = list()
        listProbabilisticDistributions = list()
        inputList = inputProcessesDistributions
        for i, inp in enumerate(inputList) :
            if isinstance(inp, openturns.DistributionImplementation):
                name = inp.getName()
                assert name is not 'Unnamed', "Please give a name to your distributions through the setName() method..."
                if name in listNames:
                    print('Possible duplicata of 2 variable names...')
                    newName = name+str(i)+'_'+str(uuid.uuid4())
                    print('New random name assigned:\n',name,'-->',newName)
                    name = newName
                    inp.setName(name)
                listNames.append(name)
                listProbabilisticDistributions.extend([inp])


            if isinstance(inp, openturns.Process) :
                name = inp.getName()
                assert name is not 'Unnamed', "Please give a name to your Processes through the setName() method..."
                if name in listNames:
                    print('Possible duplicata of 2 variable names...')
                    newName = name+str(i)+'_'+str(uuid.uuid4())
                    print('New random name assigned:\n',name,'-->',newName)
                    name = newName
                    inp.setName(name)
                listNames.append(name)
                try :
                    inp.getFieldProjectionOnEigenmodes()
                    inp.getDecompositionAsRandomVector()
                except :
                    _ = inp.getSample(ntemp)
                    inp.getFieldProjectionOnEigenmodes()
                    inp.getDecompositionAsRandomVector()
                processAsRandVect = inp.decompositionAsRandomVector.getRandVectorAsOtNormalsList()
                self.NdGaussianProcessList.append([i, inp]) #so we mkeep in memory the position in the function arguments
                listProbabilisticDistributions.extend(processAsRandVect)
        print('Composed distribution built with processes and distributions:',' '.join(listNames))
        composedDistribution = openturns.ComposedDistribution(listProbabilisticDistributions)
        return composedDistribution

    def liftFieldFromKLDistribution(self, KLComposedDistribution):
        fieldPositions     = [self.NdGaussianProcessList[i][0] for i in range(len(self.NdGaussianProcessList))]
        numberModesFields  = [int(self.NdGaussianProcessList[i][1].decompositionAsRandomVector.n_modes) for i in range(len(self.NdGaussianProcessList))]
        fieldNames         = [self.NdGaussianProcessList[i][1].getName() for i in range(len(self.NdGaussianProcessList))]
        listInputVars      = list()
        idxStp             = 0
        tempCompo          = numpy.asarray(KLComposedDistribution)
        for k in range(self.inputDim):
            if k in fieldPositions :
                idx        = int(numpy.argwhere(numpy.asarray(fieldPositions)==k))
                idxStpPrev = idxStp
                idxStp     += numberModesFields[idx]-1
                process    = self.NdGaussianProcessList[fieldPositions[k]][1]
                field      = process.liftDistributionToField(tempCompo[...,k+idxStpPrev:k+idxStpPrev+idxStp].tolist())
                field      = numpy.asarray(field).tolist()
                listInputVars.append(field)
            else :
                listInputVars.append(tempCompo[...,k+idxStp])
        return listInputVars

    def getInputVariablesName(self):
        self.inputVarNames.clear()
        for inp in self.inputProcessesDistributions :
            self.inputVarNames.append(inp.getName())
        print('Input Variables are (without Karhunen Loeve Decomposition) :\n'," ".join(self.inputVarNames),'\n')

    def getOutputVariablesName(self): 
        sortedKeys = sorted(self.outputDict , key = lambda x : self.outputDict[x]['position'])
        self.outputVarNames.clear()
        for key in sortedKeys :
            try : 
                nameOutput = self.outputDict[key]['name']
                self.outputVarNames.append(nameOutput)
            except KeyError :
                print('Error in your output dictionary')
        print('Output Variables are :\n',self.outputVarNames,'\n')

    def getTotalOutputDimension(self):
        outputDict = self.outputDict
        n_outputs  = len(outputDict.keys())
        shapeList  = list()
        for key in outputDict.keys() : 
            shapeList.append(list(outputDict[key]['shape']))
        tot_dim = 0
        dim_perOut = list()
        for shape in shapeList : 
            size = numpy.prod(numpy.array(list(shape)))
            tot_dim = tot_dim + size
            dim_perOut.append(size)
        return tot_dim, dim_perOut, shapeList

    def getKLDecompositionVarNames(self):
        fieldPositions    = [self.NdGaussianProcessList[i][0] for i in range(len(self.NdGaussianProcessList))]
        numberModesFields = [int(self.NdGaussianProcessList[i][1].decompositionAsRandomVector.n_modes) for i in range(len(self.NdGaussianProcessList))]
        fieldNames        = [self.NdGaussianProcessList[i][1].getName() for i in range(len(self.NdGaussianProcessList))]
        RVNames           = [self.NdGaussianProcessList[i][1].decompositionAsRandomVector.nameList for i in range(len(self.NdGaussianProcessList))]
        namesList         = self.inputVarNames
        namesArray        = numpy.asarray(namesList)
        reIdx = 0
        for i in range(len(namesList)):
            name = namesList[i]
            if name in fieldNames :
                idx        = int(numpy.argwhere(numpy.asarray(fieldNames)==name))
                namesArray = numpy.hstack([namesArray[:i+reIdx],numpy.asarray(RVNames[idx]),namesArray[1+i+reIdx:]])
                reIdx      += numberModesFields[idx]-1
        return namesArray.tolist()
    
    def outputListToMatrix(self, outputList):
        '''Flattens a list of ndarrays, sharing the same first dimension
        '''
        # Should be ok...
        outputList    = deepcopy(outputList)
        flatArrayList = list()
        n_tot         = int(numpy.array(outputList[0]).shape[0])
        print('Converting list of outputs into matrix: ')
        el = 1
        for array in outputList :
            array     = numpy.array(array) #to be sure
            shapeArr  = array.shape
            print('Element ', el ,' has shape ', array.shape)
            dimNew    = int(numpy.prod(shapeArr[1:]))
            arrayFlat = numpy.reshape(array, [n_tot, dimNew])
            flatArrayList.append(arrayFlat)
            el += 1
        flatArray = numpy.hstack(flatArrayList)
        print('Final shape matrix: ', flatArray.shape)

        return flatArray

    def matrixToOutputList(self, matrix):
        '''Tranform the flattened image of the output back into it's original
        shape
        '''
        n_outputs = len(self.outputDict.keys())
        tot_dim, dim_perOut, shapeList = self.getTotalOutputDimension()
        assert matrix.shape[1] == tot_dim, "Should be the same if from same function"
        outputList = list()
        increment = 0
        print('Transforming matrix of shape ',matrix.shape)
        print('Into list of Ndarrays according to output definition')
        for i in range(n_outputs) : 
            flattenedOutput = matrix[...,increment : increment+dim_perOut[i]]
            print('Dimension output ',i, ' is ',dim_perOut[i])
            print('Intermediary shape is :', flattenedOutput.shape)
            shape = sum([[matrix.shape[0]],shapeList[i]],[])
            print('new shape is: ',shape)
            reshapedOutput  = numpy.squeeze(numpy.reshape(flattenedOutput, shape))
            increment       += dim_perOut[i]-1
            outputList.append(reshapedOutput)
            print('Output element ',i,' has shape ', reshapedOutput.shape)
        return outputList

    def _exec(self, X):
        #X = deepcopy(X)
        inputProcessNRVs = self.liftFieldFromKLDistribution(X)
        return self.PythonFunction(*inputProcessNRVs)

    def _exec_sample(self, X):
        #X = deepcopy(X)
        inputProcessNRVs = self.liftFieldFromKLDistribution(X)
        return self.PythonFunctionSample(*inputProcessNRVs)
















#######################################################################################
#######################################################################################
#######################################################################################

class senstivityAnalysisWrapper(object):
    '''Wrapper to easily have access to the different sensitivty indices

    Necessary as the NdGaussianProcessSensitivityAnalysis.getSensitivityAnalysisResults 
    method returns a list of objects, that can be openturns sensitivityAnalysis objects,
    or arrays of those same objects. Thus the original methods can't be directly used.


    Arguments
    ---------
    sensitivityAnalysisList : list
        list of np.arrays of openturns.sensitivityAnalysis objects

    _description : list of strings
        list of all the variable names of the sensitivity analysis
        (name of the Karhunen Loève variables and other RVs)
    '''
    def __init__(self, sensitivityAnalysisList, description):
        self.sensitivityAnalysisList = sensitivityAnalysisList
        self._description            = description
        self._dimInput               = len(self._description)
        self._dimOutput              = len(self.sensitivityAnalysisList)
        self.firstOrderIndices       = None 
   
    def getFirstOrderIndices(self):
        '''Returns a list of the sensitivty indices arrays (Sobol)
        '''
        firstOrderIndicesList = list()
        for i in range(self._dimOutput):
            sensitivityOutput = self.sensitivityAnalysisList[i]
            assert type(sensitivityOutput) == numpy.ndarray , ""
            shape_output      = sensitivityOutput.shape
            flat_shape        = int(numpy.prod(shape_output))
            flattenedOutput   = numpy.reshape(sensitivityOutput, list([flat_shape]))
            tempIndArray      = numpy.empty(list([self._dimInput,flat_shape]))
            for k in range(flat_shape):
                tempIndArray[...,k] = numpy.array(flattenedOutput[k].getFirstOrderIndices())
            reshapeIndicesArray = numpy.reshape(tempIndArray,list([self._dimInput]).extend(list(shape_output)))
            firstOrderIndicesList.append(reshapeIndicesArray)
        self.firstOrderIndices  = firstOrderIndicesList



#######################################################################################
#######################################################################################
#######################################################################################

@atexit.register
def cleanAtExit() :
    dirName = './tempNpArrayMaps'
    try :
        if os.path.isdir(dirName) == True:
            shutil.rmtree(dirName, ignore_errors=True)
        gc.collect()
    except :
        gc.collect()

class FunctionError(Exception):
    pass


'''
import NdGaussianProcessSensitivity as ngps

'''

