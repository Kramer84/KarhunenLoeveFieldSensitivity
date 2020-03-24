import openturns
import numpy
import NdGaussianProcessConstructor as ngpc
import atexit


class NdGaussianProcessSensitivityAnalysis(object):
    '''Custom class to do sensitivity analysis on complex models
        
        This class will guide through the process of doing sensitivity
        analysis on functions that take as an input random variables
        and Gaussian Processes, and output fields as well as scalars.

        To make the analysis possible, some informations are required in 
        advance, mainly the shape of the inputs, the laws or processes 
        that these inputs follow, as well as the nature of the outputs.
        This informations are added to the sensitivity analysis by using
        the different methods in the right order.
        
        The function indicated can be any python function, but not the 
        openTurnsPythonFunction type, as this is done internaly. The
        function also needs to take as an input the full field, not the 
        modified function taking in decompositions like kahrunen loeve.

        Attributes:
            listfOfProcessesAndDistributions    : nested dict
                dictionary containing the parameters for the inputs
                    dictModel = {'process_1' :
                                 {
                                  'nameProcess'     : str, 
                                    #name to identify variable after analysis
                                  'positionProcess' : int  
                                    #position of argument in function
                                  'shapeGrid'       : list 
                                    #shape of discrete grid the field is defined on
                                  'covarianceModel' : dict 
                                    #dictionary as in NdGaussianProcessConstructor
                                  'trendFunction'   : list 
                                    # [['var1', ...],'symbolicFunctionOrConstant']  
                                 },
                                 variable_2' :
                                 {
                                  'nameRV'       : str,
                                  'position'     : int,
                                  'meanAndStd'   : list
                                 },
                                }
            outputVariables   : nested dict
                dictionary containg the parameters for the output processes
                    dictModel = {'outputField_1' :
                                 {
                                  'nameField'    : str,
                                  'position'     : int, #position in output
                                  'shape'        : list
                                 },
                                }
 funcModel         : pythonFunction
                python function of our model, taking as input RVs and processes
    '''
    def __init__(self, listfOfProcessesAndDistributions : list, 
                outputVariables : dict, funcSample, funcSolo, sampleSize : int):
        self.listfOfProcessesAndDistributions = listfOfProcessesAndDistributions
        self.outputVariables  = outputVariables
        self.functionSample   = funcSample
        self.functionSolo     = funcSolo
        self.inputDictionary  = self.setInputsFromNormalDistributionsAndNdGaussianProcesses(self.listfOfProcessesAndDistributions)
        self.wrappedFunction  = OpenturnsPythonFunctionWrapper(self.functionSample,
                                                               self.functionSolo, 
                                                               self.inputDictionary,
                                                               self.outputVariables)
        self.sampleSize          = sampleSize
        self.errorWNans          = 0
        self.sobolBatchSize      = None
        self._inputDesignNC      = None
        self.inputDesign         = None
        self._outputDesignListNC = None 
        self.outputDesignList    = None 
        self._designsWErrors     = None



    def makeExperiment(self):
        self.prepareSobolIndicesExperiment()
        self.getOutputDesignAndPostprocess()

    def prepareSobolIndicesExperiment(self):
        composedDistribution    = self.wrappedFunction.KLComposedDistribution
        size                    = self.sampleSize
        sobolExperiment         = openturns.SobolIndicesExperiment(composedDistribution, size)
        inputDesign             = sobolExperiment.generate()
        inputDesign.setDescription(self.wrappedFunction.getInputDescription())
        sobolBatchSize          = len(inputDesign)
        print('len(inputDesign) = ', sobolBatchSize)
        self.sobolBatchSize     = sobolBatchSize
        self._inputDesignNC     = inputDesign
        self.inputDesign        = inputDesign

    def getOutputDesignAndPostprocess(self):
        self.errorWNans  = 0
        assert self._inputDesignNC is not None, ""
        assert self.functionSample is not None or self.functionSolo is not None , ""
        if self.functionSample is not None : 
            outputDesign     = self.wrappedFunction(self._inputDesignNC)
        else :
          ## We first have to implement the multiprocessing of the single evaluation model
            raise NotImplementedError
        n_outputs = len(self.outputVariables.keys())
        if n_outputs >1 :
            outputDesignList     = [outputDesign[i] for i in range(n_outputs)]
        else : 
            outputDesignList     = [outputDesign]
        self._outputDesignListNC = outputDesignList
        self.outputDesignList    = outputDesignList
        self._postProcessOutputDesign()

    def _postProcessOutputDesign(self):
        '''To check if there are nan values and replace them with new realizations
           We not only have to erase the value with the nan, but all the corresponding
           permutations. 
        '''
        composedDistribution = self.wrappedFunction.KLComposedDistribution
        size                 = self.sampleSize
        dimensionInput       = composedDistribution.getDimension()
        dimensionOutput      = len(self.outputVariables.keys())
        outputList           = self.outputDesignList
        ## We flatten all the realisation of each sample, to check if we have np.nans
        outputMatrixFlattend = self.flattenTupleOfArray(outputList)
        whereNan             = numpy.argwhere(numpy.isnan(outputMatrixFlattend))[...,0]
        columnIdx            = numpy.squeeze(numpy.unique(whereNan))
        n_nans               = len(columnIdx.tolist())
        outputZipped         = [list(a) for a in zip(*outputList)]
        self.errorWNans      += n_nans
        inputArray           = numpy.array(self._inputDesignNC)
        self._designsWErrors = inputArray[columnIdx, ...]
        if n_nans > 0:
            print('There were ',n_nans, ' errors (numpy.nan) while processing, trying to regenerate missing outputs \n')
            for i  in range(n_nans):
                idxInSample               = columnIdx[i]%size
                idx2Change                = numpy.arange(dimensionInput+2)*size + idxInSample
                newInputDes, newOutputDes = self._regenerate_missing_vals_safe()
                outputDesNewZip           = [list(a) for a in zip(*newOutputDes)]
                for q in range(len(idx2Change)):
                    p                   = idx2Change[i]
                    self.inputDesign[p] = newInputDes[q]
                    outputZipped[p]     = outputDesNewZip[q]
            self.outputDesignList   = [np.array(X) for X in zip(*outputZipped)] 
        else :
            print('No errors while processing, the function has returned no np.nan.')

    def _regenerate_missing_vals_safe(self): 
        composedDistribution = self.wrappedFunction.KLComposedDistribution
        exit                 = 1
        tries                = 0
        while exit != 0:
            sobolReg  = openturns.SobolIndicesExperiment(composedDistribution, 1)
            inputDes  = sobolReg.generate()
            outputDes = self.wrappedFunction(inputDes)
            exit      = len(numpy.argwhere(numpy.isnan(numpy.array(outputDes))))
            tries += 1
            if tries == 50 :
                print('Error with function')
                raise FunctionError('The function used does only return nans, done over 50 loops')
        return inputDes, outputDes


    def flattenTupleOfArray(self, tuple):
        '''Flattens a tuple of ndarrays, sharing the same first dimension
        '''
        # Should be ok...
        flatArrayList = list()
        n_tot         = int(numpy.array(tuple[0]).shape[0])
        for array in tuple :
            array     = numpy.array(array) #to be sure
            shapeArr  = array.shape
            dimNew    = int(numpy.prod(shapeArr[1:]))
            arrayFlat = numpy.reshape(array, [n_tot, dimNew])
            flatArrayList.append(arrayFlat)
        flatArray = numpy.hstack(flatArrayList)
        return flatArray

    def getSobolIndiciesKLCoefs(self):
        '''get sobol indices for each element of the output
        As the output should be a list of fields and scalars, this step will
        return a list of scalar sobol indices and field sobol indices
        '''
        size             = self.sampleSize
        inputDesign      = self.inputDesign
        dimensionInput   = len(inputDesign[0])
        dimensionOutput  = len(self.outputVariables.keys())
        outputDesignList = self.outputDesignList
        sensitivityIndicesList = list()
        n_tot  = size*(2+dimensionInput)
        for k in range(dimensionOutput):
            outputDesign      = numpy.array(outputDesignList[k])
            shapeDesign       = outputDesign.shape 
            shapeNew          = int(numpy.prod(shapeDesign[1:]))
            shapeDesResh      = numpy.reshape(outputDesign, [n_tot, shapeNew])
            sensitivityIdList = [openturns.SaltelliSensitivityAlgorithm(inputDesign, numpy.expand_dims(shapeDesResh[...,i], 1), size) for i in range(shapeNew)]
            sensitivityIdArr  = numpy.array(sensitivityIdList)
            endShape          = shapeDesign
            endShape[0]       = dimensionInput
            sensitivityField  = numpy.reshape(sensitivityIdArr, endShape)
            print('Shape sensitivity field : ',sensitivityField.shape)
            sensitivityIndicesList.append(sensitivityField)
        return sensitivityIndicesList

    def setInputsFromNormalDistributionsAndNdGaussianProcesses(self, listfOfProcessesAndDistributions):
        '''Function to transform list of Process object into a dictionary, as used in the 
        OpenturnsPythonFunctionWrapper class
        '''
        dictOfInputs        = dict()
        n_inputRvsProcesses = len(listfOfProcessesAndDistributions)
        for i in range(n_inputRvsProcesses):
            if type(listfOfProcessesAndDistributions[i]) == ngpc.NdGaussianProcessConstructor :
                process = listfOfProcessesAndDistributions[i]
                varDict = {
                           'nameProcess'     : process.getName(), 
                             #name to identify variable after analysis
                           'position' : i ,
                             #position of argument in function
                           'shapeGrid'       : process._grid_shape ,
                             #shape of discrete grid the field is defined on
                           'covarianceModel' : process._covarianceModelDict,
                             #dictionary as in NdGaussianProcessConstructor
                           'trendFunction'   : [process._trendArgs, process._trendFunc] 
                             # [['var1', ...],'symbolicFunctionOrConstant']  
                          }
                varName               = 'var' + str(i)
                dictOfInputs[varName] = varDict
            elif type(listfOfProcessesAndDistributions[i]) == ngpc.NormalDistribution :
                randomVar             = listfOfProcessesAndDistributions[i]
                varDict               = dict({
                                         'nameRV'     : randomVar.getName(),
                                         'position'   : i,
                                         'meanAndStd' : [randomVar.mean, randomVar.variance]
                                        })
                varName               = 'var' + str(i)
                dictOfInputs[varName] = varDict

            else :
                print('''
                      Make sure that the input processes and RVs are in the right order and are from \n 
                      type NdGaussianProcessConstructor.NdGaussianProcessConstructor or \n   
                      type NdGaussianProcessConstructor.NormalDistribution
                      ''')
                raise TypeError 
        return dictOfInputs


    def setNumberOutputProcesses(self, numberOutputProcesses : int):
        assert numberOutputProcesses >= 0, "has to be positive"
        self.numberOutputProcesses = numberOutputProcesses

    def setNumberOutputRVs(self, numberRVs : int):
        assert numberRVs >= 0, "has to be positive"
        self.numberOutputRVs = numberRVs

    def setFunctionSample(self, wrapFunction):
        '''Python function taking RVs and Processes as samples

        Note
        ----funcModel
        the function's arguments order is the one defined in
        self.InputProcesses and self.InputRVs
        '''
        self.funcModel = wrapFunction

    def shuffleVariablesSobol(self, ):
        pass

    def getDistributionFromKLcoeffs(self, KL_coefs):
        pass

    def KarhunenLoeveSVDAlgorithm(self, numpy_array, process_sample=None, nbModes=15, threshold = 0.0001, centeredFlag = False):
        '''Function to get Kahrunen Loeve decomposition from samples stored in array
        '''
        if process_sample is None :
            process_sample = self.processSampleFromNpArray(numpy_array)
        FL_SVD = openturns.KarhunenLoeveSVDAlgorithm(process_sample, threshold, centeredFlag)
        #FL_SVD.setNbModes(nbModes)
        FL_SVD.run()
        return FL_SVD

    def processSampleFromNpArray(self, numpy_array):
        '''Function to get a field out of a numpy array representing a sample
        
        As it is a sample we will have a collection of fields
        ''' 
        #first build mesh of same size than array, given that each [i,:] is a sample
        arr_shape = list(numpy_array[0,...].shape)
        dimension = len(arr_shape)
        assert(len(arr_shape)<5), "dimension can not be greater than 4 \n=> NotImplementedError"
        #we build a unit grid
        grid_shape = [[0, arr_shape[i], arr_shape[i]] for i in range(dimension)]
        otMesh      = self.buildMesh(grid_shape, dimension)
        field_list = [openturns.Field(otMesh,numpy.expand_dims(numpy_array[i,...].flatten(order='C'), axis=1).tolist()) for i in range(numpy_array.shape[0])]
        process_sample = openturns.ProcessSample(otMesh, 0, dimension)
        [process_sample.add(field_list[i]) for i in range(len(field_list))]
        return process_sample

    def buildMesh(self, grid_shape : list, dimension : int):
        '''Function to set the grid on which the process will be defined
        
        Arguments
        ---------
        grid_shape : list
            List containing the lower bound, the length and the number of elements
            for each dimension. Ex: [[x0 , Lx , Nx], **]
        '''
        n_intervals     = [grid_shape[i][2]-1               for i in range(dimension)]
        low_bounds      = [grid_shape[i][0]                 for i in range(dimension)]
        lengths         = [grid_shape[i][1]-1               for i in range(dimension)]
        high_bounds     = [low_bounds[i] + lengths[i]       for i in range(dimension)]
        mesherObj       = openturns.IntervalMesher(n_intervals)
        grid_interval   = openturns.Interval(low_bounds, high_bounds)
        mesh            = mesherObj.build(grid_interval)
        mesh.setName(str(dimension)+'D_Grid')
        return mesh





variablesDict = {
                 'var1' :
                 {
                  'nameProcess'     : 'E_', 
                    #name to identify variable after analysis
                  'position' : 0 ,
                    #position of argument in function
                  'shapeGrid'       : [[0,1000,100],] ,
                    #shape of discrete grid the field is defined on
                  'covarianceModel' : {
                                  'NameModel' :'MaternModel',
                                  'amplitude' :5000.,
                                  'scale'     :300.,
                                  'nu'        :13/3
                                      },
                    #dictionary as in NdGaussianProcessConstructor
                  'trendFunction'   : [['x'],210000] 
                    # [['var1', ...],'symbolicFunctionOrConstant']  
                 },

                 'var2' :
                 {
                  'nameProcess'     : 'D_', 
                    #name to identify variable after analysis
                  'position' : 1  ,
                    #position of argument in function
                  'shapeGrid'       : [[0,1000,100],] ,
                    #shape of discrete grid the field is defined on
                  'covarianceModel' : {
                                  'NameModel' :'MaternModel',
                                  'amplitude' :.3,
                                  'scale'     :250.,
                                  'nu'        :7.4/3
                                      },
                    #dictionary as in NdGaussianProcessConstructor
                  'trendFunction'   : [['x'],10] 
                    # [['var1', ...],'symbolicFunctionOrConstant']  
                 },

                 'var3' :
                 {
                  'nameRV'     : 'Rho',
                  'position'   : 2,
                  'meanAndStd' : [7850, 250]
                 },

                 'var4' :
                 {
                  'nameRV'     : 'FP',
                  'position'   : 3,
                  'meanAndStd' : [500, 50]
                 },

                 'var5' :
                 {
                  'nameRV'     : 'FN',
                  'position'   : 4,
                  'meanAndStd' : [100, 15]
                 }               
                }

outputDict   = {'out1' :
                {
                 'name'     : 'VonMisesStress',
                 'position' : 0,
                 'shape'    : (100,)  
                }
               }


import RandomBeamGenerationClass as rbgc
functionSample = rbgc.RandomBeam_anastruct.multiprocessBatchField
functionSolo   = rbgc.RandomBeam_anastruct.multiprocessBatchField

class OpenturnsPythonFunctionWrapper(openturns.OpenTURNSPythonFunction):
    def __init__(self, functionSample = functionSample, 
                       functionSolo   = None,  
                       inputDict      = variablesDict, 
                       outputDict     = outputDict):
        self.inputDict              = inputDict
        self.outputDict             = outputDict
        self.PythonFunctionSample   = functionSample
        self.PythonFunction         = functionSolo
        self.NdGaussianProcessList  = list()
        self.inputVarNames          = list()
        self.outputVarNames         = list()
        self._inputVarOrdering      = None
        self.getInputVariablesName()
        self.getOutputVariablesName()
        self.inputDim               = len(self.inputVarNames)
        self.outputDim              = len(self.outputVarNames)
        self.KLComposedDistribution = self.get_KL_decompositionAndRvs()
        self.inputVarNamesKL        = self.getKLDecompositionVarNames()
        self.inputDimKL             = len(self.inputVarNamesKL)
        super(OpenturnsPythonFunctionWrapper, self).__init__(self.inputDimKL, self.outputDim)
        self.setInputDescription(self.inputVarNamesKL)
        self.setOutputDescription(self.outputVarNames)

    def get_KL_decompositionAndRvs(self):
        listOfRandomVars     = list()
        for variable in self._inputVarOrdering :
            listOfRandomVars.extend(self.getListKLDecompoOrRVFromProcessDict(self.inputDict[variable]))
        composedDistribution = openturns.ComposedDistribution(listOfRandomVars)
        return composedDistribution

    def liftKLComposedDistributionAsFieldAndRvs(self, KLComposedDistribution):
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

    def getListKLDecompoOrRVFromProcessDict(self, processDict, nSamples=5000):
        if 'covarianceModel' in processDict :
            nameProcess     = processDict['nameProcess']     
            position        = processDict['position']         
            shapeGrid       = processDict['shapeGrid']   
            covarianceModel = processDict['covarianceModel']         
            trendFunction   = processDict['trendFunction']       
            process = ngpc.NdGaussianProcessConstructor(dimension        = len(shapeGrid),
                                                        grid_shape       = shapeGrid,
                                                        covariance_model = covarianceModel,
                                                        trend_arguments  = trendFunction[0],
                                                        trend_function   = trendFunction[1])
            process.setName(nameProcess)
            _ = process.getSample(int(nSamples))
            process.getFieldProjectionOnEigenmodes()
            process.getDecompositionAsRandomVector()
            processAsRandVect = process.decompositionAsRandomVector.getRandVectorAsOtNormalsList()
            self.NdGaussianProcessList.append([position, process]) #so we mkeep in memory the position in the function arguments
            return processAsRandVect
       
        elif 'meanAndStd' in processDict :
            mu, sigma          = processDict['meanAndStd']
            NormalDistribution = ngpc.NormalDistribution(mu    = mu,
                                                         sigma = sigma,
                                                         name  = processDict['nameRV'])
            return  [NormalDistribution]

    def getInputVariablesName(self):
        sortedKeys = sorted(self.inputDict , key = lambda x : self.inputDict[x]['position'])
        self._inputVarOrdering = sortedKeys
        for key in sortedKeys :
            try : 
                nameInput      = self.inputDict[key]['nameRV']
                self.inputVarNames.append(nameInput)
            except KeyError :
                try :
                    nameInput  = self.inputDict[key]['nameProcess']
                    self.inputVarNames.append(nameInput)
                except :
                    print('Error in your input dictionary')

    def getOutputVariablesName(self): 
        sortedKeys = sorted(self.outputDict , key = lambda x : self.outputDict[x]['position'])
        for key in sortedKeys :
            try : 
                nameOutput = self.outputDict[key]['name']
                self.outputVarNames.append(nameOutput)
            except KeyError :
                print('Error in your output dictionary')

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

    def _exec(self, X):
        inputProcessNRVs = self.liftKLComposedDistributionAsFieldAndRvs(X)
        return self.PythonFunction(*inputProcessNRVs)

    def _exec_sample(self, X):
        inputProcessNRVs = self.liftKLComposedDistributionAsFieldAndRvs(X)
        return self.PythonFunctionSample(*inputProcessNRVs)



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







'''
    def instanceThroughDictionary(self):
        assert(self.inputDictionary is not None),"Please enter dictionary of inputs"
        keyString = list(self.inputDictionary.keys())
        self.setKeysInputProcess(keyString)
        nRv     = 0
        nProc   = 0
        augmentedDictionary = dict() #To add more data to the inputs
        sampleSizes         = list()
        #to make sure each variable (Rv or Field) has the same number of samples
        for key in keyString :
            dim         = len(numpy.squeeze(self.inputDictionary[key]).shape)-1 # -1 to get rid of the sample dimension
            sampleSize  = len(numpy.squeeze(self.inputDictionary[key]))
            augmentedDictionary[key] = {'sample'     : self.inputDictionary[key],
                                        'dimension ' : dim,
                                        'sampleSize' : sampleSize}
            sampleSizes.append(sampleSize)
            if dim>0:
                nRv +=1
            else:
                nProc +=1
        try:
            assert(numpy.min(sampleSizes)==numpy.max(sampleSizes)),"There should be the same sample size for each variable"
        except AssertionError :
            augmentedDictionary = self.normalizeSamples(augmentedDictionary, sampleSizes)

        self.setNumberInputRVs(nRv)
        self.setNumberInputProcesses(nProc)
        self._augmentedDict = augmentedDictionary

    def setKeysInputProcess(self, keyStrings : list):
        assert (type(keyStrings[i])==str for i in range(len(keyStrings))) , "keyString holds the names of the keys in the dictionary."
        self.keyString = keyString


    def normalizeSamples(self, augmentedDictionary, sampleSizes):
        end_len = numpy.min(sampleSizes)
        for key in self.keyString :
            augmentedDictionary[key]['sample']      = augmentedDictionary[key]['sample'][:end_len,...]
            augmentedDictionary[key]['sampleSize']  = end_len
        return augmentedDictionary

'''