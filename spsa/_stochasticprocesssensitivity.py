__version__ = '0.1'
__author__ = 'Kristof Attila S.'
__date__  = '22.06.20'

__all__ = ['StochasticProcessSensitivityAnalysis', 
           'OpenturnsPythonFunctionWrapper']

import atexit
import gc
import uuid 
from typing import Callable, List, Tuple, Optional, Any, Union
from copy   import deepcopy, copy

import openturns
try :
    import numpy
except ImportError :
    from .tinynumpy import tinynumpy as numpy
from ._stochasticprocessexperimentgeneration import \
                                    StochasticProcessSensitivityExperiment
from ._stochasticprocesssensitivityindices   import \
                                    SobolIndicesStochasticProcessAlgorithm

class StochasticProcessSensitivityAnalysis(object):
    r"""Custom class to do sensitivity analysis on complex models
        
        This class will guide through the process of doing sensitivity
        analysis on functions that take as an input random variables
        and Gaussian Processes, and output fields as well as scalars.

        To make the analysis possible, some informations are required in
        advance, mainly the shape of the inputs, the laws or processes
        that these inputs follow, as well as the nature of the outputs.
        This informations are added by passing a list of objects defined in
        an other class: spsa.StochasticProcessConstructor

        The function indicated can be any python function, but not the
        openTurnsPythonFunction type, as this is done internaly. The
        function also needs to take as an input the full field, not the
        modified function taking in decompositions like kahrunen loeve.

        Attributes
        ----------
        processesDistributions: list
            list of spsa.StochasticProcessConstructor and scalar
            probabilistic openturns distributions
        outputVariables: nested dict
            dictionary containg the parameters for the output processes
                >>> dictModel = {'outputField_1' :
                >>>              {
                >>>               'name' : str,
                >>>               'position' : int, #position in output
                >>>               'shape' : list
                >>>              },
                >>>             }
        batchFunction: pythonFunction
            function taking as an input samples of rvs (vectors) 
            and samples of fields (ndarrays) for multiprocessing)
        singleFunction: pythonFunction
            function taking as a input rvs (scalars) and fields
        size: int
            size of the sample for the sensitivity analysis

        Notes
        -----
        The functions are expected to malfunction sometimes and to return 
        nan's. In this case, the algorithm automatically regenerates new 
        inputs for the generation and replaces the nan's with new values 

    """
    def __init__(self, processesDistributions : Optional[list] = None ,
                       outputVariables : Optional[dict] = None ,  
                       batchFunction : Optional[Callable] = None ,
                       singleFunction : Optional[Callable] = None ,
                       size : Optional[int] = None):

        self.processesDistributions = processesDistributions   
        self.outputVariables        = outputVariables 
        self.batchFunction          = batchFunction                
        self.singleFunction         = singleFunction               
        self.size                   = size                        
        
        self._wrappedFunction = None   #wrapper around functions passed 
        
        self.sobolBatchSize             = None
        self.inputDesign                = None
        self.outputDesignList           = None 
        self.SensitivityAnalysisResults = None 
        
        self._errorWNans         = 0
        self._inputDesignNC      = None
        self._outputDesignListNC = None  
        self._designsWErrors     = None

        state = self._getState()
        if (state[0] is True) and (state[1] is True) \
                    and ((state[2] is True) or (state[3] is True)) \
                                                and (state[4] is True):
            self._wrapFunc()

    def _wrapFunc(self):
        wrapFunc = OpenturnsPythonFunctionWrapper(
                   batchFunction          = self.batchFunction,
                   singleFunction         = self.singleFunction, 
                   processesDistributions = self.processesDistributions,
                   outputDict             = self.outputVariables)
        self._wrappedFunction = wrapFunc
        print('Program initialised, ready for sensitivity analysis.\
         You can now proceed to prepare the Sobol indices experiment\n')

    def _getState(self):
        B1 = self.processesDistributions!=None
        B2 = self.outputVariables!=None
        B3 = self.batchFunction!=None
        B4 = self.singleFunction!=None
        B5 = self.size!=None
        return B1, B2, B3, B4, B5


    ##########################################################################
    ##################
    #############         Everything concerning the experiment generation and
    ########           model evaluation.
    ######

    def run(self, **kwargs):
        r"""Method to generate the Sobol' Experiment, solve it and correct the 
        output

        Notes
        -----
        This run method can take a multitude of arguments, according to the 
        different sampling strategies. 

            <1> : only SimulatedAnnealingLHS

            <2> : only LowDiscrepancySequence

        Parameters
        ----------
        generationType: int, optional
            order = {1:Random, 2:LHS, 3:LowDiscrepancySequence, 
            4:SimulatedAnnealingLHS}
                Parameter to set the desired sampling strategy
        SpaceFilling: str, optional, <1>                 
            order = {'SpaceFillingC2', 'SpaceFillingMinDist',
            'SpaceFillingPhiP'}
                Parameter to set the space filling strategy, refer to 
                openTURNS documentation
        p: int, optional, <1>
            default 50 => refer to openTURNS documentation for more details
        TemperatureProfile: str, optional, <1>                 
            order = {'GeometricProfile', 'LinearProfile'}
                Refer to openTURNS documentation for more details
        sequence: str, optional, <2>
            order = {'Sobol','Faure','Halton','ReverseHalton','Haselgrove'}
             
        """
        try :
            self.prepareSobolIndicesExperiment(**kwargs)
            self.getOutputDesignAndPostprocess(**kwargs)
        except :
            if self._wrappedFunction is None :
                print('Please initialize the attributes of the class by using\
                 the .set methods')
                state = self._getState()
                attributes = ['list of processes & distributions', 
                              'output variables dictionary', 
                              'batch python function', 
                              'single python function', 'sample size']


    def prepareSobolIndicesExperiment(self, **kwargs):
        '''Method to generate the experiment for the sensitivity analysis.

        Parameters
        ----------
         ~ same as .run() method ~
        '''
        if 'generationType' not in kwargs : generationType = 1 
        else : generationType = kwargs['generationType'] 
        sobolExperiment = StochasticProcessSensitivityExperiment(
                                size = self.size, 
                                OTPyFunctionWrapper = self._wrappedFunction, 
                                generationType = generationType
                                )
        inputDesign             = sobolExperiment.generate(**kwargs)
        sobolBatchSize          = len(inputDesign)
        print('number of samples for sobol experiment =', \
                                                         sobolBatchSize, '\n')
        self.sobolBatchSize     = sobolBatchSize
        self._inputDesignNC     = inputDesign
        print('input design shape is: ',inputDesign.shape)

    def getOutputDesignAndPostprocess(self, **kwargs):
        '''Method to evaluate the model on the experiment. Has to be done 
        once the experiment is generated

        Parameters
        ----------
         ~ same as .run() method ~
        '''
        assert self._inputDesignNC is not None
        assert self.batchFunction is not None \
                or self.singleFunction is not None 
        if self.batchFunction is not None :
            print('DOING EVAL!!!')
            inputDes         = copy(self._inputDesignNC)
            outputDesign     = self._wrappedFunction(inputDes)
        else :
          # We first have to implement the multiprocessing of the single 
          # evaluation model
            print('There was an error...')
            raise NotImplementedError
        n_outputs = len(self.outputVariables.keys())
        if n_outputs >1 :
            outputDesignList = [numpy.array(
                                   outputDesign[i]) for i in range(n_outputs)]
        else : 
            outputDesignList     = [numpy.array(outputDesign)]
        self._outputDesignListNC = outputDesignList
        self._postProcessOutputDesign(**kwargs)

    def _postProcessOutputDesign(self, **kwargs):
        '''To check if there are nan values and replace them with new 
           realizations
           We not only have to erase the value with the nan, but all the 
           corresponding permutations. 

        Parameters
        ----------
         ~ same as .run() method ~
        '''
        outputList = copy(self._outputDesignListNC)
        # flatten all realisation of each sample, to check if we have np.nans
        outputMatrix = self._wrappedFunction.outputListToMatrix(
                        outputList)
        inputArray = numpy.array(deepcopy(self._inputDesignNC)) 
        composedDistribution = self._wrappedFunction.KLComposedDistribution
        N = self.size
        d_implicit = int(inputArray.shape[0]/N)-2                             # Any type of input shape being a multiple > 2 of N
        d_inputKL = composedDistribution.getDimension()                       # dimension of the karhunen loeve composed distribution
        combinedMatrix = numpy.hstack([inputArray, outputMatrix]).copy()      # of size (N_samples, inputDim*outputDim)
        combinedMatrix0 = combinedMatrix.copy()
        whereNan = numpy.argwhere(
                                    numpy.isnan(
                                             deepcopy(combinedMatrix)))[...,0]
        columnIdx            = numpy.atleast_1d(numpy.squeeze(
                                                      numpy.unique(whereNan)))
        print('Columns where nan : ', columnIdx,'\n')
        n_nans               = len(columnIdx)
        self._errorWNans      += n_nans
        self._designsWErrors = inputArray[columnIdx, ...]
        if n_nans > 0:
            print('There were',n_nans, 'errors (numpy.nan) while processing,\
                                     trying to regenerate missing outputs \n')
            for i  in range(n_nans):
                idx2Change   = numpy.arange(d_implicit+2)*N + columnIdx[i]%N
                print('index to change: ',idx2Change)
                newCombinedMatrix = self._regenerate_missing_vals_safe(
                                                                     **kwargs)
                for q in range(len(idx2Change)):
                    p                      = idx2Change[q]
                    combinedMatrix[p, ...] = newCombinedMatrix[q, ...]
            try :
                assert numpy.allclose(
                    combinedMatrix0,combinedMatrix, equal_nan=True) == False,\
                             "after correction they should be some difference"
                assert numpy.isnan(combinedMatrix).any() == False,\
                                                  "should have no nan anymore"
            except : 
                whereNan = numpy.argwhere(numpy.isnan(
                                    deepcopy(
                                        combinedMatrix)))[...,0]
                columnIdx = numpy.atleast_1d(numpy.squeeze(
                                            numpy.unique(
                                                    whereNan)))
                print('columns where still nan', columnIdx)

            print(' - Post replacement assertion passed - \n')
            inputArray  = combinedMatrix[..., :d_inputKL]    
            outputArray = combinedMatrix[..., d_inputKL:]
            inputSample = openturns.Sample(inputArray)
            inputSample.setDescription(
                                  self._wrappedFunction.getInputDescription())
            self.outputDesignList = self._wrappedFunction.matrixToOutputList(
                                                                  outputArray
                                                                  )
            self.inputDesign      = inputSample
        else :
            self.outputDesignList = self._outputDesignListNC
            self.inputDesign      = self._inputDesignNC
            print('\nNo errors while processing, \
                the function has returned no np.nan.\n')

    def _regenerate_missing_vals_safe(self, **kwargs): 
        '''Method to generate and evaluate new samples of size N=1 to
        fill the outputs of the model where we have nans

        Parameters
        ----------
         ~ same as .run() method ~
        '''
        if 'generationType' not in kwargs : generationType = 1 
        else : generationType = kwargs['generationType']         
        composedDistribution = self._wrappedFunction.KLComposedDistribution
        exit                 = 1
        tries                = 0
        while exit != 0:
            sobolReg      = StochasticProcessSensitivityExperiment(
                                                        1, 
                                                        self._wrappedFunction, 
                                                        generationType
                                                        )
            inputDes      = sobolReg.generate(**kwargs)
            outputDes     = self._wrappedFunction(inputDes)
            inputDes      = numpy.array(inputDes).tolist()
            outputDesFlat = self._wrappedFunction.outputListToMatrix(outputDes)
            #should be 0 when there is no nan
            exit  = len(numpy.atleast_1d(numpy.squeeze(
                                    numpy.argwhere(
                                        numpy.isnan(outputDesFlat))[...,0])))
            tries += 1
            if tries == 50 :
                print('Error with function')
                raise FunctionError('The function used does only return nans,\
                    done over 50 loops')

        return numpy.hstack([inputDes, outputDesFlat])

    ##########################################################################
    ##################
    #############          Basic functions to set the attributes of the class
    ######## 
    ######

    def setOutput(self, outputDict : dict) -> None:
        '''set dictionary containing data about the output variables name, 
        their positions in the functions output, as well as their dimension
        '''
        assert type(outputDict) is dict ,""
        s00, s01, s02, s03, s04 = self._getState()
        self.outputVariables    = outputDict       
        s10, s11, s12, s13, s14 = self._getState()
        state0 = (s00 or s10)
        state1 = (s01 or s11)
        state2 = (s02 or s12)
        state3 = (s03 or s13)
        state4 = (s04 or s14) 
        if (state0 is True) \
            and (state1 is True) \
            and ((state2 is True) or (state3 is True)) \
            and (state4 is True):
            self._wrapFunc() 

    def setProcessesDistributions(
                                 self, processesDistributions : List) -> None:
        '''set list of input processes and distributions.

        Notes
        -----
        You can use any openTURNS distribution (Normal, Arcsine...), but if 
        using Stochastic processes you have to use the class contained in 
        StochasticProcessConstructor to create and define them
        '''
        assert type(processesDistributions) is list ,""
        s00, s01, s02, s03, s04 = self._getState()
        self.processesDistributions = processesDistributionsN
        s10, s11, s12, s13, s14 = self._getState()
        state0 = (s00 or s10)
        state1 = (s01 or s11)
        state2 = (s02 or s12)
        state3 = (s03 or s13)
        state4 = (s04 or s14) 
        if (state0 is True) \
            and (state1 is True) \
            and ((state2 is True) or (state3 is True)) \
            and (state4 is True):
            self._wrapFunc() 

    def setSize(self, N : int) -> None:
        '''Method to set the size of the samples to generate
        '''
        assert (type(N) is int) and (N > 0), \
                                         "size can only be a positive integer"
        s00, s01, s02, s03, s04 = self._getState()
        self.size = N
        s10, s11, s12, s13, s14 = self._getState()
        state0 = (s00 or s10)
        state1 = (s01 or s11)
        state2 = (s02 or s12)
        state3 = (s03 or s13)
        state4 = (s04 or s14) 
        if (state0 is True) \
            and (state1 is True) \
            and ((state2 is True) or (state3 is True)) \
            and (state4 is True):
            self._wrapFunc() 

    def setBatchFunc(self, batchFunction : Callable = None) -> None:
        '''method to set a python function taking as an input random variables
        and fields in batches
        '''
        s00, s01, s02, s03, s04 = self._getState()
        self.batchFunction = batchFunction
        s10, s11, s12, s13, s14 = self._getState()
        state0 = (s00 or s10)
        state1 = (s01 or s11)
        state2 = (s02 or s12)
        state3 = (s03 or s13)
        state4 = (s04 or s14) 
        if (state0 is True) \
            and (state1 is True) \
            and ((state2 is True) or (state3 is True)) \
            and (state4 is True):
            self._wrapFunc() 

    def setSingleFunc(self, singleFunction: Callable = None) -> None:
        '''method to set a python function taking as an input random variables
        and fields
        '''
        s00, s01, s02, s03, s04 = self._getState()
        self.singleFunction = singleFunction
        s10, s11, s12, s13, s14 = self._getState()
        state0 = (s00 or s10)
        state1 = (s01 or s11)
        state2 = (s02 or s12)
        state3 = (s03 or s13)
        state4 = (s04 or s14) 
        if (state0 is True) \
            and (state1 is True) \
            and ((state2 is True) or (state3 is True)) \
            and (state4 is True):
            self._wrapFunc() 


    ##########################################################################
    ##################
    #############             Functions to retrieve the sensitivity analysis
    ########              results.
    ######

    def getSensitivityAnalysisResults(
                                     self, method : str = 'Saltelli') -> None:
        '''method to execute the sensitivity analysis and store it's results 
        in a dictionnary

        Parameters
        ----------
        method : str
            string specifying the Sobol' estimator (default : Saltelli)
        '''
        methods = ['Jansen', 'Saltelli', 'Mauntz-Kucherenko', 'Martinez']
        assert method in methods, '''
Argument has to be one of:
['Jansen','Saltelli','Mauntz-Kucherenko','Martinez']'''
        print('You have chosen to use the',method,'method')
        print('There are',len(self.outputDesignList),'distinct outputs')
        sensitivityAnalysisDict = dict()
        for k, key in enumerate(self.outputVariables.keys()):
            sensitivityAnalysis = SobolIndicesStochasticProcessAlgorithm(
                                  self.outputDesignList[k], self.size, method)
            try : 
                subKeys             = list(self.outputVariables[key].keys())
                nameKey             = str()
                for subKey in subKeys : 
                    if ('name' in subKey) or ('Name' in subKey) or (
                        'var' in subKey) or ('descri' in subKey) or (
                        'Var' in subKey):
                        nameKey = subKey
                name = self.outputVariables[key][nameKey]
            except :
                print("'name' key not found in outputVariables dictionary")
                name = 'tempName_'+str(k)
            sensitivityAnalysis.setInputDescription(
                [self.processesDistributions[i].getName() for i in range(
                len(self.processesDistributions))])
            sensitivityAnalysis.setName(name)
            sensitivityAnalysis._runAlgorithm()
            sensitivityAnalysisDict[k] = sensitivityAnalysis
            sensitivityAnalysisDict[name] = sensitivityAnalysis
            print(sensitivityAnalysis)

        self.SensitivityAnalysisResults = sensitivityAnalysisDict

    ##########################################################################
    ##################
    #############      Utility functions, to retrieve the covariance function
    ########       of a field you would only have samples of.
    ######

    def KarhunenLoeveSVDAlgorithm(self, ndarray : numpy.ndarray, 
                                             process_sample  = None, 
                                                threshold      = 0.0001, 
                                                    centeredFlag = False):
        '''Function to get Kahrunen Loeve decomposition from samples stored in 
        an array.
        Allows to get a representation of a field and it's beahaviour only by 
        looking at samples
        '''
        if process_sample is None :
            process_sample = self.getProcessSampleFromNumpyArray(ndarray)
        KLresult = openturns.KarhunenLoeveSVDAlgorithm(process_sample, 
                                                            threshold, 
                                                            centeredFlag)
        KLresult.run()
        return KLresult

    def getProcessSampleFromNumpyArray(
                     self, ndarray : numpy.array) -> openturns.ProcessSample :
        '''Function to get a field out of a numpy array representing a sample
        As it is a sample we will have a collection of fields
        ''' 
        arr_shape = list(ndarray[0,...].shape)
        dimension = len(arr_shape)
        if len(arr_shape)<5 : raise NotImplementedError # Maximum 4 dimensions
        # We build a unit grid
        grid_shape = [[0, arr_shape[i], arr_shape[i]] for i in range(dimension)]
        otMesh      = self.getMesh(grid_shape, dimension)
        field_list = [openturns.Field(
                        otMesh,numpy.expand_dims(
                          ndarray[i,...].flatten(
                            order='C'), axis=1).tolist(
                        )) for i in range(numpy_array.shape[0])]
        process_sample = openturns.ProcessSample(otMesh, 0, dimension)
        [process_sample.add(field_list[i]) for i in range(len(field_list))]
        return process_sample

    def getMesh(self, grid_shape : List[Tuple[float,float,int],], 
                         dimension : Optional[int] = None) -> openturns.Mesh :
        '''Function to set the grid on which the process will be defined
        
        Parameters
        ----------
        grid_shape : list
            List containing the lower bound, the length and the number of 
            elements for each dimension. Ex:
            >>> [[x0 , Lx , Nx], **]
        dimension : int
            dimension of the mesh
        '''
        n_intervals     = [grid_shape[i][2]-1 for i in range(dimension)]
        low_bounds      = [grid_shape[i][0]   for i in range(dimension)]
        lengths         = [grid_shape[i][1]-1 for i in range(dimension)]
        high_bounds     = [low_bounds[i] + lengths[i] for i in range(dimension)]
        mesherObj       = openturns.IntervalMesher(n_intervals)
        grid_interval   = openturns.Interval(low_bounds, high_bounds)
        mesh            = mesherObj.build(grid_interval)
        mesh.setName(str(dimension)+'D_Grid')
        return mesh





##############################################################################
##############################################################################
##############################################################################
######
###
#              Wrapper for the python functions, to translate between the KL
#              coefficients and the stochastic fields 
###
#####
##############################################################################
##############################################################################
##############################################################################

class OpenturnsPythonFunctionWrapper(openturns.OpenTURNSPythonFunction):
    r"""Wrapper for python functions that take as an argument stochastic fields,
    as well as scalar probabilistic distributions. The wrapper creates an 
    interface to a new function, that only takes in scalar values, by 
    decomposing the stochastic processes using Karhunen-Loeve

    Notes
    -----
    This class is used internaly by the StochasticProcessSensitivityAnalysis 
    class, so the order of decomposition of the Processes hs not to be known 
    and all is tracked internaly relatively robustly. 

    Parameters
    ----------
    functionSample: python function
        function taking as an input random fields and scalars, specialy
        optimized for batch processing
    functionSolo: python function
        same function than above, but only working with one sample at once 
    processesDistributions: list 
        list of distributions and processes (processes are defined with the 
        StochasticProcessConstructor)
    outputDict: dict
        dictionary containing data about the outputs of the function. The 
        dimension as well as the output position has to be indicated

    Attributes
    ----------


    """
    def __init__(self, batchFunction : Optional[Callable] = None, 
                       singleFunction : Optional[Callable] = None,
                       processesDistributions : Optional[list] = None, 
                       outputDict : Optional[dict] = None):
        self.processesDistributions = processesDistributions
        self.outputDict = outputDict
        self.PythonBatchFunction = batchFunction
        self.PythonSingleFunction = singleFunction

        self.StochasticProcessList = list()
        self.inputVarNames = list()
        self.outputVarNames = list()
        self.getInputVariablesName()
        self.getOutputVariablesName()
        self.inputDim = len(self.inputVarNames)
        self.outputDim = len(self.outputVarNames)
        self.KLComposedDistribution = self.composeFromProcessAndDistribution(
                                        self.processesDistributions)
        self.inputVarNamesKL = self.getKLDecompositionVarNames()
        self.inputDimKL = len(self.inputVarNamesKL)
        super(OpenturnsPythonFunctionWrapper, self).__init__(
                                              self.inputDimKL, self.outputDim)
        self.setInputDescription(self.inputVarNamesKL)
        self.setOutputDescription(self.outputVarNames)

    def composeFromProcessAndDistribution(self, 
                                          processesDistributions : list,
                                          ntemp : int = 1750):
        r"""Using a list of ordered openturns distributions and custom defined 
        Process with the StochasticProcessConstructor class we construct a 
        vector of scalar distributions, according to the distributions and the 
        Karhunen-Loeve decomposition of the processes.

        Parameters
        ----------
        processesDistributions: list
            list of probabilistic distributions as well as Processes 

        Returns
        -------
        composedDistribution: openturns.ComposedDistribution
            random vector of various probabilistic distributions, in the order 
            of the input list and with Processes decomposed as random 
            variables
        """
        listNames = list()
        listProbabilisticDistributions = list()
        inputList = processesDistributions
        for i, inp in enumerate(inputList) :
            if isinstance(inp, openturns.DistributionImplementation):
                name = inp.getName()
                assert name != 'Unnamed',\
                    "give a name to distributions through setName() method"
                if name in listNames:
                    print('Possible duplicata of 2 variable names...')
                    newName = name+str(i)+'_'+str(uuid.uuid4())
                    print('New random name assigned:\n',name,'-->',newName)
                    name = newName
                    inp.setName(name)
                listNames.append(name)
                listProbabilisticDistributions.extend([inp])


            if isinstance(inp, openturns.ProcessImplementation) :
                name = inp.getName()
                assert name != 'Unnamed',\
                    "Give a name to Processes through setName() method"
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
                self.StochasticProcessList.append([i, inp]) #so we mkeep in memory the position in the function arguments
                listProbabilisticDistributions.extend(processAsRandVect)
        print('Composed distribution built with processes and distributions:','; '.join(listNames))
        composedDistribution = openturns.ComposedDistribution(listProbabilisticDistributions)
        return composedDistribution

    def liftFieldFromKLDistribution(self, KLComposedDistribution):
        '''Function to transform an 2D array of random variables (as we have allways more than one sample)
        into a collection of random variables and fields, according to the Karhunen-Loeve decomposition
        '''
        print('Lifting')
        fieldPositions     = [self.StochasticProcessList[i][0] for i in range(len(self.StochasticProcessList))]
        numberModesFields  = [int(self.StochasticProcessList[i][1].decompositionAsRandomVector.n_modes) for i in range(len(self.StochasticProcessList))]
        fieldNames         = [self.StochasticProcessList[i][1].getName() for i in range(len(self.StochasticProcessList))]
        listInputVars      = list()
        idxStp             = 0
        tempCompo          = numpy.asarray(KLComposedDistribution)
        print('Before loop OK, input dim is',self.inputDim)
        print('fieldPositions are',fieldPositions)
        for k in range(self.inputDim):
            if k in fieldPositions :
                print('wtf')
                idx        = int(numpy.argwhere(numpy.asarray(fieldPositions)==k))
                idxStpPrev = idxStp
                idxStp     += numberModesFields[idx]
                process    = self.StochasticProcessList[fieldPositions[k]][1]
                try :
                    print('idxStpPrev=',idxStpPrev,'idxStp=',idxStp, 'k=',k)
                    field      = process.liftDistributionToField(tempCompo[...,k+idxStpPrev:k+idxStpPrev+idxStp].tolist())
                except Exception as e:
                    print('got exception',e)
                    raise e
                print('lifted')
                field      = numpy.asarray(field).tolist()
                listInputVars.append(field)
                print('Appended')
            else :
                listInputVars.append(tempCompo[...,k+idxStp])
                print('Appended')
        print('Lifted')
        return listInputVars

    def getInputVariablesName(self):
        '''Get the name of the input variables and fields
        '''
        self.inputVarNames.clear()
        for inp in self.processesDistributions :
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
        '''As we don't know how many output variables the function returns, and the dimension of
        each of theme, we analyse it again  
        '''
        outputDict = self.outputDict
        n_outputs  = len(outputDict.keys())
        shapeList  = list()
        for key in outputDict.keys() : 
            shapeList.append(list(outputDict[key]['shape']))
        tot_dim    = 0
        dim_perOut = list()
        for shape in shapeList : 
            size    = numpy.prod(numpy.array(list(shape)))
            tot_dim = tot_dim + size
            dim_perOut.append(size)
        return tot_dim, dim_perOut, shapeList

    def getKLDecompositionVarNames(self):
        '''function to retrieve new set of variable names, once the function
        is depending on the KL decomposition
        '''
        fieldPositions    = [self.StochasticProcessList[i][0] for i in range(len(self.StochasticProcessList))]
        numberModesFields = [int(self.StochasticProcessList[i][1].decompositionAsRandomVector.n_modes) for i in range(len(self.StochasticProcessList))]
        fieldNames        = [self.StochasticProcessList[i][1].getName() for i in range(len(self.StochasticProcessList))]
        RVNames           = [self.StochasticProcessList[i][1].decompositionAsRandomVector.nameList for i in range(len(self.StochasticProcessList))]
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
        outputList    = deepcopy(outputList)
        flatArrayList = list()
        n_tot         = int(numpy.array(outputList[0]).shape[0])
        print('Converting list of outputs into matrix...')
        el = 1
        for array in outputList :
            array     = numpy.array(array) #to be sure
            shapeArr  = array.shape
            print('Output variable', el ,'has shape', array.shape)
            dimNew    = int(numpy.prod(shapeArr[1:]))
            arrayFlat = numpy.reshape(array, [n_tot, dimNew])
            flatArrayList.append(arrayFlat)
            el += 1
        flatArray = numpy.hstack(flatArrayList)
        print('Final matrix shape:', flatArray.shape)

        return flatArray

    def matrixToOutputList(self, matrix):
        '''Tranform the flattened image of the output back into it's original
        shape
        '''
        n_outputs                      = len(self.outputDict.keys())
        tot_dim, dim_perOut, shapeList = self.getTotalOutputDimension()
        assert matrix.shape[1] == tot_dim, "Should be the same if from same function"
        outputList = list()
        increment  = 0
        print('Transforming matrix of shape ',matrix.shape)
        print('Into list of Ndarrays according to output definition...')
        for i in range(n_outputs) : 
            flattenedOutput = matrix[...,increment : increment+dim_perOut[i]]
            shape           = sum([[matrix.shape[0]],shapeList[i]],[])
            reshapedOutput  = numpy.squeeze(numpy.reshape(flattenedOutput, shape))
            increment       += dim_perOut[i]-1
            outputList.append(reshapedOutput)
        return outputList

    def _exec(self, X):
        '''single evaluation, X is a sequence of float, returns a tuple of float,
        arrays and/or matrices, according to the function

        Notes
        -----
        These functions are now using the Karhunen-Loeve decomposition
        '''
        print('calling solo function')
        inputProcessNRVs = self.liftFieldFromKLDistribution(X)
        return self.PythonSingleFunction(*inputProcessNRVs)

    def _exec_sample(self, X):
        '''multiple evaluations, X is a 2-d sequence of float, returns a 2-d sequence of floats,
        ndarrays and/or matrices, according to the function

        Notes
        -----
        These functions are now using the Karhunen-Loeve decomposition
        '''
        print('calling sample function 000')
        inputProcessNRVs = self.liftFieldFromKLDistribution(X)
        print('inputs are:',inputProcessNRVs)

        return self.PythonBatchFunction(*inputProcessNRVs)


class FunctionError(Exception):
    pass
