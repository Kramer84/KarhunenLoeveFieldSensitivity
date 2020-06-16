import openturns
import numpy 
import os
import gc
import tempfile     #for storing the samples as temporary memmaps
import shutil
from   joblib       import  Parallel, delayed, cpu_count

class NdGaussianProcessConstructor(openturns.Process):
    '''Class to create up to 4-dimensional gaussian processes.

    It has a bit more flexibility than the low level openturns classes, 
    as we can set the different covariance models, the shape of the grid 
    and the trend function. You can instanciate the class through the 
    init arguments, or by using the different set methods.

    Note
    ----
    The class can be created without any arguments in the __init__ function
    and then be set through the different .set methods.

    Parameters
    ----------
    dimension : int, >= 1
        The dimension of the gaussian process
    covariance_model : dict
        dictionary containing the parameters to pass for the model creation.
    grid_shape : list, len(grid_shape) = dimension
        List containing the [[first value, length, number of elements],**] 
        for each dimension
    trend_arguments : list 
        List containg a string variable for each dimension 
        Ex : for 3 dimensions, trend_arguments = ['x','y','z']
    trend_function : str / float
        Argument for the symbolic function, using the arguments defined above.
    verbosity : int or else
        0 to display message on class instance

    Attributes
    ----------
    dimension               : int, >= 1
        The dimension of the gaussian process
    covarianceModel         : openturns.statistics.{modelName}
        openturns model of the covariance
    mesh                    : openturns.geom.Mesh
        grid on which the process is built
    TrendTransform           : openturns.func.TrendTransform
        function to define a trend on the process. If constant
        it is the mean of the process
    GaussianProcess         : openturns.model_process.GaussianProcess
        openturns representation of the final stochastic model
    _covarianceModelDict    : dict
        dictionary containing the parameters to pass for the model creation.
    _grid_shape             : list
        List containing the [[first value, length, number of elements],**] 
    _trendArgs              : list 
        List containg a string variable for each dimension, ex: ['x','y','z']
    _trendFunc              : str / float
        Argument for the symbolic function

    ##########################################################################
    Example : 1D Process
    --------------------
    2 ways of constructing the process :
    => through the .__init__ method 
    P_1D = BuildNdGaussianProcess(
                        dimension           = 1, 
                        grid_shape          = [[0,10,100],],
                        covariance_model    = {'Model':'MaternModel', 
                                            'amplitude':10000.,
                                            'scale':1.5, 'nu':5/2})
                        trend_arguments     = ['x'],
                        trend_function      = 210000
                                )

    => through the constructor methods
    P_1D = BuildNdGaussianProcess()
    P_1D.setDimension(1)
    P_1D.setGrid([[0,10,100],])
    P_1D.setCovarianceModel({'Model':'MaternModel',
                            'amplitude':10000.,
                            'scale':1.5,'nu':5/2})
    P_1D.setTrend(['x'],210000)   #Constant trend <=> Mean process
    P_1D.setGaussianProcess()
    ##########################################################################
    '''

    def __init__(self, 
                dimension:int      = None, covariance_model:str = None,     
                grid_shape:list    = None, trend_arguments:list = None, 
                trend_function:str = None):
        super().__init__()
        self.dimension                      = dimension
        self._covarianceModelDict           = covariance_model
        self._grid_shape                    = grid_shape
        self.covarianceModel                = None
        self.mesh                           = None
        self._trendArgs                     = trend_arguments
        self._trendFunc                     = trend_function
        self.TrendTransform                 = None
        self.GaussianProcess                = None
        self.KarhunenLoeveResult            = None
        self.sample_map                     = None 
        self.fieldSampleEigenmodeProjection = None
        self.decompositionAsRandomVector    = None
        self._buildIfArgsInInit()

    def _buildIfArgsInInit(self):
        '''To instanciate the function if the arguments are passed through __init__
        '''
        if self._grid_shape             is not None and self.dimension is not None:
            self.setGrid(self._grid_shape)
        if self._covarianceModelDict    is not None and self.dimension is not None:
            self.setCovarianceModel(self._covarianceModelDict)
        if self.mesh           is not None and self.dimension is not None and self._trendArgs is not None and self._trendFunc is not None:
            self.setTrend(self._trendArgs, self._trendFunc)
        if self.mesh           is not None and self.covarianceModel is not None:
            self.setGaussianProcess()
        
    def __help__(self):
        # help method
        helpStr='''
                USAGE:\n\nThis class allows for the creation of multi dimensional stochastic processes.\n
                It is possible de set the dimension of the process, the covariance model, 
                as well as the shape of the grid of the multidimensional process.\n
                Finally, it also possible to set a trend function (can be constant)\n\n
                Usage is as following : \n
                -- First, set the desired dimension of the process with the self.setDimension() method\n
                -- Second, set the shape of the grid mon which the process will the modelled, with the self.setGrid() method:\n
                -- => for 2 dimensions : \n'
                   grid = [[X0 : float, X_length : float, X_steps : int], [Y0 : float, Y_length : float, Y_steps : int]]\n
                   Other dimensions are added through the same list.\n\n
                -- Third, the covariance model is set through the self.setCovarianceModel method. \n
                   Four type of models are availabe : AbsoluteExponential, SquaredExponential, MaternModel, ExponentialModel.\n
                   The arguments of the model; amplitude and scale, (also nu for matern) are passed in the form of a dictionary:\n
                   model args = {"amplitude" : float , "scale" : float , "nu" : float}\n\n
                -- Fourth, a trend function can be set, that has to have the same dimension (at least in the input arguments) 
                than the dimension of the process. The function can be a constant, and so set the mean for the process.\n
                   The trend functions are symbolic, so the input arguments are in this form ["x","y","z"], and the function is an integer or 
                symbolic function using the same parameters than defined in the arguments.\n
                   Finally, the gaussian process is created using the self.setGaussianProcess method, and adapts itself to the presence or not 
                of a trend function.\n\n
                    The class can then be used to generate realizations of the gaussian process.\n\n
                '''
        print(helpStr)

    def __del__(self):
        del(self.GaussianProcess)
        del(self.sample_map)
        del(self.KarhunenLoeveResult)
        del(self.fieldSampleEigenmodeProjection)
        del(self.decompositionAsRandomVector)
        del(self)
        gc.collect()

    def setDimension(self, dim : int):
        '''Function to set the dimension of the process, has to be set first
        either here or through the __init__ arguments
        '''
        assert dim>0, 'the dimension has to be positive and non zero'
        self.dimension = dim 

    def setGrid(self, grid_shape : list):
        '''Function to set the grid on which the process will be defined
        
        Arguments
        ---------
        grid_shape : list
            List containing the lower bound, the length and the number of elements
            for each dimension. Ex: [[x0 , Lx , Nx], **]
        '''
        assert self.dimension is not None, 'first set dimension with self.setDimension method'
        assert type(grid_shape) == list, 'the grids shape has to be in a list'
        assert(len(grid_shape)  == self.dimension and type(grid_shape[0]) is list), 'check self.printHelp method' 
        
        n_intervals     = [grid_shape[i][2]             for i in range(self.dimension)]
        low_bounds      = [grid_shape[i][0]             for i in range(self.dimension)]
        lengths         = [grid_shape[i][1]             for i in range(self.dimension)]
        high_bounds     = [low_bounds[i] + lengths[i]   for i in range(self.dimension)]
        mesherObj       = openturns.IntervalMesher(n_intervals)
        grid_interval   = openturns.Interval(low_bounds, high_bounds)
        mesh            = mesherObj.build(grid_interval)
        mesh.setName(str(self.dimension)+'D_Grid')
        self.mesh       = mesh
        self.setMesh(mesh)

    def setMesh(self, mesh):
        '''Sets the openturns mesh.
        '''
        self.mesh = mesh 

    def setCovarianceModel(self, covarianceModelDict : dict = None, covarianceModel = None):
        '''Function to set the covariance model of the stochastic process. Two possible constructors,
        one with a dictionary of keys representing the model and its parameters, or directly one of openTURNS models
        
        Arguments
        ---------
        covarianceModelDict : dict
            Dictionary containing the name of the process and the parameters:
            {'Model' : str, ,'amplitude':float, 'scale':float, 'nu':float}
        covarianceModel : openturns.statistics.CovarianceModels*
            openTURNS covariance model object, already constructed
        '''
        OTCovarModels = {'AbsoluteExponential'            : (openturns.AbsoluteExponential,           ['spatialDim','scale','amplitude'], [1, 2, 23]),          
                         'SquaredExponential'             : (openturns.SquaredExponential,            ['spatialDim','scale','amplitude'],[1, 2, 23] ),    
                         'MaternModel'                    : (openturns.MaternModel,                   ['spatialDim','scale','amplitude' ,'nu'], [1, 24, 234]),
                         'ExponentialModel'               : (openturns.ExponentialModel,              ['spatialDim','scale','amplitude' ,'spatialCorrelation' ,'spatialCovariance'], [1, 23, 234, 235]),
                         'DiracCovarianceModel'           : (openturns.DiracCovarianceModel,          ['spatialDim','amplitude' ,'spatialCorrelation' ,'spatialCovariance'], [1, 12, 123, 14]),
                         'ExponentiallyDampedCosineModel' : (openturns.ExponentiallyDampedCosineModel,['spatialDim','scale' ,'amplitude' ,'f' ], [1, 234]),     
                         'FractionalBrownianMotionModel'  : (openturns.FractionalBrownianMotionModel, ['scale' ,'amplitude' ,'exponent' ,'eta' ,'rho'], [123, 12345]),
                         'GeneralizedExponential'         : (openturns.GeneralizedExponential,        ['spatialDim', 'scale', 'amplitude', 'p'], [1, 24, 234 ]),        
                         'ProductCovarianceModel'         : (openturns.ProductCovarianceModel,        ['coll'], [1]),
                         'RankMCovarianceModel'           : (openturns.RankMCovarianceModel,          ['inputDimension','variance','basis','covariance'], [1, 23, 42]),
                         'SphericalModel'                 : (openturns.SphericalModel,                ['spatialDim','scale','amplitude','radius'], [1, 23, 234]),            
                         'TensorizedCovarianceModel'      : (openturns.TensorizedCovarianceModel,     ['coll'], [1]),
                         'UserDefinedCovarianceModel'     : (openturns.UserDefinedCovarianceModel,    ['mesh','matrix'], [12])
                         }

        if covarianceModel is not None :
            assert type(covarianceModel) in list(zip(*OTCovarModels.values()))[0], str(covarianceModel) + " is not implemented in openturns yet" 
            self.covarianceModel = covarianceModel

        else : 
            self._covarianceModelDict = covarianceModelDict
            assert self.dimension is not None,                     "first set dimension with self.setDimension method"
            assert(covarianceModelDict['Model'] in OTCovarModels), "only use the same model names than those existing in openTURNS"
            print("Assessing the right constructor choice from input dictionary for",covarianceModelDict['Model'])
            dictValues = OTCovarModels[covarianceModelDict['Model']]  #Here we choose the values of one of the keys in OTCovarModels
            constructorPossible = [False]*len(dictValues[2]) #Here we check all the possible constructors
            for i in range(len(dictValues[2])):
                numStr = str(dictValues[2][i])
                for j in range(len(numStr)) :
                    digit = int(numStr[j])-1
                    try : 
                        assert dictValues[1][digit] in covarianceModelDict,""
                        if j == int(len(numStr)-1) :
                            constructorPossible[i] = True
                            break
                    except AssertionError :
                        break
            if any(constructorPossible) == False : #So no constructor is available :
                print('The parameters for the',covarianceModelDict['Model'],'model, only allows these: (check your syntax!)\n'," ".join(dictValues[1]))
            else :
                choice = None
                for k,p in reversed(list(enumerate(constructorPossible))):
                    if p is True :
                        choice = k
                        break 
            constructor = [covarianceModelDict[dictValues[1][int(i)-1]] for i in str(dictValues[2][choice])]
            print('Choosen constructor is: (',", ".join([dictValues[1][int(digit)-1] for digit in str(dictValues[2][choice])]),') => ',constructor)
            try:
                covarianceModel = OTCovarModels[covarianceModelDict['Model']][0](*constructor)
            except:
                print('Please check your input parameters with the openTURNS documentation, as some values have to be points rather than numbers: \n 5 --> [5]  ""brackets!!"" ')
            covarianceModel.setName(covarianceModelDict['Model'])
            self.covarianceModel = covarianceModel

    def setTrend(self, arguments : list, funcOrConst = 0):
        '''Function to set the trend transform of the process

        Note
        ----
        1. If the trend function is constant it only sets the mean of the process
        2. The grid and the dimension have to be already defined

        Arguments
        ---------
        arguments : list 
            List of str, each representing a dimension of the process
        funcOrConst : str / float
            Str or Float representing either the symbolic function of the trend
            or a constant representing the mean
        '''
        funcCst = None
        assert self.mesh is not None, 'first set grid with self.setGrid method'
        assert len(arguments)==self.dimension, 'there have to be as many arguments than dimensions'
        if type(funcOrConst) is list:
            print('function is list')
            funcCst = funcOrConst[0]
            if type(funcCst) is list :
                funcCst = funcCst[0]
        else : 
            funcCst = funcOrConst
        if type(funcCst) in [float, int, str] :
            if type(funcCst) != str : 
                funcCst = str(funcCst)
        assert(type(funcCst) is str), 'function has to be written as a symbolic function' 

        self._trendArgs    = arguments 
        self._trendFunc    = funcCst
        print('trend function args: ',self._trendArgs,' trend function: ', self._trendFunc,'\n')
        print('Please be aware that the number of elements in the argument list has to be the same as the dimension of the process: ', self.dimension)
        symbolicFunction   = openturns.SymbolicFunction(self._trendArgs, [self._trendFunc])
        TrendTransform      = openturns.TrendTransform(symbolicFunction, self.mesh)
        self.TrendTransform = TrendTransform

    def setGaussianProcess(self):
        '''Function to set the Gaussian Process

        Note
        ----
        The grid and the covariance model have to be already defined
        '''
        assert(self.mesh is not None and self.covarianceModel is not None), "first instantiate grid and covariance model"
        if self.TrendTransform is not None :
            self.GaussianProcess = openturns.GaussianProcess(self.TrendTransform, self.covarianceModel, self.mesh)
        else :
            self.GaussianProcess = openturns.GaussianProcess(self.covarianceModel, self.mesh)
        self.GaussianProcess.setName(str(self.dimension)+'D_Gaussian_Process')

## Everything concerning Karhunen Loève
###################################################################################################

    def getKarhunenLoeveDecomposition(self, method = 'P1Algorithm',  threshold = 1e-4, getResult = False, **kwargs):
        '''Function to get the Karhunen Loève decomposition of the gaussian process, using 
        the P1 approximation

        Note
        ----
        Based on the openturns example : Metamodel of a field function
        '''
        assert(self.GaussianProcess is not None and self.mesh is not None), "First create process"
        assert method in ['P1Algorithm', 'QuadratureAlgorithm'], "Methods available : 'P1Algorithm', 'QuadratureAlgorithm'"
        if method is 'P1Algorithm' :
            KarhunenLoeveAlgorithm = openturns.KarhunenLoeveP1Algorithm(self.mesh,
                                                                    self.getCovarianceModel(),
                                                                    threshold)

        if method is 'QuadratureAlgorithm' :
            domain = openturns.MeshDomain(self.mesh)
            bounds = openturns.Interval(list(zip(*list(zip(*self.grid_shape))[:-1])))
            covariance = self.getCovarianceModel()
            if 'marginalDegree' in kwargs :
                marginalDegree = kwargs['marginalDegree']
            else :
                marginalDegree = 20 #arbitrary number
                print('Missing kwarg marginalDegree, default fixed at 20')
            print('threshold is :',threshold)
            KarhunenLoeveAlgorithm = openturns.KarhunenLoeveQuadratureAlgorithm(domain,
                                                                            bounds,
                                                                            covariance,
                                                                            marginalDegree,
                                                                            threshold)

        KarhunenLoeveAlgorithm.setName('Karhunen Loeve Metamodel')
        KarhunenLoeveAlgorithm.run()
        KL_algo_results             = KarhunenLoeveAlgorithm.getResult()
        self.KarhunenLoeveResult   = KL_algo_results
        if getResult == True :
            return KL_algo_results

    def getFieldProjectionOnEigenmodes(self, ProcessSample=None):
        ''' for each element of a field sample get the corresponding set of random variables
        '''
        if self.KarhunenLoeveResult is None:
            self.getKarhunenLoeveDecomposition()    
        assert(self.sample_map is not None or ProcessSample is not None), ""
        if ProcessSample is None : 
            ProcessSample       = self.getProcessSampleFromNumpyArray(self.sample_map)
        if type(ProcessSample) == numpy.ndarray :
            ProcessSample = self.getProcessSampleFromNumpyArray(ProcessSample)
        KL_eigenmodes           = self.KarhunenLoeveResult.project(ProcessSample)
        self.fieldSampleEigenmodeProjection = numpy.array(KL_eigenmodes)

    def getDecompositionAsRandomVector(self, optName = None):
        '''Builds a random vector representative of the gaussian process

        the random vector is from the class : RandomNormalVector defined below
        '''
        assert self.fieldSampleEigenmodeProjection is not None ,""
        if type(self.getName()) == str :
            optName = self.getName()
        meanDistri  = self.fieldSampleEigenmodeProjection.mean(axis=0)
        stdDistri   = self.fieldSampleEigenmodeProjection.std(axis=0)
        eigenmodesDistribution = RandomNormalVector(len(meanDistri), optName)
        eigenmodesDistribution.setMean(meanDistri)
        eigenmodesDistribution.setStd(stdDistri)
        self.decompositionAsRandomVector = eigenmodesDistribution
        cleanAtExit() #to remove temporary arrays

    def liftDistributionToField(self, randomVector):
        '''transforms a randomVector or collection of random vectors into gaussian fields
        '''
        try : 
            assert self.KarhunenLoeveResult is not None, "first run self.getKarhunenLoeveDecomposition()"
        except :
            self.getKarhunenLoeveDecomposition()
        if type(randomVector[0]) == float : # if it is only one realisation
            field = self.KarhunenLoeveResult.liftAsField(randomVector)
            return field 
        elif type(randomVector[0]) == list: # if it is a collection of realisations
            dimension = len(randomVector)
            field_list = [self.KarhunenLoeveResult.liftAsField(randomVector[i]) for i in range(dimension)]
            process_sample = openturns.ProcessSample(self.mesh, 0, dimension)
            [process_sample.add(field_list[i]) for i in range(len(field_list))]
            return process_sample

## Generic over-loaded methods
###################################################################################################
    def getProcessSampleFromNumpyArray(self, ndarray:numpy.ndarray=None)->openturns.ProcessSample:
        '''Transform an numpy array containing samples into openturns ProcessSample
        ''' 
        arr_shape = list(ndarray[0,...].shape)
        dimension = len(arr_shape)
        assert(dimension<5), "dimension can not be greater than 4 \n=> NotImplementedError"
        field_list = [openturns.Field(self.mesh,numpy.expand_dims(ndarray[i,...].flatten(order='C'), axis=1).tolist()) for i in range(ndarray.shape[0])]
        process_sample = openturns.ProcessSample(self.mesh, 0, dimension)
        [process_sample.add(field_list[i]) for i in range(len(field_list))]
        return process_sample


    def getMeanProcess(self):
        '''Generic function to get the mean of the model
        '''
        try: 
            mean_process = float(*self._trendFunc)
        except:
            mean_process = 0
        return mean_process

    # overloading of methods
    def getCovarianceModel(self):
        '''Accessor to the covariance model.
        '''
        return self.covarianceModel
    
    def getSample(self, size : int,  getAsArray = False):
        '''Get n realizations of the process.
        '''
        assert(self.GaussianProcess is not None),""
        sample_ot = self.GaussianProcess.getSample(size)
        self.sample_map = np_as_tmp_map(numpy.asarray(sample_ot))
        if getAsArray == True : 
            return numpy.asarray(self.sample_map) 
        else :
            return sample_ot

    def setSample(self, npSample):
        '''Set a sample, used to construct the random vector
        of Karhunen Loeve coefficients
        '''
        self.sample_map = np_as_tmp_map(npSample)

    def getRealization(self, getAsArray = False):
        '''Get a realization of the process.
        '''
        assert(self.GaussianProcess is not None),""
        realization = self.GaussianProcess.getRealization()
        if getAsArray == True : 
            return numpy.asarray(realization.getValues())
        else :
            return realization

    def getTrend(self):
        '''Accessor to the trend.
        '''
        return self.TrendTransform

    def getMesh(self):
        '''Get the mesh.
        '''
        return self.mesh

    def getDimension(self):
        '''Get the dimension of the domain D.
        '''
        return self.dimension 



## Class representing a vector of normal distributions, all with 1 as variance but centered around 
## independent means
###################################################################################################
###################################################################################################
###################################################################################################

class RandomNormalVector(openturns.PythonRandomVector):
    '''class holding a random normal vector, used to get relaizations that can later be transformed
    in random fields
    '''
    def __init__(self, n_modes : int, optName = None):
        super(RandomNormalVector, self).__init__(n_modes)
        self.nameList = None
        self.n_modes  = int(n_modes)
        self.mean     = numpy.zeros(n_modes)
        self.stdDev   = numpy.ones(n_modes)
        self.setDescription(self.getNameList(optName))

    def setMean(self, meanArr):
        self.mean = meanArr

    def setStd(self, stdArr):
        self.stdDev = stdArr

    def getNameList(self, optName = None):
        nameStr     = 'xi_'
        if optName is not None :
                nameStr = str(optName)+nameStr
        namelist    = list()
        [namelist.append(nameStr+str(i)) for i in range(self.n_modes)]
        self.nameList = namelist
        return namelist

    def getRealization(self):
        numpy.random.seed()
        X=numpy.random.normal(self.mean,self.stdDev,self.n_modes)
        return X.tolist()

    def getSample(self, size):
        numpy.random.seed()
        means = numpy.tile(numpy.expand_dims(self.mean, axis=0),[size,1])
        stds  = numpy.tile(numpy.expand_dims(self.stdDev, axis=0),[size,1])
        X     = numpy.random.normal(means,stds,[size,self.n_modes])      
        return X.tolist()

    def getMean(self):
        return self.mean.tolist()

    def getCovariance(self):
        return numpy.eye(self.n_modes).tolist()

    def getRandVectorAsDict(self):
        randDict = dict()
        for key, item1, item2 in zip(self.getDescription(), self.mean, self.stdDev) :
            randDict[key] = [item1, item2]
        return randDict

    def getRandVectorAsOtNormalsList(self):
        randDict = self.getRandVectorAsDict()
        listOtNormals = list()
        for key in list(randDict.keys()):
            listOtNormals.append(openturns.Normal(randDict[key][0], randDict[key][1]))
        return listOtNormals


## Redifinition of the normal distribution, random number generation with numpy seems more optimal
###################################################################################################
###################################################################################################
###################################################################################################

class NormalDistribution(openturns.dist_bundle2.Normal):
    '''class defining the normal distribution
    '''
    def __init__(self,mu : float, sigma : float, name = None, seed = None):
        super(NormalDistribution, self).__init__(mu, sigma)
        self.variance   = sigma
        self.mean       = mu
        self.sample     = None
        self.seed       = seed
        self.setName(name)

    def getRealization(self):
        numpy.random.seed(self.seed)
        X=numpy.random.normal(self.mean,self.variance)
        return float(X)

    def getSample(self, size):
        numpy.random.seed(self.seed)
        X=numpy.random.normal(self.mean,self.variance, size)
        self.sample = X
        return X.tolist()  

## Redifinition of the uniform distribution, random number generation with numpy seems more optimal
###################################################################################################
###################################################################################################
###################################################################################################

class UniformDistribution(openturns.dist_bundle3.Uniform):
    '''class defining the uniform distribution
    '''
    def __init__(self, lower : float, upper : float, name = None, seed = None):
        super(UniformDistribution, self).__init__(lower, upper)
        self.lower      = lower
        self.upper      = upper
        self.sample     = None
        self.seed       = seed
        self.setName(name)

    def getRealization(self):
        numpy.random.seed(self.seed)
        X=numpy.random.uniform(self.lower,self.upper)
        return float(X)

    def getSample(self, size):
        numpy.random.seed(self.seed)
        X=numpy.random.uniform(self.lower,self.upper, size)
        self.sample = X
        return X.tolist()  

def cleanAtExit() :
    dirName = './tempNpArrayMaps'
    try :
        if os.path.isdir(dirName) == True:
            shutil.rmtree(dirName, ignore_errors=True)
        gc.collect()
    except :
        gc.collect()

#######################################################################################
#######################################################################################
#######################################################################################
######
###                To store the samples as a temporary memmap, thus avoiding to 
#                  over-fill RAM - SOON DEPRECIATED

class tempmap(numpy.memmap):
    """
    Extension of numpy memmap to automatically map to a file stored in temporary directory.
    Usefull as a fast storage option when numpy arrays become large and we just want to do some quick experimental stuff.
    """
    def __new__(subtype, dtype=numpy.uint8, mode='w+', offset=0,
                shape=None, order='C'):
        dirName             = './tempNpArrayMaps'
        if os.path.isdir(dirName) == False:
            os.mkdir(dirName)
        ntf                 = tempfile.NamedTemporaryFile(suffix='_tempMap', prefix='npArray_', dir = dirName)
        self                = numpy.memmap.__new__(subtype, ntf, dtype, mode, offset, shape, order)
        self.tempDir        = dirName
        self.temp_file_obj  = ntf
        return self

    def __del__(self):
        try :
            if hasattr(self,'temp_file_obj') and self.temp_file_obj is not None:
                self.temp_file_obj.close()
                del self.temp_file_obj
                gc.collect()
        except :
            gc.collect()

def np_as_tmp_map(nparray):
    tmpmap      = tempmap(dtype=nparray.dtype, mode='w+', shape=nparray.shape)
    tmpmap[...] = nparray
    return tmpmap
#                  
###
#####
#######################################################################################
#######################################################################################
#######################################################################################
