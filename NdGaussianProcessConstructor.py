import openturns
import numpy 
import os
import gc
import tempfile #for storing the samples as temporary memmap
import shutil
from   joblib       import Parallel, delayed, cpu_count
import customWraps  as cw

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
    mesh           : openturns.geom.Mesh
        grid on which the process is built
    trendFunction           : openturns.func.TrendTransform
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
                        covariance_model    = {'NameModel':'MaternModel', 
                                            'amplitude':10000.,
                                            'scale':1.5, 'nu':5/2})
                        trend_arguments     = ['x'],
                        trend_function      = 210000,
                        verbosity           = 1
                                )

    => through the constructor methods
    P_1D = BuildNdGaussianProcess(verbosity=1)
    P_1D.setDimension(1)
    P_1D.setGrid([[0,10,100],])
    P_1D.setCovarianceModel({'NameModel':'MaternModel',
                            'amplitude':10000.,
                            'scale':1.5,'nu':5/2})
    P_1D.setTrendFunction(['x'],210000)   #Constant trend <=> Mean process
    P_1D.setGaussianProcess()
    ##########################################################################
    '''

    def __init__(self, dimension  = None, covariance_model = None,     
                grid_shape        = None, trend_arguments  = None, 
                trend_function    = None):
        super().__init__()
        self.dimension                      = dimension
        self._covarianceModelDict           = covariance_model
        self._grid_shape                    = grid_shape
        self.covarianceModel                = None
        self.mesh                           = None
        self._trendArgs                     = trend_arguments
        self._trendFunc                     = trend_function
        self.trendFunction                  = None
        self.GaussianProcess                = None
        self.resultsKarhunenLoeve           = None
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
            self.setTrendFunction(self._trendArgs, self._trendFunc)
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
        del(self.resultsKarhunenLoeve)
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
        '''Set the openturns mesh.
        '''
        self.mesh = mesh 

    def setCovarianceModel(self, model_args : dict):
        '''Function to set the covariance model of the stochastic process.
        
        Arguments
        ---------
        model_args : dict
            Dictionary containing the name of the process and the parameters:
            {'NameModel' : str, ,'amplitude':float, 'scale':float, 'nu':float}
        '''
        self._covarianceModelDict = model_args
        modelDict = {'AbsoluteExponential' : openturns.AbsoluteExponential,
                    'SquaredExponential'   : openturns.SquaredExponential,
                    'MaternModel'          : openturns.MaternModel,
                    'ExponentialModel'     : openturns.ExponentialModel}

        assert self.dimension is not None, 'first set dimension with self.setDimension method'
        assert(model_args['NameModel'] in modelDict), "check the self.printHelp method to check possible covariance models"
        assert("amplitude" in model_args and "scale" in model_args), "check the self.printHelp method"
        assert(model_args['amplitude'] > 0 and model_args['scale'] > 0), "scale and amplitude have to be both positive"
        
        scale       = [model_args['scale']]*self.dimension
        amplitude   = [model_args['amplitude']]
        if  model_args['NameModel'] == 'MaternModel':
            assert("nu" in model_args), 'include the "nu" factor when using the MaternModel'
            assert(model_args['nu'] > 0), "nu factor has to be positive"
            nu = model_args['nu']
            covarianceModel = modelDict[model_args['NameModel']](scale, amplitude, nu)
        else : 
            covarianceModel = modelDict[model_args['NameModel']](scale, amplitude)
        covarianceModel.setName(model_args['NameModel'])
        self.covarianceModel = covarianceModel

    def setTrendFunction(self, arguments : list, funcOrConst):
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
        trendFunction      = openturns.TrendTransform(symbolicFunction, self.mesh)
        self.trendFunction = trendFunction

    def setGaussianProcess(self):
        '''Function to set the Gaussian Process

        Note
        ----
        The grid and the covariance model have to be already defined
        '''
        assert(self.mesh is not None and self.covarianceModel is not None), "first instantiate grid and covariance model"
        if self.trendFunction is not None :
            self.GaussianProcess = openturns.GaussianProcess(self.trendFunction, self.covarianceModel, self.mesh)
        else :
            self.GaussianProcess = openturns.GaussianProcess(self.covarianceModel, self.mesh)
        self.GaussianProcess.setName(str(self.dimension)+'D_Gaussian_Process')

## Everything concerning Karhunen Loève
###################################################################################################

    def getMetamodelProcessKarhunenLoeve(self, threshold = 0.0001, getResult = False):
        '''Function to get the Karhunen Loève decomposition of the gaussian process.

        Note
        ----
        Based on the openturns example : Metamodel of a field function
        '''
        assert(self.GaussianProcess is not None and self.mesh is not None), "First create process"
        KL_algorithm        = openturns.KarhunenLoeveP1Algorithm(self.mesh,
                                                                 self.getCovarianceModel(),
                                                                 threshold)

        KL_algorithm.setName('Karhunen Loeve Metamodel')
        KL_algorithm.run()
        KL_algo_results             = KL_algorithm.getResult()
        self.resultsKarhunenLoeve   = KL_algo_results
        if getResult == True :
            return KL_algo_results

    def getFieldProjectionOnEigenmodes(self, ProcessSample=None):
        ''' for each element of a field sample get the corresponding set of random variables
        '''
        if self.resultsKarhunenLoeve is None:
            self.getMetamodelProcessKarhunenLoeve()    
        assert(self.sample_map is not None or ProcessSample is not None), ""
        if ProcessSample is None : 
            ProcessSample       = self.ndarray2ProcessSample(self.sample_map)
        if type(ProcessSample) == numpy.ndarray :
            ProcessSample = self.ndarray2ProcessSample(ProcessSample)
        KL_eigenmodes           = self.resultsKarhunenLoeve.project(ProcessSample)
        self.fieldSampleEigenmodeProjection = numpy.array(KL_eigenmodes)

    def getDecompositionAsRandomVector(self, optName = None):
        '''Builds a random vector representative of the gaussian process

        the random vector is from the class : RandomNormalVector defined below
        '''
        assert self.fieldSampleEigenmodeProjection is not None ,""
        if type(self.getName()) == str :
            optName = self.getName()
        meanDistri = self.fieldSampleEigenmodeProjection.mean(axis=0)
        stdDistri  = self.fieldSampleEigenmodeProjection.std(axis=0)
        eigenmodesDistribution = RandomNormalVector(len(meanDistri), optName)
        eigenmodesDistribution.setMean(meanDistri)
        eigenmodesDistribution.setStd(stdDistri)
        self.decompositionAsRandomVector = eigenmodesDistribution

    def liftDistributionToField(self, randomVector):
        '''transforms a randomVector or collection of random vectors into gaussian fields
        '''
        try : 
            assert self.resultsKarhunenLoeve is not None, "first run self.getMetamodelProcessKarhunenLoeve()"
        except :
            self.getMetamodelProcessKarhunenLoeve()
        if type(randomVector[0]) == float : # if it is only one realisation
            field = self.resultsKarhunenLoeve.liftAsField(randomVector)
            return field 
        elif type(randomVector[0]) == list: # if it is a collection of realisations
            dimension = len(randomVector)
            field_list = [self.resultsKarhunenLoeve.liftAsField(randomVector[i]) for i in range(dimension)]
            process_sample = openturns.ProcessSample(self.mesh, 0, dimension)
            [process_sample.add(field_list[i]) for i in range(len(field_list))]
            return process_sample


## Generic over-loaded methods
###################################################################################################
    def ndarray2ProcessSample(self, ndarray):
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
        return self.trendFunction

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
    '''class to create a random vector that will mimic the gaussian field
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
    def __init__(self,mu : float, sigma : float, name = None):
        super(NormalDistribution, self).__init__(mu, sigma)
        self.variance   = sigma
        self.mean       = mu
        self.sample     = None
        self.setName(name)

    def getRealization(self):
        numpy.random.seed()
        X=numpy.random.normal(self.mean,self.variance)
        return float(X)

    def getSample(self, size):
        numpy.random.seed()
        X=numpy.random.normal(self.mean,self.variance, size)
        self.sample = X
        return X.tolist()  

## To store the samples as a temporary memmap, thus avoiding to over-fill RAM
###################################################################################################
###################################################################################################
###################################################################################################

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



'''
### example :

import NdGaussianProcessConstructor as ngpc
from importlib import reload

process_1D = ngpc.NdGaussianProcessConstructor()
process_1D.setDimension(1)
process_1D.setGrid([[0,10,100],])
process_1D.setCovarianceModel({'NameModel':'MaternModel','amplitude':10000.,'scale':1.5,'nu':5/2})
process_1D.setTrendFunction(['x'],210000)
process_1D.setGaussianProcess()


import NdGaussianProcessConstructor as ngpc
from importlib import reload
process_1D = ngpc.NdGaussianProcessConstructor(dimension=1,grid_shape=[[0,10,100],],covariance_model={'NameModel':'MaternModel','amplitude':10000.,'scale':1.5,'nu':5/2},trend_arguments=['x'],trend_function=210000)
process_1D.setName('myprocess_')
_=process_1D.getSample(50000)
process_1D.getFieldProjectionOnEigenmodes()
process_1D.getDecompositionAsRandomVector

'''