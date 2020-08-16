import os
import gc
import openturns
import numpy
from collections.abc import Iterable

__version__ = '0.1'
__author__ = 'Kristof Attila S.'
__date__ = '22.06.20'

__all__ = ['StochasticProcessConstructor']


class StochasticProcessConstructor(openturns.Process):
    '''Class to create up to 4-dimensional gaussian processes.

    It has a bit more flexibility than the low level openturns classes,
    as we can set the different covariance models, the shape of the grid
    and the trend function.

    Parameters
    ----------
    dimension: int, >= 1
        The dimension of the gaussian process
    covariance_model: dict
        dictionary containing the parameters to pass for the model creation.
    grid_shape: list, len(grid_shape) = dimension
        List containing the [[first value, length, number of elements],**]
        for each dimension
    trend_arguments: list
        List containg a string variable for each dimension
        Ex: for 3 dimensions, trend_arguments = ['x','y','z']
    trend_function: str / float
        Argument for the symbolic function, using the arguments defined above.
    verbosity: int or else
        0 to display message on class instance

    Attributes
    ----------
    dimension: int, >= 1
        The dimension of the gaussian process
    covarianceModel: openturns.statistics.{modelName}
        openturns model of the covariance
    mesh: openturns.geom.Mesh
        grid on which the process is built
    TrendTransform: openturns.func.TrendTransform
        function to define a trend on the process. If constant
        it is the mean of the process
    GaussianProcess: openturns.model_process.GaussianProcess
        openturns representation of the final stochastic model
    _covarianceMode: dict
        dictionary containing the parameters to pass for the model creation.
    _grid_shape: list
        List containing the [[first value, length, number of elements],**]
    _trendArgs: list
        List containg a string variable for each dimension, ex: ['x','y','z']
    _trendFunc: str / float
        Argument for the symbolic function

    Notes
    -----
    The class can be created without any arguments in the __init__ function
    and then be set through the different .set methods.

    Examples
    --------
    2 ways of constructing the process.
    - Through the .__init__ method:

    >>> # In the case of a 1D process
    >>> P_1D = BuildNdGaussianProcess(
    >>>     dimension        = 1,
    >>>     grid_shape       = [[0,10,100],],
    >>>     covariance_model = {'Model':'MaternModel',
    >>>                      'amplitude':10000.,
    >>>                      'scale':1.5, 'nu':5/2})
    >>>     trend_arguments  = ['x'],
    >>>     trend_function   = 210000)

    - Through the constructor methods:

    >>> P_1D = BuildNdGaussianProcess()
    >>> P_1D.setDimension(1)
    >>> P_1D.setGrid([[0,10,100],])
    >>> P_1D.setCovarianceModel({'Model':'MaternModel',
    >>>                         'amplitude':10000.,
    >>>                         'scale':1.5,'nu':5/2})
    >>> # constant trend == mean of process
    >>> P_1D.setTrend(['x'],210000)
    >>> P_1D.setGaussianProcess()
    '''

    def __init__(self,
                 dimension: int = None, covariance_model: str = None,
                 grid_shape: list = None, trend_arguments: list = None,
                 trend_function: str = None, verbosity: int = 0):
        super().__init__()
        self.dimension = dimension
        self._covarianceModelDict = covariance_model
        self._grid_shape = grid_shape
        self.shape = None
        self.extent = None
        self.covarianceModel = None
        self.mesh = None
        self._trendArgs = trend_arguments
        self._trendFunc = trend_function
        self.TrendTransform = None
        self.GaussianProcess = None
        self.KarhunenLoeveResult = None
        self.sampleBuffer = None
        self.fieldSampleEigenmodeProjection = None
        self.decompositionAsRandomVector = None
        self.verbosity = verbosity
        # threshold for the SVD algorithm
        self.threshold = 1e-3
        self._buildIfArgsInInit()

    def _buildIfArgsInInit(self):
        '''To instanciate the function if the arguments are passed through
        __init__'''
        if self._grid_shape is not None and self.dimension is not None:
            self.setGrid(self._grid_shape)
        if self._covarianceModelDict is not None \
           and self.dimension is not None:
            self.setCovarianceModel(self._covarianceModelDict)
        if self.mesh is not None and self.dimension is not None \
           and self._trendArgs is not None and self._trendFunc is not None:
            self.setTrend(self._trendArgs, self._trendFunc)
        if self.mesh is not None and self.covarianceModel is not None:
            try:
                self.setGaussianProcess()
            except Exception as e:
                print("Exception, Gaussian Process was not set. Log:")
                print(e)

    def __repr__(self):
        if self._grid_shape is not None:
            shape = str([self._grid_shape[i][2] for i in range(
                                                            self.dimension)])
        else:
            shape = 'UNKNOWN'
        if self.covarianceModel is not None:
            covaModel = self.covarianceModel.getName()
        else:
            covaModel = 'UNKNOWN'
        helpStr = '''
HELP:

This class allows for the creation of multi dimensional stochastic processes\n
The process is defined through the specification of:
    - The mesh on top of wich the model is built.
    - The covariance model
    - A trend function (optional), default is '0'
Usage is as following: \n

0. Create the process:

 >>> import spsa
 >>> process=spsa.StochasticProcessConstructor()

1. Set the dimension of the process with the self.setDimension() method:\n

 >>> process.setDimension(2)

2. Set the processes grid shape  with the self.setGrid() method:\n

 >>> #2 dimensional grid with  10 x 10 elements of 1 x 1 unit size: \n'
 >>> process.setGrid([[0,1,10],[0,1,10],])

3. Set Covariance model through the self.setCovarianceModel() method: \n

Note
----
All openTURNS covariance model names are passable as a string to the keywords
of a dictionary describing the processes model
Here the 'MaternModel' is used as an example. For other Models refer to the
openTURNS documentation. https://openturns.github.io/
The covariance model can also be passed as an already constructed openturns
CovarianceModel object

 >>> process.setCovarianceModel(
 >>>   {"Model":"MaternModel",
 >>>   "amplitude": float,
 >>>   "scale": float,
 >>>   "nu": float})

4. Optional. Set trend function through the setTrend() method.

 >>> process.setTrend(arguments = ['X','Y'],
 >>>   trendOrFunc = '-0.5*X+0.5*Y+1')

5. Set the gaussian process using the self.setGaussianProcess() method

 >>> process.setGaussianProcess()

6. Generate single realizations or get samples:

 >>> otField = process.getRealization()
 >>> npField = process.getRealization(True) #output as numpy array
 >>> otSample = process.getSample(N)
 >>> npSample = process.getSample(N,True) #output as numpy array
'''
        reprStr = ' '.join(['Stochastic',
                            'Process',
                            'on',
                            'grid',
                            'of',
                            'shape',
                            shape,
                            'and',
                            covaModel,
                            'covariance',
                            'Model',
                            helpStr])
        return reprStr

    def __del__(self):
        del(self.GaussianProcess)
        del(self.sampleBuffer)
        del(self.KarhunenLoeveResult)
        del(self.fieldSampleEigenmodeProjection)
        del(self.decompositionAsRandomVector)
        del(self)
        gc.collect()

    def setDimension(self, dim: int):
        '''Function to set the dimension of the process, has to be set first
        either here or through the __init__ arguments
        '''
        assert dim > 0, 'the dimension has to be positive and non zero'
        self.dimension = dim

    def setGrid(self, grid_shape: list):
        '''Function to set the grid on which the process will be defined

        Arguments
        ---------
        grid_shape: list
            List containing the lower bound, the length and the number of
            elements for each dimension. Ex: [[x0 , Lx , Nx], **]
        '''
        assert (self.dimension is not None,
                'first set dimension with self.setDimension method')
        assert type(grid_shape) == list, 'the grids shape has to be in a list'
        assert(len(grid_shape) == self.dimension
               and isinstance(grid_shape[0], list),
               'check self.printHelp method')

        self._grid_shape = grid_shape
        n_intervals = [int(grid_shape[i][2]) for i in range(
                                                            self.dimension)]
        low_bounds = [grid_shape[i][0] for i in range(self.dimension)]
        lengths = [grid_shape[i][1] for i in range(self.dimension)]
        high_bounds = [low_bounds[i] + lengths[i] for i in range(
                                                              self.dimension)]
        mesherObj = openturns.IntervalMesher(n_intervals)
        grid_interval = openturns.Interval(low_bounds, high_bounds)
        mesh = mesherObj.build(grid_interval)
        mesh.setName(str(self.dimension)+'D_Grid')
        self.mesh = mesh
        self.setMesh(mesh)
        shape = numpy.squeeze(n_intervals)+1
        self.shape = shape.tolist()
        self.extent = numpy.ravel(numpy.asarray(
                                  list(zip(low_bounds,
                                           high_bounds)))).tolist()

    def setMesh(self, mesh):
        '''Sets the openturns mesh.
        '''
        self.mesh = mesh

    def setCovarianceModel(self,
                           covarianceModelDict: dict = None,
                           covarianceModel=None):
        '''Function to set the covariance model of the stochastic process.
        Two possible constructors, one with a dictionary of keys representing
        the model and its parameters, or directly one of openTURNS models

        Arguments
        ---------
        covarianceModelDict: dict
            Dictionary containing the name of the process and the parameters.
            The name of the parameter has to be the same as used in openturns.
            Order, as well as the number of arguments does not matter.

        covarianceModel: openturns.statistics.CovarianceModels*
            openTURNS covariance model object, already constructed

        Notes
        -----
        Only one of the two constructors is necessary.
        '''

        # This dictionary contains everything 'bout the openturns convariance
        # models.
        # The key is the models name as a string, and the values are the model
        # itself, a list of each argument accepted by the covariance model,
        # a list of ints that contain information about each combination of
        # the arguments and their order. finally a list containing info 'bout
        # if the argument should be in form of a list or a float/int or
        # something else...
        ot = openturns
        OTCovarModels = {
            'AbsoluteExponential':
                (ot.AbsoluteExponential,
                 ['spatialDim', 'scale', 'amplitude'],
                 [1, 2, 23], [0, 1, 1]),
            'SquaredExponential':
                (ot.SquaredExponential,
                 ['spatialDim', 'scale', 'amplitude'],
                 [1, 2, 23], [0, 1, 1]),
            'MaternModel':
                (ot.MaternModel,
                 ['spatialDim', 'scale', 'amplitude', 'nu'],
                 [1, 24, 234], [0, 1, 1, 0]),
            'ExponentialModel':
                (ot.ExponentialModel,
                 ['spatialDim', 'scale',
                  'amplitude', 'spatialCorrelation',
                  'spatialCovariance'],
                 [1, 23, 234, 235], [0, 1, 1, 2, 2]),
            'DiracCovarianceModel':
                (ot.DiracCovarianceModel,
                 ['spatialDim', 'amplitude', 'spatialCorrelation',
                  'spatialCovariance'],
                 [1, 12, 123, 14], [0, 1, 2, 2]),
            'ExponentiallyDampedCosineModel':
                (ot.ExponentiallyDampedCosineModel,
                 ['spatialDim', 'scale', 'amplitude', 'f'],
                 [1, 234], [0, 1, 1, 0]),
            'FractionalBrownianMotionModel':
                (ot.FractionalBrownianMotionModel,
                 ['scale', 'amplitude', 'exponent', 'eta', 'rho'],
                 [123, 12345], [1, 1, 0, 0, 0]),
            'GeneralizedExponential':
                (ot.GeneralizedExponential,
                 ['spatialDim', 'scale', 'amplitude', 'p'],
                 [1, 24, 234], [1, 1, 1, 0]),
            'ProductCovarianceModel':
                (ot.ProductCovarianceModel,
                 ['coll'],
                 [1], [1]),
            'RankMCovarianceModel':
                (ot.RankMCovarianceModel,
                 ['inputDimension', 'variance', 'basis', 'covariance'],
                 [1, 23, 42], [-1, -1, -1, -1]),
            'SphericalModel':
                (ot.SphericalModel,
                 ['spatialDim', 'scale', 'amplitude', 'radius'],
                 [1, 23, 234], [0, 1, 1, 1]),
            'TensorizedCovarianceModel':
                (ot.TensorizedCovarianceModel,
                 ['coll'], [1], [-1]),
            'UserDefinedCovarianceModel':
                (ot.UserDefinedCovarianceModel,
                 ['mesh', 'matrix'], [12], [2, 2])
                             }
        dataTypes = {0: "int/float",
                     1: "list",
                     2: "openTURNS object",
                     -1: "unknown"}

        if covarianceModel is not None:
            modelTypes = list(zip(*OTCovarModels.values()))[0]
            assert type(covarianceModel) in modelTypes, " ".join(
                        [str(covarianceModel),
                         'is', 'not', 'implemented',
                         'in', 'openturns', 'yet'])
            self.covarianceModel = covarianceModel

        else:
            self._covarianceModelDict = covarianceModelDict
            assert self.dimension is not None, (
                   "first set dimension with self.setDimension method")
            assert covarianceModelDict['Model'] in OTCovarModels, (
                   "only use model names existing in openTURNS")

            if self.verbosity > 2:
                print(' '.join(['Assessing',
                                'the',
                                'right',
                                'constructor',
                                'choice',
                                'from',
                                'input',
                                'dictionary',
                                'for',
                                covarianceModelDict['Model']]))
            # Here we choose the values of one of the keys in OTCovarModels
            dictValues = OTCovarModels[covarianceModelDict['Model']]
            # Here we check all the possible constructors
            constructorPossible = [False]*len(dictValues[2])
            for i in range(len(dictValues[2])):
                numStr = str(dictValues[2][i])
                for j in range(len(numStr)):
                    digit = int(numStr[j])-1
                    try:
                        # this assertion does not take in acount if
                        # the letters are capitalized or not
                        # a new class could be implemented that returns
                        # True for (TexT == text)
                        assert dictValues[1][digit] in covarianceModelDict
                        if j == int(len(numStr)-1):
                            constructorPossible[i] = True
                            break
                    except AssertionError:
                        break
            # If no constructor is available:
            if any(constructorPossible) is False:
                print('The parameters for the',
                      covarianceModelDict['Model'],
                      'model, only allows these: (check your syntax!)\n',
                      " ".join(dictValues[1]))
            else:
                choice = None
                for k, p in reversed(list(enumerate(constructorPossible))):
                    if p is True:
                        choice = k
                        break
            constructor = list()
            # Beacause a string is an iterator and we iterate over each letter
            for i in str(dictValues[2][choice]):
                constructor.append([covarianceModelDict[
                                                    dictValues[1][int(i)-1]]])
            try:
                if self.verbosity > 1:
                    print('Choosen constructor is: (',
                          ", ".join(
                            [dictValues[1][int(digit)-1] for digit in str(
                             dictValues[2][choice])]),
                          ') => ',
                          constructor)
                    print('''
WARNING: if your process is multidimensional, some inputs for the covariance
model have to be multidimensional''')
                    print('''
For example, the scale parameter has the same number of arguments than the
dimension.''')
                    print('''
If you forget it, you will have no error message, it just won't work''')
                covarianceModel = OTCovarModels[
                                    covarianceModelDict['Model']][0](
                                                            *constructor)
            except Exception as e:
                print(''.join([' ']*10), ' ___    ERROR    ___')
                print(''.join([' ']*10), '        |  |     ')
                print(''.join([' ']*10), '       \\|  |/   ')
                print(''.join([' ']*10), '        \\  /    ')
                print(''.join([' ']*10), '         \\/  \n ')
                print('''
Please check your input parameters with the openTURNS documentation,
some values have to be points (with brackets) rather than numbers.
5 ==> [5]''')
                print('''
in the case of some parameters as scale and amplitude, the values have to be
points (so with brackets) 5 ==> [5]''')
                print('For this constructor the types should be:\n',
                      ", ".join([dictValues[1][int(digit)-1]
                                + ': '
                                + dataTypes[
                                    dictValues[3][
                                        int(digit)-1]] for digit in str(
                                                    dictValues[2][choice])]))
                return None
            covarianceModel.setName(covarianceModelDict['Model'])
            self.covarianceModel = covarianceModel
        try:
            self.setGaussianProcess()
        except Exception as e:
            print("Gaussian Process not set. Verifiy parameters")
            print(e)

    def setTrend(self, arguments: Iterable, funcOrConst=0):
        '''Function to set the trend transform of the process

        Notes
        -----
        1. If the trend function is constant it only sets the mean of the
           process
        2. The grid and the dimension have to be already defined

        Parameters
        ----------
        arguments: list
            List of str, each representing a dimension of the process
        funcOrConst: str / float
            Str or Float representing either the symbolic function of the
            trend or a constant representing the mean
        '''
        funcCst = None
        assert self.mesh is not None, 'set grid with self.setGrid method'
        assert len(arguments) == self.dimension, (
                'there have to be as many arguments than dimensions')
        if isinstance(funcOrConst, Iterable):
            print('function is list')
            funcCst = funcOrConst[0]
            if isinstance(funcCst, Iterable):
                funcCst = funcCst[0]
        else:
            funcCst = funcOrConst
        if isinstance(funcCst, (float, int, str)):
            if not isinstance(funcCst, str):
                funcCst = str(funcCst)
        assert isinstance(funcCst, str), (
                'function has to be written as a symbolic function')
        self._trendArgs = arguments
        self._trendFunc = funcCst
        if self.verbosity > 0:
            print('trend function args:', self._trendArgs,
                  'trend function:', self._trendFunc)
        if self.verbosity > 1:
            print('''
Please be aware that the number of elements in the argument list has to be
the same as the dimension of the process: ''',
                  self.dimension)
        symbolicFunction = openturns.SymbolicFunction(self._trendArgs,
                                                      [self._trendFunc])
        TrendTransform = openturns.TrendTransform(symbolicFunction, self.mesh)
        self.TrendTransform = TrendTransform

    def setGaussianProcess(self):
        '''Function to set the Gaussian Process

        Note
        ----
        The grid and the covariance model have to be already defined
        '''
        assert (self.mesh is not None and self.covarianceModel is not None), (
                "first instantiate grid and covariance model")
        if self.TrendTransform is not None:
            if self.verbosity > 1:
                print('Creating Gaussian Process with trend transform ...')
            self.GaussianProcess = openturns.GaussianProcess(
                                                self.TrendTransform,
                                                self.covarianceModel,
                                                self.mesh)
        else:
            self.GaussianProcess = openturns.GaussianProcess(
                                              self.covarianceModel, self.mesh)
        self.GaussianProcess.setName(str(self.dimension)+'D_Gaussian_Process')

# Everything concerning Karhunen Loève
##############################################################################

    def getKarhunenLoeveDecomposition(self, method='P1Algorithm',
                                      threshold=1e-4, getResult=False,
                                      **kwargs):
        '''Function to get the Karhunen Loève decomposition of the gaussian
        process, using the P1 approximation

        Notes
        -----
        Based on the openturns example: Metamodel of a field function
        '''
        try:
            assert(
                self.GaussianProcess is not None and self.mesh is not None), (
                "First create process")
        except AssertionError:
            if self.mesh is None:
                raise Exception("Can't create process without grid")
            if self.GaussianProcess is None:
                self.setGaussianProcess()
        assert method in ['P1Algorithm', 'QuadratureAlgorithm'], (
                "Methods available: 'P1Algorithm', 'QuadratureAlgorithm'")
        if method == 'P1Algorithm':
            KLAlgorithm = openturns.KarhunenLoeveP1Algorithm(
                            self.mesh,
                            self.getCovarianceModel(),
                            threshold)

        if method == 'QuadratureAlgorithm':
            domain = openturns.MeshDomain(self.mesh)
            bounds = openturns.Interval(
                        list(zip(*list(zip(*self.grid_shape))[:-1])))
            covariance = self.getCovarianceModel()
            if 'marginalDegree' in kwargs:
                marginalDegree = kwargs['marginalDegree']
            else:
                marginalDegree = 20  # arbitrary number
                print('Missing kwarg marginalDegree, default fixed at 20')
            print('threshold is:', threshold)
            KLAlgorithm = openturns.KarhunenLoeveQuadratureAlgorithm(
                            domain,
                            bounds,
                            covariance,
                            marginalDegree,
                            threshold)
        KLAlgorithm.setName('Karhunen Loeve Metamodel')
        KLAlgorithm.run()
        KL_algo_results = KLAlgorithm.getResult()
        self.KarhunenLoeveResult = KL_algo_results
        if getResult is True:
            return KL_algo_results

    def getKarhunenLoeveSVDAlgorithmOnSampleResults(self,
                                                    processSample,
                                                    getResult=False):
        '''Build the Karhunen Loeve decomposition using a sample, without
        prior knowledge about the covariance function of the Process'''
        assert self.mesh is not None, "first instanciate dimension and mesh"
        if isinstance(processSample, numpy.ndarray):
            processSample = self.getProcessSampleFromNumpyArray(processSample)
        assert isinstance(processSample, openturns.ProcessSample)
        self.sampleBuffer = processSample
        klDecomposition = openturns.KarhunenLoeveSVDAlgorithm(
                            processSample, self.threshold)
        klDecomposition.run()
        klDecompositionResult = klDecomposition.getResult()
        self.KarhunenLoeveResult = klDecompositionResult
        klDecompositionEigenmodes = self.KarhunenLoeveResult.project(
                                        processSample)
        self.fieldSampleEigenmodeProjection = numpy.array(
                                                   klDecompositionEigenmodes)
        self.getDecompositionAsRandomVector('Unnamed process decomposition')
        if getResult is True:
            return klDecompositionResult

    def getFieldProjectionOnEigenmodes(self, ProcessSample=None):
        ''' for each element of a field sample get the corresponding set of
        random variables'''
        if self.KarhunenLoeveResult is None:
            self.getKarhunenLoeveDecomposition()
        assert(self.sampleBuffer is not None or ProcessSample is not None)
        if ProcessSample is None:
            ProcessSample = self.getProcessSampleFromNumpyArray(
                                self.sampleBuffer)
        if isinstance(ProcessSample, numpy.ndarray):
            ProcessSample = self.getProcessSampleFromNumpyArray(ProcessSample)
        KL_eigenmodes = self.KarhunenLoeveResult.project(ProcessSample)
        self.fieldSampleEigenmodeProjection = numpy.array(KL_eigenmodes)

    def getDecompositionAsRandomVector(self, optName=None):
        '''Builds a random vector representative of the gaussian process

        Notes
        -----
        Random vector is from the class: RandomNormalVector defined below
        '''
        assert self.fieldSampleEigenmodeProjection is not None
        if isinstance(self.getName(), str):
            optName = self.getName()
        meanDistri = self.fieldSampleEigenmodeProjection.mean(axis=0)
        stdDistri = self.fieldSampleEigenmodeProjection.std(axis=0)
        eigenmodesDistribution = RandomNormalVector(len(meanDistri), optName)
        eigenmodesDistribution.setMean(meanDistri)
        eigenmodesDistribution.setStd(stdDistri)
        self.decompositionAsRandomVector = eigenmodesDistribution
        self.sampleBuffer = None

    def liftDistributionToField(self, randomVector):
        '''transforms a randomVector or collection of random vectors
        into one or multiple gaussian fields
        '''
        try:
            assert self.KarhunenLoeveResult is not None, (
                    "first run self.getKarhunenLoeveDecomposition()")
        except Exception as e:
            self.getKarhunenLoeveDecomposition()
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

# Generic over-loaded methods
##############################################################################
    def getProcessSampleFromNumpyArray(self,
                                       ndarray: numpy.ndarray = None) -> (
                                        openturns.ProcessSample):
        '''Transform an numpy array containing samples into openturns
        ProcessSample objects
        '''
        arr_shape = list(ndarray[0, ...].shape)
        dimension = len(arr_shape)
        assert(dimension < 5), (
            "dimension can not be greater than 4 \n=> NotImplementedError")
        field_list = list()
        for i in range(ndarray.shape[0]):
            field_list.append(openturns.Field(
                self.mesh, numpy.expand_dims(
                    ndarray[i, ...].flatten(order='C'), axis=1).tolist()))
        process_sample = openturns.ProcessSample(self.mesh, 0, dimension)
        [process_sample.add(field_list[i]) for i in range(len(field_list))]
        return process_sample

    def getMeanProcess(self):
        '''Generic function to get the mean of the model

        Returns
        -------
        mean_process = float
            mean of the process, if there is no non-constant trend function
        '''
        try:
            mean_process = float(*self._trendFunc)
        except Exception:
            mean_process = 0
        return mean_process

    # overloading of methods
    def getCovarianceModel(self):
        '''Accessor to the covariance model.

        Returns
        -------
        self.covarianceModel: openturns.CovarianceModel
            covariance model of the stochastic process
        '''
        return self.covarianceModel

    def getSample(self, size: int,  getAsArray=False):
        '''Get n realizations of the process.

        Parameters
        ----------
        getAsArray: bool
            if flag is set to True, returns the realization as a ndarray and
            not a openturns object
        '''
        if self.GaussianProcess is not None:
            sample_ot = self.GaussianProcess.getSample(size)
            self.sampleBuffer = numpy.asarray(sample_ot)
            if len(self.extent) == 2*len(self.shape):
                self.extent.pop()
                self.extent.pop()
                self.extent.append(self.sampleBuffer.min())
                self.extent.append(self.sampleBuffer.max())
            if getAsArray is True:
                array = numpy.asarray(self.sampleBuffer)
                array = numpy.reshape(array, [size, *self.shape], order='F')
                return array
            else:
                return sample_ot
        elif self.KarhunenLoeveResult is not None:
            if self.decompositionAsRandomVector is not None:
                rvRea = self.decompositionAsRandomVector.getSample(size)
                sample_ot = self.liftDistributionToField(rvRea)
                self.sampleBuffer = numpy.asarray(sample_ot)
                if len(self.extent) == 2*len(self.shape):
                    self.extent.pop()
                    self.extent.pop()
                    self.extent.append(self.sampleBuffer.min())
                    self.extent.append(self.sampleBuffer.max())
                if getAsArray is True:
                    array = numpy.asarray(self.sampleBuffer)
                    array = numpy.reshape(
                                array, [size, *self.shape], order='F')
                    return array
                else:
                    return sample_ot
            else:
                print('Sample could not be created')
                print('Gaussian Process was not set')
                print('Or model not learnt with Karhunen-Loeve')
                return None

    def setSample(self, npSample: numpy.array):
        '''Set a sample, used to construct the random vector of Karhunen Loeve
        coefficients.
        '''
        self.sampleBuffer = npSample

    def getRealization(self, getAsArray=False):
        '''Get a realization of the process.

        Parameters
        ----------
        getAsArray: bool
            if flag is set to True, returns the realization as a ndarray and
            not a openturns object
        '''
        if self.GaussianProcess is not None and self.shape is not None:
            realization = self.GaussianProcess.getRealization()
            if len(self.extent) == 2*len(self.shape):
                self.extent.pop()
                self.extent.pop()
                self.extent.append(realization.getValues().getMin()[0])
                self.extent.append(realization.getValues().getMax()[0])
            if getAsArray is True:
                array = numpy.asarray(realization.getValues())
                return numpy.reshape(array, self.shape, order='F')
            else:
                return realization
        elif self.KarhunenLoeveResult is not None:
            if self.decompositionAsRandomVector is not None:
                rvRea = self.decompositionAsRandomVector.getRealization()
                sample_ot = self.liftDistributionToField(rvRea)
                sample_np = numpy.asarray(sample_ot)
                if len(self.extent) == 2*len(self.shape):
                    self.extent.pop()
                    self.extent.pop()
                    self.extent.append(sample_np.min())
                    self.extent.append(sample_np.max())
                if getAsArray is True:
                    array = numpy.asarray(sample_np)
                    array = numpy.reshape(array, self.shape, order='F')
                    return array
                else:
                    return sample_ot
            else:
                print('Realization could not be generated. Reasons:')
                print('Gaussian Process was not set')
                print('Model was not learnt with Karhunen-Loeve')
                return None

    def getTrend(self):
        '''Accessor to the trend.


        Returns
        -------
        self.TrendTransform: openturns.TrendTransform
        '''
        return self.TrendTransform

    def getMesh(self):
        '''Get the mesh.

        Returns
        -------
        self.mesh: openturns.Mesh
        '''
        return self.mesh

    def getDimension(self):
        '''Get the dimension of the domain D.

        Returns
        -------
        self.dimension: int
            dimension of the process
        '''
        return self.dimension

# Class representing a vector of normal distributions, all with 1 as variance,
# but centered around independent means
##############################################################################


class RandomNormalVector(openturns.PythonRandomVector):
    '''class holding a random normal vector, used to get relaizations that can
    later be transformed into random fields
    '''
    def __init__(self, n_modes: int, optName=None):
        super(RandomNormalVector, self).__init__(n_modes)
        self.nameList = None
        self.n_modes = int(n_modes)
        self.mean = numpy.zeros(n_modes)
        self.stdDev = numpy.ones(n_modes)
        self.setDescription(self.getNameList(optName))

    def setMean(self, meanArr):
        self.mean = meanArr

    def setStd(self, stdArr):
        self.stdDev = stdArr

    def getNameList(self, optName=None):
        nameStr = 'xi_'
        if optName is not None:
            nameStr = str(optName) + nameStr
        namelist = list()
        [namelist.append(nameStr + str(i)) for i in range(self.n_modes)]
        self.nameList = namelist
        return namelist

    def getRealization(self):
        numpy.random.seed()
        X = numpy.random.normal(self.mean, self.stdDev, self.n_modes)
        return X.tolist()

    def getSample(self, size):
        numpy.random.seed()
        means = numpy.tile(numpy.expand_dims(self.mean, axis=0), [size, 1])
        stds = numpy.tile(numpy.expand_dims(self.stdDev, axis=0), [size, 1])
        X = numpy.random.normal(means, stds, [size, self.n_modes])
        return X.tolist()

    def getMean(self):
        return self.mean.tolist()

    def getCovariance(self):
        return numpy.eye(self.n_modes).tolist()

    def getRandVectorAsDict(self):
        randDict = dict()
        for key, item1, item2 in zip(self.getDescription(),
                                     self.mean, self.stdDev):
            randDict[key] = [item1, item2]
        return randDict

    def getRandVectorAsOtNormalsList(self):
        randDict = self.getRandVectorAsDict()
        listOtNormals = list()
        for key in list(randDict.keys()):
            listOtNormals.append(openturns.Normal(randDict[key][0],
                                                  randDict[key][1]))
        return listOtNormals
