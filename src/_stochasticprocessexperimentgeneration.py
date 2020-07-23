__version__ = '0.1'
__author__ = 'Kristof Attila S.'
__date__ = '22.06.20'

import openturns
import numpy


'''Codes for generating samples for the Monte-Carlo experiement. The generation
is done with the prior knowledge about which of the variables is a field or
scalar.

This has little implication on the Latin Hypercube sampling itself, but will 
change the way we shuffle to retrieve the conditional variances.

To know which variable belongs to which type of physical quantity, this class 
works will work exclusively with the 
NdGaussianProcessSensitivity.OpenturnsPythonFunctionWrapper, from which we need 
the KLComposedDistribution attribute, as well as the  inputVarNames and 
inputVarNamesKL. 

This can later be modified to work with other inputs as well.
'''

__all__ = ['StochasticProcessSensitivityExperiment']

class StochasticProcessSensitivityExperiment(object):
    '''Class to generate experiments for the sensitivity analysis.

    This uses the fact that we have fields decomposed in a series of
    random variables (with karhunen loeve), but that only the conditional
    variance knowing all of those variables is needed, and that there is no
    physical meaning to the conditional variance knowing only one of those
    decomposed variables.

    This generation begins similarly to other experiments, generating a big
    sample of size 2*N that we decompose in two samples A and B of size N.
    This generation can be done entirely randomely, or using specific sampling 
    methods, as LHS, LowDiscrepancySequence or SimulatedAnnealingLHS.

    The difference lies in the way we 'shuffle' our two samples. For a classical 
    RV representing a mono-dimensional physical quantitiy, we take the column 
    representing this quantity in the matrix B and replace the correspoding 
    values in A, thus creating a new matrix, that we append to our samples. 
    But in the case of a gaussian field represented by a series of random 
    variables, we take all those variables in B and put them in A, but we do 
    not take them individually, as in a classical experiment.

    Generation options :
    1 : Random
    2 : LHS
    3 : LowDiscrepancySequence
    4 : SimulatedAnnealingLHS
    '''
    genTypes = {1: 'Random', 2: 'LHS', 3: 'LowDiscrepancySequence', 
                                                    4: 'SimulatedAnnealingLHS'}

    def __init__(self, size=None, OTPyFunctionWrapper=None, generationType=1):
        self.OTPyFunctionWrapper = None
        self.N = None
        self._genType = generationType

        self.composedDistribution = None
        self.inputVarNames = list()
        self.inputVarNamesKL = list()

        print('Generation types are:\n1 : Random (default)\n2 : LHS\n\
            3 : LowDiscrepancySequence\n4 : SimulatedAnnealingLHS')
        print('You choose', self.genTypes[self._genType], 'generation')
        if size is not None:
            self.setSize(size)
        if OTPyFunctionWrapper is not None:
            self.setOTPyFunctionWrapper(OTPyFunctionWrapper)
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
        assert (self.OTPyFunctionWrapper is not None) and \
               (self.N is not None), 
                    "Please intialise sample size and PythonFunction wrapper"
        self.generateSample(**kwargs)
        self.getDataFieldAndRV()
        self.getExperiment()
        return self.experimentSample

    def setSize(self, N):
        '''set size of the samples 
        '''
        assert (type(N) is int) and (N > 0), 
                                    "Sample size can only be positive integer"
        if self.N is None:
            self.N = N
        else:
            self.N = N
            self.sample_A = self.sample_B = self.experimentSample = None

    def setOTPyFunctionWrapper(self, OTPyFunctionWrapper):
        '''set the wrapped function from the NdGaussianProcessSensitivity;
        '''
        self.OTPyFunctionWrapper = OTPyFunctionWrapper
        self.inputVarNames = self.OTPyFunctionWrapper.inputVarNames
        self.inputVarNamesKL = self.OTPyFunctionWrapper.inputVarNamesKL
        self.composedDistribution = self.OTPyFunctionWrapper.KLComposedDistribution
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
        N = self.N
        self.experimentSample = numpy.tile(self.sample_A, [2 + n_vars, 1])
        self.experimentSample[N:2 * N, ...] = self.sample_B
        jump = 2 * N
        jumpDim = 0
        for i in range(n_vars):
            self.experimentSample[jump + N*i:jump + N*(i+1), jumpDim:jumpDim + self.dataMixSamples[i]] = \
                self.sample_B[...,                   jumpDim:jumpDim + self.dataMixSamples[i]]
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
        N2 = 2 * self.N
        if method is 1:
            sample = distribution.getSample(N2)
        elif (method is 2) or (method is 4):
            lhsExp = openturns.LHSExperiment(distribution,
                                             N2,
                                             False,  # alwaysShuffle
                                             True)  # randomShift
            if method is 2:
                sample = lhsExp.generate()
            if method is 4:
                lhsExp.setAlwaysShuffle(True)
                if 'SpaceFilling' in kwargs:
                    if kwargs['SpaceFilling'] is 'SpaceFillingC2':
                        spaceFill = openturns.SpaceFillingC2
                    if kwargs['SpaceFilling'] is 'SpaceFillingMinDist':
                        spaceFill = openturns.SpaceFillingMinDist
                    if kwargs['SpaceFilling'] is 'SpaceFillingPhiP':
                        spaceFill = openturns.SpaceFillingPhiP
                        if 'p' in kwargs:
                            if (type(kwargs['p']) is int) or (type(kwargs['p']) is float):
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
                    if kwargs['TemperatureProfile'] is 'GeometricProfile':
                        geomProfile = openturns.GeometricProfile(10.0, 0.95, 2000)  # Default value
                    if kwargs['TemperatureProfile'] is 'LinearProfile':
                        geomProfile = openturns.LinearProfile(10.0, 100)
                else:
                    print("undefined parameter 'TemperatureProfile', setting default GeometricProfile")
                    geomProfile = openturns.GeometricProfile(10.0, 0.95, 2000)
                optimalLHSAlgorithm = openturns.SimulatedAnnealingLHS(
                    lhsExp, geomProfile, spaceFill())
                sample = optimalLHSAlgorithm.generate()
        elif method is 3:
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
        self.sample_A = sample[:self.N, :]
        self.sample_B = sample[self.N:, :]
