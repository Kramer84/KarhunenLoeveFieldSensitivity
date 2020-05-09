import openturns 
import numpy 


'''Here we are going to generate samples for the Monte-Carlo experiement,
knowing that the variables that we are generating are a mix of random-variables
representing Physical variables and random-variables used to reconstruct stochastic
field. 
This has little implication of the Latin Hypercube sampling itself, but will change the
way we shuffle to retrieve the conditional variances.

To know which variable belongs to which type of physical quantity, this class works will
work exclusively with the NdGaussianProcessSensitivity.OpenturnsPythonFunctionWrapper
, from which we need the KLComposedDistribution attribute, as well as the inputVarNames
and inputVarNamesKL. Later, this can later be modified to work with other inputs as well.
'''


class NdGaussianProcessExperiment(object):
    '''Class to generate experiments for the sensitivity analysis.

    This uses the fact that we have fields decomposed in a series of
    random variables (with karhunen loeve), but that only the conditional
    variance knowing all of those variables is needed, and that there is no
    physical meaning to the conditional variance knowing only one of those
    decomposed variables.

    This generation begins similarly to other experiments, generating a big
    sample of size 2*N that we decompose in two samples A and B of size N.
    This generation can be done entirely randomely, or using specific sampling methods,
    as LHS, LowDiscrepancySequence or SimulatedAnnealingLHS.

    The difference lies in the way we 'shuffle' our two samples. For a classical RV
    representing a mono-dimensional physical quantitiy, we take the column representing this
    quantity in the matrix B and replace the correspoding values in A, thus creating a new matrix,
    that we append to our samples. But in the case of a gaussian field represented by a series
    of random variables, we take all those variables in B and put them in A, but we do not take them
    individually, as in a classical experiment.

    Generation options :
    1 : Random
    2 : LHS
    3 : LowDiscrepancySequence
    4 : SimulatedAnnealingLHS
    '''

    def __init__(self, sampleSize = None, OTPyFunctionWrapper = None):
        self.OTPyFunctionWrapper  = OTPyFunctionWrapper
        self.composedDistribution = None
        self.inputVarNames        = list()
        self.inputVarNamesKL      = list()
        self.N                    = sampleSize
        self._genType             = 1
        print('Generation types:\n1 : Random (default)\n2 : LHS\n3 : LowDiscrepancySequence\n4 : SimulatedAnnealingLHS')
        # here we come to the samples
        self.sample_A             = None
        self.sample_B             = None
        self.dataMixSamples       = list()
        self.experimentSample     = None

    def setSampleSize(self, N):
        if self.N is None :
            self.N = N 
        else :
            self.N = N 
            self.sample_A = self.sample_B = self.experimentSample = None

    def setOTPyFunctionWrapper(self, OTPyFunctionWrapper):
        self.OTPyFunctionWrapper  = OTPyFunctionWrapper
        self.inputVarNames        = self.OTPyFunctionWrapper.inputVarNames
        self.inputVarNamesKL      = self.OTPyFunctionWrapper.inputVarNamesKL
        self.composedDistribution = self.OTPyFunctionWrapper.KLComposedDistribution

    def setGenType(self, arg):
        arg = int(arg)
        if arg not in [1,2,3,4]:
            print('Generation types:\n1 : Random (default)\n2 : LHS\n3 : LowDiscrepancySequence\n4 : SimulatedAnnealingLHS')
            raise TypeError
        self._genType = arg

    def getDataFieldAndRV(self):
        '''Here we analyse the names of the variables, to know which columns
        belong to RVs or Fields
        '''
        n_vars = len(self.inputVarNames)
        n_vars_KL = len(self.inputVarNamesKL)
        self.dataMixSamples =  list()
        for i in range(n_vars):
            k = 0
            timesInList = 0
            jump = NdGaussianProcessExperiment.ramp(sum(self.dataMixSamples)-i)
            while self.inputVarNamesKL[i+k+jump].startswith(self.inputVarNames[i]):
                timesInList += 1
                k += 1
                print('i=',i,'k=',k,'jump=',jump)
                if i+k+jump == n_vars_KL:
                    break   
            self.dataMixSamples.append(timesInList)

    def getExperiment(self):
        n_vars = len(self.inputVarNames)
        N = self.N
        self.experimentSample = numpy.tile(self.sample_A,[2+n_vars,1])
        self.experimentSample[:N,...] = self.sample_B
        jump = 2*N
        jumpDim = 0
        for i in range(n_vars):
            self.experimentSample[jump+N*i:jump+N*(i+1), jumpDim:jumpDim+self.dataMixSamples[i]] = \
                    self.sample_B[...,                   jumpDim:jumpDim+self.dataMixSamples[i]]
            jumpDim += self.dataMixSamples[i]

    @staticmethod
    def ramp(X):
        if X >= 0: return X
        else:      return 0

    def generateSample(self, **kwargs):
        distribution = self.composedDistribution
        method       = self._genType
        N2           = 2*self.N 
        if   method is 1 :
            sample = distribution.getSample(N2)
        elif (method is 2) or (method is 4) :
            lhsExp = openturns.LHSExperiment(distribution, 
                                             N2, 
                                             False, #alwaysShuffle
                                             True) #randomShift
            if method is 2 :
                sample = lhsExp.generate()
            if method is 4 :
                lhsExp.setAlwaysShuffle(True)
                if 'SpaceFilling' in kwargs :
                    if kwargs['SpaceFilling'] is 'SpaceFillingC2':      spaceFill = openturns.SpaceFillingC2
                    if kwargs['SpaceFilling'] is 'SpaceFillingMinDist': spaceFill = openturns.SpaceFillingMinDist
                    if kwargs['SpaceFilling'] is 'SpaceFillingPhiP':    
                        spaceFill = openturns.SpaceFillingPhiP
                        if 'p' in kwargs : 
                            if (type(kwargs['p']) is int) or (type(kwargs['p']) is float) : p = int(kwargs['p'])
                            else : 
                                print('Wrong type for p parameter in SpaceFillingPhiP algorithm, setting to default p = 50')
                                p = 50
                        else : 
                            print('undefined parameter p in SpaceFillingPhiP algorithm, setting to default p = 50')
                            p = 50
                else : 
                    print("undefined parameter 'SpaceFilling', setting to default 'SpaceFillingC2'")
                    spaceFill = openturns.SpaceFillingC2
                if 'TemperatureProfile' in kwargs : 
                    if kwargs['TemperatureProfile'] is 'GeometricProfile': geomProfile = openturns.GeometricProfile(10.0, 0.95, 2000) #Default value
                    if kwargs['TemperatureProfile'] is 'LinearProfile':    geomProfile = openturns.LinearProfile(10.0, 100)
                else :
                    print("undefined parameter 'TemperatureProfile', setting default GeometricProfile")
                    geomProfile = openturns.GeometricProfile(10.0, 0.95, 2000)
                optimalLHSAlgorithm = openturns.SimulatedAnnealingLHS(lhsExp, geomProfile, spaceFill())
                sample = optimalLHSAlgorithm.generate()
        elif method is 3 : 
            restart = True 
            if 'sequence' in kwargs :
                if kwargs['sequence'] == 'Faure':         seq = openturns.FaureSequenc 
                if kwargs['sequence'] == 'Halton':        seq = openturns.HaltonSequence
                if kwargs['sequence'] == 'ReverseHalton': seq = openturns.ReverseHaltonSequence
                if kwargs['sequence'] == 'Haselgrove':    seq = openturns.HaselgroveSequence
                if kwargs['sequence'] == 'Sobol':         seq = openturns.SobolSequence
            else :
                print('sequence undefined for low discrepancy experiment, setting default to SobolSequence')
                print("possible vals for 'sequence' argument:\n    ['Faure','Halton','ReverseHalton','Haselgrove','Sobol']")
                seq = openturns.SobolSequence
            LDExperiment = openturns.LowDiscrepancyExperiment(seq(), 
                                                              distribution,
                                                              N2,
                                                              True)
            sample = LDExperiment.generate()
        sample = numpy.array(sample)
        self.sample_A = sample[:self.N,:]
        self.sample_B = sample[self.N:,:]



'''
import NdGaussianProcessSensitivity as ngps
import NdGaussianProcessConstructor as ngpc
# Classes utilitaires
import numpy                        as np
import openturns                    as ot
import matplotlib.pyplot            as plt
from   importlib                import reload 

# on importe aussi les fonctions à étudier
import RandomBeamGenerationClass    as rbgc


# process governing the young modulus for each element      (MPa)
process_E = ngpc.NdGaussianProcessConstructor(dimension=1,
                                              grid_shape=[[0,1000,100],],
                                              covariance_model={'NameModel':'MaternModel',
                                                                'amplitude':5000.,
                                                                'scale':300,
                                                                'nu':13/3},
                                              trend_arguments=['x'],trend_function=210000)
process_E.setName('E_')


# process governing the diameter for each element          (mm)
process_D = ngpc.NdGaussianProcessConstructor(dimension=1,
                                              grid_shape=[[0,1000,100],],
                                              covariance_model={'NameModel':'MaternModel',
                                                                'amplitude':.3,
                                                                'scale':250,
                                                                'nu':7.4/3},
                                              trend_arguments=['x'],trend_function=10)
process_D.setName('D_')


# random variable for the density of the material (kg/m³)
rho         = 7850.
sigma       = 250
nameD       = 'Rho'
RV_Rho = ngpc.NormalDistribution(mu = rho, sigma = sigma, name = nameD)


# random variable for the position of the force   (mm) 
middle       = 500
sigma_f      = 50
namePos     = 'FP'
RV_Fpos = ngpc.NormalDistribution(mu = middle, sigma = sigma_f, name = namePos)


# random variable for the norm of the force    (N)
muForce       = 100
sigma_Fnor    = 1.5
nameNor       = 'FN'
RV_Fnorm = ngpc.NormalDistribution(mu = muForce, sigma = sigma_Fnor, name = nameNor)

from importlib import reload
reload(ngps)
outputVariables = {'out1' :
                   {
                         'name'     : 'VonMisesStress',
                         'position' : 0,
                         'shape'    : (102,)  
                    },
                   'out2' :
                   {
                        'name'      : 'maxDeflection',
                        'position'  : 1,
                        'shape'     : (1,)
                   }
                  }

functionWrapper = rbgc.sampleAndSoloFunctionWrapper(process_E, process_D, RV_Rho, RV_Fpos, RV_Fnorm)

inputVarList = [process_E, process_D, RV_Rho, RV_Fpos, RV_Fnorm]

soloFunction   = functionWrapper.randomBeamFunctionSolo
sampleFunction = functionWrapper.randomBeamFunctionSample
##
size           = 100 ## Number of samples for our sobol indicies experiment (kept low here to make things faster)
##
reload(ngps)
##
processSensitivityAnalysis = ngps.NdGaussianProcessSensitivityAnalysis(inputVarList, 
                                                                       outputVariables,
                                                                       sampleFunction,
                                                                       soloFunction,
                                                                       size)

import NdGaussianProcessExperimentGeneration as ngpeg
reload(ngpeg)

test = ngpeg.NdGaussianProcessExperiment()

test.setSampleSize(100)

test.setGenType(2)
test.generateSample()
'''