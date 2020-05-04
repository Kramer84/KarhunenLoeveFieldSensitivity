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
        self.experimentSample     = None

    def setSampleSize(self, N):
        if self.N is None :
            self.N = N 
        else :
            self.N = N 
            self.sample_A = self.sample_B = self.experimentSample = None

    def setOTPyFunctionWrapper(self, OTPyFunctionWrapper):
        if self.OTPyFunctionWrapper is None : 
            self.OTPyFunctionWrapper  = OTPyFunctionWrapper
            self.inputVarNames        = self.OTPyFunctionWrapper.inputVarNames
            self.inputVarNamesKL      = self.OTPyFunctionWrapper.inputVarNamesKL
            self.composedDistribution = self.OTPyFunctionWrapper.KLComposedDistribution
        else :
            raise NotImplementedError

    def setGenType(self, arg)
        arg = int(arg)
        if arg not in [1,2,3,4]:
            print('Generation types:\n1 : Random (default)\n2 : LHS\n3 : LowDiscrepancySequence\n4 : SimulatedAnnealingLHS')
            raise TypeError
        self._genType = arg

    def generateSample(self, **kwargs):
        distribution = self.composedDistribution
        method       = self._genType
        methodDict   = {1 : 'random', 2 : openturns.LHSExperiment, 3 : openturns.LowDiscrepancySequence, 4 :  openturns.SimulatedAnnealingLHS}
        N            = self.N 
        if method is 1 :
            sample = distribution.getSample(N)
        elif method is 2 :
            lhsExp = openturns.LHSExperiment(distribution, 
                                             N, 
                                             alwaysShuffle = False,
                                             randomShift = True)
            sample = lhsExp.generate()



