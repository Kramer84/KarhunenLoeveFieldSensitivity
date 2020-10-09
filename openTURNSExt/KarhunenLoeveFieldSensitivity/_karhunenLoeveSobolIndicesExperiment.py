__author__ = 'Kristof Attila S.'
__version__ = '0.1'
__date__  = '17.09.20'

__all__ = ['KarhunenLoeveSobolIndicesExperiment']

from copy import deepcopy
import openturns as ot 

class KarhunenLoeveSobolIndicesExperiment(ot.SobolIndicesExperiment):
    def __init__(self, AggregatedKarhunenLoeveResults=None, size=None, 
                                                        second_order=False):
        self._AKLR = AggregatedKarhunenLoeveResults
        self.size = None

        self.__visibility__ = True
        self.__name__ = 'Unnamed'
        self.__shadowedId__ = None
        self.__computeSecondOrder__ = second_order
        self.composedDistribution = None
        self.inputVarNames = list()
        self.inputVarNamesKL = list()
        self._modesPerProcess = list()

        if size is not None:
            self.setSize(size)
        if AggregatedKarhunenLoeveResults is not None:
            self.setAggregatedKLResults(AggregatedKarhunenLoeveResults)

        # here we come to the samples (our outputs)
        self._sample_A = None
        self._sample_B = None
        self._experimentSample = None

    def generate(self, **kwargs):
        '''generate final sample with A and b mixed
        '''
        assert (self._AKLR is not None) and \
               (self.size is not None), \
                    "Please intialise sample size and PythonFunction wrapper"
        self._generateSample(**kwargs)

        self._mixSamples()
        self._experimentSample.setDescription(self.inputVarNamesKL)
        return self._experimentSample

    def generateWithWeights(self, **kwargs):
        pass

    def getClassName(self):
        return self.__class__.__name__

    def getAggregatedKLResults(self):
        if self._AKLR is not None:
            return self._AKLR
        else :
            return None

    def getId(self):
        return id(self)

    def getName(self):
        return self.__name__

    def getShadowedId(self):
        return self.__shadowedId__

    def getSize(self):
        if self._experimentSample is None :
            return 0
        else :
            return len(self._experimentSample)

    def getVisibility(self):
        return self.__visibility__

    def hasName(self):
        if len(self.__name__)==0 or self.__name__ is None:
            return False
        else : 
            return True

    def hasUniformWeights(self):
        return None

    def hasVisibleName(self):
        if self.__name__ == 'Unnamed' or len(self.__name__)==0:
            return False
        else :
            return True

    def setAggregatedKLResults(self, AggrKLRes=None):
        self._AKLR = AggrKLRes
        self.inputVarNames = self._AKLR._subNames
        self.inputVarNamesKL = self._AKLR._modeDescription
        self.composedDistribution = ot.ComposedDistribution(
                                     [ot.Normal()]*self._AKLR.getSizeModes())
        self._modesPerProcess = self._AKLR._modesPerProcess

    def setName(self, name):
        self.__name__ = str(name)

    def setShadowedId(self,ids):
        self.__shadowedId__ = ids

    def setSize(self, N):
        '''set size of the samples 
        '''
        assert isinstance(N,int) and N>0, \
                "Sample size can only be positive integer"
        if self.size is None:
            self.size = N
        else:
            self.size = N
            self._sample_A = self._sample_B = self._experimentSample = None

    def _mixSamples(self):
        '''Here we mix the samples together 
        '''
        n_vars = self._AKLR._N
        N = self.size
        self._experimentSample = deepcopy(self._sample_A)
        print('Samples A and B of size {} and dimension {}'.format(
                                                self._sample_A.getSize(),
                                                self._sample_A.getDimension()))
        [[self._experimentSample.add(
            self._sample_A[j]) for j in range(
                len(self._sample_A))] for _ in range(1+n_vars)]  
        self._experimentSample[:,N:2 * N] = self._sample_B
        jump = 2 * N
        jumpDim = 0
        for i in range(n_vars):
            self._experimentSample[jump + N*i:jump + N*(i+1), 
              jumpDim:jumpDim+self._modesPerProcess[i]] = \
                self._sample_B[:,jumpDim:jumpDim + self._modesPerProcess[i]]
            jumpDim += self._modesPerProcess[i]

        if self.__computeSecondOrder__ == True and n_vars>2 : 
            print('Generating samples for the second order indices')
            # when cxomputing second order indices we add n_vars*size elements to 
            # the experiment sample. 
            # Here we mix columns of A into B 
            [[self._experimentSample.add(
                self._sample_B[j]) for j in range(
                    len(self._sample_B))] for _ in range(n_vars)]
            jump = N*(2+n_vars)
            jumpDim = 0
            for i in range(n_vars):
                self._experimentSample[jump + N*i:jump + N*(i+1), 
                  jumpDim:jumpDim+self._modesPerProcess[i]] = \
                    self._sample_A[:,jumpDim:jumpDim + self._modesPerProcess[i]]
                jumpDim += self._modesPerProcess[i]
            print('Experiment for second order generated')
        print('Experiment of size {} and dimension {}'.format(
                                                self._experimentSample.getSize(),
                                                self._experimentSample.getDimension()))

    def _ramp(self, X):
        '''simple _ramp function
        '''
        if X >= 0 : return X
        else : return 0

    def _generateSample(self, **kwargs):
        '''Generation of two samples A and B using diverse methods
        '''
        distribution = self.composedDistribution
        if 'method' in kwargs :
            method = kwargs['method']
        else : 
            method = 'MonteCarlo'
        N2 = 2 * self.size
        if method == 'MonteCarlo':
            sample = distribution.getSample(N2)
        elif method == 'LHS':
            lhsExp = ot.LHSExperiment(distribution,
                                             N2,
                                             False,  # alwaysShuffle
                                             True)  # randomShift
            sample = lhsExp.generate()
        elif method == 'QMC':
            restart = True
            if 'sequence' in kwargs:
                if kwargs['sequence'] == 'Faure':
                    seq = ot.FaureSequenc
                if kwargs['sequence'] == 'Halton':
                    seq = ot.HaltonSequence
                if kwargs['sequence'] == 'ReverseHalton':
                    seq = ot.ReverseHaltonSequence
                if kwargs['sequence'] == 'Haselgrove':
                    seq = ot.HaselgroveSequence
                if kwargs['sequence'] == 'Sobol':
                    seq = ot.SobolSequence
            else:
                print(
'sequence undefined for low discrepancy experiment, default: SobolSequence')
                print(
"'sequence' arguments: 'Faure','Halton','ReverseHalton','Haselgrove','Sobol'")
                seq = ot.SobolSequence
            LDExperiment = ot.LowDiscrepancyExperiment(seq(),
                                                              distribution,
                                                              N2,
                                                              True)
            LDExperiment.setRandomize(False)
            sample = LDExperiment.generate()
        sample = ot.Sample(sample)
        self._sample_A = sample[:self.size, :]
        self._sample_B = sample[self.size:, :]
