from copy import deepcopy
import openturns as ot 

class KarhunenLoeveSobolIndicesExperiment(object):
    def __init__(self, AggregatedKarhunenLoeveResults=None, size=None, second_order=False):
        self.AKLR = AggregatedKarhunenLoeveResults
        self.size = None

        self.composedDistribution = None
        self.inputVarNames = list()
        self.inputVarNamesKL = list()
        self._dataMixSamples = list()

        if size is not None:
            self.setSize(size)
        if AggregatedKarhunenLoeveResults is not None:
            self._setKaruhnenLoeveResults(AggregatedKarhunenLoeveResults)

        # here we come to the samples (our outputs)
        self._sample_A = None
        self._sample_B = None
        self._experimentSample = None

    def generate(self, **kwargs):
        '''generate final sample with A and b mixed
        '''
        assert (self.AKLR is not None) and \
               (self.size is not None), \
                    "Please intialise sample size and PythonFunction wrapper"
        self._generateSample(**kwargs)

        self._mixSamples()
        return self._experimentSample

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

    def _setKaruhnenLoeveResults(self, AKLR):
        '''set the wrapped function from the NdGaussianProcessSensitivity;
        '''
        self.AKLR = AKLR
        self.inputVarNames = self.AKLR._subNames
        self.inputVarNamesKL = self.AKLR._modeDescription
        self.composedDistribution = ot.ComposedDistribution([ot.Normal()]*self.AKLR.getSizeModes())
        self._dataMixSamples = self.AKLR._modesPerProcess


    def _mixSamples(self):
        '''Here we mix the samples together 
        '''
        n_vars = self.AKLR._N
        N = self.size
        self._experimentSample = deepcopy(self._sample_A)
        print(self._sample_A)
        [[self._experimentSample.add(self._sample_A[j]) for j in range(len(self._sample_A))] for _ in range(2+n_vars)]
        print(self._experimentSample)
        print(self._sample_B)
        self._experimentSample[:,N:2 * N] = self._sample_B
        jump = 2 * N
        jumpDim = 0
        for i in range(n_vars):
            self._experimentSample[jump + N*i:jump + N*(i+1), 
              jumpDim:jumpDim+self._dataMixSamples[i]] = \
                self._sample_B[:,jumpDim:jumpDim + self._dataMixSamples[i]]
            jumpDim += self._dataMixSamples[i]

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
                print('sequence undefined for low discrepancy experiment, setting default to SobolSequence')
                print("possible vals for 'sequence' argument:\n\
                    ['Faure','Halton','ReverseHalton','Haselgrove','Sobol']")
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
