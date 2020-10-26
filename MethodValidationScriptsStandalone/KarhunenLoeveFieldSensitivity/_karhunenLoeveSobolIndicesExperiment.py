__author__ = 'Kristof Attila S.'
__version__ = '0.1'
__date__  = '17.09.20'

__all__ = ['KarhunenLoeveSobolIndicesExperiment']

from copy import deepcopy
import openturns as ot 

class KarhunenLoeveSobolIndicesExperiment(ot.SobolIndicesExperiment):
    def __init__(self, AggregatedKarhunenLoeveResults=None, size=None, 
                                                        second_order=False):
        self.__AKLR__ = AggregatedKarhunenLoeveResults
        self.size = None

        self.__visibility__ = True
        self.__name__ = 'Unnamed'
        self.__shadowedId__ = None
        self.__computeSecondOrder__ = second_order
        self.composedDistribution = None
        self.inputVarNames = list()
        self.inputVarNamesKL = list()
        self.__mode_count__ = list()

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
        assert (self.__AKLR__ is not None) and \
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
        if self.__AKLR__ is not None:
            return self.__AKLR__
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
        self.__AKLR__ = AggrKLRes
        self.inputVarNames = self.__AKLR__.__process_distribution_description__
        self.inputVarNamesKL = self.__AKLR__.__mode_description__
        self.composedDistribution = ot.ComposedDistribution(
                                     [ot.Normal()]*self.__AKLR__.getSizeModes())
        self.__mode_count__ = self.__AKLR__.__mode_count__

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
        n_vars = self.__AKLR__._N
        n_modes = self.__AKLR__.getSizeModes()
        N = self.size
        if self.__computeSecondOrder__ == False or n_vars<=2 : 
            N_tot = int(N*(2+n_vars))
            self._experimentSample = ot.Sample(N_tot, n_modes)
            self._experimentSample[:N,:] = self._sample_A[:,:]
            print('Samples A and B of size {} and dimension {}'.format(
                                                self._sample_A.getSize(),
                                                self._sample_A.getDimension()))
            self._experimentSample[N:2 * N,:] = self._sample_B[:,:]
            jmp = 2 * N # As the first two blocks are sample A and B
            jmpDim = 0
            for i in range(n_vars):
                self._experimentSample[jmp + N*i : jmp + N*(i+1), :] = self._sample_A[:, :]
                self._experimentSample[jmp + N*i : jmp + N*(i+1), 
                             jmpDim : jmpDim+self.__mode_count__[i]] = self._sample_B[:, jmpDim : jmpDim + self.__mode_count__[i]]
                jmpDim += self.__mode_count__[i]

        elif self.__computeSecondOrder__ == True and n_vars>2 : 
            print('Generating samples for the second order indices')
            # when cxomputing second order indices we add n_vars*size elements to 
            # the experiment sample. 
            # Here we mix columns of A into B 
            N_tot = int(N*(2+2*n_vars))
            self._experimentSample = ot.Sample(N_tot, n_modes)
            self._experimentSample[:N,:] = self._sample_A[:,:]
            print('Samples A and B of size {} and dimension {}'.format(
                                                self._sample_A.getSize(),
                                                self._sample_A.getDimension()))
            self._experimentSample[N:2 * N,:] = self._sample_B[:,:]
            jmp = 2 * N # As the first two blocks are sample A and B
            jmpDim = 0
            for i in range(n_vars):
                self._experimentSample[jmp + N*i : jmp + N*(i+1), :] = self._sample_A[:,:]
                self._experimentSample[jmp + N*i : jmp + N*(i+1), 
                                       jmpDim : jmpDim+self.__mode_count__[i]] = \
                        self._sample_B[: , jmpDim : jmpDim + self.__mode_count__[i]]
                jmpDim += self.__mode_count__[i]
            jmp = int(N*(2+n_vars))
            jmpDim = 0
            for i in range(n_vars):
                self._experimentSample[jmp + N*i : jmp + N*(i+1), :] = self._sample_B[:,:]
                self._experimentSample[jmp + N*i : jmp + N*(i+1), 
                  jmpDim:jmpDim+self.__mode_count__[i]] = \
                        self._sample_A[:,jmpDim:jmpDim + self.__mode_count__[i]]
                jmpDim += self.__mode_count__[i]
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