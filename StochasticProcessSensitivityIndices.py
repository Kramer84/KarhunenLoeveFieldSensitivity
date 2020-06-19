import openturns 
import numpy 
from   typing    import Callable, List, Tuple, Optional, Any, Union
import StochasticProcessSobolIndicesAlgorithm.StochasticProcessSobolIndicesAlgorithmBase as spsia

class SobolIndicesStochasticProcessAlgorithm(openturns.SobolIndicesAlgorithmImplementation):
    sobolEngine = spsia.StochasticProcessSobolIndicesAlgorithmBase()
    def __init__(self, outputDesign : Union[openturns.Sample, numpy.array], N : int) -> None:
        self.outputDesign       = outputDesign
        self.sampleSize         = N 

        self.dim                = int(self.outputDesign.shape[0]/N)
        print('Implicit dimension =', self.dim)

        self.confidenceLevel    = 0.975
        
        self.inputDescription   = openturns.Description(['X'+str(i) for i in range(self.dim)])
        print('Implicit description:',self.inputDescription)

        super(SobolIndicesStochasticProcessAlgorithm, self).__init__()

        self._FirstOrderIndices = None
        self._TotalOrderIndices = None
        
        self._varFirstOrder     = None 
        self._varTotalOrder     = None        


    def _runAlgorithm(self, method : str ='Saltelli') -> None:
        if (self._FirstOrderIndices is None) and (self._TotalOrderIndices is None) and (self._varFirstOrder is None) and (self._varTotalOrder is None) :
            self._FirstOrderIndices, self._TotalOrderIndices, self._varFirstOrder , self._varTotalOrder = self.sobolEngine.getSobolIndices(self.outputDesign, self.sampleSize , method)

    def setInputDescription(self, names : Union[List[str], str]) -> None:
        if type(names) is str:
            self.inputDescription = openturns.Description([names])
        else :
            self.inputDescription = openturns.Description(names)

    def getInputDescription(self) -> List[str]:
        return self.inputDescription

    def setMethod(self, method : str = 'Saltelli') -> None:
        self.method = method

    def getmethod(self) -> str:
        return self.method

    ###########################################################################
    ################### Overloaded functions ##################################
    ###########################################################################

    def getFirstOrderIndices(self) -> numpy.array:
        assert self.outputDesign is not None, "You need a sample to work on"
        self._runAlgorithm()
        return self._FirstOrderIndices

    def getTotalOrderIndices(self) -> numpy.array :
        assert self.outputDesign is not None, "You need a sample to work on"
        self._runAlgorithm()
        return self._TotalOrderIndices

    def getFirstOrderIndicesInterval(self) -> float :
        assert self.outputDesign is not None, "You need a sample to work on"
        self._runAlgorithm()
        confidence = numpy.sqrt(self._varFirstOrder)*openturns.Normal().computeQuantile(self.confidenceLevel)[0]
        print('half of the length of the symetric confidence interval ')
        return confidence

    def getTotalorderIndicesInterval(self) -> float :
        assert self.outputDesign is not None, "You need a sample to work on"
        self._runAlgorithm()
        confidence = numpy.sqrt(self._varTotalOrder)*openturns.Normal().computeQuantile(self.confidenceLevel)[0]
        print('half of the length of the symetric confidence interval ')       
        return confidence

    def setDimension(self,dim : int) -> None:
        '''Dimension of the inputs => how many different
        sobol indices we will calculate 
        '''
        self.dim = dim

    def setConfidenceLevel(self, confidenceLevel : float = 0.975) -> None :
        '''method to set desired confidence level for the 
        plot 
        '''
        self.confidenceLevel = confidenceLevel

    def getConfidenceLevel(self) -> float :
        return self.confidenceLevel

    def draw(self, *args : Any) -> None :
        fsiab.plotSobolIndicesWithErr(self._FirstOrderIndices, 
                            self.getFirstOrderIndicesInterval(), 
                            self.inputDescription, 
                            self.dim, 
                            self._TotalOrderIndices, 
                            self.getTotalorderIndicesInterval())



