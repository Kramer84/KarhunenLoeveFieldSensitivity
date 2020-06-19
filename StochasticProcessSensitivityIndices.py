import openturns 
import numpy 
from   typing             import Callable, List, Tuple, Optional, Any, Union
import FieldSobolIndicesAlgorithm.FieldSobolIndicesAlgorithmBase as fsiab

class SobolIndicesStochasticProcessAlgorithm(openturns.SobolIndicesAlgorithmImplementation):
    sobolEngine = fsiab.FieldSobolIndicesAlgorithmBase()
    def __init__(self, outputDesign, N):
        self.outputDesign       = outputDesign
        self.sampleSize         = N 

        self.dim                = int(self.outputDesign.shape[0]/N)
        print('Implicit dimension =', self.dim)

        self.confidenceLevel    = 0.975
        
        self.inputDescription   = openturns.Description(['X'+str(i) for i in range(self.dim)])
        print('Implicit description:',self.inputDescription)

        super(SaltelliSensitivityAlgorithmField, self).__init__()

        self._FirstOrderIndices = None
        self._TotalOrderIndices = None
        
        self._varFirstOrder     = None 
        self._varTotalOrder     = None        


    def _runAlgorithm(self, method='Saltelli'):
        if (self._FirstOrderIndices is None) and (self._TotalOrderIndices is None) and (self._varFirstOrder is None) and (self._varTotalOrder is None) :
            self._FirstOrderIndices, self._TotalOrderIndices, self._varFirstOrder , self._varTotalOrder = self.sobolEngine.getSobolIndices(self.outputDesign, self.sampleSize , method)

    def setInputDescription(self, names):
        if type(names) is str:
            self.inputDescription = openturns.Description([names])
        else :
            self.inputDescription = openturns.Description(names)

    def getInputDescription(self):
        return self.inputDescription

    def setMethod(self, method):
        self.method = method

    def getmethod(self):
        return self.method

    ###########################################################################
    ################### Overloaded functions ##################################
    ###########################################################################

    def getFirstOrderIndices(self):
        assert self.outputDesign is not None, "You need a sample to work on"
        self._runAlgorithm()
        return self._FirstOrderIndices

    def getTotalOrderIndices(self):
        assert self.outputDesign is not None, "You need a sample to work on"
        self._runAlgorithm()
        return self._TotalOrderIndices

    def getFirstOrderIndicesInterval(self):
        assert self.outputDesign is not None, "You need a sample to work on"
        self._runAlgorithm()
        confidence = numpy.sqrt(self._varFirstOrder)*openturns.Normal().computeQuantile(self.confidenceLevel)[0]
        print('half of the length of the symetric confidence interval ')
        return confidence

    def getTotalorderIndicesInterval(self):
        assert self.outputDesign is not None, "You need a sample to work on"
        self._runAlgorithm()
        confidence = numpy.sqrt(self._varTotalOrder)*openturns.Normal().computeQuantile(self.confidenceLevel)[0]
        print('half of the length of the symetric confidence interval ')       
        return confidence

    def setDimension(self,dim):
        self.dim = dim

    def setConfidenceLevel(self, confidenceLevel):
        self.confidenceLevel = confidenceLevel

    def getConfidenceLevel(self):
        return self.confidenceLevel

    def draw(self, *args):
        fsiab.plotSobolIndicesWithErr(self._FirstOrderIndices, 
                            self.getFirstOrderIndicesInterval(), 
                            self.inputDescription, 
                            self.dim, 
                            self._TotalOrderIndices, 
                            self.getTotalorderIndicesInterval())



