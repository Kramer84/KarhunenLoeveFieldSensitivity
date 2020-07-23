__version__ = '0.1'
__author__ = 'Kristof Attila S.'
__date__ = '22.06.20'

from typing import Callable, List, Tuple, Optional, Any, Union

import openturns
import numpy

import StochasticProcessSobolIndicesAlgorithmBase as SPSIA

__all__ = ['SobolIndicesStochasticProcessAlgorithm']

class SobolIndicesStochasticProcessAlgorithm(
                        openturns.SobolIndicesAlgorithmImplementation):
    '''Base class for sensitivity analysis, without usage of the inputDesign. 

    Note
    ----
    Code written according to PEP 484 specification, with explicit hints in 
    methods. This reduces the need to detail the method in help. 

    Parameters
    ----------
    outputDesign : numpy.array
        numpy matriix containing the samples Y_A, Y_B, Y_Ei, Y_En, ...
    N : int 
        size of one sample
    method : str
        string with information about method for the sensitivity analysis 

    Attributes
    ----------
    outputDesign : numpy.array
        same as above
    sampleSize : int
        sames as N
    method : str
        same as above
    confidenceLevel : float (between 0 and 1)
        Confidence level of the estimator, important for plotting
    inputDescription : List[str]
        list containing the names of the different input dimensions
    _FirstOrderIndices : numpy.array
        data containing the first order Sobol' estimators
    _TotalOrderIndices : numpy.array
        data containing the total order Sobol' estimators
    _varFirstOrder : numpy.array
        variance of the First Order Sobol Estimator
    _varTotalOrder : numpy.array
        variance of the Total order Sobol Estimator

    Methods 
    -------
    setInputDescription:
        sets the name of each dimension of the input
    getInputDescription:
        accessor to the input dimension names
    setMethod:
        method to set the estimator calculus strategy
    getMethod:
        accessor to the method (default : saltelli)
    getFirstOrderIndices:
        accessor to the first order indices of the modeol
    getTotalOrderIndices:
        accessor to the total order indices
    getFirstOrderIndicesInterval:
        accessor to the variance of the first order Sobol indices
    getTotalorderIndicesInterval:
        accessor to the variance of the total order Sobol indices
    setDimension: 
        method to set the dimension of the input. usually deduced automatically
    setConfidenceLevel: 
        set method for confidence level of estimator (default : 0.95)
    getConfidenceLevel:
        accessor to the confidence level 
    draw:
        method to draw the indices, in up to three dimensions
    '''
    sobolEngine = SPSIA.StochasticProcessSobolIndicesAlgorithmBase()

    def __init__(self, outputDesign: Union[openturns.Sample, numpy.array],
                 N: int, method: str = 'Saltelli') -> None:
        self.outputDesign = outputDesign
        self.sampleSize = N
        self.method = method

        # -2 as the we have two samples A and B
        self.dim = int(self.outputDesign.shape[0] / N) - 2
        print('Implicit dimension =', self.dim)

        self.confidenceLevel = 0.95

        self.inputDescription = openturns.Description(
            ['X' + str(i) for i in range(self.dim)])
        print('Implicit description:', self.inputDescription)

        super(SobolIndicesStochasticProcessAlgorithm, self).__init__()

        self._FirstOrderIndices = None
        self._TotalOrderIndices = None

        self._varFirstOrder = None
        self._varTotalOrder = None

    def _runAlgorithm(self) -> None:
        if (self._FirstOrderIndices is None) and (self._TotalOrderIndices is None) \
           and (self._varFirstOrder is None) and (self._varTotalOrder is None):
            self._FirstOrderIndices, self._TotalOrderIndices, \
                self._varFirstOrder, self._varTotalOrder = self.sobolEngine.getSobolIndices(
                    self.outputDesign, self.sampleSize, self.method)

    def setInputDescription(self, names: Union[List[str], str]) -> None:  
        if type(names) is str:
            self.inputDescription = openturns.Description([names])
        else:
            self.inputDescription = openturns.Description(names)

    def getInputDescription(self) -> List[str]:
        return self.inputDescription

    def setMethod(self, method: str = 'Saltelli') -> None:
        self.method = method

    def getMethod(self) -> str:
        return self.method

    ###########################################################################
    ################### Overloaded functions ##################################
    ###########################################################################

    def getFirstOrderIndices(self) -> numpy.array:
        assert self.outputDesign is not None, "You need a sample to work on"
        self._runAlgorithm()
        return self._FirstOrderIndices

    def getTotalOrderIndices(self) -> numpy.array:
        assert self.outputDesign is not None, "You need a sample to work on"
        self._runAlgorithm()
        return self._TotalOrderIndices

    def getFirstOrderIndicesInterval(self) -> float:
        assert self.outputDesign is not None, "You need a sample to work on"
        self._runAlgorithm()
        confidence = numpy.sqrt(
            self._varFirstOrder) * openturns.Normal().computeQuantile(
                                                        self.confidenceLevel)[0]
        return confidence

    def getTotalorderIndicesInterval(self) -> float:
        assert self.outputDesign is not None, "You need a sample to work on"
        self._runAlgorithm()
        confidence = numpy.sqrt(
            self._varTotalOrder) * openturns.Normal().computeQuantile(
                                                        self.confidenceLevel)[0]
        return confidence

    def setDimension(self, dim: int) -> None:
        '''Dimension of the inputs => how many different
        sobol indices we will calculate 
        '''
        self.dim = dim

    def setConfidenceLevel(self, confidenceLevel: float = 0.975) -> None:
        '''method to set desired confidence level for the 
        plot 
        '''
        self.confidenceLevel = confidenceLevel

    def getConfidenceLevel(self) -> float:
        return self.confidenceLevel

    def draw(self, *args: Any) -> None:
        SPSIA.plotSobolIndicesWithErr(S=self._FirstOrderIndices,
                                    errS=self.getFirstOrderIndicesInterval(),
                                    varNames=self.inputDescription,
                                    n_dims=self.dim,
                                    Stot=self._TotalOrderIndices,
                                    errStot=self.getTotalorderIndicesInterval())
