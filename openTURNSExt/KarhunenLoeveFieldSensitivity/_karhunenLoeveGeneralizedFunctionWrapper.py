import openturns as ot 

class KarhunenLoeveGeneralizedFunctionWrapper(ot.OpenTURNSPythonPointToFieldFunction):
    '''Class allowing to rewrite any function taking as an input a list
    of fields, samples or ProcessSamples as only dependent of a vector.
    The function passed should return a list of fields.

    Note
    ----
    The usage of this wrapper implies having already done the karhunen loeve
    decomposition of the list of the Processes with wich we will feed the 
    function. The input dimension of this function is dependent of the 
    order of the Karhunen Loeve decomposition. 
    '''
    def __init__(self, AggregatedKarhunenLoeveResults=None, func=None, 
        func_sample=None, outputDim=None):
        self.func = func
        self.func_sample = func_sample
        self.AKLR = AggregatedKarhunenLoeveResults
        inputDim = self.AKLR.getSizeModes()
        super(KarhunenLoeveGeneralizedFunctionWrapper,self).__init__(inputDim,
                                                                  ot.Mesh(),
                                                                  outputDim)
        self.setInputDescription(self.AKLR._modeDescription)

    def _exec(self, X):
        assert len(X)==self.getInputDimension()
        inputFields = self.AKLR.liftAsField(X)
        #evaluating ...
        result = self.func(inputFields)
        return result

    def _exec_sample(self, X):
        assert len(X[0])==self.getInputDimension()
        inputProcessSamples = self.AKLR.liftAsProcessSample(X)
        try :
            result = self.func_sample(inputProcessSamples)
            return result
        except Exception as e:
            raise e
