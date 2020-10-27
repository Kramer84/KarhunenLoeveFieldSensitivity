import _karhunenLoeveGeneralizedFunctionWrapper as klgfw
import unittest
import openturns as ot 
from copy import deepcopy
import spsa
import numpy as np
'''Class to test if everything works OK 
'''

class DummyFuncResults :
    dim = 25
    size = 1000
    np.random.seed(125)
    ListOfPoints = [ot.Point(np.random.random(size))]*dim
    BigSample =  ot.Sample(np.random.random((size,dim)))
    NumpySample = np.random.random((size,dim))
    ListNumpySamples = [np.random.random((size,dim)),
                        np.random.random((size,dim-2)),
                        np.random.random((size,dim+3)),
                        np.random.random((size,dim-15))]




class Test_karhunenLoeveGeneralizedFunctionWrapper(unittest.TestCase):

    def setUp(self):
        self.pts = DummyFuncResults.ListOfPoints
        self.smp = DummyFuncResults.BigSample
        self.npsmp = DummyFuncResults.NumpySample
        self.nplst = DummyFuncResults.ListNumpySamples


    def testTransformations(self):
        X = klgfw.convertIntoProcessSample(self.pts)
        print("For self.pts :",X )
        X = klgfw.convertIntoProcessSample(self.smp)
        print("For self.smp :",X )
        X = klgfw.convertIntoProcessSample(self.npsmp)
        print("For self.npsmp :", X)
        X = klgfw.convertIntoProcessSample(self.nplst)
        print("For self.nplst :", X)

if __name__ == '__main__':
    unittest.main()


'''
def stack(listArrays):
    X = listArrays[0]
    nElem = [len(listArrays)]
    nElem.extend(X.shape) 
    Xf = X.copy().repeat(nElem[0]).reshape(nElem,order='F')
    for i in range(nElem[0]):
        Xf[i,...]=listArrays[i][...]
    return Xf

from collections import *
from numbers import Complex, Integral, Real, Rational, Number
def roundList(X):
    try : 
        return [roundList(x) for x in X]
    except TypeError:
        return round(X,4)

def list2str(L):
    L = roundList(L)
    try :
        if isinstance(L[0],(Complex, Integral, Real, Rational, Number, str)):
            return str(L)
        elif isinstance(L[0],(Iterable, Sequence)):
            return [list2str(ll) for ll in L]
    except : 
        return list2str(L)

'''