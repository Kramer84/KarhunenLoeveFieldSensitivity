__author__ = 'Kristof Attila S.'
__version__ = '0.1'
__date__  = '17.09.20'

__all__ = ['KarhunenLoeveGeneralizedFunctionWrapper']

import openturns as ot
from collections import Iterable, UserList, Sequence
from copy import copy
from numbers import Complex, Integral, Real, Rational, Number

class KarhunenLoeveGeneralizedFunctionWrapper(object):
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
        func_sample=None, n_outputs=1):
        self.func = func
        self.func_sample = func_sample
        self.__AKLR__ = AggregatedKarhunenLoeveResults
        self.__nOutputs__ = n_outputs
        self.__inputDim__ = 0
        self._inputDescription = ot.Description.BuildDefault(self.__inputDim__, 'X_')
        self._outputDescription = ot.Description.BuildDefault(self.__nOutputs__, 'Y_')
        self.__calls__ = 0
        self.__name__ = 'Unnamed'
        self.__setDefaultState__()

    def __setDefaultState__(self):
        try :
            if (self.func is not None or self.func_sample is not None) and self.__AKLR__ is not None :
                self.__inputDim__ = self.__AKLR__.getSizeModes()
                self.setInputDescription(ot.Description(self.__AKLR__.__mode_description__))
                self.setOutputDescription(ot.Description.BuildDefault(self.__nOutputs__, 'Y_'))
            else :
                self.func         = None
                self.func_sample  = None
                self.__AKLR__     = None
                self.__nOutputs__ = 0
                self.__inputDim__ = 0
                self._inputDescription = ot.Description()
                self._outputDescription = ot.Description()
        except Exception as e:
            print('Check if your aggregated karhunen loeve result object is correct')
            raise e

    def __call__(self, X):
        if isinstance(X, (ot.Point)) or (
            hasattr(X, '__getitem__') and not hasattr(X[0], '__getitem__')):
            return self._exec(X)
        else :
            return self._exec_sample(X)

    def _exec(self, X):
        assert len(X)==self.getInputDimension()
        inputFields = self.__AKLR__.liftAsField(X)
        #evaluating ...
        try :
            result = self.func(inputFields)
        except :
            try :
                result = self.func(*inputFields)
            except TypeError as te:
                print('did not manage to evaluate single function')
                raise te
        result = CustomList.atLeastList(result)
        result = self._convert_exec_ot(result)
        self.__calls__+=1
        return result

    def _exec_sample(self, X):
        assert len(X[0])==self.getInputDimension()
        inputProcessSamples = self.__AKLR__.liftAsProcessSample(X)
        try :
            result = self.func_sample(inputProcessSamples)
        except :
            try :
                result = self.func_sample(*inputProcessSamples)
            except TypeError as te:
                print('did not manage to evaluate batch function')
                raise te
        result = CustomList.atLeastList(result)
        result = self._convert_exec_sample_ot(result)
        self.__calls__ += X.__len__()
        return result


    def _convert_exec_ot(self, output):
        '''For a singular evaluation, the function must only return scalars,
        points or fields.'''
        print(
'''Using the single evaluation function. Assumes that the outputs are in the
same order than for the batch evaluation function. This one should only
return Points, Fields, Lists or numpy arrays.''')
        outputList = []
        if len(output) != len(self._outputDescription) :
            self.__nOutputs__ = len(output)
            self.setOutputDescription(ot.Description.BuildDefault(self.__nOutputs__, 'Y_'))
            print("shapes mismatched")
        for i, element in enumerate(output) :
            if isinstance(element, (ot.Point, ot.Field)):
                element.setName(self._outputDescription[i])
                outputList.append(element)
                try : dim = element.getDimension()
                except : dim = element.getMesh().getDimension()
                print(
'Element {} of the output tuple returns elements of type {} of dimension {}'.format(
                      i, element.__class__.__name__ ,dim))
            elif isinstance(element, (Sequence, Iterable)):
                intermElem = CustomList(element)
                shape = intermElem.shape
                dtype = intermElem.dtype
                assert dtype is not None, 'If None the list is not homogenous'
                if isinstance(dtype(), (Complex, Integral, Real, Rational, Number, str)):
                    intermElem.recurse2list()
                    if len(shape) >= 2 :
                        print(
'Element {} of the output tuple returns fields of dimension {}'.format(i,len(shape)))
                        intermElem.flatten()
                        element = ot.Field(self._buildMesh(self._getGridShape(shape)),
                                           [[elem] for elem in intermElem])
                        element.setName(self._outputDescription[i])
                        outputList.append(element)
                    if len(shape) == 1 :
                        print(
'Element {} of the output tuple returns points of dimension {}'.format(i,shape[0]))
                        intermElem.recurse2list()
                        intermElem.flatten()
                        element = ot.Point(intermElem)
                        element.setName(self._outputDescription[i])
                        outputList.append(element)
                else :
                    print('Do not use non-numerical dtypes in your objects')
                    print('Wrong dtype is: ',dtype.__name__)
            elif isinstance(element, (Complex, Integral, Real, Rational, Number, str)):
                print(
'Element {} of the output tuple returns unique {}'.format(i,type(element).__name__))
                outputList.append(element)
            elif isinstance(element, (ot.Sample, ot.ProcessSample)):
                print(
'ONLY _exec_sample FUNCTION MUST RETURN ot.Sample OR ot.ProcessSample OBJECTS!!')
                raise TypeError
            else :
                print(
'Element is {} of type {}'.format(element, type(element).__name__))
                raise NotImplementedError
        return outputList

    def _convert_exec_sample_ot(self, output):
        '''For a singular evaluation, the function must only return
        any combination of scalars, points or fields.'''
        print(
'''Using the batch evaluation function. Assumes that the outputs are in the
same order than for the single evaluation function. This one should only
return ProcessSamples, Samples, Lists or numpy arrays.''')
        outputList = []
        if len(output) != len(self._outputDescription) :
            self.__nOutputs__ = len(output)
            self.setOutputDescription(ot.Description.BuildDefault(self.__nOutputs__, 'Y_'))
        for i, element in enumerate(output) :
            if isinstance(element, (ot.Sample, ot.ProcessSample)):
                element.setName(self._outputDescription[i])
                outputList.append(element)
                print(
'Element {} of the output tuple returns elements of type {} of dimension {}'.format(
                      i, element.__class__.__name__ ,element.getDimension()))
            elif isinstance(element, (Sequence, Iterable)):
                print(
'Element is iterable, assumes that first dimension is size of sample')
                intermElem = CustomList(element)
                intermElem.recurse2list()
                shape = intermElem.shape
                dtype = intermElem.dtype
                print('Shape is {} and dtype is {}'.format(shape,dtype))
                sampleSize = shape[0]
                subSample = [CustomList(intermElem[j]) for j in range(sampleSize)]
                assert dtype is not None, 'If None the list is not homogenous'
                if isinstance(dtype(), (Complex, Integral, Real, Rational, Number, str)):
                    if len(shape) >= 2 :
                        print(
'Element {} of the output tuple returns process samples of dimension {}'.format(i,len(shape)-1))
                        mesh = self._buildMesh(self._getGridShape(shape[1:]))
                        subSample = [subSample[j].flatten() for j in range(sampleSize)]
                        procsample = ot.ProcessSample(mesh, 0, len(shape)-1)
                        for j in range(sampleSize):
                            procsample.add(ot.Field(mesh, [[elem] for elem in subSample[j].data]))
                        procsample.setName(self._outputDescription[i])
                        outputList.append(procsample)
                    elif len(shape) == 1 :
                        print(
'Element {} of the output tuple returns samples of dimension {}'.format(i,1))
                        element = ot.Sample([[dat] for dat in intermElem.data])
                        element.setName(self._outputDescription[i])
                        outputList.append(element)
                else :
                    print('Do not use non-numerical dtypes in your objects')
                    print('Wrong dtype is: ',dtype.__name__)
            elif isinstance(element, ot.Point):
                print(
'Element {} of the output tuple returns samples of dimension 1'.format(i,type(element).__name__))
                element = ot.Sample([[element[j]] for j in range(len(element))])
                element.setName(self._outputDescription[i])
                outputList.append(element)
            elif isinstance(element, ot.Field):
                print(
'ONLY _exec_sample FUNCTION MUST RETURN ot.Sample OR ot.ProcessSample OBJECTS!!')
                raise TypeError
            else :
                print(
'Element is {} of type {}'.format(element, element.__class__.__name__))
                raise NotImplementedError
        return outputList

    def _getGridShape(self, shape=()):
        return [[0,1,shape[dim]-1] for dim in range(len(shape))]

    def _buildMesh(self,grid_shape):
        dimension = len(grid_shape)
        n_intervals = [int(grid_shape[i][2]) for i in range(dimension)]
        low_bounds = [grid_shape[i][0] for i in range(dimension)]
        lengths = [grid_shape[i][1] for i in range(dimension)]
        high_bounds = [low_bounds[i] + lengths[i] for i in range(dimension)]
        mesherObj = ot.IntervalMesher(n_intervals)
        grid_interval = ot.Interval(low_bounds, high_bounds)
        mesh = mesherObj.build(grid_interval)
        mesh.setName(str(dimension)+'D_Grid')
        return mesh

    def getCallsNumber(self):
        return self.__calls__

    def getClassName(self):
        return self.__class__.__name__

    def getId(self):
        return id(self)

    def getImplementation(self):
        print('custom implementation')
        return None

    def getInputDescription(self):
        return self._inputDescription

    def getInputDimension(self):
        return self.__inputDim__

    def getMarginal(self):
        print('custom implementation')
        return None

    def getName(self):
        return self.__name__

    def getOutputDescription(self):
        return self._outputDescription

    def setNumberOutputs(self, N):
        self.__nOutputs__ = N

    def getNumberOutputs(self):
        return self.__nOutputs__

    def getOutputDimension(self):
        return self.outputDim

    def setInputDescription(self, description):
        self._inputDescription.clear()
        self._inputDescription = ot.Description(list(description))


    def setName(self, name):
        self.__name__ = name

    def setNumberOutputs(self, N):
        self.__nOutputs__ = N

    def setOutputDescription(self, description):
        assert len(description) == self.__nOutputs__
        "You must specify the name for each separate output, not for each element of each output"
        self._outputDescription.clear()
        self._outputDescription = ot.Description(description)


##############################################################################
##############################################################################
##############################################################################


class CustomList(UserList):
    '''List-like object, but with methods allowing to find indexes and values
    inside of a certain threshold, or to iterate over the different sub-dimensions
    of the list and to convert any sub-iterable into a list
    '''
    def __init__(self, data=list()):
        data = CustomList.atLeastList(data)
        super(CustomList, self).__init__(data)
        self.shape = None
        self.dtype = None
        self._getShapeDType()
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)
    def __getslice__(self, i, j):
        return self.data[i,j]
    def __add__(self, other):
        assert isinstance(other, Iterable), "nopppeee"
        other=list(other)
        data = copy(self.data)
        data.extend(other)
        return CustomList(data)
    def __repr__(self):
        return 'Custom list object with data:\n'+self.data.__repr__()
    def index(self, val, thresh=1e-5):
        dif = [abs(self.data[i]-val) for i in range(self.__len__())]
        idx = [dif[i]<=thresh for i in range(self.__len__())]
        try:
            return idx.index(True)
        except ValueError:
            return None
    def count(self, val, thresh=1e-5):
        dif = [abs(self.data[i]-val) for i in range(self.__len__())]
        idx = [dif[i]<=thresh for i in range(self.__len__())]
        return sum(idx)
    def clear(self):
        self.data = list()
    def pop(self):
        self.data.pop()
    def reverse(self):
        self.data = self.data.reverse()
        return CustomList(self.data)
    def append(self, val):
        self.data.append(val)
    def extend(self, lst):
        assert isinstance(lst, Iterable), 'TypeError'
        self.data.extend(lst)
    def copy(self):
        return CustomList(copy(self.data))
    def sort(self):
        self.data = sorted(self.data)
    def argsort(self):
        return sorted(range(len(copy(self.data))),
                                         key=lambda k: copy(self.data)[k])
    def getOrderedUnique(self):
        # custom function, returning the unique values of the list in
        # ascending order
        newList = sorted(list(set(copy(self.data))))
        return CustomList(newList)
    def all_same(self, items=None):
        #Checks if all items of a list are the same
        if items is None : items = self.data
        return all(x == items[0] for x in items)
    def _getShapeDType(self):
        L = self.data
        isHomogenous = self.all_same([type(L[i]) for i in range(len(L))])
        isIterable = all([(isinstance(L[i],Iterable) and not isinstance(L[i],str)) for i in range(len(L))])
        notEmpty = len(L)!=0
        dtype = None
        shape = []
        if len(L)==0 :
            shape.append(0)
            self.shape = tuple(shape)
            self.dtype = dtype
            return None
        if isHomogenous :
            dtype = L[0].__class__
        shape.append(len(L))
        while isHomogenous and isIterable and notEmpty:
            L = L[0]
            notEmpty = len(L)!=0
            isHomogenous = self.all_same([type(L[i]) for i in range(len(L))])
            isIterable = all([(isinstance(L[i],Iterable) and not isinstance(L[i],str)) for i in range(len(L))])
            if isHomogenous :
                dtype = L[0].__class__
            else :
                dtype = None
            shape.append(len(L))
        self.shape = tuple(shape)
        self.dtype = dtype
    def recurse2list(self):
        self.data = CustomList._iterable2list(self.data)
        self._getShapeDType()
    @staticmethod
    def _iterable2list(X):
        try :
            return [CustomList._iterable2list(x) for x in X]
        except TypeError:
            return X
    def flatten(self, L=None):
        if L is None: L = self.data
        if len(L)!=1 and not isinstance(L, (str, bytes)):
            L = list(CustomList._yielder(L))
        return CustomList(L)
    @staticmethod
    def _yielder(L):
        for el in L:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                yield from CustomList._yielder(el)
            else:
                yield el
    @staticmethod
    def atLeastList(elem):
        if isinstance(elem, Iterable) and not isinstance(elem,(str,bytes)):
            if elem.__class__.__module__ == 'numpy':
                if len(elem.shape)==1:
                    return [elem]
                else :
                    return list(elem)
            else:
                return list(elem)
        else :
            return [elem]
