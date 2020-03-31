from functools  import wraps
from time       import time

def timing(func):
    '''
    prints the executuion time of a function
    '''
    @wraps(func)
    def wrap_func(*args, **kwargs):
        t0           = time()
        result       = func(*args)
        t1           = time()
        print('timed ',round(t1-t0,9), ' s for function "',func.__name__,'"')
        return result
    return wrap_func


def safeFunction(func):
    doc = func.__doc__
    @wraps(func)
    def wrap_func(*args, **kwargs):
        try :
            results = func(*args)
            return results
        except :
            print('Error occured')
            print('Type function  : ', type(func))
            print('Type arguments : ',[type(arg) for arg in args])
            print('Help function  : ',doc)
    return wrap_func


