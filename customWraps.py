from functools import wraps
import logging
from time import time

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



def exceptionLogger():
    """
    Creates a logging object and returns it
    """
    logger = logging.getLogger("example_logger")
    logger.setLevel(logging.INFO)
    # create the logging file handler
    fh = logging.FileHandler("/path/to/test.log")
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    # add handler to logger object
    logger.addHandler(fh)
    return logger


def exception(function):
    """
    A decorator that wraps the passed in function and logs 
    exceptions should one occur
    """
    @wraps(function)
    def exception(*args, **kwargs):
        logger = exceptionLogger()
        try:
            return function(*args, **kwargs)
        except:
            # log the exception
            err = "There was an exception in  "
            err += function.__name__
            logger.exception(err)
            # re-raise the exception
            raise
    return wrapper

