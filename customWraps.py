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
5.20,7.00,7.23,7.88,7.95,8.23,8.25,9.03,9.25,9.56,9.83,10.00,10.00,10.03,10.25,10.40,10.50,10.75,10.80,11.00,11.10,11.29,11.30,11.50,11.50,12.00,12.00,12.10,12.15,12.17,12.27,12.33,12.64,13.00,13.43,13.50,13.53,13.67,13.75,14.00,14.00,14.00,14.10,14.15,14.50,14.50,14.50,14.60,14.80,14.90,15.00,15.00,15.10,15.14,16.00,16.33,16.34,16.50,16.97,17.00,17.00,17.50,17.57,18.00,18.10,18.30,19.38,19.40,20.00