import functools
#from copy import deepcopy, copy
# import time
import os
# from collections.abc import Sequence, Iterable
from fractions import Fraction
import numpy as np
import pandas as pd
import openturns as ot


sample_path = './sample_storage'
if not os.path.isdir(sample_path):
    os.mkdir(sample_path)


def get_fem_vertices(min_vertices, max_vertices, n_elements):

    interval = ot.Interval([min_vertices], [max_vertices])
    mesher = ot.IntervalMesher([n_elements])
    fem_vertices = mesher.build(interval)
    return fem_vertices


def get_process_kl_decomposition(
        mean, coef_var=None, amplitude=None, scale=0, nu=1, mesh=None, dimension=1,
        name='', threshold=1e-3):

    # for matern model only
    if amplitude is None and coef_var is not None:
        amplitude = [float(mean*coef_var)]*dimension
    else:
        amplitude = [float(amplitude)]*dimension
    scale = [float(scale)]*dimension
    model = ot.MaternModel(scale, amplitude, float(nu))
    # Karhunen Loeve decomposition of process
    algorithm = ot.KarhunenLoeveP1Algorithm(mesh, model, threshold)
    algorithm.run()
    results = algorithm.getResult()
    results.setName(name)
    return results


def getRandomNormalVector(AggregatedKLRes):

    nModes = AggregatedKLRes.getSizeModes()  # the number of elements in the input vector of our KL wrapped model
    RandomNormalVector = ot.ComposedDistribution([ot.Normal()] * nModes)  #
    RandomNormalVector.setDescription(AggregatedKLRes._getModeDescription())
    return RandomNormalVector


def function_with_save_wrapper(func):
    @functools.wraps(func)
    def new_function_wrapper(arg, file_name):
        if not (os.path.isfile(os.path.join(sample_path, file_name+'.csv')) \
                and os.path.isfile(os.path.join(sample_path, file_name+'_VM'+'.csv')) \
                and os.path.isfile(os.path.join(sample_path, file_name+'_MD'+'.csv'))):
            output = func(arg)
            vonMises = ot.Sample(np.array(np.stack([np.squeeze(np.asarray(output[0][i])) for i in range(len(output[0]))])))
            maxDefl = output[1][0]
            arg.exportToCSVFile(os.path.join(sample_path,file_name+'.csv'), ';')
            vonMises.exportToCSVFile(os.path.join(sample_path,file_name+'_VM'+'.csv'), ';')
            maxDefl.exportToCSVFile(os.path.join(sample_path,file_name+'_MD'+'.csv'), ';')
            return arg, vonMises, maxDefl
        else :
            print('function was already evaluated, reloading data')
            doe = pd.read_csv(os.path.join(sample_path, file_name+'.csv'), sep=';')
            vonMises = pd.read_csv(os.path.join(sample_path, file_name+'_VM'+'.csv'), sep=';', header=None)
            maxDefl = pd.read_csv(os.path.join(sample_path, file_name+'_MD'+'.csv'), sep=';', header=None)
            sample_doe = ot.Sample(doe.values)
            sample_doe.setDescription(doe.columns)
            sample_vonMises = ot.Sample(vonMises.values)
            sample_vonMises.setDescription(ot.Description.BuildDefault(len(vonMises.columns), 'VM_'))
            sample_maxDefl = ot.Sample(maxDefl.values)
            sample_maxDefl.setDescription(ot.Description.BuildDefault(len(maxDefl.columns), 'MD_'))
            return sample_doe, sample_vonMises, sample_maxDefl
    return new_function_wrapper


def optimizedLHS(distribution, size, seed):
    ot.RandomGenerator.SetSeed(seed)
    lhs = ot.LHSExperiment(distribution, size, True, True)
    lhs_optimise = ot.SimulatedAnnealingLHS(lhs)
    lhs_sample = lhs_optimise.generate()
    return lhs_sample


