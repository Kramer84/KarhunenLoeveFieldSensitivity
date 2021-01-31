import functools
from copy import deepcopy, copy
import time
import os
from collections.abc import Sequence, Iterable
from fractions import Fraction

import numpy as np
import pandas as pd
import openturns as ot

import pythontools as pt
sample_path = './sample_storage'
if not os.path.isdir(sample_path):
    os.mkdir(sample_path)

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

try :
    import pythontools as pt
except ImportError as ie :
    raise ie
import KarhunenLoeveFieldSensitivity as klfs
from notebooksAndExamples import beamExample as MODEL
ot.ResourceMap.SetAsString('KrigingAlgorithm-LinearAlgebra', 'HMAT')

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer

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

def getRandomNormalVector(AggregatedKLRes):
    nModes = AggregatedKLRes.getSizeModes()  # the number of elements in the input vector of our KL wrapped model
    RandomNormalVector = ot.ComposedDistribution([ot.Normal()] * nModes)  #
    RandomNormalVector.setDescription(AggregatedKLRes._getModeDescription())
    return RandomNormalVector

def get_fem_vertices(min_vertices, max_vertices, n_elements):
    interval = ot.Interval([min_vertices],[max_vertices])
    mesher = ot.IntervalMesher([n_elements])
    fem_vertices = mesher.build(interval)
    return fem_vertices

def sequentialFunctionEvaulation(func, sample, limit=50000):
    #This function evaulates func so that the input samples
    # size is never over limit
    sample_size = sample.getSize()
    if sample_size > limit:
        rest = sample_size%limit
        base_size = limit
        n_reps = int((sample_size-rest)/base_size)
        outputs = list()
        for i in range(n_reps):
            outputs.append(func(sample[i*base_size:(i+1)*base_size,:]))
        if rest > 0:
            outputs.append(func(sample[(n_reps+1)*base_size:(n_reps+1)*base_size+rest,:]))
        outputSample = outputs[0]
        [outputSample.add(outputs[i]) for i in range(1,len(outputs))]
        return outputSample
    else :
        outputSample = func(sample)
        return outputSample

def get_process_kl_decomposition(mean, coef_var=None, amplitude=None, scale=0, nu=1, mesh=None, dimension=1, name='', threshold= 1e-3):
    # for matern model only
    if amplitude is None and coef_var is not None:
        amplitude = [float(mean*coef_var)]*dimension
    else :
        amplitude = [float(amplitude)]*dimension
    scale = [float(scale)]*dimension
    model = ot.MaternModel(scale, amplitude, float(nu))
    # Karhunen Loeve decomposition of process
    algorithm = ot.KarhunenLoeveP1Algorithm(mesh, model, threshold)
    algorithm.run()
    results = algorithm.getResult()
    results.setName(name)
    return results


# Little helper class for optimized lhs :
@timer
def optimizedLHS(distribution, size, seed):
    ot.RandomGenerator.SetSeed(seed)
    lhs = ot.LHSExperiment(distribution, size, True, True)
    lhs_optimise = ot.SimulatedAnnealingLHS(lhs)
    lhs_sample = lhs_optimise.generate()
    return lhs_sample

@timer
def getSample(distribution, size, seed):
    ot.RandomGenerator.SetSeed(seed)
    sample = distribution.getSample(size)
    sample.setDescription(distribution.getDescription())
    return sample

@timer
def getSobolExperiment(aggregated_karhunen_loeve_results, size, seed, secondOrder = False):
    ot.RandomGenerator.SetSeed(seed)
    experiment = klfs.KarhunenLoeveSobolIndicesExperiment(aggregated_karhunen_loeve_results, size, secondOrder)
    sobolExp = experiment.generate()
    return sobolExp, experiment



def all_same(items=None):
    #Checks if all items of a list are the same
    return all(x == items[0] for x in items)



def isValidSobolIndicesExperiment(sample_like, size, second_order = False):
    try :
        sample = np.asarray(sample_like)
    except :
        print('Could not convert sample to numpy array')
        raise TypeError
    N = sample.shape[0]
    print('N is', N)
    dim = sample.shape[1]
    Y_A = sample[:size]
    Y_B = sample[size:2*size]
    N_indice = int(N/size - 2)
    assert N%size==0,"wrong sample size"
    print('Simplified view of the sobol indices experiments structure')
    print('There are {} dimensions and {} sobol indices to calculate'.format(dim, N_indice))
    assert np.where(Y_A==Y_B,True,False).any() == False #Here we check that there is no similarity at all  between A and B
    tot_lines = list()
    try :
        for i in range(N_indice+2):
            Y_E = sample[i*size:(i+1)*size]
            tot_cols = [True]*dim
            col_where_A = list(set(np.argwhere(Y_A==Y_E)[:,1]))
            line_where_A = list(set(np.argwhere(Y_A==Y_E)[:,0]))
            if len(line_where_A)==size : OK = True
            for co in col_where_A :
                tot_cols[co] = False
            if OK :
                if len(col_where_A)==dim and all(tot_cols):
                    sl = ['A_'+str(j) for j in range(dim)]
                elif len(col_where_A)==0 and all_same(tot_cols):
                    sl = ['B_'+str(j) for j in range(dim)]
                else :
                    sl = ['B_'+str(j) for j in range(dim)]
                    for k in range(len(col_where_A)):
                        sl[col_where_A[k]] = 'A_'+str(col_where_A[k])
                l = "  ,    ".join(sl)
                l = '    '+l
            if not OK :
                print('Error')
                return False
            tot_lines.append(l)
        repres = ' \n\n'.join(tot_lines)
        repres = '\n'+repres
        print(repres)
        return True
    except :
        return False



def ereaseNanFromSample(sample_in, sample_out, N , secondOrder = False):
    sampOut = np.array(sample_out,copy=True, subok=False)
    if secondOrder == False :
        n_vars = int(sampOut.shape[0]/N) - 2
        print('n vars is', n_vars)
        N_max = int(N*(n_vars + 2))
        N_var = N_max/N
    else :
        n_vars = int(sampOut.shape[0]/(N*2)) - 1
        print('n vars is', n_vars)
        N_max = int(N*(2*n_vars + 2))
        N_var = N_max/N

    print('N_max is', N_max)
    argNan = np.argwhere(np.isnan(sampOut))[:,0].tolist()
    print('args where nan : ',argNan)
    toErease = set()
    for arg in argNan:
        whereToErease = list(range(arg%N, N_max, N))
        [toErease.add(elem) for elem in whereToErease]
    whereToErease = list(toErease)
    print('Where we are erasing:', whereToErease)
    N -= int(len(whereToErease)/N_var)
    for idx in sorted(whereToErease)[::-1]:
        sample_in.erase(idx)
        sample_out.erase(idx)
    print('N is now: ',N)
    return N



class metamodeling_kriging :
    def __init__(self, inSample, outSample, **kwargs):
        self.input_sample = inSample
        self.output_sample = outSample
        self.__default_kriging__ = None
        self.__kriging_theta__ = None
        self.__kriging_results__ = None
        self.__kriging_metamodel__ = None
        self.__kriging_model__ = None
        self.__size_multistart__ = kwargs['size_multistart'] if 'size_multistart' in kwargs else 5
        self.__lb__ = kwargs['lower_bound'] if 'lower_bound' in kwargs else None
        self.__ub__ = kwargs['upper_bound'] if 'upper_bound' in kwargs else None
        self.__optim_type__ = kwargs['optim_type'] if 'optim_type' in kwargs else 'best_start'
        self.validation_results = __validation_results__()

    def _build_default(self):
        self.__default_kriging__ = pt.build_default_kriging_algo(
                                    input_sample  = self.input_sample,
                                    output_sample = self.output_sample,
                                    basis         = None,
                                    covariance_model = None,
                                    noise         = None)

    def _estimate_theta(self):
        self.__kriging_theta__ = pt.estimate_kriging_theta(
                            algo_kriging = self.__default_kriging__,
                            lower_bound = self.__lb__,
                            upper_bound = self.__ub__,
                            size        = self.__size_multistart__,
                            optim_type  = self.__optim_type__)

    def _get_results_metamodel(self):
        if isinstance(self.__kriging_theta__,(Sequence,Iterable, list)):
            self.__kriging_results__ = [_kt.getResult() for _kt in self.__kriging_theta__]
            self.__kriging_metamodel__ = [_km.getMetaModel() for _km in self.__kriging_results__]
            self.__kriging_model__ = [_km.getModel() for _km in self.__kriging_results__]

        else :
            self.__kriging_results__ = self.__kriging_theta__.getResult()
            self.__kriging_metamodel__ = self.__kriging_results__.getMetaModel()
            self.__kriging_model__ = self.__kriging_results__.getModel()


    def run(self):
        self._build_default()
        self._estimate_theta()
        self._get_results_metamodel()
        print('Done !')

    def getKrigingResult(self):
        return self.__kriging_results__

    def getKrigingMetaModel(self):
        return self.__kriging_metamodel__

    def _check_clean_nans(self, sampleIn, sampleOut):
        whereNan = list(set(np.argwhere(np.isnan(sampleOut))[:,0]))
        if len(whereNan)>0:
            print('NaN values found at index:',whereNan)
            [(sampleOut.erase(int(val)), sampleIn.erase(int(val))) for val in whereNan]

    def getMetaModelValidation(self, sample_in_validation, sample_out_validation):
        assert self.__kriging_metamodel__ is not None, "Please first run calculus"
        assert len(sample_in_validation) == len(sample_out_validation)
        self._check_clean_nans(sample_in_validation, sample_out_validation)
        self.validation_results.clear()
        if isinstance(self.__kriging_metamodel__,(Sequence,Iterable, list)):
            for i, model in enumerate(self.__kriging_metamodel__):
                validation = ot.MetaModelValidation(sample_in_validation,
                                                    sample_out_validation[:,i],
                                                    self.__kriging_metamodel__[i])
                R2 = validation.computePredictivityFactor()
                residual = validation.getResidualSample()
                graph = validation.drawValidation()
                self.validation_results.addGraph(graph)
                self.validation_results.addR2(R2)
                self.validation_results.addResidual(residual)
        else :
            validation = ot.MetaModelValidation(sample_in_validation,
                                                sample_out_validation[:,0],
                                                self.__kriging_metamodel__)
            R2 = validation.computePredictivityFactor()
            residual = validation.getResidualSample()
            graph = validation.drawValidation()
            self.validation_results.addGraph(graph)
            self.validation_results.addR2(R2)
            self.validation_results.addResidual(residual)



class __validation_results__(object) :
    def __init__(self):
        self.__R2__ = []
        self.__residuals__ = []
        self.__graphs__ = []

    def clear(self):
        self.__R2__.clear()
        self.__residuals__.clear()
        self.__graphs__.clear()

    def addGraph(self, graph):
        self.__graphs__.append(graph)

    def addR2(self, R2):
        self.__R2__.append(R2)

    def addResidual(self, residual):
        self.__residuals__.append(residual)

    def getGraphs(self):
        for graph in self.__graphs__ :
            ot.Show(graph)

    def getResiduals(self):
        theGraph = ot.Graph('Residuals','varying dimension','residual',True,'')
        theCurve = ot.Curve(list(range(len(self.__residuals__))),
                            self.__residuals__, 'residuals')
        theGraph.add(theCurve)
        ot.Show(theGraph)

    def getR2(self):
        theGraph = ot.Graph('R2','varying dimension','residual',True,'')
        r2_sampl = ot.Sample(self.__R2__)
        R2 = r2_sampl.asPoint()
        print('R2:',R2)
        return R2



#Basic parameters of the problem
dim_field          = 1
n_elements   = 99
min_vertices = 0    #mm
max_vertices = 1000 #mm

if not os.path.isdir('./meta_analysis_results'):
    os.mkdir('./meta_analysis_results')

class _metamodel_parameter_routine:
    #routines for the calculus of multiple sobol indices, with
    #varying thresholds and varying sizes for the LHS DOE used
    #for creating the metamodel


    def __init__(self, variance_young,
                       scale_young,
                       variance_diam,
                       scale_diam,
                       var_f_pos,
                       var_f_norm):
        self.variance_young = variance_young
        self.scale_young = scale_young
        self.variance_diam = variance_diam
        self.scale_diam = scale_diam
        self.var_f_pos = var_f_pos
        self.var_f_norm = var_f_norm

        #This parameters are the means and are fixed
        self.mean_young = 210000 #MPa
        self.mean_diam = 10 #mm
        self.mean_f_pos = 500 #mm
        self.mean_f_norm = 100 #N

        #parameters of the beam model
        self.fem_vertices = get_fem_vertices(min_vertices, max_vertices, n_elements)

        # The two scalar random variables :
        self.pos_force = ot.Normal(self.mean_f_pos, self.var_f_pos)
        self.pos_force.setName('F_Pos_')

        self.norm_force  = ot.Normal(self.mean_f_norm, self.var_f_norm)
        self.norm_force.setName('F_Norm_')

        #The different thresholds that we test
        self.threshold_list = [1e-1,1e-3,1e-5]

        self.NU_list = [1/2,5/2,100]
        self._base_name = None
        self._sub_dir = None

    @timer
    def _exec_routine(self):
        #Here in the rountine we are going to calculate the sobol indices
        #corresponding to the different combination of variances and scales.
        #For each of these sobol indices, we try diverse thresholds for the
        #KL decomposition, diverse NU parameters for the matern model, as well
        #as divese LHS sizes, for the calculus the metamodel, and calculus of
        #the sobol indices on this metamodel.
        self.getBasePathAndFileName()
        name_base = copy(self._base_name)

        #first we iterate over the thresholds:
        for threshold in self.threshold_list :
            thresh_str = np.format_float_scientific(threshold, precision = 1,exp_digits=1)
            print('threshold is',thresh_str)
            sub_sub_dir = os.path.join(self.sub_dir ,'threshold{}'.format(thresh_str))
            if not os.path.isdir(sub_sub_dir):
                os.mkdir(sub_sub_dir)

            #then we iterate over the nus:
            for nu in self.NU_list :
                nu_str = str(round(nu, 3))
                csv_name = 'NU{}'.format(nu_str)+'.csv'
                csv_file_path = os.path.join(sub_sub_dir,csv_name)
                kl_results_E, kl_results_D = self.getKLDecompositionYoungDiam(threshold, nu)
                AggregatedKLRes = self.getAggregatedKLResults(kl_results_E, kl_results_D)
                FEM_model = self.getFEMModel(AggregatedKLRes)
                result_sample = self._exec_sub_routine(AggregatedKLRes, FEM_model)
                result_sample.exportToCSVFile(csv_file_path,';')


    def _exec_sub_routine(self, AggregatedKLRes, FEM_model):
        # Here we do the calulus work.
        # First we calculate the "real sobol indices" on the real
        # model with a sample size of 2000
        # Then we calculate the response of the model to the different LHS
        # samples, and each metamodel associated to it
        # Then we recalculate the sobol indices with the metamodel
        modes = AggregatedKLRes.__mode_count__
        N_KL_Young = modes[0]
        N_KL_Diam = modes[1]
        RandomNormalVector = getRandomNormalVector(AggregatedKLRes)
        SEED0 = 948546882996
        SEED1 = 98577599025
        SEED2 = 911745283
        SEED3 = 68071771823
        SEED4 = 387349900932
        SEED5 = 1275728859
        doe_sobol_experiment_N1000, _ = getSobolExperiment(AggregatedKLRes, 1000, SEED0)
        doe_metasobol_experiment_N50000, _ = getSobolExperiment(AggregatedKLRes, 50000, SEED5)

        #Here we create the design of experiments (DOE) for the metamodels and the validation
        doe_kriging_LHS25 = optimizedLHS(RandomNormalVector, 25, SEED1)
        doe_kriging_LHS50 = optimizedLHS(RandomNormalVector, 50, SEED2)
        doe_kriging_LHS100 = optimizedLHS(RandomNormalVector, 100, SEED3)
        doe_kriging_valid = optimizedLHS(RandomNormalVector, 600, SEED4)
        #now we evaluate the function for the metamodels
        doe_kriging_LHS25_VM, doe_kriging_LHS25_MD = FEM_model(doe_kriging_LHS25)
        doe_kriging_LHS50_VM, doe_kriging_LHS50_MD = FEM_model(doe_kriging_LHS50)
        doe_kriging_LHS100_VM, doe_kriging_LHS100_MD = FEM_model(doe_kriging_LHS100)
        #And here we evaluate the function for the Sobol Indices
        doe_sobol_experiment_N1000_VM, doe_sobol_experiment_N1000_MD = FEM_model(doe_sobol_experiment_N1000)
        doe_kriging_valid_VM, doe_kriging_valid_MD = FEM_model(doe_kriging_valid)
        R2_LHS25, kriging_model_LHS25 = self.metamodel_kriging_validation(doe_kriging_LHS25, doe_kriging_LHS25_MD, doe_kriging_valid, doe_kriging_valid_MD)
        R2_LHS50, kriging_model_LHS50 = self.metamodel_kriging_validation(doe_kriging_LHS50, doe_kriging_LHS50_MD, doe_kriging_valid, doe_kriging_valid_MD)
        R2_LHS100, kriging_model_LHS100 = self.metamodel_kriging_validation(doe_kriging_LHS100, doe_kriging_LHS100_MD, doe_kriging_valid, doe_kriging_valid_MD)
        #Now we have to calculate the sobol indices
        result_point_DOE1000 = self.getSensitivityAnalysisResults(doe_sobol_experiment_N1000, doe_sobol_experiment_N1000_MD, 1000, N_KL_Young, N_KL_Diam)

        print('Sobol indices for 50000 size - metamodel')
        print('LHS25')
        result_point_DOE50000_LHS25 = self.getSensitivityAnalysisResultsMetamodel(kriging_model_LHS25, doe_metasobol_experiment_N50000, 50000, 25, N_KL_Young, N_KL_Diam, R2_LHS25)
        print('LHS50')
        result_point_DOE50000_LHS50 = self.getSensitivityAnalysisResultsMetamodel(kriging_model_LHS50, doe_metasobol_experiment_N50000, 50000, 50, N_KL_Young, N_KL_Diam, R2_LHS25)
        print('LHS100')
        result_point_DOE50000_LHS100 = self.getSensitivityAnalysisResultsMetamodel(kriging_model_LHS100, doe_metasobol_experiment_N50000, 50000, 100, N_KL_Young, N_KL_Diam, R2_LHS25)

        description = result_point_DOE1000.getDescription()
        ResultsSample = ot.Sample([result_point_DOE1000, result_point_DOE50000_LHS25, result_point_DOE50000_LHS50, result_point_DOE50000_LHS100])
        ResultsSample.setDescription(description)
        return ResultsSample

    def exporteSampleAsCsv(self, listSamples, listNames):
        if not os.path.isdir('./segfault_analysis/debug_storage'):
            os.mkdir('./segfault_analysis/debug_storage')
        assert len(listSamples)==len(listNames)
        for i, sample in enumerate(listSamples):
            sample.exportToCSVFile(os.path.join('./segfault_analysis/debug_storage',listNames[i]))


    def getSensitivityAnalysisResults(self, sample_in, sample_out, size, N_KL_Young, N_KL_Diam):
        dim = sample_in.getDimension() # Dimensoin of the KL input vector
        sensitivity_analysis = klfs.SobolKarhunenLoeveFieldSensitivityAlgorithm()
        sensitivity_analysis.setDesign(sample_in, sample_out, size)
        sensitivity_analysis.setEstimator(ot.MartinezSensitivityAlgorithm())
        FO_indices = sensitivity_analysis.getFirstOrderIndices()[0]
        conf_level = sensitivity_analysis.getFirstOrderIndicesInterval()[0]
        lb = conf_level.getLowerBound()
        ub = conf_level.getUpperBound()
        result_point = ot.PointWithDescription([('meta',0.),('N_LHS',-1.),('size',size),('kl_dimension',dim),
                                             ('N_KL_Young',N_KL_Young), ('N_KL_Diam',N_KL_Diam), ('R2',-1),
                                             ('SI_E',round(FO_indices[0][0],5)),('SI_E_lb',round(lb[0],5)),('SI_E_ub',round(ub[0],5)),
                                             ('SI_D',round(FO_indices[1][0],5)),('SI_D_lb',round(lb[1],5)),('SI_D_ub',round(ub[1],5)),
                                             ('SI_FP',round(FO_indices[2][0],5)),('SI_FP_lb',round(lb[2],5)),('SI_FP_ub',round(ub[2],5)),
                                             ('SI_FN',round(FO_indices[3][0],5)),('SI_FN_lb',round(lb[3],5)),('SI_FN_ub',round(ub[3],5))])
        print('------------- RESULTS ------------')
        print('------------ REAL MODEL ----------')
        print('----------- SIZE DOE = {} -----------'.format(str(int(size))))
        print(result_point)
        return result_point


    def getSensitivityAnalysisResultsMetamodel(self, metamodel, doe_sobol, size, N_LHS, N_KL_Young, N_KL_Diam, R2):
        dim = doe_sobol.getDimension()
        try :
            meta_response = sequentialFunctionEvaulation(metamodel.__kriging_metamodel__, doe_sobol)
        except Exception as e:
            print('Caugt exception:\n',e)
            print('filling response with zeros...')
            meta_response = ot.Sample(np.zeros((doe_sobol.getSize(),1)))
        sensitivity_analysis = klfs.SobolKarhunenLoeveFieldSensitivityAlgorithm()
        sensitivity_analysis.setDesign(doe_sobol, meta_response, size)
        sensitivity_analysis.setEstimator(ot.MartinezSensitivityAlgorithm())
        FO_indices = sensitivity_analysis.getFirstOrderIndices()[0]
        conf_level = sensitivity_analysis.getFirstOrderIndicesInterval()[0]
        lb = conf_level.getLowerBound()
        ub = conf_level.getUpperBound()
        result_point = ot.PointWithDescription([('meta',1.),('N_LHS',N_LHS),('size',size),('kl_dimension',dim),
                                             ('N_KL_Young',N_KL_Young), ('N_KL_Diam',N_KL_Diam), ('R2',R2),
                                             ('SI_E',FO_indices[0][0]),('SI_E_lb',lb[0]),('SI_E_ub',ub[0]),
                                             ('SI_D',FO_indices[1][0]),('SI_D_lb',lb[1]),('SI_D_ub',ub[1]),
                                             ('SI_FP',FO_indices[2][0]),('SI_FP_lb',lb[2]),('SI_FP_ub',ub[2]),
                                             ('SI_FN',FO_indices[3][0]),('SI_FN_lb',lb[3]),('SI_FN_ub',ub[3])])
        print('------------- RESULTS ------------')
        print('------------ META MODEL ----------')
        print('-----------SIZE LHS : {} ---------'.format(int(N_LHS)))
        print(result_point)
        return result_point

    def metamodel_kriging_validation(self, doe_in, doe_out, doe_validation_in, doe_validation_out):
        kriging_model = metamodeling_kriging(doe_in, doe_out,
                    optim_type='multi_start', size_multistart = 10)
        kriging_model.run()
        kriging_model.getMetaModelValidation(doe_validation_in, doe_validation_out)
        R2 = kriging_model.validation_results.getR2()[0] #for now works only with 1D outputs
        return R2, kriging_model

    def getBasePathAndFileName(self):
        base_string ='EXP_'+'VE'+str(int(self.variance_young))+'_'+'SE'+str(int(self.scale_young))\
            +'_'+'VD'+str(round(self.variance_diam,4)) +'_'+'SD'+str(int(self.scale_diam))\
            +'_'+'VFP'+str(round(self.var_f_pos,3))    +'_'+'VFN'+str(round(self.var_f_norm,3))+'_'
        self.sub_dir = './meta_analysis_results/'+base_string
        self._base_name = base_string
        if not os.path.isdir(self.sub_dir):
            os.mkdir(self.sub_dir)

    def getKLDecompositionYoungDiam(self, threshold, nu):
        kl_results_E = get_process_kl_decomposition(
                        mean = self.mean_young, amplitude = self.variance_young , scale = self.scale_young,
                        nu = nu, mesh = self.fem_vertices, dimension = dim_field,
                        name = 'E_', threshold = threshold)

        kl_results_D = get_process_kl_decomposition(
                        mean = self.mean_diam, amplitude = self.variance_diam, scale = self.scale_diam,
                        nu = nu, mesh = self.fem_vertices, dimension = dim_field,
                        name = 'D_', threshold = threshold)
        return kl_results_E, kl_results_D

    def getAggregatedKLResults(self, kl_results_E, kl_results_D):
        kl_results_list = [kl_results_E, kl_results_D, self.pos_force, self.norm_force]
        AggregatedKLRes = klfs.AggregatedKarhunenLoeveResults(kl_results_list)
        AggregatedKLRes.setMean(0, self.mean_young) # At other indices the means are initialized from the distributions
        AggregatedKLRes.setMean(1, self.mean_diam)
        AggregatedKLRes.setLiftWithMean(True)
        return AggregatedKLRes

    def getFEMModel(self, AggregatedKLRes):
        # definition of the model :
        _MODEL = MODEL.PureBeam()
        # initialization of the function wrapper :
        FUNC = klfs.KarhunenLoeveGeneralizedFunctionWrapper(
                    AggregatedKarhunenLoeveResults = AggregatedKLRes,
                    func        = None,
                    func_sample = _MODEL.batchEval,
                    n_outputs   = 2)
        def FUNCwrap(arg):
            output = FUNC(arg)
            vonMises = ot.Sample(np.array(np.stack([np.squeeze(np.asarray(output[0][i])) for i in range(len(output[0]))])))
            maxDefl = output[1][0]
            return vonMises, maxDefl
        return FUNCwrap


class globalMetaParameterAnalysis:
    def __init__(self,):
        # Variance distributions :
        SEED = 43515687355
        var_ampl_Young = ot.Uniform(100, 13000)
        var_scale_Young = ot.Uniform(50,1000)
        var_ampl_Diam = ot.Uniform(.01, .5)
        var_scale_Diam = ot.Uniform(50,1000)
        var_var_pos = ot.Uniform(1,100)
        var_var_norm = ot.Uniform(1,20)
        ComposedMetaParamDistribution = ot.ComposedDistribution([var_ampl_Young, var_scale_Young, var_ampl_Diam, var_scale_Diam, var_var_pos, var_var_norm])
        RandomNormalVector = ot.ComposedDistribution([ot.Normal()] * 6) #Vector used for the LHS generation
        normalizedLHS = optimizedLHS(RandomNormalVector, 20, SEED)
        inverseIsoProbaTransform = ComposedMetaParamDistribution.getInverseIsoProbabilisticTransformation()
        realLHS = inverseIsoProbaTransform(normalizedLHS)
        print('realLHS is',realLHS)
        self.run_experience(realLHS)

    @timer
    def run_experience(self, LHS):
        for doe in LHS:
            l = list(doe)
            experiment = _metamodel_parameter_routine(*l)
            experiment._exec_routine()
            print('--------- Time for one experiment above -------\n\n\n\n\n')


if __name__=='__main__':
    routine = globalMetaParameterAnalysis()

'''
import _base_algorithms as ba
test = ba._metamodel_parameter_routine(10000,250,.3,250,10,10)
test._exec_routine()

import _base_algorithms as ba
routine = ba.globalMetaParameterAnalysis()


'''

'''
Est-ce qu'il est pertinent d'utiliser un métamodèle pour calculer les indices de Sobol' ?
    - > Oui! Mais, ...


Quelle précision du Kriegage faut il avoir pour parvenir à des indices de Sobol' cohérents en passant par le MM ?

    -> Refaire calcul, recuperer R2, Q2 + enregistre 'kriging_theta'

Comment évoluent les indices de Sobol lorsqu'on fait varier les paramètres des modèles de covariance des champs?
    -> PairPlot (Seaborn) ! Ou pandas (
        -> Abscisse : (amplitudes, scales ou vars ) , ordonnées : SobolIndice , couleur : Sobol


Comment influe la "complexité" du champ stochastique sur l'approximation des indices de Sobol' par le métamodèle?
    - > Approximation du champ par KL (en dehors de l'analyse de sensibilité)

    - > Montrer que l'utilisation du kriegage diminue drastiquement le nombre d'appels au
     modèle réel

    - > Montrer à partir de quelle différence entre la taille du LHS et la dimension du vecteur CNR (Xi)  l'approximation de l'indice de Sobol' debient trop mauvaise  (BOF)

'''