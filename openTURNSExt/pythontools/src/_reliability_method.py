#-*- coding: utf-8 -*-

__all__ = ['run_monte_carlo', 'run_importance_sampling', 'run_subset', 'run_FORM',
           'run_directional', 'SystemEvent']

import openturns as ot
import numpy as np
from distutils.version import LooseVersion

class SystemEvent():
    """
    Return the system event either union or intersection

    Parameters
    ----------
    list_event : list of :class:`openturns.Event`
        The list of event that compose the intersection or the union.
    failure_domain : string
        The type of failure domain : "union" or "intersection"

    Returns
    -------
    systemEvent : :class:`openturns.Event`
        The unique system event.
    """

    def __init__(self, list_event, failure_domain):
        self.list_event = list_event
        self.failure_domain = failure_domain

        n_event = len(list_event)
        distribution = list_event[0].getAntecedent().getDistribution()

        # par convention g_composite < 0 --> failure
        convention_function = ot.SymbolicFunction(['g', 'coef', 'threshold'],
                                                  ['coef*(g - threshold)'])
        model_list = ot.FunctionCollection(len(list_event))
        model_convention_list = ot.FunctionCollection(len(list_event))
        perf_function_list = ot.FunctionCollection(len(list_event))
        for i, event in enumerate(list_event):
            model_list[i] = event.getFunction()
            # permet de revenir à contrainte > 0 (cohérent déf pb d'optim)
            if event.getOperator().getImplementation().getClassName() in ['Less', 'LessOrEqual']:
                coef = 1
            elif event.getOperator().getImplementation().getClassName() in ['Greater', 'GreaterOrEqual']:
                coef = -1

            threshold = event.getThreshold()
            # function qui retourne g tel que def = g < 0
            perf_function_list[i] = ot.ParametricFunction(convention_function, [1, 2], [coef, threshold])
            # function g(X) tel que g < 0
            model_convention_list[i] = ot.ComposedFunction(perf_function_list[i], model_list[i])

        if failure_domain == 'union':
            # build the composite function : min(g0, ..., gn) < 0 and event
            text_function = 'min(g0'
        elif failure_domain == 'intersection':
            # build the composite function : max(g0, ..., gn) < 0 and event
            text_function = 'max(g0'

        # complete the symbolic function text
        for i in range(1, len(list_event)):
            text_function += ',g%s' %i
        text_function += ')'


        symbolic_function = ot.SymbolicFunction(ot.Description.BuildDefault(len(list_event), 'g'),
                                                [text_function])
        composite_function = ot.ComposedFunction(symbolic_function,
                                    ot.AggregatedFunction(model_convention_list))
        self.system_event = ot.ThresholdEvent(ot.CompositeRandomVector(composite_function, 
                                     ot.RandomVector(distribution)),
                                     ot.Less(), 0.)

    def getSystemEvent(self):
        """
        Return the system event
        """
        return self.system_event

    def getEventList(self):
        """
        Return the list of events
        """
        return self.list_event

    def getFailureDomain(self):
        """
        Return the type of failure domain
        """
        return self.failure_domain

def run_monte_carlo(event, coef_var=0.1, outer_sampling=10000, block_size=10,
                   seed=1234, logfile=False, verbose=False):
    
    """
    Run a Monte Carlo simulation.

    Parameters
    ----------
    event : :class:`:class:`openturns.Event` or SystemEvent
        The failure event.
    coef_var : float
         The target coefficient of variation.
    outer_sampling : int
        The maximum number of outer iterations.
        Nb of iterations = outer_sampling x block_size. 
    block_size : int
        The number of samples send to evaluate simultaneously.
    seed : int
        Seed for the openturns random generator.
    logfile : bool
        Enable or not to write the log in MonteCarlo.log file.
    verbose : bool
        Enable or not the display of the result.
    """

    if type(event) == SystemEvent:
        # case with the limit state defined as an intersection or a union of the event
        new_event = event.getSystemEvent()
    elif type(event) == ot.ThresholdEvent:
        new_event = event

    # Initialize the random generator
    ot.RandomGenerator.SetSeed(seed)

    # Gestion du log pour recuperer les infos lors du FORM run
    if logfile:
        ot.Log.Show(ot.Log.INFO + ot.Log.WARN + ot.Log.USER)
        ot.Log.SetFile('MonteCarlo.log')

    #Run Monte Carlo simulation
    simulation = ot.ProbabilitySimulationAlgorithm(new_event)
    simulation.setMaximumCoefficientOfVariation(coef_var)
    simulation.setMaximumOuterSampling(outer_sampling)
    simulation.setBlockSize(block_size)

    # try:
    simulation.run()
    # except Exception as e:
    #     dump_cache(new_event.getFunction(), 'Cache/physicalModelMathFunction')
    #     raise e

    result = simulation.getResult()

    textResultat = '' \
        '\n Result - Monte Carlo'+ \
        '\n --------------------------------------------------------------------' \
        '\n Probability of failure = {:.2e}'.format(result.getProbabilityEstimate()) + \
        '\n Coefficient of varation = {:0.3f} '.format(result.getCoefficientOfVariation()) + \
        '\n 95 % Confidence length = {:.2e}'.format(result.getConfidenceLength(0.95)) + \
        '\n Number of calls = {} '.format(result.getOuterSampling()*result.getBlockSize()) + \
        '\n --------------------------------------------------------------------'

    if verbose:
        print(textResultat)
    if logfile:
        ot.Log.User(textResultat)
        ot.Log.User('-------------------------------------------------------------------')

    return result


def run_directional(event, root_strategy=ot.SafeAndSlow(),
                    sampling_strategy=ot.RandomDirection(), maximum_distance=8,
                    step_size=1, coef_var=0.1, outer_sampling=10000, block_size=10,
                    seed=1234, logfile=False, verbose=False):
    
    """
    Run a directional simulation.

    Parameters
    ----------
    event : :class:`openturns.Event` or SystemEvent
        The failure event.
    root_strategy : :class:`openturns.RootStrategy`
        The strategy to search the root point (RiskyAndFast, MediumSafe, SafeAndSlow)
    sampling_strategy : :class:`openturns.SamplingStrategy`
        The strategy to sample the search directions (RandomDirection, OrthogonalDirection)
    maximum_distance : float
        The distance in the standard space to stop searching for roots.
    step_size : float > 0
        The length of each segment inside which the root research is performed.
    coef_var : float
         The target coefficient of variation.
    outer_sampling : int
        The maximum number of outer iterations.
        Nb of iterations = outer_sampling x block_size. 
    block_size : int
        The number of samples send to evaluate simultaneously.
    seed : int
        Seed for the openturns random generator.
    logfile : bool
        Enable or not to write the log in Directional.log file.
    verbose : bool
        Enable or not the display of the result.
    """

    if type(event) == SystemEvent:
        # case with the limit state defined as an intersection or a union of the event
        new_event = event.getSystemEvent()
    elif type(event) == ot.ThresholdEvent:
        new_event = event

    # Initialize the random generator
    ot.RandomGenerator.SetSeed(seed)

    # Gestion du log pour recuperer les infos lors du FORM run
    if logfile:
        ot.Log.Show(ot.Log.INFO + ot.Log.WARN + ot.Log.USER)
        ot.Log.SetFile('Directional.log')

    # options of the root strategy
    root_strategy.setMaximumDistance(maximum_distance)
    root_strategy.setStepSize(step_size)

    #Run Monte Carlo simulation
    simulation = ot.DirectionalSampling(new_event)
    simulation.setRootStrategy(root_strategy)
    simulation.setSamplingStrategy(sampling_strategy)
    simulation.setMaximumCoefficientOfVariation(coef_var)
    simulation.setMaximumOuterSampling(outer_sampling)
    simulation.setBlockSize(block_size)

    # try:
    simulation.run()
    # except Exception as e:
    #     dump_cache(model, 'Cache/physicalModelMathFunction')
    #     raise e

    result = simulation.getResult()

    textResultat = '' \
        '\n Result - Directional sampling'+ \
        '\n --------------------------------------------------------------------' \
        '\n Probability of failure = {:.2e}'.format(result.getProbabilityEstimate()) + \
        '\n Coefficient of varation = {:0.3f} '.format(result.getCoefficientOfVariation()) + \
        '\n 95 % Confidence length = {:.2e}'.format(result.getConfidenceLength(0.95)) + \
        '\n Number of calls = {} '.format(result.getOuterSampling()*result.getBlockSize()) + \
        '\n --------------------------------------------------------------------'

    if verbose:
        print(textResultat)
    if logfile:
        ot.Log.User(textResultat)
        ot.Log.User('-------------------------------------------------------------------')

    return result


def run_importance_sampling(event, pstar, sd=1., coef_var=0.1, outer_sampling=1000,
                           block_size=10, seed=1234, logfile=False, verbose=False):
    """
    Run an importance sampling simulation.

    Parameters
    ----------
    event : :class:`openturns.Event` or SystemEvent
        The failure event.
    pstar : list of points
        Design points in the standard space where to centered the instrumental
        distribution.
    sd : positive float
        The standard deviation of the instrumental distribution.
    coef_var : float
         The target coefficient of variation.
    outer_sampling : int
        The maximum number of outer iterations.
        Nb of iterations = outerSampling x block_size. 
    block_size : int
        The number of samples send to evaluate simultaneously.
    seed : int
        Seed for the openturns random generator.
    logfile : bool
        Enable or not to write the log in ImportanceSampling.log file.
    verbose : bool
        Enable or not the display of the result.
    """
 
    if type(event) == SystemEvent:
        # case with the limit state defined as an intersection or a union of the event
        new_event = event.getSystemEvent()
    elif type(event) == ot.ThresholdEvent:
        new_event = event

    # Initialize the random generator
    ot.RandomGenerator.SetSeed(seed)

    # Gestion du log pour recuperer les infos lors du FORM run
    if logfile:
        ot.Log.Show(ot.Log.INFO + ot.Log.WARN + ot.Log.USER)
        ot.Log.SetFile('ImportanceSampling.log')

    dim = new_event.getFunction().getInputDimension()
    pstar = np.atleast_2d(pstar)
    n_pstar = pstar.shape[0]

    stdev = [sd] * dim
    corr = ot.CorrelationMatrix(dim)
    if n_pstar > 1:
        distribution_list = list()
        for point in pstar:
            distribution_list.append(ot.Normal(point, stdev, corr))
        instrumental_distribution = ot.Mixture(distribution_list)
    elif n_pstar == 1:
        instrumental_distribution = ot.Normal(pstar[0], stdev, corr)

    #Run importance sampling simulation
    instrumental_experiment = ot.ImportanceSamplingExperiment(instrumental_distribution)
    standard_event = ot.StandardEvent(new_event)
    simulation = ot.ProbabilitySimulationAlgorithm(standard_event, instrumental_experiment)
    simulation.setMaximumOuterSampling(outer_sampling)
    simulation.setBlockSize(block_size)
    simulation.setMaximumCoefficientOfVariation(coef_var)

    # try:
    simulation.run()
    # except Exception as e:
    #     dump_cache(model, 'Cache/physicalModelMathFunction')
    #     raise e

    result = simulation.getResult()


    textResultat = '' \
        '\n Result - Importance Sampling'+ \
        '\n --------------------------------------------------------------------' \
        '\n Probability of failure = {:.2e}'.format(result.getProbabilityEstimate()) + \
        '\n Coefficient of varation = {:0.3f} '.format(result.getCoefficientOfVariation()) + \
        '\n 95 % Confidence length = {:.2e}'.format(result.getConfidenceLength(0.95)) + \
        '\n Number of calls = {} '.format(result.getOuterSampling()*result.getBlockSize()) + \
        '\n --------------------------------------------------------------------'

    if verbose:
        print(textResultat)
    if logfile:
        ot.Log.User(textResultat)
        ot.Log.User('-------------------------------------------------------------------')

    return result


def run_subset(event, conditional_pf=0.1, iterationPerStep=2000, 
               block_size=20, seed=1234, logfile=False, verbose=False):
    """
    Run a Subset simulation.

    Parameters
    ----------
    event : :class:`openturns.Event` or SystemEvent
        The failure event.
    conditional_pf : 0 < float < 1
        The target probability value at each step.
    iterationPerStep : int
        The number of iterations at each step.
    block_size : int
        The number of samples send to evaluate simultaneously.
    seed : int
        Seed for the openturns random generator.
    logfile : bool
        Enable or not to write the log in Subset.log file.
    verbose : bool
        Enable or not the display of the result.
    """

    if type(event) == SystemEvent:
        # case with the limit state defined as an intersection or a union of the event
        new_event = event.getSystemEvent()
    elif type(event) == ot.ThresholdEvent:
        new_event = event

    # Initialize the random generator
    ot.RandomGenerator.SetSeed(seed)

    if logfile:
        ot.Log.Show(ot.Log.INFO + ot.Log.WARN + ot.Log.USER)
        ot.Log.SetFile('Subset.log')

    # Create a simulation algorithm
    simulation = ot.SubsetSampling(new_event)
    simulation.setConvergenceStrategy(ot.HistoryStrategy(ot.Compact(50)))
    simulation.setBlockSize(block_size)
    simulation.setMaximumOuterSampling(int(iterationPerStep/simulation.getBlockSize()))
    simulation.setConditionalProbability(conditional_pf)

    # try:
    simulation.run()
    # except Exceptione:
        # dump_cache(model, 'Cache/physicalModelMathFunction')
        # raise e

    result = simulation.getResult()
    SS_evaluation_number = block_size * result.getOuterSampling() * simulation.getNumberOfSteps()

    textResultat = '' \
        '\n Result - Method Subset' \
        '\n --------------------------------------------------------------------' \
        '\n Failure probability = {:.2e}'.format(result.getProbabilityEstimate()) + \
        '\n Coefficient of variation = {:0.3f}'.format(result.getCoefficientOfVariation()) + \
        '\n 95 % Confidence length = {:.2e}'.format(result.getConfidenceLength(0.95)) + \
        '\n Number of evaluations = {}'.format(SS_evaluation_number) + \
        '\n Number of steps = {}'.format(simulation.getNumberOfSteps()) + \
        '\n ------------------------------------------------------------------- ' 

    if verbose:
        print(textResultat)
    if logfile:
        ot.Log.User(textResultat)
        ot.Log.User('-------------------------------------------------------------------')

    return simulation

def run_FORM_simple(event, nearest_point_algo='AbdoRackwitz',
              n_max_iteration=100, eps=[1e-4]*4, physical_starting_point=None,
              use_multi_form=False, seed=1234,
              logfile=False, verbose=False):
    
    """
    Run a FORM approximation.

    Parameters
    ----------
    event : :class:`openturns.Event`
        The failure event.
    nearest_point_algo : str
        Type of the optimization algorithm. It must be 'AbdoRackwitz', 'SQP' or
        'Cobyla'.
    n_max_iteration : int
        The maximum number of iterations.
    eps = sequence of float
        The stopping criterion value of the optimization algorithm. Order is 
        absolute error, relative error, residual error, constraint error.
    physical_starting_point : sequence of float
        The starting point of the algorithm. Default is the median values.
    use_multi_form : bool
        Choose to use the MultiFORM algorithm instead of classical FORM in order
        to find several design points.
    seed : int
        Seed for the openturns random generator.
    logfile : bool
        Enable or not to write the log in FORM.log file.
    verbose : bool
        Enable or not the display of the result.
    """
    
    distribution = event.getAntecedent().getDistribution()

    # Initialize the random generator
    ot.RandomGenerator.SetSeed(seed)

    #Defintion of the nearest point algorithm
    if nearest_point_algo=='AbdoRackwitz':
        algo = ot.AbdoRackwitz()
        #spec = algo.getSpecificParameters()
        #spec.setTau(0.5)
        #algo.setSpecificParameters(spec)
    elif nearest_point_algo=='Cobyla':
        algo = ot.Cobyla()
    elif nearest_point_algo=='SQP':
        algo = ot.SQP()
    else:
        raise NameError("Nearest point algorithm name must be \
                            'AbdoRackwitz', 'Cobyla' or 'SQP'.")

    eps = np.array(eps)
    algo.setMaximumAbsoluteError(eps[0])
    algo.setMaximumRelativeError(eps[1])
    algo.setMaximumResidualError(eps[2])
    algo.setMaximumConstraintError(eps[3])
    algo.setMaximumIterationNumber(n_max_iteration)

    #Set the physical starting point of the Nearest point algorithm to the mediane value
    if physical_starting_point is None:
        dim = event.getFunction().getInputDimension()
        physical_starting_point = ot.Point(dim)
        for i in range(dim):
            marginal = distribution.getMarginal(i)
            physical_starting_point[i] = marginal.computeQuantile(0.5)[0]

    # Gestion du log pour recuperer les infos lors du FORM run
    if logfile:
        ot.Log.Show(ot.Log.INFO + ot.Log.WARN + ot.Log.USER)
        ot.Log.SetFile('FORM.log')
        ot.Log.User('Physical Starting Point : {}'.format(physical_starting_point))

    #Run FORM method
    if use_multi_form:
        approximation = ot.MultiFORM(algo, event, physical_starting_point)
    else:
        approximation = ot.FORM(algo, event, physical_starting_point)

    # try:
    approximation.run()
    # except Exception as e:
    #     dump_cache(model, 'Cache/physicalModelMathFunction')
    #     raise e

    result = approximation.getResult()
    if use_multi_form:
        iter_number = ' + '.join(str(nb)
                        for nb in [form_res.getOptimizationResult().getEvaluationNumber()
                        for form_res in result.getFORMResultCollection()])
        standard_space_design_point = a = [form_res.getStandardSpaceDesignPoint()
                        for form_res in result.getFORMResultCollection()]
    else:
        iter_number = str(result.getOptimizationResult().getIterationNumber())
        standard_space_design_point = result.getStandardSpaceDesignPoint()

    eval_number = ot.MemoizeFunction(event.getFunction()).getInputHistory().getSize()

    textResultat = '' \
        '\n Result - FORM (' + nearest_point_algo +')'+ \
        '\n --------------------------------------------------------------------' \
        '\n Probability of failure : %0.2e' % result.getEventProbability() + \
        '\n Generalised reliability index: %.4f' % result.getGeneralisedReliabilityIndex() + \
        '\n Number of iterations = {}'.format(iter_number) + \
        '\n Number of function calls = {} '.format(eval_number) + \
        '\n Standard space design point:\n{}'.format(standard_space_design_point) + \
        '\n --------------------------------------------------------------------'
        # '\n List of points: {}'.format(distribution.getDescription()) + \
        # '\n Physical space design point: %s' % result.getPhysicalSpaceDesignPoint() + \
        # '\n ----------------------------------------------------------------------------- ' + \

    if verbose:
        print(textResultat)
    if logfile:
        ot.Log.User(textResultat)

    # for i in xrange(result.getImportanceFactors().getSize()):
    #     IF = 'Importance factor %s: %.2f' % (distribution.getDescription()[i], 
    #                                 result.getImportanceFactors()[i])
    #     # print(IF)
    #     if logfile:
    #         ot.Log.User(IF)

    if logfile:
        ot.Log.User('-------------------------------------------------------------------')

    return approximation


def run_FORM_multi_constraint(event_list, nearest_point_algo='LD_SLSQP',
              n_max_iteration=100, eps=[1e-4]*4, physical_starting_point=None, seed=1234,
              logfile=False, verbose=False):
    
    """
    Run a FORM system approximation based on a multi constraint resolution.

    Parameters
    ----------
    event_list : list of :class:`openturns.Event`
        The failure events.
    nearest_point_algo : str
        Type of the optimization algorithm, defaut is LD_SLSQP
    n_max_iteration : int
        The maximum number of iterations.
    eps = sequence of float
        The stopping criterion value of the optimization algorithm. Order is 
        absolute error, relative error, residual error, constraint error.
    physical_starting_point : sequence of float
        The starting point of the algorithm. Default is the median values.
    seed : int
        Seed for the openturns random generator.
    logfile : bool
        Enable or not to write the log in FORM.log file.
    verbose : bool
        Enable or not the display of the result.
    """


    # Initialize the random generator
    ot.RandomGenerator.SetSeed(seed)

    #Defintion of the nearest point algorithm
    if nearest_point_algo not in ['LD_SLSQP', 'LD_MMA', 'AUGLAG', 'AUGLAG_EQ', 'LN_COBYLA',
        'GN_ISRES', 'LD_AUGLAG', 'LD_AUGLAG_EQ', 'LN_AUGLAG', 'LN_AUGLAG_EQ',
        'GN_ORIG_DIRECT']:
        ot.Log.Warn('Algorithm not known : change nearest point algorithm to "LD_SLSQP".')
        nearest_point_algo = 'LD_SLSQP'
        # raise NameError("Nearest point algorithm name must be 'LD_SLSQP'.")

    distribution = event_list[0].getAntecedent().getDistribution()
    #Set the physical starting point of the Nearest point algorithm to the mediane value
    if physical_starting_point is None:
        dim = distribution.getDimension()
        physical_starting_point = ot.Point(dim)
        for i in range(dim):
            marginal = distribution.getMarginal(i)
            physical_starting_point[i] = marginal.computeQuantile(0.5)[0]

    # Gestion du log pour recuperer les infos lors du FORM run
    if logfile:
        ot.Log.Show(ot.Log.INFO + ot.Log.WARN + ot.Log.USER)
        ot.Log.SetFile('FORM.log')
        ot.Log.User('Physical Starting Point : {}'.format(physical_starting_point))

    ################### Construction du problème d'optimisation ################
    # Spécification de la fonction contrainte, dans l'espace standard !!!
    transf_inverse = distribution.getInverseIsoProbabilisticTransformation()

    # Définition du problème d'optimisation
    gt = ot.ComposedFunction(event_list[0].getFunction(), transf_inverse)
    fake_optimProblem = ot.NearestPointProblem(gt, 0.)
    objective = fake_optimProblem.getObjective()
    
    # construction des contraintes dans l'espace standard
    convention_function = ot.SymbolicFunction(['g', 'coef', 'threshold'],
                                              ['coef*(g - threshold)'])
    model_list = ot.FunctionCollection(len(event_list))
    model_standard_list = ot.FunctionCollection(len(event_list))
    model_convention_list = ot.FunctionCollection(len(event_list))
    constraint_list = ot.FunctionCollection(len(event_list))
    perf_function_list = ot.FunctionCollection(len(event_list))
    constraint_function_list = ot.FunctionCollection(len(event_list))
    for i, event in enumerate(event_list):
        model_list[i] = event.getFunction()
        # permet de revenir à contrainte > 0 (cohérent déf pb d'optim)
        if event.getOperator().getImplementation().getClassName() in ['Less', 'LessOrEqual']:
            coef_constraint = -1
            coef_perf = 1
        elif event.getOperator().getImplementation().getClassName() in ['Greater', 'GreaterOrEqual']:
            coef_constraint = 1
            coef_perf = -1

        threshold = event.getThreshold()
        # function qui retourne g tel que def = g < 0
        perf_function_list[i] = ot.ParametricFunction(convention_function, [1, 2], [coef_perf, threshold])
        constraint_function_list[i] = ot.ParametricFunction(convention_function, [1, 2], [coef_constraint, threshold])
        # function G(U)
        model_standard_list[i] = ot.ComposedFunction(model_list[i], transf_inverse)
        # function g(U) tel que g > 0
        constraint_list[i] = ot.ComposedFunction(constraint_function_list[i],
                                                 model_standard_list[i])
        # function g(X) tel que g < 0
        model_convention_list[i] = ot.ComposedFunction(perf_function_list[i], model_list[i])

    # build the optimization problem with multi constraints
    aggregated_constraint = ot.MemoizeFunction(ot.AggregatedFunction(constraint_list))

    #Initialize cache and history
    aggregated_constraint.enableHistory()
    aggregated_constraint.clearHistory()
    aggregated_constraint.enableCache()

    problem_multi_constraint = ot.OptimizationProblem(objective,
                                                      ot.Function(),
                                                      aggregated_constraint,
                                                      ot.Interval())

    # Algorithme SLSQP
    algo_multi_constraint = ot.NLopt(problem_multi_constraint, nearest_point_algo)
    eps = np.array(eps)
    algo_multi_constraint.setMaximumAbsoluteError(eps[0])
    algo_multi_constraint.setMaximumRelativeError(eps[1])
    algo_multi_constraint.setMaximumResidualError(eps[2])
    algo_multi_constraint.setMaximumConstraintError(eps[3])
    algo_multi_constraint.setMaximumEvaluationNumber(10000)

    if LooseVersion(ot.__version__) >= "1.8":
        algo_multi_constraint.setMaximumIterationNumber(n_max_iteration)
    else:
        algo_multi_constraint.setMaximumIterationsNumber(n_max_iteration)

    # Point de départ
    standardStartPoint = distribution.getIsoProbabilisticTransformation()(
                                                        physical_starting_point)
    algo_multi_constraint.setStartingPoint(standardStartPoint)

    # Lancement
    algo_multi_constraint.run()

    optimResult = algo_multi_constraint.getResult()

    # build the composite function : max(g0, ..., gn) < 0 and event
    text_function = 'max(g0'
    for i in range(1, len(event_list)):
        text_function += ',g%s' %i
    text_function += ')'
    max_function = ot.SymbolicFunction(ot.Description.BuildDefault(len(event_list), 'g'),
                                       [text_function])
    composite_function = ot.ComposedFunction(max_function,
                                ot.AggregatedFunction(model_convention_list))
    composite_event = ot.ThresholdEvent(ot.CompositeRandomVector(composite_function, 
                               ot.RandomVector(distribution)),
                               ot.Less(), 0.) 

    result = ot.FORMResult(optimResult.getOptimalPoint(), composite_event,
                           composite_function(transf_inverse(optimResult.getOptimalPoint()))[0] < 0.)

    if LooseVersion(ot.__version__) >= "1.7":
        iter_number = optimResult.getIterationNumber()
    elif LooseVersion(ot.__version__) == "1.6":
        iter_number = optimResult.getIterationsNumber()

    textResultat = '' \
        '\n Result - FORM (' + nearest_point_algo +')'+ \
        '\n --------------------------------------------------------------------' \
        '\n Probability of failure : %0.2e' % result.getEventProbability() + \
        '\n Generalised reliability index: %.4f' % result.getGeneralisedReliabilityIndex() + \
        '\n Number of iterations = {} '.format(iter_number) + \
        '\n Number of function calls = {} '.format(aggregated_constraint.getInputHistory().getSize()) + \
        '\n Standard space design point: %s' % result.getStandardSpaceDesignPoint() + \
        '\n --------------------------------------------------------------------'
        # '\n List of points: {}'.format(distribution.getDescription()) + \
        # '\n Physical space design point: %s' % result.getPhysicalSpaceDesignPoint() + \
        # '\n ----------------------------------------------------------------------------- ' + \

    if verbose:
        print(textResultat)
    if logfile:
        ot.Log.User(textResultat)

    # for i in xrange(result.getImportanceFactors().getSize()):
    #     IF = 'Importance factor %s: %.2f' % (distribution.getDescription()[i], 
    #                                 result.getImportanceFactors()[i])
    #     # print(IF)
    #     if logfile:
    #         ot.Log.User(IF)

    if logfile:
        ot.Log.User('-------------------------------------------------------------------')

    return result



class FORMsystem():
    """
    This class aims at computing the probability of failure of a structure with
    several most probable failure points.

    Parameters:
    -----------
    FORMruns : list of openturns.FORM object
        The list of the FORM objects which have been run.
    failure_domain : string
        Type of failure domain form : either 'union' or 'intersection'.
    """
    def __init__(self, FORMruns, failure_domain):

        self._nFORM = len(FORMruns)
        self._FORMresult = [FORMruns[i].getResult() for i in range(self._nFORM)]
        self._beta = None
        self._ustar = None
        self._pstar = None
        self._alpha = None
        self._rho = None
        self._failure_domain = failure_domain
        
    def _getBeta(self):

        if self._ustar is None:
            self.getStandardSpaceDesignPoint()

        # self._beta = [self._FORMresult[i].getGeneralisedReliabilityIndex() for i 
        #              in range(self._nFORM)]

        # I do not why but beta is not equal to ustar.norm() in some cases so i
        # compute it manually
        self._beta = []
        for i, ustar in enumerate(self._ustar):
            if self._FORMresult[i].getIsStandardPointOriginInFailureSpace():
                self._beta.append(-ustar.norm())
            else:
                self._beta.append(ustar.norm())

        return self._beta

    def getStandardSpaceDesignPoint(self):
        self._ustar = [self._FORMresult[i].getStandardSpaceDesignPoint() for i 
                     in range(self._nFORM)]
        return self._ustar

    def getPhysicalSpaceDesignPoint(self):
        self._pstar = [self._FORMresult[i].getPhysicalSpaceDesignPoint() for i 
                     in range(self._nFORM)]
        return self._pstar

    def getAlpha(self):

        if self._beta is None:
            self._getBeta()

        if self._ustar is None:
            self.getStandardSpaceDesignPoint()

        self._alpha = [self._ustar[i] / (-self._beta[i]) for i in
                      range(len(self._ustar))]

        return self._alpha

    def _getRho(self):

        if self._alpha is None:
            self.getAlpha()

        dim = len(self._alpha)
        self._rho = ot.CorrelationMatrix(dim)

        for i in range(dim):
            for j in range(dim):
                if i < j:
                    self._rho[i, j] = np.dot(self._alpha[i], self._alpha[j])

        return self._rho

    def getEventProbability(self):
        """
        Accessor the event probability

        Returns
        -------
        pf : float
            The FORM system probability.
        """


        if self._rho is None:
            self._getRho()

        dim = len(self._alpha)
        multiNor = ot.Normal([0] * dim, [1] * dim, self._rho)

        if self._failure_domain == 'union':
            pf = 1 - multiNor.computeCDF(self._beta)
        elif self._failure_domain =='intersection':
            pf = multiNor.computeCDF(-np.array(self._beta))

        return pf

    def getHasoferReliabilityIndex(self):
        """
        Accessor to the equivalent reliability index

        Returns
        -------
        beta : float
            The reliability index.
        """
        return ot.DistFunc.qNormal(1. - self.getEventProbability())


def run_FORM(event, algo_multi_constraint=False, **kwargs):
    """
    Run a FORM approximation system or not.

    Parameters
    ----------
    event : :class:`openturns.Event` or SystemEvent
        The failure event or the list of event defining the limit state.
    nearest_point_algo : str
        Type of the optimization algorithm. It must be 'AbdoRackwitz', 'SQP' or
        'Cobyla' if not algo_multi_constraint, otherwise most of the NLopt algo
        are available.
    algo_multi_constraint : bool
        When SystemEvent, use or not the optimization problem with multi constraint
        definition to find P*. Not compatible with option "use_multi_form".
    use_multi_form : bool
        Choose to use the MultiFORM algorithm instead of classical FORM in order
        to find several design points.
    n_max_iteration : int
        The maximum number of iterations.
    eps = sequence of float
        The stopping criterion value of the optimization algorithm. Order is 
        absolute error, relative error, residual error, constraint error.
        When a system of event is given and the algo multi contraint is not used,
        eps can be a list of list where each sub list are the parameter for
        each sub event; the size must match then.
    physical_starting_point : sequence of float
        The starting point of the algorithm. Default is the median values.
    seed : int
        Seed for the openturns random generator.
    logfile : bool
        Enable or not to write the log in FORM.log file.
    verbose : bool
        Enable or not the display of the result.
    """

    if type(event) == SystemEvent:
        # case with the limit state defined as an intersection or a union of the event
        event_list = event.getEventList()
        failure_domain = event.getFailureDomain()
        

        if failure_domain == 'union' and algo_multi_constraint:
            raise Exception('The multi constraint algorithm must be used when the failure ' + \
                            'domain is an intersection.')
 
        if algo_multi_constraint:
            return run_FORM_multi_constraint(event_list, **kwargs)
        else:
            use_sub_eps = False
            if 'eps' in kwargs:
                if type(kwargs['eps'][0]) is list:
                    if len(kwargs['eps']) == len(event_list):
                        use_sub_eps = True
                        eps_list = kwargs['eps']
                    else:
                        raise Exception('The number of "eps" list must match the size of the system event.')
                        
            FORMruns = []
            for i, event_solo in enumerate(event_list):
                if use_sub_eps:
                    kwargs['eps'] = eps_list[i]
                FORMruns.append(run_FORM_simple(event_solo, **kwargs))

            form_system = FORMsystem(FORMruns, failure_domain)

            textResultat = '' \
                '\n Result - FORM system'+ \
                '\n --------------------------------------------------------------------' \
                '\n Probability of failure : %0.2e' % form_system.getEventProbability() + \
                '\n Generalised reliability index: %.4f' % form_system.getHasoferReliabilityIndex() + \
                '\n Standard space design point:'

            for point in form_system.getStandardSpaceDesignPoint():
                textResultat += '\n%s' % point

            textResultat += '\n --------------------------------------------------------------------'

            if 'verbose' in kwargs:
                if kwargs['verbose']:
                    print(textResultat)
            if 'logfile' in kwargs:
                if kwargs['logfile']:
                    ot.Log.User(textResultat)

            return form_system

    elif type(event) == ot.ThresholdEvent:
        return run_FORM_simple(event, **kwargs).getResult()