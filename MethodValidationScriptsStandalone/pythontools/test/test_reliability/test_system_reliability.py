from __future__ import print_function, division
import numpy.testing as testing
import pythontools
import openturns as ot

R = ot.Normal(7.7, .55)
S = ot.Normal(1.1, .55)
distribution = ot.ComposedDistribution([R, S])

g1 = ot.SymbolicFunction(['r', 's'], ['-r^2 - s^3 + 90'])
g2 = ot.SymbolicFunction(['r', 's'], ['r^2 - s^3 -20'])

rv_g1 = ot.CompositeRandomVector(g1, ot.RandomVector(distribution))
rv_g2 = ot.CompositeRandomVector(g2, ot.RandomVector(distribution))

failure1 = ot.ThresholdEvent(rv_g1, ot.LessOrEqual(), 0.)
failure2 = ot.ThresholdEvent(rv_g2, ot.LessOrEqual(), 0.)
    
##### UNION #####
union_event = pythontools.SystemEvent([failure1, failure2], 'union')

## MONTE CARLO
mc_result_union = pythontools.run_monte_carlo(event=union_event, coef_var=0.12,
                            outer_sampling=5000,
                            block_size=100, seed=871, logfile=False,
                            verbose=False)

def test_run_monte_carlo_union_proba():
    testing.assert_almost_equal(mc_result_union.getProbabilityEstimate(), 0.002651515151515153)
def test_run_monte_carlo_union_cov():
    testing.assert_almost_equal(mc_result_union.getCoefficientOfVariation(), 0.11995983832846457)
def test_run_monte_carlo_union_outer():
    testing.assert_almost_equal(mc_result_union.getOuterSampling(), 264)

## SUBSET
ss_result_union = pythontools.run_subset(event=union_event, conditional_pf=0.1,
                            iterationPerStep=4000, block_size=100,
                            seed=871, logfile=False,
                            verbose=False)

def test_run_subset_union_proba():
    testing.assert_array_almost_equal(ss_result_union.getProbabilityEstimatePerStep(),
                                      [0.1,0.01,0.0026275])
def test_run_subset_union_cov():
    testing.assert_array_almost_equal(ss_result_union.getCoefficientOfVariationPerStep(),
                                      [0,0.0877496,0.099714])
def test_run_subset_union_threshold():
    testing.assert_array_almost_equal(ss_result_union.getThresholdPerStep(),
                                      [15.848946, 5.435637, 0])

## DIRECTIONAL
ds_result_union = pythontools.run_directional(event=union_event, root_strategy=ot.SafeAndSlow(),
                            sampling_strategy=ot.RandomDirection(),
                            maximum_distance=8, step_size=1,
                            coef_var=0.1, outer_sampling=10000,
                            block_size=10, seed=871, logfile=False,
                            verbose=False)

def test_run_directional_union_proba():
    testing.assert_almost_equal(ds_result_union.getProbabilityEstimate(), 0.0026460640007847235)
def test_run_directional_union_cov():
    testing.assert_almost_equal(ds_result_union.getCoefficientOfVariation(), 0.09733527118242706)
def test_run_directional_union_outer():
    testing.assert_almost_equal(ds_result_union.getOuterSampling(), 8)

## IMPORTANCE SAMPLING
pstar = [[2.47688,1.61733], [-1.74187,3.3484]]
is_result_union = pythontools.run_importance_sampling(event=union_event, pstar=pstar,
                            sd=1., coef_var=0.1, outer_sampling=1000,
                            block_size=10, seed=1234, logfile=False,
                            verbose=False)

def test_run_is_union_proba():
    testing.assert_almost_equal(is_result_union.getProbabilityEstimate(), 0.0027592794001524965)
def test_run_is_union_cov():
    testing.assert_almost_equal(is_result_union.getCoefficientOfVariation(), 0.09974137045662601)
def test_run_is_union_outer():
    testing.assert_almost_equal(is_result_union.getOuterSampling(), 126)

# FORM
form_result_union = pythontools.run_FORM(event=union_event,
                            nearest_point_algo='AbdoRackwitz',
                            algo_multi_constraint=False,
                            n_max_iteration=300, eps=[1e-4]*4,
                            physical_starting_point=None, seed=871,
                            logfile=False, verbose=False)

def test_run_form_union_proba():
    testing.assert_almost_equal(form_result_union.getEventProbability(), 0.0016272103655481374)
def test_run_form_union_proba_1():
    testing.assert_almost_equal(form_result_union._FORMresult[0].getEventProbability(),
                                0.001547408507729085)
def test_run_form_union_proba_2():
    testing.assert_almost_equal(form_result_union._FORMresult[1].getEventProbability(),
                                8.02044144738557e-05)
def test_run_form_union_pstar_2():
    testing.assert_array_almost_equal(form_result_union.getStandardSpaceDesignPoint()[0],
                                      [2.476883,1.617331])
def test_run_form_union_pstar_2():
    testing.assert_array_almost_equal(form_result_union.getStandardSpaceDesignPoint()[1],
                                              [-1.741875,3.3484])

##### INTERSECTION #####
intersection_event = pythontools.SystemEvent([failure1, failure2], 'intersection')

## MONTE CARLO
mc_result_intersection = pythontools.run_monte_carlo(event=intersection_event, coef_var=0.12,
                            outer_sampling=5000,
                            block_size=100, seed=871, logfile=False,
                            verbose=False)

def test_run_monte_carlo_intersection_proba():
    testing.assert_almost_equal(mc_result_intersection.getProbabilityEstimate(), 1.8000000000000024e-05)
def test_run_monte_carlo_intersection_cov():
    testing.assert_almost_equal(mc_result_intersection.getCoefficientOfVariation(), 0.33499286891249136)
def test_run_monte_carlo_intersection_outer():
    testing.assert_almost_equal(mc_result_intersection.getOuterSampling(), 5000)

## SUBSET
ss_result_intersection = pythontools.run_subset(event=intersection_event, conditional_pf=0.1,
                            iterationPerStep=4000, block_size=100,
                            seed=871, logfile=False,
                            verbose=False)

def test_run_subset_intersection_proba():
    testing.assert_array_almost_equal(ss_result_intersection.getProbabilityEstimatePerStep(),
                                      [0.1,0.01,0.001,9.975e-05,1.37655e-05])
def test_run_subset_intersection_cov():
    testing.assert_array_almost_equal(ss_result_intersection.getCoefficientOfVariationPerStep(),
                                      [0,0.0905539,0.125449,0.155397,0.174406])
def test_run_subset_intersection_threshold():
    testing.assert_array_almost_equal(ss_result_intersection.getThresholdPerStep(),
                                      [33.26347078431023,26.150375250,17.061004248,8.1453050,0])

## DIRECTIONAL
ds_result_intersection = pythontools.run_directional(event=intersection_event, root_strategy=ot.SafeAndSlow(),
                            sampling_strategy=ot.RandomDirection(),
                            maximum_distance=8, step_size=1,
                            coef_var=0.1, outer_sampling=10000,
                            block_size=10, seed=871, logfile=False,
                            verbose=False)

def test_run_directional_intersection_proba():
    testing.assert_almost_equal(ds_result_intersection.getProbabilityEstimate(), 1.1937067681648873e-05)
def test_run_directional_intersection_cov():
    testing.assert_almost_equal(ds_result_intersection.getCoefficientOfVariation(), 0.09969961074927983)
def test_run_directional_intersection_outer():
    testing.assert_almost_equal(ds_result_intersection.getOuterSampling(), 84)

## IMPORTANCE SAMPLING
pstar = [-0.516003,3.94739]
is_result_intersection = pythontools.run_importance_sampling(event=intersection_event, pstar=pstar,
                            sd=1., coef_var=0.1, outer_sampling=1000,
                            block_size=10, seed=1234, logfile=False,
                            verbose=False)

def test_run_is_intersection_proba():
    testing.assert_almost_equal(is_result_intersection.getProbabilityEstimate(), 1.1634663014100047e-05)
def test_run_is_intersection_cov():
    testing.assert_almost_equal(is_result_intersection.getCoefficientOfVariation(), 0.09991908238189803)
def test_run_is_intersection_outer():
    testing.assert_almost_equal(is_result_intersection.getOuterSampling(), 103)

# FORM
form_result_intersection = pythontools.run_FORM(event=intersection_event,
                            nearest_point_algo='LD_SLSQP',
                            algo_multi_constraint=True,
                            n_max_iteration=300, eps=[1e-4]*4,
                            physical_starting_point=None, seed=871,
                            logfile=False, verbose=False)

def test_run_form_intersection_proba():
    testing.assert_almost_equal(form_result_intersection.getEventProbability(), 3.431688985411491e-05)
def test_run_form_intersection_pstar():
    testing.assert_array_almost_equal(form_result_intersection.getStandardSpaceDesignPoint(),
                                      [-0.516003,3.94739])
