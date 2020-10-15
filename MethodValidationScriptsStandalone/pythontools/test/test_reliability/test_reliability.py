from __future__ import print_function, division
import numpy.testing as testing
import pythontools
import openturns as ot

L = ot.LogNormal()
L.setParameter(ot.LogNormalMuSigmaOverMu()([5., .02, 0.]))
b = ot.LogNormal()
b.setParameter(ot.LogNormalMuSigmaOverMu()([.2, .05, 0.]))
h = ot.LogNormal()
h.setParameter(ot.LogNormalMuSigmaOverMu()([.4, .05, 0.]))
E = ot.LogNormal()
E.setParameter(ot.LogNormalMuSigmaOverMu()([3e4, .12, 0.]))
F = ot.LogNormal()
F.setParameter(ot.LogNormalMuSigmaOverMu()([.1, .20, 0.]))

distribution = ot.ComposedDistribution([L, b, h, E, F])
dim = 5

beam = ot.SymbolicFunction(['L', 'b', 'h', 'E', 'F'], ['F * L^3 / (48 * E * b * h^3 / 12)'])

output_rv = ot.CompositeRandomVector(beam, ot.RandomVector(distribution))

event = ot.ThresholdEvent(output_rv, ot.Greater(), 0.02)
        
## MONTE CARLO ##
mc_result = pythontools.run_monte_carlo(event=event, coef_var=0.12,
                            outer_sampling=5000,
                            block_size=100, seed=871, logfile=False,
                            verbose=False)

def test_monte_carlo_proba():
    testing.assert_almost_equal(mc_result.getProbabilityEstimate(), 8.208092e-4)
def test_monte_carlo_cov():
    testing.assert_almost_equal(mc_result.getCoefficientOfVariation(), 0.1192211021286)
def test_monte_carlo_outer():
    testing.assert_almost_equal(mc_result.getOuterSampling(), 865)

## SUBSET
ss_result = pythontools.run_subset(event=event, conditional_pf=0.1,
                            iterationPerStep=4000, block_size=100,
                            seed=871, logfile=False,
                            verbose=False)
def test_run_subset_proba():
    testing.assert_array_almost_equal(ss_result.getProbabilityEstimatePerStep(),
                                  [0.1,0.01,0.001,0.00082675])
def test_run_subset_cov():
    testing.assert_array_almost_equal(ss_result.getCoefficientOfVariationPerStep(),
                                      [0,0.0821584,0.11538,0.116348])
def test_run_subset_threshold():
    testing.assert_array_almost_equal(ss_result.getThresholdPerStep(),
                                      [0.0117031,0.015728,0.0196656,0.02])

## DIRECTIONAL
ds_result = pythontools.run_directional(event=event, root_strategy=ot.SafeAndSlow(),
                            sampling_strategy=ot.RandomDirection(),
                            maximum_distance=8, step_size=1,
                            coef_var=0.1, outer_sampling=10000,
                            block_size=10, seed=871, logfile=False,
                            verbose=False)

def test_run_directional_proba():
    testing.assert_almost_equal(ds_result.getProbabilityEstimate(), 0.0006989686423345521)
def test_run_directional_cov():
    testing.assert_almost_equal(ds_result.getCoefficientOfVariation(), 0.09964251238612933)
def test_run_directional_outer():
    testing.assert_almost_equal(ds_result.getOuterSampling(), 186)

# IMPORTANCE SAMPLING
pstar = [0.664754,-0.553668,-1.66101,-1.32488, 2.194357]
is_result = pythontools.run_importance_sampling(event=event, pstar=pstar,
                            sd=1., coef_var=0.1, outer_sampling=1000,
                            block_size=10, seed=1234, logfile=False,
                            verbose=False)

def test_run_is_proba():
    testing.assert_almost_equal(is_result.getProbabilityEstimate(), 0.0007210066378429345)
def test_run_is_cov():
    testing.assert_almost_equal(is_result.getCoefficientOfVariation(), 0.09947833181226208)
def test_run_is_outer():
    testing.assert_almost_equal(is_result.getOuterSampling(), 44)

## FORM
form_result = pythontools.run_FORM(event=event,
                            nearest_point_algo='AbdoRackwitz',
                            algo_multi_constraint=False,
                            n_max_iteration=300, eps=[1e-4]*4,
                            physical_starting_point=None, seed=871,
                            logfile=False, verbose=False)

def test_run_form_proba():
    testing.assert_almost_equal(form_result.getEventProbability(), 0.0007502863603353581)
def test_run_form_iteration():
    testing.assert_almost_equal(form_result.getOptimizationResult().getIterationNumber(),
                            26)
def test_run_form_pstar():
    testing.assert_array_almost_equal(form_result.getStandardSpaceDesignPoint(),
                    [0.664754,-0.553668,-1.66101,-1.32488, 2.194357])
