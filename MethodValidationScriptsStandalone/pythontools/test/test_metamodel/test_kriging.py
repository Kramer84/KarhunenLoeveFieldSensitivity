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

ot.RandomGenerator.SetSeed(654)
input_sample = distribution.getSample(150)
output_sample = beam(input_sample)

ot.RandomGenerator.SetSeed(456)
kriging = pythontools.build_default_kriging_algo(input_sample=input_sample,
                            output_sample=output_sample, basis=None,
                            covariance_model=None, noise=None)

kriging = pythontools.estimate_kriging_theta(algo_kriging=kriging,
                    lower_bound=1e-4, upper_bound=100, size=5,
                    optim_type='global')

kriging_result = kriging.getResult()
kriging_result.getCovarianceModel()

# Blas, openmp ? is making it not working
# def test_kriging_parameter():
#     testing.assert_array_almost_equal(kriging_result.getCovarianceModel().getParameter(),
#                         [11.632272, 17.362844, 14.194365, 88.44738 , 12.886452])

def test_kriging_1_compute_q2():
    testing.assert_almost_equal(
        pythontools.compute_Q2(input_sample, output_sample, kriging_result),
        0.999, decimal=3)


ot.RandomGenerator.SetSeed(456)
kriging = pythontools.build_default_kriging_algo(input_sample=input_sample,
                            output_sample=output_sample, basis=None,
                            covariance_model=None, noise=None)

kriging = pythontools.estimate_kriging_theta(algo_kriging=kriging,
                    lower_bound=1e-4, upper_bound=100, size=5,
                    optim_type='best_start')

kriging_result = kriging.getResult()
kriging_result.getCovarianceModel()

def test_kriging_2_compute_q2():
    testing.assert_almost_equal(
        pythontools.compute_Q2(input_sample, output_sample, kriging_result),
        0.999, decimal=3)


############################## MULTIPLE OUTPUT #################################

beam_multi = ot.SymbolicFunction(['L', 'b', 'h', 'E', 'F'],
                           ['F * L^3 / (48 * E * b * h^3 / 12)', 'b * h^3 / 12'])
output_sample_multi = beam_multi(input_sample)

kriging_col = pythontools.build_default_kriging_algo(input_sample=input_sample,
                            output_sample=output_sample_multi, basis=None,
                            covariance_model=None, noise=None)

kriging_col = pythontools.estimate_kriging_theta(algo_kriging=kriging_col,
                    lower_bound=1e-4, upper_bound=100, size=5,
                    optim_type='multi_start')

kriging_result_col = [kriging.getResult() for kriging in kriging_col]
[res.getCovarianceModel() for res in kriging_result_col]

def test_kriging_1_multi_1_compute_q2():
    testing.assert_almost_equal(
        pythontools.compute_Q2(input_sample, output_sample_multi[:, 0], kriging_result_col[0]),
        0.999, decimal=3)

def test_kriging_1_multi_2_compute_q2():
    testing.assert_almost_equal(
        pythontools.compute_Q2(input_sample, output_sample_multi[:, 1], kriging_result_col[1]),
        0.999, decimal=3)

kriging_col = pythontools.build_default_kriging_algo(input_sample=input_sample,
                            output_sample=output_sample_multi, basis=None,
                            covariance_model=None, noise=None)

kriging_col = pythontools.estimate_kriging_theta(algo_kriging=kriging_col,
                    lower_bound=1e-4, upper_bound=100, size=5,
                    optim_type='best_start')

kriging_result_col = [kriging.getResult() for kriging in kriging_col]
[res.getCovarianceModel() for res in kriging_result_col]

def test_kriging_2_multi_1_compute_q2():
    testing.assert_almost_equal(
        pythontools.compute_Q2(input_sample, output_sample_multi[:, 0], kriging_result_col[0]),
        0.999, decimal=3)

def test_kriging_2_multi_2_compute_q2():
    testing.assert_almost_equal(
        pythontools.compute_Q2(input_sample, output_sample_multi[:, 1], kriging_result_col[1]),
        0.999, decimal=3)