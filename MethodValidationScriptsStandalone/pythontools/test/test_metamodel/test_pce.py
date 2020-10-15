from __future__ import print_function, division
import numpy.testing as testing
import pythontools as pyto
import openturns as ot
import numpy as np
import pandas as pd

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

beam = ot.SymbolicFunction(
    ['L', 'b', 'h', 'E', 'F'], ['F * L^3 / (48 * E * b * h^3 / 12)'])

ot.RandomGenerator.SetSeed(654)
input_sample = distribution.getSample(150)
output_sample = beam(input_sample)


chaos_1 = pyto.PolynomialChaos(distribution)
chaos_2 = pyto.PolynomialChaos(distribution)
chaos_multi_1 = pyto.PolynomialChaos(distribution)
chaos_multi_2 = pyto.PolynomialChaos(distribution)

degree = 2
size = 1000

ot.RandomGenerator.SetSeed(125)
input_sample_test = distribution.getSample(50)
output_sample_test = beam(input_sample_test)

beam_multi = ot.SymbolicFunction(
    ['L', 'b', 'h', 'E', 'F'],
    ['F * L^3 / (48 * E * b * h^3 / 12)', 'b * h^3 / 12'])
output_sample_multi = beam_multi(input_sample)
output_sample_multi_test = beam_multi(input_sample_test)

chaos_meta = chaos_1.fit(input_sample, output_sample, degree, sparse=False)
chaos_meta2 = chaos_2.fit_by_integration(beam, size, degree)


chaos_multi_meta = chaos_multi_1.fit(
    input_sample, output_sample_multi, degree, sparse=False)


chaos_multi_meta2 = chaos_multi_2.fit_by_integration(
    beam_multi, size, degree)


def test_PCE_1_compute_q2():
    testing.assert_almost_equal(
        chaos_1.compute_q2().values,
        0.999, decimal=3)


def test_PCE_1_compute_r2():
    testing.assert_almost_equal(
        chaos_1.compute_r2(input_sample_test, output_sample_test).values,
        0.999, decimal=3)


def test_PCE_2_compute_r2():
    testing.assert_almost_equal(
        chaos_2.compute_r2(input_sample_test, output_sample_test).values,
        0.99, decimal=2)


# MULTIPLE OUTPUT #


def test_pce_1_multi_1_compute_q2():
    testing.assert_array_almost_equal(
        chaos_multi_1.compute_q2().values,
        [[0.999, 0.999]], decimal=3)


def test_pce_1_multi_2_compute_r2():
    testing.assert_array_almost_equal(
        chaos_multi_1.compute_r2(
            input_sample_test, output_sample_multi_test).values,
        [[0.99, 0.99]], decimal=2)


def test_pce_2_multi_2_compute_r2():
    testing.assert_array_almost_equal(
        chaos_multi_2.compute_r2(
            input_sample_test, output_sample_multi_test).values,
        [[0.999, 0.999]], decimal=2)

# Sobol indices #


sobol_1 = [[0.042, 0.045], [0.029, 0.032], [0.264, 0.277],
           [0.17, 0.182], [0.471, 0.489], [0.024, np.nan]]

sobol_2 = [[0.042, 0.045, 0., 0.], [0.029, 0.032, 0.098, 0.101],
           [0.264, 0.277, 0.899, 0.902], [0.17, 0.182, 0., 0.],
           [0.471, 0.489, 0., 0.], [0.024, np.nan, 0.002, np.nan]]


def test_get_sobol_indices():
    testing.assert_array_almost_equal(
        pd.DataFrame(chaos_1.get_sobol_indices().to_numpy()),
        sobol_1, decimal=2)


def test_multi_get_sobol_indices():
    testing.assert_array_almost_equal(
        pd.DataFrame(chaos_multi_1.get_sobol_indices().to_numpy()),
        sobol_2, decimal=2)
