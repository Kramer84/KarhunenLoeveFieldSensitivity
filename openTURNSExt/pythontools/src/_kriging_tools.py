# -*- coding: utf-8 -*-

__all__ = ['build_default_kriging_algo', 'estimate_kriging_theta',
           'compute_LOO', 'compute_Q2']

import openturns as ot
import numpy as np

def build_default_kriging_algo(input_sample, output_sample, basis=None,
                            covariance_model=None, noise=None):
    """
    Build 1D kriging algorithm without running it.

    Parameters
    ----------
    input_sample : 2d sequence of float
        The learning input sample
    output_sample : 2d sequence of float
        The learning output sample
    basis : :class:`openturns.Basis`
        The openturns basis, default is the constant basis.
    covariance_model : :class:`openturns.CovarianceModel`
        The openturns covariance model, default is anisotropic Matérn 5/2.
    noise : 2d sequence of float
        The noise associated to each output values. Default is no noise.
    
    Returns
    -------
    algo_kriging : :class:`openturns.KrigingAlgorithm` or a list of algorithm
        An openturns kriging algorithm or a list of it in case of multiple outputs.
    """

    output_dim = ot.Sample(output_sample).getDimension()
    algo_kriging_col = [object] * output_dim

    for marginal in range(output_dim):
        dim = input_sample.getDimension()
        if basis is None:
            # create linear basis
            basis = ot.ConstantBasisFactory(dim).build()

        if covariance_model is None:
            # anisotropic matern covariance model
            covariance_model  = ot.MaternModel([1]*dim, 2.5)

        algo_kriging_col[marginal] = ot.KrigingAlgorithm(input_sample,
                                            output_sample[:, marginal],
                                            covariance_model, basis, True)
        if noise is not None:
            if type(noise) is not list:
                algo_kriging_col[marginal].setNoise([noise]*input_sample.getSize())
            else:
                algo_kriging_col[marginal].setNoise(noise)

        algo_kriging_col[marginal].setOptimizeParameters(False)
        algo_kriging_col[marginal].run()

    if output_dim == 1:
        algo_kriging_col = algo_kriging_col[0]

    return algo_kriging_col


def estimate_kriging_theta(algo_kriging, lower_bound=None, upper_bound=None, size=100,
                      optim_type='multi_start'):
    """
    Estimate the kriging theta values with an initial random search using
    a Sobol sequence of size samples.

    Parameters
    ----------
    algo_kriging : :class:`openturns.KrigingAlgorithm` or a list of algorithm
        An openturns kriging algorithm or a list in case of multiple outputs.
    lower_bound : float or list of float
        The lower bound of the kriging scale parameters. If a list is given,
        it must be the size of the input dimension.
    upper_bound : float or list of float
        The upper bound of the kriging scale parameters. If a list is given,
        it must be the size of the input dimension.
    size : int
        The number of sample used for multi start or best start
    optim_type : string
        The type of optimization : multi start, best initial start, optim global.
        It must be "multi_start", "best_start", "global".
        "best_start" only find the best log likelihood from an initial doe and
        perform one optimization starting from this point.
    """
    if type(algo_kriging) is list:
        output_dim = len(algo_kriging)
    else:
        output_dim = 1
        algo_kriging = [algo_kriging]

    for marginal in range(output_dim):
        algo_kriging[marginal].setOptimizeParameters(True)
        # get input parameters of the kriging algorithm
        X = algo_kriging[marginal].getInputSample()
        Y = algo_kriging[marginal].getOutputSample()
        noise = algo_kriging[marginal].getNoise()
        
        kriging_result = algo_kriging[marginal].getResult()
        covariance_model = kriging_result.getCovarianceModel()
        basis = kriging_result.getBasisCollection()
        llf = algo_kriging[marginal].getReducedLogLikelihoodFunction()
        dim = algo_kriging[marginal].getOptimizationBounds().getDimension()

        # create uniform distribution of the parameters bounds
        if lower_bound is not None:
            if type(lower_bound) in [float, int]:
                lower_bound = [lower_bound] * dim
            elif len(lower_bound) != dim:
                raise ValueError(f'The dimension must be {dim}, here {len(lower_bound)} given.')
        else:
            lower_bound = algo_kriging[marginal].getOptimizationBounds().getLowerBound()

        if upper_bound is not None:
            if type(upper_bound) in [float, int]:
                upper_bound = [upper_bound] * dim
            elif len(lower_bound) != dim:
                raise ValueError(f'The dimension must be {dim}, here {len(upper_bound)} given.')
        else:
            upper_bound = algo_kriging[marginal].getOptimizationBounds().getUpperBound()

        dist_bound_col = []
        for i in range(dim):
            dist_bound_col += [ot.Uniform(lower_bound[i], upper_bound[i])]
        dist_bound = ot.ComposedDistribution(dist_bound_col)    
        searchInterval = ot.Interval(lower_bound, upper_bound)

        # Generate starting points with an optimized LHS
        ot.RandomGenerator.SetSeed(9832)
        lhs_experiment = ot.LHSExperiment(dist_bound, size, True, True)
        theta_start = ot.MonteCarloLHS(lhs_experiment, 1000).generate()

        if optim_type == "multi_start":
            algo_kriging[marginal].setOptimizationAlgorithm(ot.MultiStart(ot.TNC(), theta_start))

        elif optim_type == "best_start":
            # Get the best theta from the maximum llf value
            llfValue = llf(theta_start)
            indexMax = np.argmax(llfValue)
            bestTheta = theta_start[int(indexMax)]

            # update theta after random search
            covariance_model.setScale(bestTheta)

            # if covariance_model.getScale().getDimension() == dim:
            # else:
            #     # else the optimization is also on the amplitude
            #     covariance_model.setScale(bestTheta[:-1])
            #     covariance_model.setAmplitude([bestTheta[-1]])

            # Now the KrigingAlgorithm is used to optimize the likelihood using a
            # good starting point
            algo_kriging[marginal] = ot.KrigingAlgorithm(X, Y, covariance_model, basis, True)
            if noise.getDimension() != 0:
                algo_kriging[marginal].setNoise(noise)

        elif optim_type == "global":
            algo_kriging[marginal].setOptimizationAlgorithm(ot.NLopt('GN_DIRECT'))

        else:
            raise Exception('The "optim_type" parameter must be "multi_start", "best_start" or "global".')

        algo_kriging[marginal].setOptimizationBounds(searchInterval)
        algo_kriging[marginal].run()

    if output_dim == 1:
        algo_kriging = algo_kriging[0]

    return algo_kriging


def compute_LOO(input_sample, output_sample, kriging_result):
    """
    Compute the Leave One out prediction analytically.

    Parameters
    ----------
    input_sample : 2d sequence of float
        The learning input sample
    output_sample : 2d sequence of float
        The learning output sample
    kriging_result : :class:`openturns.KrigingResult`
        The optimised kriging result.
    """
    input_sample = np.array(input_sample)
    output_sample = np.array(output_sample)

    # Récupération du modèle de covariance
    cov = kriging_result.getCovarianceModel()
    # Récupération de la transformation sur le vecteur d'entrée
    t = kriging_result.getTransformation()
    # Normalisation des données
    if t.getInputDimension() == input_sample.shape[1]:
        normalized_input_sample = np.array(t(input_sample))
    else:
        normalized_input_sample = input_sample
    # Matrice de corrélation et décomposition de Cholesky
    R = cov.discretize(normalized_input_sample)
    C = R.computeCholesky()
    # Récupération du sigma²
    sigma2 = kriging_result.getCovarianceModel().getAmplitude()[0]**2
    # Récupération des coefficients et calcul de la partie déterministe pour tous les points
    basis = kriging_result.getBasisCollection()[0]
    F1 = kriging_result.getTrendCoefficients()[0]
    size = input_sample.shape[0]
    p = F1.getDimension()
    F = np.ones((size, p))
    for i in range(p):
        F[:, i] = np.hstack(basis.build(i)(normalized_input_sample))
    # Calcul de y_loo
    K = sigma2 * np.dot(C, C.transpose())
    Z = np.zeros((p, p))
    S = np.vstack([np.hstack([K, F]), np.hstack([F.T, Z])])
    S_inv = np.linalg.inv(S)
    B = S_inv[:size:, :size:]
    B_but_its_diag = B * (np.ones(B.shape) - np.eye(size))
    B_diag = np.atleast_2d(np.diag(B)).T
    y_loo = (- np.dot(B_but_its_diag / B_diag, output_sample)).ravel()
    return y_loo

def compute_Q2(input_sample, output_sample, kriging_result):
    """
    Compute the Q2 using the analytical loo prediction.

    Parameters
    ----------
    input_sample : 2d sequence of float
        The learning input sample
    output_sample : 2d sequence of float
        The learning output sample
    kriging_result : :class:`openturns.KrigingResult`
        The optimised kriging result.
    """
    y_loo = compute_LOO(input_sample, output_sample, kriging_result)
    # Calcul du Q2
    delta = (np.hstack(output_sample) - y_loo)
    return 1 - np.mean(delta**2)/np.var(output_sample)