# -*- coding: utf-8 -*-

__all__ = ['AKMethod']

import openturns as ot
import numpy as np
from ._kriging_tools import build_default_kriging_algo, estimate_kriging_theta, compute_Q2

class AKMethod():
    """
    This class implements the AKMCS or AKIS method developped by Echard et al.:
    AK-MCS: An active learning reliability method combining Kriging and Monte 
    Carlo Simulation.


    Parameters
    ----------
    event: OpenTURNS event
        Event characterizing the reliability problem.
    sampling_size: int
        Size of the Monte Carlo sampling, on which the probability of failure
        is computed.
    n_max_iteration: int
        Maximum number of iteration (~ number of points added in the DOE)
    method: str
        Kind of simulation method ('MC' or 'IS'). Default is 'MC'.
    mpfp: list of openturns.Point
        The most probable failure points in the standard space (at least one)
        used to build the instrumental distribution for the importance sampling.
    coeff_variation : float
        The maximum coefficient of variation of the failure probability when 
        running the simulation at each iteration.
    criterion : float
        Stopping criterion value. The default is set to 2 which corresponds with
        a confidence level of 95%% of being well classified for all points.
    verbose: bool
        Enable the verbosity or not.
    kriging_param : dict
        The kriging parameters that can be passed in order to personnalize the
        king of kriging model and the optimization parameters.
    """

    def __init__(self, event, sampling_size, n_max_iteration=500,
                 method='MC', mpfp=None, coeff_variation=0.1, criterion=2.,
                 verbose=True, seed=43982,
                 kriging_param={'size':100, 'optim_type':'multi_start'}):
    
        self.event = event
        self.function = event.getFunction()
        self.distribution = event.getAntecedent().getDistribution()
        self.sampling_size = sampling_size
        self.n_max_iteration = n_max_iteration
        self.verbose = verbose
        self.metamodelList = list()
        self.method = method  # 'MC' or 'IS'
        self.mpfp = mpfp
        self.coeff_variation = coeff_variation
        self.seed = seed
        self.kriging_param = kriging_param

        self._basis = None
        self._covariance_model = None
        self._noise = None
        self._dim = self.function.getInputDimension()
        self._operator = event.getOperator()
        self._threshold = event.getThreshold()
        self._stopping_criterion = criterion

        # run check and useful method
        if method == 'IS':
            if mpfp is None:
                raise Exception('If the IS methode is chosen, "mpfp" must be given.')
            else:
                self._buildInstrumentalDistribution()


    def buildKrigingModel(self, inputSample, outputSample):
        """
        Build a kriging model from an input and output sample.
        """
        # Default kriging algorithm is linear basis and anisotropic squared 
        # exponential covariance model
        algokriging = buildDefaultKrigingAlgo(inputSample, outputSample,
                                self._basis, self._covariance_model, self._noise)

        krigingResult = algokriging.getResult()
        self._basis = krigingResult.getBasisCollection()
        self._covariance_model = krigingResult.getCovarianceModel()

        Q2 = computeQ2(inputSample, outputSample, krigingResult)
        # need optimization of theta parameters or when first iteration
        if Q2 < 0.95 or len(self.metamodelList) == 0:
            if self.verbose:
                print('Theta optimization running...')
            dim_problem = algokriging.getReducedLogLikelihoodFunction().getInputDimension()
            lowerBound = [0.001] * dim_problem
            upperBound = [100] * dim_problem
            size = 1000
            algokriging = estimKrigingTheta(algokriging, lowerBound, upperBound,
                                            **self.kriging_param)
            krigingResult = algokriging.getResult()
            self._basis = krigingResult.getBasisCollection()
            self._covariance_model = krigingResult.getCovarianceModel()
        
        return algokriging


    def runMonteCarlo(self, metamodel):
        """
        Run the Monte Carlo simulation to compute the probability of failure.
        """
        
        ot.RandomGenerator.SetSeed(self.seed)
        metamodel.enableHistory()

        G = ot.RandomVector(metamodel, ot.RandomVector(self.distribution))
        failure = ot.Event(G, self._operator, self._threshold)

        monte_carlo = ot.ProbabilitySimulationAlgorithm(failure)
        monte_carlo.setMaximumCoefficientOfVariation(self.coeff_variation)
        monte_carlo.setMaximumOuterSampling(int(self.sampling_size/100))
        monte_carlo.setBlockSize(100)

        metamodel.clearHistory()
        monte_carlo.run()
        
        return monte_carlo

    def _buildInstrumentalDistribution(self):
        # Création de l'objet ImportanceSampling avec :
        #  - l'événement dans l'espace standard (--> fonction de performance définie dans l'espace standard)
        #  - la densité instrumentale
        corr = ot.IdentityMatrix(self._dim)
        self.instrumental_distribution = ot.Mixture([ot.Normal(point, corr)
                                                    for point in self.mpfp])

    def runImportanceSampling(self, metamodel):
        """
        Run the importance sampling simulation to compute the probability of failure.
        """
        
        ot.RandomGenerator.SetSeed(self.seed)
        metamodel.enableHistory()

        G = ot.RandomVector(metamodel, ot.RandomVector(self.distribution))
        failure = ot.Event(G, self._operator, self._threshold)

        instrumental_experiment = ot.ImportanceSamplingExperiment(self.instrumental_distribution)
        IS_algorithm = ot.ProbabilitySimulationAlgorithm(ot.StandardEvent(failure),
                                                         instrumental_experiment)
        IS_algorithm.setMaximumCoefficientOfVariation(self.coeff_variation)
        IS_algorithm.setMaximumOuterSampling(int(self.sampling_size/100))
        IS_algorithm.setBlockSize(100)
        
        metamodel.clearHistory()
        IS_algorithm.run()

        return IS_algorithm

    def compute_pf_bounds(self, output_MC, enrich_criterion_values,
                          inputMC=None):
        """
        Compute upper and lower bounds of pf taking into account the percentage
        of well classified points with a confidence lentgth of 95%. These points
        are those with a learning criterion value greater than 2.
        """

        output_MC = np.array(output_MC)
        enrich_criterion_values = np.array(enrich_criterion_values)

        if self._operator.getImplementation().getClassName() in ['LessOrEqual', 'Less']:
            indicatrice_inf = np.vstack(((output_MC <= self._threshold) &
                               (enrich_criterion_values > 2)).astype(float))
            indicatrice_sup = np.vstack(((output_MC <= self._threshold) |
                               ((output_MC > self._threshold) &
                                (enrich_criterion_values < 2))).astype(float))

        elif self._operator.getImplementation().getClassName() in ['GreaterOrEqual', 'Greater']:
            indicatrice_inf = np.vstack(((output_MC >= self._threshold) &
                               (enrich_criterion_values > 2)).astype(float))
            indicatrice_sup = np.vstack(((output_MC >= self._threshold) |
                               ((output_MC < self._threshold) &
                                (enrich_criterion_values < 2))).astype(float))

        if self.method == 'MC':
            pfinf_kriging = np.sum(indicatrice_inf) / indicatrice_inf.size
            pfsup_kriging = np.sum(indicatrice_sup) / indicatrice_sup.size

            cov_inf_kriging = np.sqrt((1 - pfinf_kriging) / (self.sampling_size * pfinf_kriging))
            cov_sup_kriging = np.sqrt((1 - pfsup_kriging) / (self.sampling_size * pfsup_kriging))

        elif self.method == 'IS':
            Tiso = self.distribution.getIsoProbabilisticTransformation()
            diff = np.array(ot.Normal(self._dim).computePDF(Tiso(inputMC))) / \
            np.array(self.instrumental_distribution.computePDF(Tiso(inputMC)))

            pfinf_kriging = np.sum(indicatrice_inf * diff) / indicatrice_inf.size
            pfsup_kriging = np.sum(indicatrice_sup * diff) / indicatrice_sup.size

            variance_inf_kriging = ((np.sum((indicatrice_inf * diff)**2) / indicatrice_inf.size) - 
                                    pfinf_kriging**2) / (indicatrice_inf.size - 1)
            cov_inf_kriging = np.sqrt(variance_inf_kriging) / pfinf_kriging

            variance_sup_kriging = ((np.sum((indicatrice_sup * diff)**2) / (indicatrice_sup.size)) - 
                                    pfinf_kriging**2) / (indicatrice_sup.size - 1)
            cov_sup_kriging = np.sqrt(variance_sup_kriging) / pfsup_kriging

        # Recalcule des bornes de pf en ajoutant l'intervalle de confiance à 95%.
        pfinf = pfinf_kriging * (1 - 1.96 * cov_inf_kriging)
        pfsup = pfsup_kriging * (1 + 1.96 * cov_sup_kriging)

        return pfinf_kriging, pfsup_kriging, pfinf, pfsup

    def compute_criterion_U(self, output_MC, variance_MC):

        enrich_criterion_values = np.abs(self._threshold - output_MC) / np.sqrt(variance_MC)

        # Récupération du point minimisant le critère
        umin = enrich_criterion_values.min()

        return umin, enrich_criterion_values


    def enrich_DOE(self, input_DOE, input_MC, output_MC, variance_MC,
                   enrich_criterion_values):
        """
        Add a point to the DOE which minimize the learning criterion:
        u(X) = |G(X)| / sigma(X).
        It indicates the distance in Kriging standard deviations between
        the prediction and the estimated limit state. It represents a reliability
        index on the risk of making a mistake on the sign of G considering G(x)
        with the same sign than ~G(X) (prediction with the metamodel).
        """

        enrich_criterion_values = np.abs(self._threshold - output_MC) / np.sqrt(variance_MC)


        # recherche du point minimisant le critère
        index_enrich = enrich_criterion_values.argmin()
        input_enrich = input_MC[int(index_enrich)]

        # ajout du point dans le DOE
        size_DOE = input_DOE.getSize()
        tmp_DOE = ot.Sample(size_DOE + 1, self._dim)
        tmp_DOE[:-1], tmp_DOE[-1] = input_DOE, input_enrich
        input_DOE = tmp_DOE

        return input_DOE


    def run(self, size_initial_doe=None, initial_doe=None):
        """
        Launch the AKMCS method.
        
        Parameters
        ----------
        size_initial_doe: int
            Size of the initial DOE build based on a low discrepancy sequence.
        initial_doe: 2d sequence of float
            The sample of the input used as initial doe.
            
        Notes
        -----
        If the initial_doe is given, the other parameters related to the doe
        are not taken into account.
        """

        if initial_doe is None:
            if self.method =='MC':
                # Plan d'expériences initiales : suite à faible discrépance
                initial_doe = ot.LowDiscrepancyExperiment(ot.SobolSequence(),
                                                 self.distribution,
                                                 size_initial_doe).generate()
            elif self.method =='IS':
                # Plan d'expériences initiales : suite à faible discrépance
                initial_doe = ot.LowDiscrepancyExperiment(ot.SobolSequence(),
                                                 self.instrumental_distribution,
                                                 size_initial_doe).generate()

        self.iteration = 0

        # boucle while
        while self.iteration <= self.n_max_iteration:

            self.iteration += 1
            if self.verbose:
                print('Iteration : ', self.iteration)
                
            # si première itération, DOE = initial_doe
            if self.iteration == 1:
                input_DOE = initial_doe
            
            # Calcul de la sortie par le vrai modèle
            output_DOE = self.function(input_DOE)
            
            # Construction du modèle de krigeage
            self.algo_kriging = self.buildKrigingModel(input_DOE, output_DOE)
            result_kriging = self.algo_kriging.getResult()
            Q2 = computeQ2(input_DOE, output_DOE, result_kriging)
            meta_kriging = ot.MemoizeFunction(result_kriging.getMetaModel())
            meta_kriging.enableHistory()
            self.metamodelList.append(meta_kriging)
            
            if self.method =='MC':
                # Réalisation de la simulation de Monte Carlo
                simulation = self.runMonteCarlo(meta_kriging)
            elif self.method =='IS':
                # Simulation par tirages d'importance
                simulation = self.runImportanceSampling(meta_kriging)

            # récupération des points de la simulation, entrée, sortie et 
            # calcul de la variance de krigeage
            input_MC = meta_kriging.getInputHistory()
            output_MC = np.hstack(meta_kriging.getOutputHistory())
            variance_MC = np.array([result_kriging.getConditionalCovariance(
                            input_MC[i])[0, 0] for i in range(input_MC.getSize())])
            # vérification si la variance est négative
            variance_MC[variance_MC<0] = np.finfo(float).eps
            
            # Calcul de Pf, IC
            result_simulation = simulation.getResult()
            pf = result_simulation.getProbabilityEstimate()

            # Calcul du critère d'enrichissement sur tous les points
            # Récupération de la valeur Umin
            umin, enrich_criterion_values = self.compute_criterion_U(output_MC, variance_MC)

            # Calcul des bornes supérieures et inférieures de pf en prenant en compte 
            # le pourcentage de points bien classés avec un indice de confiance à 95%.
            # et les bornes de pf en ajoutant l'intervalle de confiance à 95%.
            pfinf_kriging, pfsup_kriging, pfinf, pfsup = self.compute_pf_bounds(
                                output_MC, enrich_criterion_values, input_MC)

            if self.verbose:
                print('Validation Q2 : {:0.5f}'.format(Q2))
                print('umin = {:0.3f}'.format(umin) )
                print('Pf = {:.3e}'.format(pf))
                print('Taille MC = {}'.format(input_MC.getSize()))
                print('Pf inf (u>2) = {:.3e}'.format(pfinf_kriging))
                print('Pf sup (u>2) = {:.3e}'.format(pfsup_kriging))
                print('Pf inf (u>2 & IC 95%) = {:.3e}'.format(pfinf))
                print('Pf inf (u>2 & IC 95%) = {:.3e}'.format(pfsup))
                print('')

            # Test umin > 2
            if umin >= self._stopping_criterion:
                break

            # recherche du point minimisant le critère et ajout du point dans le DOE
            input_DOE = self.enrich_DOE(input_DOE, input_MC, output_MC,
                                        variance_MC, enrich_criterion_values)
        
        self.input_DOE = input_DOE
        self.output_DOE = output_DOE
        self.input_MC = input_MC
        self.output_MC = output_DOE

        return result_simulation

    def setBasis(self, basis):
        """
        Accessor to the kriging basis. 

        Parameters
        ----------
        basis : :py:class:`openturns.Basis`
            The basis used as trend in the kriging model.
        """
        try:
            ot.Basis(basis)
        except NotImplementedError:
            raise Exception('The given parameter is not a Basis.')
        self._basis = basis

    def getBasis(self):
        """
        Accessor to the kriging basis. 

        Returns
        -------
        basis : :py:class:`openturns.Basis`
            The basis used as trend in the kriging model. Default is a linear
            basis for the defect and constant for the other parameters.
        """
        if self._basis is None:
            print('The run method must be launched first.')
        else:
            return self._basis

    def setCovarianceModel(self, covariance_model):
        """
        Accessor to the kriging covariance model. 

        Parameters
        ----------
        covariance_model : :py:class:`openturns.CovarianceModel`
            The covariance model in the kriging model.
        """
        try:
            ot.CovarianceModel(covariance_model)
        except NotImplementedError:
            raise Exception('The given parameter is not a CovarianceModel.')
        self._covariance_model = covariance_model

    def getCovarianceModel(self):
        """
        Accessor to the kriging covariance model. 

        Returns
        -------
        covariance_model : :py:class:`openturns.CovarianceModel`
            The covariance model in the kriging model. Default is an anisotropic
            squared exponential covariance model.
        """
        if self._covariance_model is None:
            print('The run method must be launched first.')
        else:
            return self._covariance_model