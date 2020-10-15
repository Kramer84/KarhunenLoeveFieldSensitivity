# -*- coding: utf-8 -*-

__all__ = ['PolynomialChaos']

import pandas as pd
import numpy as np
import openturns as ot
from openturns.viewer import View
from ._graphics import adequation_plot
import matplotlib.pyplot as plt

#########################################################
# To do ajouter test de sobol
# To do Notebook exemples PCE
#########################################################

class PolynomialChaos():

    def __init__(self, distribution):

        self.distribution = distribution
        self.dim = distribution.getDimension()
        self._use_fit_method = False

    def _build_adaptive_strategy(self, degree):
        """Build the adaptive strategy.

        Notes
        -----
        The strategy is built based on a linear enumerate function
        with a fixed strategy.
        """
        enumerateFunction = ot.EnumerateFunction(self.dim)
        polyCol = [0.] * self.dim
        for i in range(self.dim):
            polyCol[i] = ot.StandardDistributionPolynomialFactory(
                self.distribution.getMarginal(i))

        multivariateBasis = ot.OrthogonalProductPolynomialFactory(
            polyCol, enumerateFunction)
        indexMax = enumerateFunction.getStrataCumulatedCardinal(degree)
        return ot.FixedStrategy(multivariateBasis, indexMax)

    def fit(self, input_sample, output_sample, degree, sparse=False):
        """
        Build a PC using an openturns Experiment.

        Parameters
        ----------
        input_sample : 2d sequence of float
            The input sample of the design of experiment.
        output_sample : 2d sequence of float
            The computed output sample.
        degree : int
            The degree of the polynomial chaos.
        sparse : bool
            Flag telling to use sparse polynomial or not, default is False.
        """

        adaptiveStrategy = self._build_adaptive_strategy(degree)
        if sparse:
            # Sparse chaos
            approximation_algorithm = ot.LeastSquaresMetaModelSelectionFactory(
                ot.LARS(), ot.CorrectedLeaveOneOut())
            evaluationStrategy = ot.LeastSquaresStrategy(
                approximation_algorithm)
        else:
            # regular chaos
            evaluationStrategy = ot.LeastSquaresStrategy()

        # chaos algo
        self.algo_chaos = ot.FunctionalChaosAlgorithm(
            input_sample, output_sample, self.distribution, adaptiveStrategy,
            evaluationStrategy)
        self.algo_chaos.run()
        self.input_sample = self.algo_chaos.getInputSample()
        self.output_sample = self.algo_chaos.getOutputSample()
        metamodel = self.algo_chaos.getResult().getMetaModel()
        metamodel.setOutputDescription(output_sample.getDescription())
        self._use_fit_method = True
        return metamodel

    def fit_by_integration(self, model, size, degree):
        """
        Build a PC using an openturns Experiment.

        Parameters
        ----------
        model : openturns.Function
            The model that is used to compute the output values.
        size : int
            The number of sample to generate.
        degree : int
            The degree of the polynomial chaos.
        """

        adaptiveStrategy = self._build_adaptive_strategy(degree)
        multivariateBasis = adaptiveStrategy.getBasis()
        measure = multivariateBasis.getMeasure()
        evaluationStrategy = ot.IntegrationStrategy(
            ot.GaussProductExperiment(measure, [degree] * self.dim))

        self.algo_chaos = ot.FunctionalChaosAlgorithm(
            model, self.distribution, adaptiveStrategy, evaluationStrategy)
        self.algo_chaos.run()
        self.input_sample = self.algo_chaos.getInputSample()
        self.output_sample = self.algo_chaos.getOutputSample()
        metamodel = self.algo_chaos.getResult().getMetaModel()
        metamodel.setInputDescription(self.distribution.getDescription())
        metamodel.setOutputDescription(self.output_sample.getDescription())
        self._use_fit_method = False
        return metamodel

    def compute_r2(self, input_sample=None, output_sample=None):
        """ Compute R2 based on the learning experiment or a given DOE.

        Notes
        -----
        The R2 is computed using the learning DOE if no samples are given.
        If samples are given the R2 is computed on it. The "output_sample"
        must be computed with the model used to built the metamodel using
        the given "input_sample".
        """

        if (input_sample is None and output_sample is not None) or \
           (input_sample is not None and output_sample is None):
            raise Exception(
                "Both input and output sample must be given or None.")

        result = self.algo_chaos.getResult()
        inverse_transformation = result.getInverseTransformation()
        metamodel = result.getMetaModel()

        if input_sample is None and output_sample is None:
            input_sample = inverse_transformation(self.input_sample)
            output_sample = self.output_sample

        input_sample = ot.Sample(input_sample)
        output_sample = ot.Sample(output_sample)

        output_description = self.output_sample.getDescription()

        R2 = pd.DataFrame(index=['R2'], columns=output_description)
        for marginal in range(output_sample.getDimension()):
            validation = ot.MetaModelValidation(
                input_sample, output_sample[:, marginal],
                metamodel.getMarginal(marginal))
            R2.loc['R2'][marginal] = validation.computePredictivityFactor()
        return R2

    def draw_validation(self, input_sample=None, output_sample=None,
                        savefile=True):
        """Draw validation graph based on the learning experiment
         or a given DOE.

        Notes
        -----
        The graph is plotted using the learning DOE if no samples are given.
        If samples are given the validation is based on it. The "output_sample"
        must be computed with the model used to built the metamodel using
        the given "input_sample".
        """

        if (input_sample is None and output_sample is not None) or \
           (input_sample is not None and output_sample is None):
            raise Exception(
                "Bchoth input and output sample must be given or None.")

        result = self.algo_chaos.getResult()
        inverse_transformation = result.getInverseTransformation()
        metamodel = result.getMetaModel()

        if input_sample is None and output_sample is None:
            input_sample = inverse_transformation(self.input_sample)
            output_sample = self.output_sample

        input_sample = ot.Sample(input_sample)
        output_sample = ot.Sample(output_sample)

        output_description = self.output_sample.getDescription()
#       size = input_sample.getSize()

        for marginal in range(output_sample.getDimension()):
            fig, ax = adequation_plot(
                output_sample[:, marginal],
                metamodel.getMarginal(marginal)(input_sample))
            ax.set_title('Validation for {}'.format(
                output_description[marginal]))
            fig.show()
            if savefile:
                fig.savefig('validation_chaos_{}.png'.format(
                    output_description[marginal]),
                    bbox_inches='tight', dpi=150)

    def compute_q2(self):
        """ Compute Q2 analytically.
        """
        if not self._use_fit_method:
            raise Exception(
                "compute_q2 method can not be used with a model "
                "constructed by fit_by_integration method")
        result = self.algo_chaos.getResult()
        inverse_transformation = result.getInverseTransformation()

        input_sample = np.array(inverse_transformation(self.input_sample))
        output_description = self.output_sample.getDescription()
        output_sample = np.array(self.output_sample)
        size = input_sample.shape[0]

        nb_coef = result.getCoefficients().getSize()
        metamodel = result.getMetaModel()
        reducedBasis = result.getReducedBasis()
        transformation = result.getTransformation()
        dimBasis = len(reducedBasis)
        A = np.zeros((size, dimBasis))
        for i in range(dimBasis):
            A[:, i] = np.hstack(reducedBasis[i](transformation(input_sample)))
        # Calcul de la matrice H
        gramA = np.dot(A.transpose(), A)
        H = np.dot(A, np.dot(np.linalg.inv(gramA), A.transpose()))
        Hdiag = np.vstack(H.diagonal())
        y_pred = np.array(metamodel(input_sample))
        # correction
        T = size / (size - nb_coef) * (1 + np.trace(
            np.linalg.inv(gramA / size)) / size)

        Q2 = pd.DataFrame(index=['Q2'], columns=output_description)
        for marginal in range(output_sample.shape[1]):
            # Calcul de err_loo
            delta = (output_sample[:, marginal] - y_pred[:, marginal]) / (
                1 - Hdiag)
            err_loo = np.mean(delta**2)
            # Calcul du Q2
            Q2.loc['Q2'][marginal] = 1 - T * err_loo / np.var(
                output_sample[:, marginal])
        return Q2

    def get_sobol_indices(self):
        """
        Get the first and total sobol index from the polynomial chaos.

        Returns
        -------
        sobol : pandas.DataFrame
            The values of the first and total order with the interactions.
        """
        assert hasattr(self, 'algo_chaos'), "You haven't invoked yet one of " \
                                            "the fitting method"

        self.sobol = ot.FunctionalChaosSobolIndices(
            self.algo_chaos.getResult())

        multi_column = pd.MultiIndex.from_product([list(
            self.output_sample.getDescription()),
            ['First order', 'Total order']])
        sobol = pd.DataFrame(
            index=list(self.distribution.getDescription()) + ['Interaction'],
            columns=multi_column)
        sobol.index.name = 'Input variable'

        output_description = self.output_sample.getDescription()
        for marginal in range(self.output_sample.getDimension()):
            firstSobol = np.array([self.sobol.getSobolIndex(
                [j], marginal) for j in range(self.dim)])
            totalSobol = np.array([self.sobol.getSobolTotalIndex(
                [j], marginal) for j in range(self.dim)])
            interaction = 1. - firstSobol.sum()
            if interaction < 0:
                interaction = 0
            sobol[output_description[marginal], 'First order'] = np.vstack(
                np.concatenate([firstSobol, [interaction]]))
            sobol[output_description[marginal], 'Total order'] = np.vstack(
                np.concatenate([totalSobol, [np.nan]]))
        sobol = sobol.round(3)
        return sobol

    def draw_first_order_indices(self, filter_threshold=None, savefile=True):
        """ Draw the first order indices in a pie chart.
        """
        sobol = self.get_sobol_indices()
        for name in self.output_sample.getDescription():
            # filter sobol values if they are under the given
            # threshold to no display it
            if filter_threshold:
                values = sobol[name, 'First order']
                [sobol[name, 'First order'] >= filter_threshold]

            else:
                values = sobol[name, 'First order']

            labels = list(range(values.shape[0]))
            for i in range(values.shape[0]):
                labels[i] = values.index[i] + ' : %.1f %%' % (100 * values[i])

            graphPie = ot.Pie(values)
            graphPie.setLabels(labels)
            fig = View(
                graphPie, figure_kwargs={'figsize': (6, 6)},
                axes_kwargs={
                    'title': "First order Sobol' indices for {}".format(name)},
                pie_kwargs={'explode': [0] * values.shape[0]}).getFigure()
            fig.show()
            if savefile:
                fig.savefig('sobol_first_order_{}.png'.format(name),
                            bbox_inches='tight', dpi=150)

    def draw_total_order_indices(self, filter_threshold=None, savefile=True):
        """ Draw the total order indices in a pie chart.
        """
        sobol = self.get_sobol_indices()
        for name in self.output_sample.getDescription():
            # filter sobol values if they are under the given threshold
            # to no display it
            if filter_threshold:
                values = sobol[name, 'Total order']
                [sobol[name, 'Total order'] >= filter_threshold]
            else:
                values = sobol[name, 'Total order'][:-1]

            labels = list(range(values.shape[0]))
            for i in range(values.shape[0]):
                labels[i] = values.index[i] + ' : %.1f %%' % (100 * values[i])

            graphPie = ot.Pie(values)
            graphPie.setLabels(labels)
            fig = View(graphPie, figure_kwargs={'figsize': (6, 6)},
                       axes_kwargs={
                           'title': "Total order Sobol' indices for {}".format(
                                    name)}).getFigure()
            fig.show()
            if savefile:
                fig.savefig('sobol_total_order_{}.png'.format(name),
                            bbox_inches='tight', dpi=150)

    def draw_bar_sobol_indices(self, savefile=True):
        """Draw the Sobol indices with bar representation.
        """
        # for each output
        for output_description in self.output_sample.getDescription():
            # plot
            fig = self._plot_bar_sobol(output_description, fontsize=14)
            fig.suptitle(
                "Sobol indices for {}".format(output_description), fontsize=14)
            fig.show()
            if savefile is not None:
                fig.savefig(
                    'sobol_bar_{}.png'.format(
                        output_description), bbox_inches='tight', dpi=150)

    def _plot_bar_sobol(self, output_name, legend_loc='upper left',
                        fontsize=14, ylim={}, ncol=1):
        """
        Plot in a bar representation the Sobol indices for one output.
        """

        sobol = self.get_sobol_indices()
        countOrigin = 0
        color_simp = ['red', 'blue']
        fig = plt.figure(figsize=(self.dim * 1, 10))
        ax = fig.add_subplot(111)

        for input_index in range(self.dim):
            ax.bar(countOrigin, sobol[output_name,
                   'First order'][input_index], color=color_simp[0])
            countOrigin += 1
            ax.bar(countOrigin, sobol[output_name,
                   'Total order'][input_index], color=color_simp[1])
            countOrigin += 2

        ax.set_ylabel('Sobol indices', fontsize=fontsize)
        ax.set_xticks(np.arange(0.5, countOrigin - 2, 3))
        ax.set_xticklabels(self.distribution.getDescription(), rotation=30,
                           fontsize=fontsize, horizontalalignment='right')

        ax.set_xlim(right=countOrigin)
        ax.set_ylim(**ylim)
        ax.legend(['First order', 'Total order'],
                  loc=legend_loc, fontsize=fontsize, ncol=ncol)
        ax.grid(True)
        return fig
