#-*- coding: utf-8 -*-

__all__ = ['Morris']

import openturns as ot
from openturns.viewer import View
import numpy as np
import otmorris as otm
import pandas as pd
import matplotlib.pyplot as plt

class Morris():
    """
    Run the Morris sensitivity analysis
    
    Parameters
    ----------
    
    """
    
    def __init__(self, model, distribution, level, n_trajectory, seed=39847):
        
        if level%2:
            raise ValueError('The level number must be even.')
        
        self.model = model
        self.distribution = distribution
        self.level = level
        self.n_trajectory = n_trajectory
        self.dim = distribution.getDimension()
        self.seed = seed
        
    def computeMorrisIndices(self):
        # set the seed of the generator
        ot.RandomGenerator.SetSeed(self.seed)
        
        # set the same level for each dimension
        levels = [self.level] * self.dim

        # choose the bound being the 1% and 99% quantile of the distribution
        quantile = np.array([[self.distribution.getMarginal(i).computeQuantile(0.01)[0],
                              self.distribution.getMarginal(i).computeQuantile(0.99)[0]] \
                              for i in range(self.distribution.getDimension())])
        self.bound = ot.Interval(quantile[:,0], quantile[:,1])

        # generate the Morris DOE
        experiment = otm.MorrisExperimentGrid(levels, self.bound, self.n_trajectory)
        experiment.setJumpStep([int(self.level/2)]*self.dim)
        self.input_morris = experiment.generate()

        # évaluation des points
        self.output_morris = self.model(self.input_morris)
        
        # Evaluation des indices
        self.morris = otm.Morris(self.input_morris, self.output_morris, self.bound)
    
    def getMorrisIndices(self, marginal=0):
        mu_star = self.morris.getMeanAbsoluteElementaryEffects(marginal)
        mu = self.morris.getMeanElementaryEffects(marginal)
        sigma = self.morris.getStandardDeviationElementaryEffects(marginal)
        print('Output : {}'.format(self.model.getOutputDescription()[marginal]))
        print('mu* : {}'.format(mu_star))
        print('mu : {}'.format(mu))
        print('sigma : {}'.format(sigma))
        print('\n')
    
    def getInfluentVariables(self, filter_threshold=0.05, marginal_list=None):
        """
        Get the most influent variables based on a filter
        
        Parameters
        ----------
        marginal_list : list of int or None
            The list of output marginal for which the influent variables
            must be found. If None, it means for all output marginal independently.
        filter_threshold : 0 < float < 1
            The threshold between the non influent and influent
            variable as a percentage of the maximum Morris indices.
        
        Returns
        -------
        data : pandas.DataFrame
            Returns either the list of influent variable per each output variable
            or all the influent variables for the selected output. 
        """
        if marginal_list is None:
            # case for each output independently
            influent_series_list = []
            for marginal in range(self.model.getOutputDimension()):
                mu_star = np.array(self.morris.getMeanAbsoluteElementaryEffects(marginal))
                mask = mu_star > mu_star.max()*filter_threshold
                influent_series_list.append(pd.Series(np.array(self.distribution.getDescription())[mask]))

            row_number = np.max([series.size for series in influent_series_list])
            data = pd.DataFrame(index=range(row_number), columns=self.model.getOutputDescription())
            for marginal in range(self.model.getOutputDimension()):
                data[self.model.getOutputDescription()[marginal]] = influent_series_list[marginal]
        else:
            # case for the given marginal list simutaneously
            if type(marginal_list) is not list:
                raise Exception('Parameter "marginal_list" must be a list of integer or None.')

            # create a mask or True and False containing only True if the mu_star of each
            # selected output is greater than the threshold
            mask = [False] * self.distribution.getDimension()
            for marginal in marginal_list:
                mu_star = np.array(self.morris.getMeanAbsoluteElementaryEffects(marginal))
                mask = np.logical_or(mask, mu_star > mu_star.max()*filter_threshold)

            influent_variable_series = pd.Series(np.array(self.distribution.getDescription())[mask])
            selected_output_series = pd.Series(np.array(self.model.getOutputDescription())[marginal_list])
            row_number = np.max([influent_variable_series.size, selected_output_series.size])

            data = pd.DataFrame(index=range(row_number), columns=['Selected output variables', 'Influent variables'])
            data['Selected output variables'] = selected_output_series
            data['Influent variables'] = influent_variable_series

        return data.fillna('')
    
    def drawMorrisIndices(self, filter_threshold=None, savefile=True):
        """Draw the Morris indices.
        """
        output_des = self.model.getOutputDescription()
        # for each output
        for marginal in range(self.model.getOutputDimension()):
            mu_star = self.morris.getMeanAbsoluteElementaryEffects(marginal)
            mu = self.morris.getMeanElementaryEffects(marginal)
            sigma = self.morris.getStandardDeviationElementaryEffects(marginal)
            # plot
            fig = self._plot_morris(mu_star, mu, sigma, self.distribution.getDescription(), filter_threshold)
            fig.suptitle("Morris indices for {}".format(output_des[marginal]), fontsize=14)
            fig.show()
            if savefile is not None:
                fig.savefig('morris_{}.png'.format(output_des[marginal]), bbox_inches='tight', dpi=150)
                
    def drawBarMorrisIndices(self, savefile=True):
        """Draw the Morris indices with bar representation.
        """
        output_des = self.model.getOutputDescription()
        # for each output
        for marginal in range(self.model.getOutputDimension()):
            mu_star = self.morris.getMeanAbsoluteElementaryEffects(marginal)
            mu = self.morris.getMeanElementaryEffects(marginal)
            sigma = self.morris.getStandardDeviationElementaryEffects(marginal)
            # plot
            fig = self._plot_bar_morris(mu_star, mu, sigma, self.distribution.getDescription(), fontsize=14)
            fig.suptitle("Morris indices for {}".format(output_des[marginal]), fontsize=14)
            fig.show()
            if savefile is not None:
                fig.savefig('morris_bar_{}.png'.format(output_des[marginal]), bbox_inches='tight', dpi=150)

    # Fonction d'affichage des indices
    def _plot_morris(self, mu_star, mu, sigma, labels, filter_threshold=None):
        """
        Plot the Morris indices in the mu_star / sigma and mu_star / mu space.
        """
        mu_star = np.array(mu_star)
        mu = np.array(mu)
        sigma = np.array(sigma)
        labels = np.array(labels)
        if filter_threshold:
            mask = mu_star > mu_star.max()*filter_threshold
            mu_star = mu_star[mask]
            mu = mu[mask]
            sigma = sigma[mask]
            labels = labels[mask]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        tol = 1.1
        # plot sigma vs mu_star
        ax1.plot([0, np.max([mu_star, sigma])*tol], [0, np.max([mu_star, sigma])*tol])
        ax1.plot([0, np.max([mu_star, sigma])*tol], [0, 0.5*np.max([mu_star, sigma])*tol], 'b--')
        ax1.plot(mu_star, sigma, 'ro')
        for i, name in enumerate(labels):
            ax1.annotate(name, (mu_star[i], sigma[i]), fontsize='xx-large')
        ax1.grid()
        ax1.set_xlabel(r'$\mu^*$', fontsize='xx-large')
        ax1.set_ylabel(r'$\sigma$', fontsize='xx-large')
        ax1.set_xlim(right=np.max([mu_star, sigma])*tol)
        ax1.set_ylim(top=np.max([mu_star, sigma])*tol)
        if filter_threshold:
            ax1.vlines(mu_star.max()*filter_threshold, 0, ax1.get_ylim()[1],
                       'r', 'dashed', 'Influent variables threshold at {:.0f}%'.format(filter_threshold*100), alpha=0.6)
        ax1.legend(loc='upper left')

        # plot mu vs mu_sigma
        ax2.plot([0, np.max([mu_star, sigma])*tol], [0, np.max([mu_star, mu])*tol])
        ax2.plot([0, np.max([mu_star, mu])*tol], [0, -np.max([mu_star, mu])*tol], 'b')
        ax2.plot([0, np.max([mu_star, mu])*tol], [0, 0.5*np.max([mu_star, mu])*tol], 'b--')
        ax2.plot([0, np.max([mu_star, mu])*tol], [0, -0.5*np.max([mu_star, mu])*tol], 'b--')
        ax2.plot(mu_star, mu, 'ro')
        for i, name in enumerate(labels):
            ax2.annotate(name, (mu_star[i], mu[i]), fontsize='xx-large')
        ax2.grid()
        ax2.set_xlabel(r'$\mu^*$', fontsize='xx-large')
        ax2.set_ylabel(r'$\mu$', fontsize='xx-large')
        ax2.set_xlim(right=np.max([mu_star, mu])*tol)
        ax2.set_ylim(top=np.max([mu_star, mu])*tol)
        if filter_threshold:
            ax2.vlines(mu_star.max()*filter_threshold, ax2.get_ylim()[0], ax2.get_ylim()[1],
                       'r', 'dashed', 'Influent variables threshold at {:.0f}%'.format(filter_threshold*100), alpha=0.6)
        ax2.legend(loc='upper left')
        
        return fig
    
    def _plot_bar_morris(self, list_mu_star, list_mu, list_sigma, labels, 
                         legend_loc='upper left', fontsize=14, ylim={}, ncol=1):
        """
        Plot in a bar representation the Morris indices.
        """

        countOrigin = 0
        dimension_pb = len(list_mu)
        #color = ot.BarPlot.BuildDefaultPalette(dimension_pb)
        color_simp = ['red','green','blue']
        fig = plt.figure(figsize=(dimension_pb * 1, 10))
        ax = fig.add_subplot(111)

        for i,[mu_star,mu,sigma] in enumerate(zip(list_mu_star,list_mu,list_sigma)):
            ax.bar(countOrigin, mu_star, color=color_simp[0])
            countOrigin += 1
            ax.bar(countOrigin, mu, color=color_simp[1])
            countOrigin += 1
            ax.bar(countOrigin, sigma, color=color_simp[2])
            countOrigin += 2

        ax.set_ylabel('Morris indices', fontsize=fontsize)
        ax.set_xticks(np.arange(1, countOrigin-2, 4))
        ax.set_xticklabels(labels, rotation=30, fontsize=fontsize, horizontalalignment='right')

        ax.set_xlim(right=countOrigin)
        ax.set_ylim(**ylim)
        ax.legend([r'$\mu^*$', r'$\mu$', r'$\sigma$'],loc=legend_loc, fontsize=fontsize, ncol=ncol)
        ax.grid(True)
        return fig