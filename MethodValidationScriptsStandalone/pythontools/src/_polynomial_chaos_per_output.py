#-*- coding: utf-8 -*-

__all__ = ['PolynomialChaosPerOutput']

import pandas as pd
import numpy as np
import openturns as ot
from openturns.viewer import View
from ._polynomial_chaos import PolynomialChaos
import matplotlib.pyplot as plt

class PolynomialChaosPerOutput():
    
    def __init__(self, distribution, reduced_distribution_collection):
        """
        Class that create a polynomial chaos where the input are all differents
        and return the Sobol indices.
        
        Parameters
        ----------
        distribution : openturns.Distribution
            The joint distribution of all inputs.
        reduced_distribution_collection : list of openturns.Distribution
            The list of input distributions corresponding to each output
        """
        self.distribution = distribution
        self.dim = distribution.getDimension()
        self.reduced_distribution_collection = reduced_distribution_collection
        
    def fit(self, input_sample, output_sample, degree, sparse=False):
        """
        Build a PC using an openturns Experiment.
        
        Parameters
        ----------
        input_sample : 2d sequence of float
            The input sample of the design of experiment.
        output_sample : 2d sequence of float
            The computed output sample.
        degree : int of list of int
            The degree of the polynomial chaos.
        sparse : bool
            Flag telling to use sparse polynomial or not, default is False. 
        """
        
        degree_list = np.atleast_1d(degree)
        
        output_dim = output_sample.getDimension()
        
        self.chaos_collection = [object] * output_dim
        chaos_model_collection = [object] * output_dim
        transformation_model_collection = [object] * output_dim
        composed_model_collection = [object] * output_dim

        input_data = pd.DataFrame(np.array(input_sample), columns=self.distribution.getDescription())

        for i in range(output_dim):
            output_name = output_sample.getDescription()[i]

            current_selected_input_name = list(self.reduced_distribution_collection[i].getDescription())
            transformation_model_collection[i] = ot.SymbolicFunction(self.distribution.getDescription(),
                                                                     current_selected_input_name)
            
            current_Q2 = -10000000000000.
            for i_deg, degree in enumerate(degree_list):
                tested_chaos = PolynomialChaos(self.reduced_distribution_collection[i])
                chaos_model = tested_chaos.fit(input_data[current_selected_input_name].values,
                                                                    output_sample[:, i],
                                                                    int(degree),
                                                                    sparse)

                if current_Q2 < tested_chaos.compute_Q2().values:
                    self.chaos_collection[i] = tested_chaos
                    chaos_model_collection[i] = chaos_model
                    current_Q2 = tested_chaos.compute_Q2().values
                
            # choose the chaos model with the best Q2
            composed_model_collection[i] = ot.ComposedFunction(chaos_model_collection[i], transformation_model_collection[i])
        
        self.full_metamodel = ot.AggregatedFunction(composed_model_collection)
        self.full_metamodel.setOutputDescription(output_sample.getDescription())
        return self.full_metamodel
    
    def getMetamodel(self):
        return self.full_metamodel
    
    def compute_R2(self, input_sample=None, output_sample=None):
        """ Compute R2 analytically.
        """
        
        if (input_sample is None and output_sample is not None) or \
            (input_sample is not None and output_sample is None):
            raise Exception("Both input and output sample must be given or None.")
        
        if input_sample is None and output_sample is None:
            R2 = self.chaos_collection[0].compute_R2()
            for i, chaos in enumerate(self.chaos_collection):
                if i != 0:
                    R2 = R2.join(chaos.compute_R2())
        else:
            input_sample = pd.DataFrame(np.array(input_sample), columns=self.distribution.getDescription())
        
            R2 = self.chaos_collection[0].compute_R2(
                    input_sample[list(self.reduced_distribution_collection[0].getDescription())].values,
                    output_sample[:, 0])
            for i, chaos in enumerate(self.chaos_collection):
                if i != 0:
                    R2 = R2.join(chaos.compute_R2(
                    input_sample[list(self.reduced_distribution_collection[i].getDescription())].values,
                    output_sample[:, i]))
        return R2

    def compute_Q2(self):
        """ Compute Q2 analytically.
        """
        Q2 = self.chaos_collection[0].compute_Q2()
        for i, chaos in enumerate(self.chaos_collection):
            if i != 0:
                Q2 = Q2.join(chaos.compute_Q2())
        return Q2
    
    def drawValidation(self, input_sample=None, output_sample=None, savefile=True):
        """Draw validation graph based on the learning experiment or a given DOE.
        
        Notes
        -----
        The graph is plotted using the learning DOE if no samples are given. 
        If samples are given the validation is based on it. The "output_sample"
        must be computed with the model used to built the metamodel using
        the given "input_sample".
        """
        
        
        if (input_sample is None and output_sample is not None) or \
            (input_sample is not None and output_sample is None):
            raise Exception("Both input and output sample must be given or None.")
        
        if input_sample is None and output_sample is None:
            for chaos in self.chaos_collection:
                chaos.drawValidation(input_sample, output_sample, savefile)
        else:
            input_sample = pd.DataFrame(np.array(input_sample), columns=self.distribution.getDescription())
            
            for i, chaos in enumerate(self.chaos_collection):
                chaos.drawValidation(input_sample[list(self.reduced_distribution_collection[i].getDescription())].values,
                                     output_sample[:, i],
                                     savefile)

    def getSobolIndices(self):
        """
        Get the first and total sobol index from the polynomial chaos.

        Returns
        -------
        sobol : pandas.DataFrame
            The values of the first and total order with the interactions.
        """
        sobol = pd.DataFrame(index=list(self.distribution.getDescription()) + ['Interaction'])
        sobol.index.name = 'Input variable'
        for chaos in self.chaos_collection:
            sobol = sobol.join(chaos.getSobolIndices())
        multi = pd.MultiIndex.from_tuples(sobol.columns)
        sobol.columns = multi
        return sobol.fillna('')

    def drawFirstOrderIndices(self, filter_threshold=None, savefile=True):
        """ Draw the first order indices in a pie chart.
        """
        for chaos in self.chaos_collection:
            chaos.drawFirstOrderIndices(filter_threshold, savefile)

    def drawTotalOrderIndices(self, filter_threshold=None, savefile=True):
        """ Draw the total order indices in a pie chart.
        """
        for chaos in self.chaos_collection:
            chaos.drawTotalOrderIndices(filter_threshold, savefile)

    def drawBarSobolIndices(self, savefile=True):
        """Draw the Sobol indices with bar representation.
        """
        for chaos in self.chaos_collection:
            chaos.drawBarSobolIndices(savefile)