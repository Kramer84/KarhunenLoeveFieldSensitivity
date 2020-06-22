__version__ = '0.1'
__author__ = 'Kristof Attila S.'
__date__ = '22.06.20'

from typing import Callable, List, Tuple, Optional, Any, Union
from joblib import Parallel, delayed, cpu_count
from itertools import chain

import openturns
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class StochasticProcessSobolIndicesAlgorithmBase(object):
    '''Basic methods to calculate unitary sensitivity indices.
    We first set the samples Y_A and Y_B and calculate the means and
    variances of those, so they don't have to be calculated again.
    The notations are those of A. DUMAS in the paper :
    "Lois asymptotiques des estimateurs des indices de Sobol"

    This class can accept vectors (unidimensional outputs) as well
    as matrices (multidimensional outputs)

    The agregated Sobol indices are not calculated yet
    '''
    @staticmethod
    def _centerSobolExp(SobolExperiment: numpy.array, N: int) -> Tuple[numpy.array, List]:
        '''private method centering the output desgins as well as
        preparing the data for the estimator calculus
        '''
        nSamps = int(SobolExperiment.shape[0] / N)
        N = int(N)
        inputListParallel = list()
        SobolExperiment0 = SobolExperiment
        dim = 1
        psi_fo, psi_to = StochasticProcessSobolIndicesAlgorithmBase._SymbolicSaltelliIndices(1)
        for i in range(nSamps):
            # Centering
            SobolExperiment[i * N:(i + 1) * N, ...] = numpy.subtract(
                SobolExperiment[i * N:(i + 1) * N, ...],
                SobolExperiment[i * N:(i + 1) * N, ...].mean(axis=0))
        for p in range(nSamps - 2):
            inputListParallel.append(
                (SobolExperiment[:N, ...],
                    SobolExperiment[N:2 * N, ...],
                    SobolExperiment[(2 + p) * N:(3 + p) * N, ...],
                    psi_fo, psi_to))
        return SobolExperiment, inputListParallel

    @staticmethod
    def getSobolIndices(SobolExperiment : numpy.array, N : int, method : str = 'Saltelli'):
        '''method to retrieve the Sobol' indices from an output design.
        It uses to samples A,B and a number of samples being the sample A
        with one column of sample B.
        for now only the first order indices are calculated
        '''
        expShape = SobolExperiment.shape
        nIndices = int(expShape[0] / N) - 2
        dim = expShape[1:]
        SobolExperiment0 = SobolExperiment
        if dim == ():
            dim = 1
        print('There are', nIndices, 'indices to get in', dim,
              'dimensions with', SobolExperiment[0].size, 'elements')
        SobolExperiment, inputListParallel = StochasticProcessSobolIndicesAlgorithmBase._centerSobolExp(
            SobolExperiment, N)
        if method is 'Saltelli':

            SobolIndices = [StochasticProcessSobolIndicesAlgorithmBase.SaltelliIndices(
                *inputListParallel[i]) for i in range(nIndices)]
            SobolIndices, SobolIndicesTot, VarSobolIndices, VarSobolIndicesTot = map(
                list, zip(*SobolIndices))
            print('Indices successfully calculated')
            SobolIndices = numpy.stack(SobolIndices)
            SobolIndicesTot = numpy.stack(SobolIndicesTot)
            VarSobolIndices = numpy.stack(VarSobolIndices)
            VarSobolIndicesTot = numpy.stack(VarSobolIndicesTot)
            halfConfinterSFO = VarSobolIndices
            halfConfinterSTO = VarSobolIndicesTot  # Confidence interval (half)
            SobolIndices = numpy.squeeze(SobolIndices)
            SobolIndicesTot = numpy.squeeze(SobolIndicesTot)
            halfConfinterSFO = numpy.squeeze(halfConfinterSFO)
            halfConfinterSTO = numpy.squeeze(halfConfinterSTO)

        return SobolIndices, SobolIndicesTot, halfConfinterSFO, halfConfinterSTO

    @staticmethod
    def SaltelliIndices(Y_Ac: numpy.array, Y_Bc: numpy.array,\
                Y_Ec: numpy.array, psi_fo: openturns.SymbolicFunction,\
                psi_to: openturns.SymbolicFunction) ->\
                    Tuple[numpy.array, numpy.array, numpy.array, numpy.array]:
        '''method implementing the Saltelli' way of calculating 
        Sobol Indices it only calculates the indice for one unique input variables
        at the time, but on the whole output dimensions, thanks to highly parallelized
        numpy code. 

        Attributes
        ----------
        Y_Ac : numpy.array
            sample A , centered on its mean
        Y_Bc : numpy.array
            sample B , centered on its mean
        Y_Ec : numpy.array
            sample E (A|B_i) , centered on its mean
        psi_fo : openturns.SymbolicFunction
            symbolic function to calculate the gradient of
            the estimator function
        psi_fto : openturns.SymbolicFunction
            symbolic function to calculate the gradient of
            the estimator function

        Returns
        -------
        S : numpy.array
            first order Sobol' indices
        S_tot : numpy.array
            total order Sobol' indices
        varS : numpy.array
            variance of the first order Sobol' indice
        varS_tot : numpy.array
            variance of the total order Sobol' indice

        '''
        assert (Y_Ac.shape == Y_Bc.shape ==
                Y_Ec.shape), "samples have to have same shape"
        N = Y_Ac.shape[0]
        Ni = 1. / N

        # Simplified indice as samples centered
        S = numpy.divide(Ni * numpy.sum(numpy.multiply(Y_Bc, Y_Ec), axis=0),
                         Ni * numpy.sum(numpy.square(Y_Ac), axis=0)
                         )
        S_tot = numpy.subtract(1.,
                               numpy.divide(Ni * numpy.sum(numpy.multiply(Y_Ac, Y_Ec), axis=0),
                                            Ni * numpy.sum(numpy.square(Y_Ac)))
                               )
        varS, varS_tot = StochasticProcessSobolIndicesAlgorithmBase._computeVariance(Y_Ac, Y_Bc,
                                                                                     Y_Ec, N, psi_fo, psi_to)
        return S, S_tot, varS, varS_tot

    @staticmethod
    def _SymbolicSaltelliIndices(N : int) :
        '''Method to create the symbolic python saltelli function

        Note
        ----
        N > 1 only in the case of aggregated indices

        Arguments
        ---------
        N : int
            number of nominators and denominators

        Returns
        -------
        psi_fo : openturns.SymbolicFunction
            first order symbolic function
        psi_to : openturns.SymbolicFunction
            total order symbolic function
        '''
        x, y = (openturns.Description.BuildDefault(N, 'X'),
                openturns.Description.BuildDefault(N, 'Y'))
        # in order X0, Y0, X1, Y1
        xy = list(x)
        for i, yy in enumerate(y):
            xy.insert(2 * i + 1, yy)
        symbolic_num, symbolic_denom = '', ''
        symbolic_num, symbolic_denom = ([item for sublist in zip(x, ['+'] * N) for item in sublist],
                                        [item for sublist in zip(y, ['+'] * N) for item in sublist])
        (symbolic_num.pop(), symbolic_denom.pop())
        symbolic_num = ''.join(symbolic_num)
        # psi  = (x1 + x2 + ...) / (y1 + y2 + ...).
        symbolic_denom = ''.join(symbolic_denom)
        psi_fo, psi_to = (openturns.SymbolicFunction(xy, ['(' + symbolic_num + ')/(' + symbolic_denom + ')']),
                          openturns.SymbolicFunction(xy, ['1 - ' + '(' + symbolic_num + ')/(' + symbolic_denom + ')']))
        return psi_fo, psi_to

    @staticmethod
    def _computeVariance(YAc, YBc, YEc, N, psi_fo, psi_to):
        """
        Compute the variance of the estimator sample

        Parameters
        ----------
        YAc : Sample (numpy.array)
            Centered first sample YA
        YBc : Sample (numpy.array)
            Centered second sample B        
        YEc : Sample (numpy.array)
            Centered sample from mixed matrix 
        N : int
            The size of the sample.
        psi_fo : symbolic function
            First order saltelli indices symbolic function
        psi_to : symbolic function
            Total order saltelli indices symbolic function
        """
        baseShape = YAc.shape
        # print('basic output shape is:', baseShape)
        flatDim = int(numpy.prod(baseShape[1:]))
        flatShape = [N, flatDim]
        # print('output reshaped into matrix of shape (dim<=2) ',flatShape)
        YAc = numpy.reshape(YAc, flatShape)
        YBc = numpy.reshape(YBc, flatShape)
        YEc = numpy.reshape(YEc, flatShape)

        # some intermediary calculus
        # first order:
        X_fo = numpy.squeeze(numpy.multiply(YBc, YEc))
        Y_fo = numpy.squeeze(numpy.square(YAc))

        # total order
        X_to = numpy.squeeze(numpy.multiply(YAc, YEc))
        Y_to = numpy.squeeze(numpy.square(YAc))

        varianceFO = StochasticProcessSobolIndicesAlgorithmBase._computeSobolVariance(
            X_fo, Y_fo, psi_fo, N)
        varianceTO = StochasticProcessSobolIndicesAlgorithmBase._computeSobolVariance(
            X_to, Y_to, psi_to, N)

        shape = baseShape[1:]
        if len(baseShape) <= 1:
            shape = (1,)
        varianceFO = numpy.reshape(varianceFO, shape)
        varianceTO = numpy.reshape(varianceTO, shape)
        return varianceFO, varianceTO

    @staticmethod
    def _computeSobolVariance(X, Y, psi, N):
        """
        Compute the variance of the estimators (NON agregated)

        Parameters
        ---------- 
        X : numpy.array

        Y : numpy.array

        psi : Function
            The function that computes the sobol estimates.
        N : int
            The size of the sample.
        """

        # 1 if output has only one dimension, as numpy.prod(())=1
        dims = int(numpy.prod(X.shape[1:]))
        # here we get the covariance matrix for each output

        # Lorsqu'il y a multiples dimension en sortie
        if dims > 1:
            covariance = numpy.squeeze(numpy.stack(
                [numpy.cov(X[..., i], Y[..., i], rowvar=True) for i in range(dims)]))
            mean_samp = list(zip(X.mean(axis=0), Y.mean(axis=0)))
            mean_samp_list = mean_samp
            mean_psi = numpy.stack([numpy.squeeze(numpy.asarray(
                psi.gradient(mean_samp_list[i])))for i in range(len(mean_samp_list))])
            mean_psi_temp = numpy.stack(
                [mean_psi.T, mean_psi.T]).T.transpose((0, 2, 1))
            # This line is similar to a dot product
            P2 = numpy.sum(numpy.multiply(mean_psi_temp, covariance), axis=1)
            variance = numpy.divide(
                numpy.sum(numpy.multiply(mean_psi, P2),  axis=1), N)

        # lorsqu'il y a une dimension de sortie
        else:          # Ces formules sont utilisées par exemple pour ishigami
            covariance = numpy.cov([X, Y])
            mean_samp = [X.mean(), Y.mean()]
            meanPsi = numpy.squeeze(psi.gradient(mean_samp))
            variance = numpy.matmul(meanPsi, numpy.matmul(covariance, meanPsi))
            variance = variance / N
        return variance


        ### lors de l'affichage, nous multiplions la variance par 4 ##
        ### pour couvrir l'ensemble de l'intervalle de confiance    ##
        #######################  PLOTTING   ##########################

def plotSobolIndicesWithErr(S, errS, varNames, n_dims, Stot=None, errStot=None):
    '''Function to plot the Sobol' indices with an errorbar to visualize the 
    uncertaintity in the estimator. (only first and total order yet) 

    Note
    ----
    Function is written to adapt the plotting according to the dimensions of
    the input, for higher level inputs, mayavi seems the most reasonable choice
    for plotting
    '''

    plt.style.use('classic')
    S, errS = numpy.squeeze(S), numpy.squeeze(errS)
    if Stot is not None and errStot is not None:
        Stot, errStot = numpy.squeeze(Stot), numpy.squeeze(errStot)
    assert len(
        varNames) == n_dims, "Error in the number of dimensions or variable names"
    assert S.shape == errS.shape, "There have to be as much confidence intervals as indices"
    if len(S.shape) == 1:
        print('The output is scalar')
        print('The sensitivity is measured accordingly to the', n_dims,
              'input variables, namely:\n', ' and '.join(varNames))

        lgd_elems = [mpatches.Circle((0, 0),
                                     radius=7,
                                     color='r',
                                     label='first order indices'),
                     mpatches.Circle((0, 0),
                                     radius=7,
                                     color='b',
                                     label='total order indices')]

        x = numpy.arange(n_dims)
        y = S
        yerr = errS  # to have 95%                                 #####

        fig, ax = plt.subplots()
        ax.errorbar(x, y, yerr=yerr, fmt='s', color='r', ecolor='r')
        if Stot is not None and errStot is not None:
            y2 = numpy.squeeze(Stot)
            y2err = numpy.squeeze(errStot)                        #####
            ax.errorbar(x + 0.05, y2, yerr=y2err,
                        fmt='o', color='b', ecolor='b')
        else:
            lgd_elems.pop()
        ax.legend(handles=lgd_elems, loc='upper right')
        ax.set_xticks(ticks=x)
        ax.set_xticklabels(labels=varNames)
        ax.axis(xmin=-0.5, xmax=x.max() + 0.5, ymin=-0.1, ymax=1.1)
        plt.grid()
        plt.show()

    if len(S.shape) == 2:
        print('The output is a vector')
        plt.ion()
        fig = plt.figure(figsize=(20, 10))
        # Here we dinamically build our grid according to the number of input
        # dims
        if n_dims <= 5:
            n_cols = case = 1
        elif n_dims > 5 and n_dims <= 10:
            n_cols = case = 2
        else:
            case = 3
            raise NotImplementedError

        graphList = list()
        if case is 1:
            colspan = 5
            rowspan = 2
            colTot = 5
            rowTot = 2 * n_dims
            for i in range(n_dims):
                graphList.append(
                    plt.subplot2grid((rowTot, colTot),
                                     (i * rowspan, 0),
                                     colspan=colspan,
                                     rowspan=rowspan,
                                     fig=fig))
                graphList[i].set_title(varNames[i], fontsize=10)

                dimOut = S.shape[1]
                x = numpy.arange(dimOut)
                y = S[i, ...]
                yerr = errS[i, ...]

                graphList[i].errorbar(x, y, yerr, color='r', ecolor='b')
                graphList[i].axis(xmin=-0.5, xmax=x.max() +
                                  0.5, ymin=y.min() - 0.1, ymax=y.max() + 0.1)
            fig.subplots_adjust(hspace=0.25, wspace=0.25)
            plt.tight_layout()
            fig.canvas.draw()
            plt.show()

        if case is 2:
            colspan = 5
            rowspan = 2
            colTot = 5 * 2
            rowTot = 5  # (cause we fill up at least the full left side)
            for i in range(n_dims):
                col = 0
                if i > 5:
                    col = 1
                graphList.append(
                    plt.subplot2grid((rowTot, colTot),
                                     (i * rowspan, col * 5),
                                     colspan=colspan,
                                     rowspan=rowspan,
                                     fig=fig))
                graphList[i].set_title(varNames[i], fontsize=10)

                dimOut = S.shape[1]
                x = numpy.arange(dimOut)
                y = S[i, ...]
                yerr = errS[i, ...]

                graphList[i].errorbar(x, y, yerr, color='r', ecolor='b')
                graphList[i].axis(xmin=-0.5, xmax=x.max() + 0.5,
                                  ymin=y.min() - 0.01, ymax=y.max() + 0.01)
            fig.subplots_adjust(hspace=0.25, wspace=0.25)
            plt.tight_layout()
            fig.canvas.draw()
            plt.show()

    if len(S.shape) == 3:
        print('The output is a 2D field')
    plt.style.use('default')
