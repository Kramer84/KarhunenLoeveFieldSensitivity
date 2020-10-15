#-*- coding: utf-8 -*-

__all__ = ['plot_matrix', 'adequation_plot', 'plot_histogram']

import openturns as ot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

darkblue = "#3F83AA"
lightblue = "#8DB2CF"

def plot_histogram(X, labels=None, figsize=None, n_columns=3):
    """
    Plot histrograms of a sample
    
    Parameters
    ----------
    X : sample
        The sample to use to build histograms
    labels: list of str
        The name of each columns of the sample
    figsize: (int, int)
        The size of the figure.
        Default is (6 x n_columns, 6 x n_rows)
    n_columns : int
        The number of columns in the subplot figure.
    
    Returns
    -------
    fig : matplotlib.figure
        The figure created.
    """
    
    X = np.array(X)
    n_samples, n_features = X.shape
    
    n_rows = np.ceil(n_features / n_columns)
    
    if figsize is None:
        figsize = (6*n_columns, 6*n_rows)
    
    if labels is None:
        labels = ot.Description.BuildDefault(n_features, 'y')
    
    fig = plt.figure(figsize=figsize)
    n = 0
    for i in range(n_features):
        n += 1
        ax = fig.add_subplot(n_rows, n_columns, n)
            
        n_bins = int(1 + np.log2(n_samples)) + 1
        ax.hist(X[:, i], bins=n_bins, density=True,
                cumulative=False, bottom=None,
                edgecolor='grey', color=lightblue, alpha=.5,
                label=labels[i])
        ax.legend()
    return fig

def adequation_plot(target, prediction, figname=None,
                    with_hist=False, fontsize=14, outliers=None):
    """Adequation plot between a true and predicted values.
    """
    
    target = np.array(target)
    prediction = np.array(prediction)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(target, prediction, '.', ms=5)
    if outliers is not None:
        ax.plot(target[outliers], prediction[outliers], 'r.', ms=7)
    plt.xticks(rotation=45)
    concatmin = np.concatenate([target, prediction]).min()
    if concatmin < 0:
        _min = 1.05*concatmin
    elif concatmin >= 0:
        _min = 0.95*np.concatenate([target, prediction]).min()
    _max = 1.01*np.concatenate([target, prediction]).max()
    ax.plot([_min, _max], [_min, _max], 'k:')
    ax.grid()
    ax.set_xlim(_min, _max)
    ax.set_ylim(_min, _max)
    ax.set_xlabel('Model', fontsize=fontsize)
    ax.set_ylabel('Prediction', fontsize=fontsize)
    ax.set_aspect(1.)
    if with_hist:
        divider = make_axes_locatable(ax)
        ax_x = divider.append_axes("top", 1.2, pad=.2, sharex=ax)
        ax_y = divider.append_axes("right", 1.2, pad=.2, sharey=ax)
        ax_x.hist(target, density=True, alpha=.2)
        ax_x.set_xlim(_min, _max)
        ax_y.hist(prediction, density=True, alpha=.2, orientation='horizontal')
        ax_y.set_ylim(_min, _max)
        plt.setp(ax_x.get_xticklabels() + ax_y.get_yticklabels(), visible=False)
        plt.xticks(rotation=45)
    if figname is not None:
        fig.savefig(figname, transparent=True, bbox_inches='tight')

    return fig, ax

def find_outliers(X1, X2, n=3):
    '''
    Find outliers in the prediction dataset. That is, the 'n' points for which the VGP the
    worst prediction. It returns the dynamic parameters corresponding to that points.
    '''

    diference = np.absolute(X1 - X2)
    max_indexes = np.argsort(diference, 0)[-1*n:]
    return list(max_indexes.ravel())

def plot_matrix(X, ot_distribution=None, ot_kernel=None,
                labels=None, res=1000, grid=False, cleanaxes=False,
                correlation=None, outliers=None, outliers_mark=None,
                outliers_marksize=None, figsize=None, fontsize=None,
                interspace=None, cmap=None):
    """
    Create a pair plot a given sample.

    Parameters
    ----------
    X: array_like
        The sample to plot with shape (n_samples, n_features).
    ot_distribution: :class:`openturns.Distribution`
        OpenTURNS Distribution of dimension n_features, optional.
        The underlying multivariate distribution if known.
        Default is set to None.
    ot_kernel: list of :class:`openturns.Distribution`
        A list of n_features OpenTURNS KernelSmoothing's ready for
        build, optional.
        Kernel smoothing for the margins.
        Default is set to None.
    labels: A list of n_features strings, optional.
        Variates' names for labelling X & Y axes.
        Default is set to None.
    res: int, optional.
        Number of points used for plotting the marginal PDFs.
        Default is set to 1000.
    grid: bool, optional.
        Whether a grid should be added or not.
        Default is set to False (no grid).
    correlation: string
        - None
        - Pearson: linear corrlation
        - Spearman: rank correlation
        - Kendall: pairwise correlation (Kendall tau)
    outliers: array_like
        Points to be highlited on the plot
    outliers_mark: string.
        The type of marker used to highlight the outliers

    Returns
    -------
    ax: matplotlib.Axes instance.
        A handle to the matplotlib figure.
    
    Notes
    -----
    Return a handle to a matplotlib figure containing a 'matrix plot'
    representation of the sample in X. It plots:

    - the marginal distributions on the diagonal terms,
    - the dependograms on the lower terms,
    - scatter plots on the upper terms.
    
    One may also add representation of the original distribution provided it
    is known, and/or a kernel smoothing (based on OpenTURNS).
    """
    if outliers_mark is None:
        outliers_mark = '.'

    if outliers_marksize is None:
        outliers_marksize = 20

    if figsize is None:
        figsize = (8, 8)

    if fontsize is None:
        fontsize = 12

    X = np.array(X)
    n_samples, n_features = X.shape
    if outliers is not None:
        outliers = np.array(outliers)
        n_outliers = outliers.shape[0]


    if ot_distribution is None:
        ranks = np.array(ot.Sample(X).rank())
        if outliers is not None:
            ranks_outliers = np.array(ot.Sample(outliers).rank())
    else:
        ranks = np.zeros_like(X)
        if outliers is not None:
            ranks_outliers = np.zeros_like(outliers)
        for i in range(n_features):
            ranks[:, i] = np.ravel(ot_distribution.getMarginal(i).computeCDF(
                            np.atleast_2d(X[:, i]).T))
            ranks[:, i] *= n_samples
            if outliers is not None:
                ranks_outliers[:, i] = np.ravel(ot_distribution.getMarginal(i).computeCDF(
                                np.atleast_2d(outliers[:, i]).T))
                ranks_outliers[:, i] *= n_outliers

    if correlation == 'Pearson':
        Rmatrix = ot.Sample(X).computePearsonCorrelation()
    elif correlation == 'Spearman':
        Smatrix = ot.Sample(X).computeSpearmanCorrelation()
    elif correlation == 'Kendall':
        Tmatrix = ot.Sample(X).computeKendallTau()
    fig = plt.figure(figsize=figsize)

    n = 0
    for i in range(n_features):
        for j in range(n_features):
            n += 1
            ax = fig.add_subplot(n_features, n_features, n)

            if i == j:
                n_bins = int(1 + np.log2(n_samples)) + 1
                ax.hist(X[:, j], bins=n_bins, density=True,
                        cumulative=False, bottom=None,
                        edgecolor='grey', color=lightblue, alpha=.5)
                if ot_distribution is not None:
                    Xi = ot_distribution.getMarginal(i)
                    a = Xi.getRange().getLowerBound()[0]
                    b = Xi.getRange().getUpperBound()[0]
                    middle = (a + b) / 2.
                    width = b - a
                    if Xi.computePDF(a - .1 * width / 2.) == 0.:
                        a = middle - 1.1 * width / 2.
                    if Xi.computePDF(b + .1 * width / 2.) == 0.:
                        b = middle + 1.1 * width / 2.
                    support = np.linspace(a, b, res)
                    pdf = Xi.computePDF(np.atleast_2d(support).T)
                    ax.plot(support, pdf, color='b', alpha=.5, lw=1.5)
                if ot_kernel is not None:
                    Xi = ot_kernel[i].build(np.atleast_2d(X[:, i]).T)
                    if ot_distribution is None:
                        a = Xi.getRange().getLowerBound()[0]
                        b = Xi.getRange().getUpperBound()[0]
                        support = np.linspace(a, b, res)
                    pdf = Xi.computePDF(np.atleast_2d(support).T)
                    ax.plot(support, pdf, color='r', alpha=.5, lw=1.5)

                if outliers is not None and ot_distribution is not None:
                    ax.scatter(outliers[:, i], Xi.computePDF(np.atleast_2d(outliers[:, i]).T),
                               marker='o', color=darkblue, s=15)


            elif i < j:
                ax.scatter(X[:, j], X[:, i],
                           marker='o', color=lightblue, alpha=0.75,
                           s=6. / n_features)

                if outliers is not None:
                    if cmap is None:
                        ax.scatter(outliers[:, j], outliers[:, i],
                                   marker=outliers_mark, s=outliers_marksize,
                                   color=darkblue)
                    else:
                        ax.scatter(outliers[:, j], outliers[:, i],
                                   marker=outliers_mark, s=outliers_marksize,
                                   c=range(len(outliers[:, j])), cmap=cmap)
            else:
                ax.scatter(ranks[:, j].astype(float) / n_samples,
                           ranks[:, i].astype(float) / n_samples,
                           marker='o', color=lightblue, alpha=.75,
                           s=6. / n_features)

                if outliers is not None:
                    if cmap is None:
                        ax.scatter(ranks_outliers[:, j].astype(float) / n_outliers,
                                   ranks_outliers[:, i].astype(float) / n_outliers,
                                   marker=outliers_mark, s=outliers_marksize,
                                   color=darkblue)
                    else:
                        ax.scatter(ranks_outliers[:, j].astype(float) / n_outliers,
                                   ranks_outliers[:, i].astype(float) / n_outliers,
                                   marker=outliers_mark, s=outliers_marksize,
                                   c=range(len(outliers[:, j])), cmap=cmap)

                # =======================================================
                # ===================== correlation =====================
                # =======================================================

                if correlation == 'Pearson':
                    ax.text(0.5, 0.5,
                            '$\\rho = %.2f$' % Rmatrix[i, j],
                            fontsize=12 - .25 * n_features, color='red',
                            backgroundcolor='w',
                            verticalalignment='center',
                            horizontalalignment='center')
                elif correlation == 'Spearman':
                    ax.text(0.5, 0.5,
                            '$\\rho_S = %.2f$' % Smatrix[i, j],
                            fontsize=12 - .25 * n_features, color='red',
                            backgroundcolor='w',
                            verticalalignment='center',
                            horizontalalignment='center')
                elif correlation == 'Kendall':
                    ax.text(0.5, 0.5,
                            '$\\tau = %.2f$' % Tmatrix[i, j],
                            fontsize=12 - .25 * n_features, color='red',
                            backgroundcolor='w',
                            verticalalignment='center',
                            horizontalalignment='center')

            # ==========================================================
            # ===================== x and y limits =====================
            # ==========================================================

            if i > j:
                ax.set_xlim([0., 1.])
            else:
                support_x = max(X[:, j]) - min(X[:, j])
                xmin = min(X[:, j]) - 0.05 * support_x
                xmax = max(X[:, j]) + 0.05 * support_x
                ax.set_xlim(xmin, xmax)

            if i == j:
                ax.set_ylim([0., ax.get_ylim()[1]])
            elif i < j:
                support_y = max(X[:, i]) - min(X[:, i])
                ax.set_ylim(min(X[:, i]) - 0.05 * support_y,
                            max(X[:, i]) + 0.05 * support_y)
            else:
                ax.set_ylim([0., 1.])
            # =========================================================
            # ====================== Tick labels ======================
            # =========================================================

            if cleanaxes:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # XTICKS
                if i == 0:
                    ax.xaxis.tick_top()
                    xticks = [ax.get_xlim()[0], np.mean(ax.get_xlim()), ax.get_xlim()[1]]
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticks, fontsize=fontsize, rotation=70)
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))
                elif i == n_features - 1 and j != n_features - 1:
                    ax.set_xticks([0., 1.])
                    ax.set_xticklabels([0, 1], fontsize=fontsize)
                else:
                    ax.set_xticks([])

                # YTICKS
                if j == 0 and i != 0:
                    ax.set_yticks([0., 1.])
                    ax.set_yticklabels([0, 1], fontsize=fontsize)
                elif j ==  n_features - 1 and i != n_features - 1:
                    ax.yaxis.tick_right()
                    yticks = [ax.get_ylim()[0], np.mean(ax.get_ylim()), ax.get_ylim()[1]]
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(yticks, fontsize=fontsize)
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))
                else:
                    ax.set_yticks([])

            # ====================================================
            # ====================== Labels ======================
            # ====================================================

            if labels is not None:
                if j == 0:
                    ax.set_ylabel(labels[i])
                if i == n_features - 1:
                    ax.set_xlabel(labels[j])

            ax.grid(b=grid, which='both', axis='both')

    if interspace is not None:
        plt.subplots_adjust(wspace=interspace, hspace=interspace)
    return fig

