import numpy as np
import pandas as pd
import scipy
import sys
import scanpy as sc
import os
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.stats as stats

def volcano_plot(x, y, *, FC_thresh=None, pval_thresh=None):
    x = x.copy()
    y = y.copy()

    if FC_thresh:
        above_FC = x >= FC_thresh
        below_FC = x <= -FC_thresh
        true_FC = np.any((above_FC, below_FC), axis=0)

    if pval_thresh:
        true_pval = y <= pval_thresh

    if np.any(y==0):
        y.loc[y==0] = 1e-300


    if FC_thresh and pval_thresh:
        above_thresh = np.all((true_FC, true_pval), axis=0)
        upregulated = np.all((above_FC, true_pval), axis=0)
        downregulated = np.all((below_FC, true_pval), axis=0)

        c = np.full(np.shape(x)[0], 'grey')
        c[upregulated] = 'red'
        c[downregulated] = 'blue'

        plt.figure()
        plt.scatter(x, -np.log10(y), c=c, s=1)
        plt.axvline(FC_thresh)
        plt.axvline(-FC_thresh)
        plt.axhline(-np.log10(pval_thresh))
        ax = plt.gca()
        up = np.sum(upregulated)
        down = np.sum(downregulated)
        plt.text(0.9, 0.9, up, color='red',
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.text(0.1, 0.9, down, color='blue',
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)



    elif FC_thresh:
        above_thresh = true_FC
        c = np.full(np.shape(x)[0], 'grey')
        c[above_thresh] = 'red'

        plt.figure()
        plt.scatter(x, -np.log10(y), c=c)
        plt.axvline(FC_thresh)
        plt.axvline(-FC_thresh)

    elif pval_thresh:
        above_thresh = pval_thresh
        c = np.full(np.shape(x)[0], 'grey')
        c[above_thresh] = 'red'

        plt.figure()
        plt.scatter(x, -np.log10(y), c=c)
        plt.axhline(-np.log10(pval_thresh))

    else:
        c = np.full(np.shape(x)[0], 'black')
        plt.figure()
        plt.scatter(x, -np.log10(y), c=c)

    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 Adjusted P-value')
    return ax


def density_scatter_sc(adata, x_var, y_var, *, x_thresh=None, y_thresh=None, groupby=None, use_raw=True):
    """ Function to plot scatter plots and color-code by density
    Parameters
    ----------
    adata: anndata object with annotations
    x : data to plot on x-axis (string)
    y : data to plot on y-axis (string)
    x_thresh: (optional) threshold for positive cells on x-axis
    y_thresh: (optional) threshold for positive cells on y-axis

    """
    # TODO: fix this function
    ##Density-colored scatter
    # Define groups
    if groupby is None:
        groupby = 'sample'
        namedict = {'M0': 'Control', 'M1': r'LPS+IFN$\gamma$', 'M2': 'IL-4', 'M1+M2': r'LPS+IFN$\gamma$+IL-4'}

    groups = adata.obs[groupby].unique()
    n_groups = groups.shape[0]
    fig, axt = plt.subplots(1, n_groups, sharex=True, sharey=True, figsize=(3.5 * n_groups, 3.5))
    if groups.shape[0] > 1:
        plt.setp(axt.flat, aspect='equal', adjustable='box')

    # Set up collecting bins
    b_right = []
    top_left = []
    top_right = []

    for group, ax in zip(groups, fig.axes):
        adata_temp = adata[adata.obs[groupby] == group].copy()
        if use_raw:
            x = adata_temp.raw[:, x_var].X
            y = adata_temp.raw[:, y_var].X
        else:
            x = adata_temp[:, x_var].X
            y = adata_temp[:, y_var].X

        # Define gaussian kde
        xy = np.vstack([x, y])
        z = stats.gaussian_kde(xy, bw_method=.2)(xy)

        ax.scatter(x, y, c=z, s=10, label=group, cmap='viridis')

        if groupby is None:
            ax.set_title(namedict[group])
        else:
            ax.set_title(group)

        ax.set(xlabel=x_var, ylabel=y_var)
        ax.label_outer()
        plt.tight_layout()

        if x_thresh is not None:
            if y_thresh is None:
                raise ValueError('Pass both thresholds!')
            else:
                xpos = np.array(x > x_thresh)
                xneg = np.array(x <= x_thresh)
                ypos = np.array(y > y_thresh)
                yneg = np.array(y <= y_thresh)
                bleft = (xneg & yneg).mean() * 100
                bright = (xpos & yneg).mean() * 100
                tright = (xpos & ypos).mean() * 100
                tleft = (xneg & ypos).mean() * 100

                b_right.append(bright)
                top_left.append(tleft)
                top_right.append(tright)

                ax.axhline(y=y_thresh, color='black', linewidth=1, linestyle='--')
                ax.axvline(x=x_thresh, color='black', linewidth=1, linestyle='--')
                ax.text(0.05, 0.05, round(bleft, 2), transform=ax.transAxes, fontsize=10,
                        verticalalignment='bottom', horizontalalignment='left')
                ax.text(0.95, 0.05, round(bright, 2), transform=ax.transAxes, fontsize=10,
                        verticalalignment='bottom', horizontalalignment='right')
                ax.text(0.95, 0.95, round(tright, 2), transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='right')
                ax.text(0.05, 0.95, round(tleft, 2), transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='left')

    return fig
    # # Save Figure
    # fname, fext = os.path.splitext(fileName)
    # dirname = '{}-sqfigures-t-{}/'.format(fname, round(trsh, 1))
    # if not os.path.exists(dirname):
    #     os.makedirs(dirname)
    #
    # savename = '{}{} vs {} scatter plot.pdf'.format(dirname, varx, vary)
    # plt.savefig(savename, bbox_inches='tight')
    # print('----Figure saved in: {}'.format(savename))


def density_plot_sc(data, x_var, y_var, *, x_thresh=None, y_thresh=None, groupby=None, use_raw=True):
    """ Function to plot scatter plots and overlay density
    Parameters
    ----------
    data: anndata object with annotations
    x : data to plot on x-axis (string)
    y : data to plot on y-axis (string)
    x_thresh: (optional) threshold for positive cells on x-axis
    y_thresh: (optional) threshold for positive cells on y-axis

    """

    ##Density-colored
    # Define groups
    if groupby is None:
        raise ValueError("Please specify what to groupby")
        # groupby='sample'
        # namedict = {'M0': 'Control', 'M1': r'LPS+IFN$\gamma$', 'M2': 'IL-4', 'M1+M2': r'LPS+IFN$\gamma$+IL-4'}
    data = data.copy()
    groups = data.obs[groupby].unique()
    n_groups = groups.shape[0]
    fig, axt = plt.subplots(1, n_groups, sharex=True, sharey=True, figsize=(4*n_groups, 4))
    if groups.shape[0] > 1:
        plt.setp(axt.flat, aspect='equal', adjustable='box')


    #Set up collecting bins
    b_right = []
    top_left = []
    top_right = []

    for group, ax in zip(groups, fig.axes):
        adata_temp = data[data.obs[groupby] == group].copy()
        if use_raw:
            x = adata_temp.raw.obs_vector(x_var)
            y = adata_temp.raw.obs_vector(y_var)
        else:
            x = adata_temp.obs_vector(x_var)
            y = adata_temp.obs_vector(y_var)

        # Define gaussian kde
        # xy = np.vstack([x, y])
        # z = stats.gaussian_kde(xy)(xy)
        ax.scatter(x, y, s=3, c='black', alpha=1, edgecolors="None")
        sb.kdeplot(x, y, cmap='viridis', shade=True, bw=.15, n_levels=120, ax=ax, shade_lowest=False)
        # sb.kdeplot(x, y, cmap='viridis', shade=True, bw_adjust=.15, n_levels=120, ax=ax, shade_lowest=False) # new version

        for c in ax.collections: #This is done to prevent aliasing problems that make pdf transparent
            c.set_edgecolor("face")
        # ax.set_axisbelow(True)

        ax.grid(False)
        ax.set_title(group)
        # if groupby == 'sample':
        #     ax.set_title(namedict[group])
        # else:
        #     ax.set_title(group)

        ax.set(xlabel=x_var, ylabel=y_var)
        ax.tick_params(axis='both', labelsize=11)
        ax.label_outer()
        plt.tight_layout()

        if x_thresh is not None:
            if y_thresh is None:
                raise ValueError('Pass both thresholds!')
            else:
                xpos = np.array(x > x_thresh)
                xneg = np.array(x <= x_thresh)
                ypos = np.array(y > y_thresh)
                yneg = np.array(y <= y_thresh)
                bleft = (xneg & yneg).mean() * 100
                bright = (xpos & yneg).mean() * 100
                tright = (xpos & ypos).mean() * 100
                tleft = (xneg & ypos).mean() * 100

                b_right.append(bright)
                top_left.append(tleft)
                top_right.append(tright)

                ax.axhline(y=y_thresh, color='black', linewidth=0.8, alpha=0.5, linestyle='--')
                ax.axvline(x=x_thresh, color='black', linewidth=0.8, alpha=0.5, linestyle='--')
                # Add text
                text_color = 'darkblue'
                # ax.text(0.05, 0.05, round(bleft, 2), transform=ax.transAxes, fontsize=10,
                #         verticalalignment='bottom', horizontalalignment='left')
                # ax.text(0.95, 0.05, round(bright, 2), transform=ax.transAxes, fontsize=10,
                #         verticalalignment='bottom', horizontalalignment='right')
                # ax.text(0.95, 0.95, round(tright, 2), transform=ax.transAxes, fontsize=10,
                #         verticalalignment='top', horizontalalignment='right')
                # ax.text(0.05, 0.95, round(tleft, 2), transform=ax.transAxes, fontsize=10,
                #         verticalalignment='top', horizontalalignment='left')
                #

                ax.text(-0.01, 0.005, round(bleft, 1), transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', horizontalalignment='right', color=text_color)
                ax.text(1.01, 0.005, round(bright, 1), transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', horizontalalignment='left', color=text_color)
                ax.text(1.01, 1.005, round(tright, 1), transform=ax.transAxes, fontsize=12,
                        verticalalignment='bottom', horizontalalignment='left', color=text_color)
                ax.text(-0.01, 1.005, round(tleft, 1), transform=ax.transAxes, fontsize=12,
                        verticalalignment='bottom', horizontalalignment='right', color=text_color)
    return fig

    # # Save Figure
    # fname, fext = os.path.splitext(fileName)
    # dirname = '{}-sqfigures-t-{}/'.format(fname, round(trsh, 1))
    # if not os.path.exists(dirname):
    #     os.makedirs(dirname)
    #
    # savename = '{}{} vs {} scatter plot.pdf'.format(dirname, varx, vary)
    # plt.savefig(savename, bbox_inches='tight')
    # print('----Figure saved in: {}'.format(savename))


def scatter_identity(x,
                     y,
                     *,
                     log=False,
                     **kwargs):


    """ Scatter plot function that also draws and identity line
    Parameters
    ----------
    x : data to plot on x-axis (array-like)
    y : data to plot on y-axis (array-like)
    log : choose whether to plot on loglog scape or not
    **kwargs: arguments to pass to plt.scatter
    """
    plt.scatter(x,y,**kwargs)
    ax = plt.gca()
    if log:
        plt.loglog()
    # Make identity plot
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # Change limits to make plot equal
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.plot(lims, lims, 'k--', alpha=0.6, zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', 'box')
    # Change x and y ticks to least amount of numbers
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    if xticks.shape[0] < yticks.shape[0]:
        ax.set_yticks(xticks)
    else:
        ax.set_xticks(yticks)

    # Re-set axis after setting ticks
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', 'box')

    # Change plotting to make graph prettier
    ax.set_axisbelow(True)
    ax.grid(False)
    plt.tight_layout()
    return ax


