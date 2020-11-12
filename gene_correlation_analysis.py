import numpy as np
import pandas as pd
import scanpy as sc
import os
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
from scipy import stats
import time


def geneset_correlation(*, geneset=None, n_genes=None, method=None):
    """
    Runs geneset correlation analysis on defined set of gene signatures.
    Only works with the "unique_core" genesets, or the down_regulated geneset

    :param geneset: {unique_core, down_regulated, all_genes}.
    Unique_core are M1 or M2 core genes induced over control, unique to that stimulation.
    Down_regulated are genes that are down_regulated by costimulation, that were originally unique_core genes
    All_genes are all the original genes in the dataset.

    n_genes: {int, or 'all'}
    If all, it will use all the genes in that specific dataset. When selecting all_genes as the dataset, n_genes must
    be 'all' as well. This option is not working at the moment.

    method: {'spearman', 'kendall', or 'pearson'} don't use pearson as this is not linear

    :return:
    """

    # Set settings
    sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
    sc.settings.set_figure_params(scanpy=True, dpi=80, dpi_save=600, color_map='viridis', vector_friendly=False)
    sc.settings.autoshow = False
    # Set matplotlib to plot axis below to avoid grid on top of plots
    plt.rc('axes', axisbelow=True)

    fig_dir_o = 'figures/'
    sc.settings.figdir = fig_dir_o
    # Data set directory
    data_dir = fig_dir_o + 'gene_programs/'

    results_path = 'write/'
    results_file = 'bmdm_object.h5ad'

    ####################################################################################################################
    ####################################################################################################################
    # Read results file
    adata = sc.read(results_path + results_file)
    # Rename samples
    newnames = ['Control', r'LPS+IFN$\gamma$', 'IL-4', 'LPS+IFN$\gamma$\n+IL-4']
    adata.obs['sample2'] = adata.obs['sample'].copy()
    adata.rename_categories('sample2', newnames)

    namedict = {'M0': 'Control', 'M1': r'LPS+IFN$\gamma$', 'M2': 'IL-4', 'M1+M2': 'LPS+IFN$\gamma$\n+IL-4'}
    ####################################################################################################################
    ####################################################################################################################
    # Set n_genes if empty
    if not n_genes and geneset != 'all_genes':
        n_genes = 50

    if not method:
        method = 'spearman'

    # Read genesets based on parameter
    if geneset == 'unique_core':
        fig_dir = 'figures/Figure1/'
        geneset_file = data_dir + 'unique_programs.pickle'
        print("***\nRunning {} correlation analysis with {} genes\n***".format(method, n_genes))
    else:
        raise ValueError("Please pass unique_core as geneset")

    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    sc.settings.figdir = fig_dir

    if geneset == 'all_genes':
        numgenes = adata.raw.var.shape[0]
        all_unique_genes = []
    else:
        with open(geneset_file, 'rb') as f:
            unique_genes = pickle.load(f)

        #Print number of M1 and M2 genes
        print("Number of M1 genes: {}\n"
              "Number of M2 genes: {}".format(unique_genes['M1'].shape[0], unique_genes['M2'].shape[0]))

        # Make a unique set sorted by FC
        unique_genes_FC = {}
        for key in unique_genes.keys():
            unique_genes_FC[key] = unique_genes[key].sort_values('logfoldchanges', ascending=False)

        # Heatmap of M1 and M2 genes
        if n_genes == 'all':
            all_unique_genes = np.concatenate((unique_genes['M1']['names'],
                                               unique_genes['M2']['names']))
            n_genes_M1 = unique_genes['M1']['names'].shape[0]
            n_genes_M2 = unique_genes['M2']['names'].shape[0]
        else:
            all_unique_genes = np.concatenate((unique_genes['M1']['names'][0:n_genes],
                                           unique_genes['M2']['names'][0:n_genes]))

    print("*** Starting correlation analysis ***")
    M1 = adata[adata.obs['sample'] == "M1"]
    M2 = adata[adata.obs['sample'] == "M2"]
    M12 = adata[adata.obs['sample'] == "M1+M2"]
    # color_dict = {'M0': 'blue', 'M1': 'orange', 'M2': 'green', 'M1+M2': 'red'}
    # row_colors = adata.obs['sample'].map(color_dict)

    if n_genes == 'all':
        corr_colors = np.empty(n_genes_M1 + n_genes_M2, dtype=np.dtype(('U', 10)))
        corr_colors[0:n_genes_M1] = 'orange'  # M1 genes
        corr_colors[n_genes_M1:] = 'blue'  # M2 genes
    else:
        corr_colors = np.empty(n_genes*2, dtype=np.dtype(('U', 10)))
        corr_colors[0:n_genes] = 'orange' #M1 genes
        corr_colors[n_genes:] = 'blue' #M2 genes

    adatas = {"M1": M1,
              "M2": M2,
              "M1+M2": M12}

    stims = ["M1", "M2", "M1+M2"]
    fig, axt = plt.subplots(1, 3, figsize=(22, 8))
    corr_matrices = {}
    corr_pvals = {}
    for stim, ax in zip(stims, axt):
        df = pd.DataFrame(adatas[stim].raw[:,all_unique_genes].X.toarray(), columns=all_unique_genes,
                               index=adatas[stim].obs_names)

        if method == 'spearman':
            tic = time.time()
            coeff, pval = stats.spearmanr(df, axis=0)  # This is by far the fastest version
            corr = pd.DataFrame(coeff, columns=df.columns, index=df.columns)
            corr_p = pd.DataFrame(pval, columns=df.columns, index=df.columns)
            toc = time.time()
            print("Done calculating masked correlation for {}: {:.2f}".format(stim, toc - tic))
            corr.mask(corr_p > 0.05, inplace=True) # filter out correlations with p > 0.05
        else:
            tic = time.time()
            corr = df.corr(method=method)
            toc = time.time()
            print("Done calculating correlation for {}: {:.2f}".format(stim, toc - tic))

        sb.heatmap(corr, cbar=True, cmap='PiYG', vmin=-.5, vmax=.5, ax=ax, square=True, xticklabels=True, yticklabels=True,
                   cbar_kws={"shrink": 0.7}, center=0)
        ax.tick_params(axis='both', which='major', labelsize=4)

        if n_genes == 'all':
            for i, j in zip(ax.get_xticklabels()[0:n_genes_M1], ax.get_yticklabels()[0:n_genes_M1]):
                i.set_color("orange"), j.set_color("orange")
            for i, j in zip(ax.get_xticklabels()[n_genes_M1:], ax.get_yticklabels()[n_genes_M1:]):
                i.set_color("blue"), j.set_color("blue")
        else:
            for i,j in zip(ax.get_xticklabels()[0:n_genes], ax.get_yticklabels()[0:n_genes]):
                i.set_color("orange"), j.set_color("orange")
            for i, j in zip(ax.get_xticklabels()[n_genes:], ax.get_yticklabels()[n_genes:]):
                i.set_color("blue"), j.set_color("blue")

        ax.set_title(namedict[stim])

        corr_matrices[stim] = corr.copy()
        corr_pvals[stim] = corr_p.copy()

    with open(data_dir + "correlation_matrices.pickle", "wb") as f:
        pickle.dump(corr_matrices, f)

    plt.tight_layout()
    plt.savefig(fig_dir + 'Fig1e_All_correlations.png', bbox_inches='tight'), plt.close()

    return M12, corr_matrices, corr_pvals