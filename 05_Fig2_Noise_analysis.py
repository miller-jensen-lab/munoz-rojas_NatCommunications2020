import pickle
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
import single_cell_functions.sc_plots as scplts

""" Run noise analysis for Figure 2"""

# Parameters
fig_dir_o = 'figures/'
sc.settings.figdir = fig_dir_o

# Set settings
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(scanpy=True, dpi=80, dpi_save=600, color_map='viridis', vector_friendly=False)
sc.settings.autoshow = False

results_path = 'write/'
results_file = 'bmdm_object.h5ad'

# Set plotting settings
plt.rc('axes', axisbelow=True)


########################################################################################################################
########################################################################################################################

def _get_mean_var(X):
    # - using sklearn.StandardScaler throws an error related to
    #   int to long trafo for very large matrices
    # - using X.multiply is slower
    if True:
        mean = X.mean(axis=0)
        if issparse(X):
            mean_sq = X.multiply(X).mean(axis=0)
            mean = mean.A1
            mean_sq = mean_sq.A1
        else:
            mean_sq = np.multiply(X, X).mean(axis=0)
        # enforce R convention (unbiased estimator) for variance
        var = (mean_sq - mean ** 2) * (X.shape[0] / (X.shape[0] - 1))
    return mean, var


########################################################################################################################
# Read files
########################################################################################################################
adata = sc.read(results_path + results_file)

# Re-assign colors
vega20 = sc.pl.palettes.default_20
new_colors = ['dimgray', vega20[1], vega20[0], vega20[2]]
adata.uns['sample2_colors'] = new_colors

# Re-name categories for proper plotting
newnames = ['Control', 'LPS+IFNg', 'IL-4', 'LPS+IFNg\n+IL-4']
adata.obs['sample2'] = adata.obs['sample'].copy()
adata.rename_categories('sample2', newnames)

genesets_path = 'figures/gene_programs/'
with open(genesets_path + 'down_in_mixed_unique.pickle', 'rb') as f:
    d_down = pickle.load(f)

with open(genesets_path + 'up_in_mixed_unique.pickle', 'rb') as f:
    d_up = pickle.load(f)

########################################################################################################################
########################################################################################################################
fig_dir = fig_dir_o + 'Figure2/'
sc.settings.figdir = fig_dir

# Make M1 and M1+M2 anndatas
adatas = {}
adatas['M1'] = adata[adata.obs['sample'] == "M1"].copy()
adatas['M2'] = adata[adata.obs['sample'] == "M2"].copy()
adatas['M1+M2'] = adata[adata.obs['sample'] == "M1+M2"].copy()

# Get Fano for M1 and M1+M2
# note- - previous version did not use log-space data, here we are keeping the matrix in log space

for stim in adatas:
    means, var = _get_mean_var(adatas[stim].raw.X)
    means[means == 0] = np.nan
    adatas[stim].raw.var['fano'] = var / means

# Get change in noise
adatas['M1+M2'].raw.var['fano_change_M1'] = adatas['M1+M2'].raw.var['fano'] - adatas['M1'].raw.var['fano']
adatas['M1+M2'].raw.var['fano_change_M2'] = adatas['M1+M2'].raw.var['fano'] - adatas['M2'].raw.var['fano']

# #####
# Plots
stim_names = {'M1': 'LPS+IFN-g',
              'M2': 'IL-4',
              'M1+M2': 'LPS+IFN-g+IL-4'}
logplot = False
for ss in ['M1', 'M2']:
    print("*** Plotting noise analysis figures for {} ***".format(ss))

    # Plot fano factor in single stim vs fano factor in M1+M2 to visualize large changes
    # Color by down-regulated in M1+M2
    c = np.full(np.shape(adatas['M1+M2'].raw)[1], 'lightgrey')
    z = np.full(np.shape(adatas['M1+M2'].raw)[1], 0)
    up = np.in1d(adatas['M1+M2'].raw.var_names, d_up[ss]['names'])
    down = np.in1d(adatas['M1+M2'].raw.var_names, d_down[ss]['names'])
    c[up] = 'red'
    c[down] = 'blue'
    z[up] = 1
    z[down] = 2
    # Sort by color
    z_ind = np.argsort(z)
    ax = scplts.scatter_identity(adatas[ss].raw.var['fano'][z_ind], adatas['M1+M2'].raw.var['fano'][z_ind], log=logplot,
                                 s=3, c=c[z_ind], alpha=1, edgecolors='None')
    plt.xlabel('Fano factor in {}'.format(stim_names[ss]))
    plt.ylabel('Fano factor in Co-stim')
    marker_red = mlines.Line2D([], [], color='red', marker='o', linestyle='', markersize=3, label='up-regulated')
    marker_blue = mlines.Line2D([], [], color='blue', marker='o', linestyle='', markersize=3, label='down-regulated')
    plt.legend(handles=[marker_red, marker_blue], fontsize=10)
    plt.savefig(fig_dir + "Fig2d_Fano_change_vs_{}_colored.pdf".format(ss), bbox_inches='tight'), plt.close()

    # Add fano information to downregulated and upregulated genes
    d_down[ss]['fano'] = adatas['M1+M2'].raw.var.loc[d_down[ss]['names'], 'fano'].values
    d_down[ss]['fano_change_{}'.format(ss)] = adatas['M1+M2'].raw.var.loc[d_down[ss]['names'],
                                                                          'fano_change_{}'.format(ss)].values

    d_up[ss]['fano'] = adatas['M1+M2'].raw.var.loc[d_up[ss]['names'], 'fano'].values
    d_up[ss]['fano_change_{}'.format(ss)] = adatas['M1+M2'].raw.var.loc[d_up[ss]['names'],
                                                                        'fano_change_{}'.format(ss)].values

    d_all = pd.concat([d_down[ss], d_up[ss]], axis=0)

    print("*** Plotting violin plots for noise analysis for {} ***".format(ss))
    # Change save settings to improve output figures
    sc.settings.set_figure_params(scanpy=True, dpi=80, dpi_save=600, color_map='viridis', vector_friendly=False)


    # Sort by highest change in fano factor in down-regulated genes
    d_down[ss].sort_values('fano_change_{}'.format(ss), ascending=False, inplace=True)

    # Plot genes with top 15 fano factor change
    axes = sc.pl.stacked_violin(adata, d_down[ss]['names'][0:15].values, groupby='sample2', use_raw=True,
                                dendrogram=False,
                                stripplot=True, jitter=True, size=0.2)
    axes = scplts.utils.equal_stacked_axis(axes)
    plt.savefig(fig_dir + 'FigS2c_stacked_violin_{}_down_highfanochange.png'.format(ss), bbox_inches='tight')
    plt.close()

    # Plot fano change in top 15
    d_down[ss][0:15][::-1].plot.barh(y='fano_change_{}'.format(ss), x='names', legend=None, fontsize=12)
    plt.xlabel('Change in Noise (Costim - {})'.format(stim_names[ss]), size=12)
    plt.ylabel(None)
    plt.grid(False)
    # Make square
    ax = plt.gca()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1 - x0) / (y1 - y0))
    # plt.tight_layout()
    plt.savefig(fig_dir + 'Fig2e_{}_High_fano_change.pdf'.format(ss), bbox_inches='tight'), plt.close()
