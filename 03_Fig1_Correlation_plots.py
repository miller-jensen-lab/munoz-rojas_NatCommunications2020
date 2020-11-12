import gene_correlation_analysis as gcorr
import single_cell_functions.sc_plots as scplts
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd

"""
This scripts runs the correlation analysis for Figure 1 and Figure S1
"""

# Parameters
filtered = True
fig_dir_o = 'figures/'
sc.settings.figdir = fig_dir_o
# Data set directory
data_dir = fig_dir_o + 'gene_programs/'
fig_dir = fig_dir_o + 'Figure1/'

# Set settings
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(scanpy=True, dpi=80, dpi_save=600, color_map='viridis', vector_friendly=False)
sc.settings.autoshow = False

results_path = 'write/'
results_file = 'bmdm_object.h5ad'

########################################################################################################################
# Figure 1 - correlation analysis
########################################################################################################################
_, corr_matrices, corr_pvals = gcorr.geneset_correlation(geneset='unique_core', n_genes=50, method='spearman')

########################################################################################################################
# Figure S1d - Genes with max and min correlation
########################################################################################################################
adata = sc.read(results_path + results_file)
# Rename samples
newnames = ['Control', 'LPS+IFNg', 'IL-4', 'LPS+IFN-g\n+IL-4']
adata.obs['sample2'] = adata.obs['sample'].copy()
adata.rename_categories('sample2', newnames)

# Density plots to show positive and negative correlation - these are the top and bottom correlations in M1+M2 cells
x_vars = ['Ifit2', 'Cflar']
y_vars = ['Rsad2', 'Dok2']
for x, y in zip(x_vars, y_vars):
    scplts.density_plot_sc(adata, x, y, groupby='sample2')
    plt.savefig(fig_dir + "FigS1d_{}vs{}.pdf".format(x, y), bbox_inches='tight')
    plt.close()

# Get actual correlation values
stims = corr_matrices.keys()
correlations = pd.DataFrame(index=['LowC', 'HighC'], columns=stims)

for stim in stims:
    correlations.loc['LowC', stim] = corr_matrices[stim].loc['Cflar', 'Dok2']
    correlations.loc['HighC', stim] = corr_matrices[stim].loc['Ifit2', 'Rsad2']

print(correlations)
