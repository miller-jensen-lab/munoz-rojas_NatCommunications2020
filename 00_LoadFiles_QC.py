import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy
import seaborn as sb
from rpy2.rinterface import RRuntimeWarning
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

""" Loads sample, performs QC and normalization and concatenates into one anndata object
"""

# Set settings
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(scanpy=True, dpi=80, dpi_save=600, color_map='viridis', vector_friendly=False)
sc.settings.autoshow = False #set to True if running interactively

fig_dir = 'figures/'
sc.settings.figdir = fig_dir

if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

# Print versions
sc.logging.print_version_and_date()
sc.logging.print_versions()

results_path = 'write/'
results_file = 'bmdm_object_raw.h5ad'

if not os.path.isdir(results_path):
    os.makedirs(results_path)

# Automatically convert rpy2 outputs to pandas dataframes
pandas2ri.activate()
warnings.filterwarnings("ignore", category=RRuntimeWarning)

########################################################################################################################
########################################################################################################################
# ###### Quality control and merging of samples ######
# Load files and merge them together
# ### Note: Here is where you specify the path to the folder containing the 10x matrices (output from cellranger).
# These are located in the GEO submission accompanying the paper: GSE161125
paths = ['../data/M0-H/filtered_gene_bc_matrices/mm10/',
         '../data/M1-H/filtered_gene_bc_matrices/mm10/',
         '../data/M2-H/filtered_gene_bc_matrices/mm10/',
         '../data/M1_M2-H/filtered_gene_bc_matrices/mm10/']
samples = ['M0', 'M1', 'M2', 'M1+M2']

adatas = []
###################
# Set filtering parameters based on previous plots
filt_param = {key: '' for key in samples}
filt_param['M0'] = {'min_counts': 10000,
                    'max_counts': 60000,
                    'min_genes': 2600,
                    'max_genes': 6200}

filt_param['M1'] = {'min_counts': 10000,
                    'max_counts': 60000,
                    'min_genes': 2000,
                    'max_genes': 5800}

filt_param['M2'] = {'min_counts': 10000,
                    'max_counts': 60000,
                    'min_genes': 2500,
                    'max_genes': 6100}

filt_param['M1+M2'] = {'min_counts': 10000,
                       'max_counts': 70000,
                       'min_genes': 2500,
                       'max_genes': 6200}

for path1, sample in zip(paths, samples):
    fig_dir = f'figures/QC/{sample}/'
    sc.settings.figdir = fig_dir
    adata = sc.read_10x_mtx(path1, var_names='gene_symbols', cache=True)
    mito_genes = adata.var_names.str.startswith('mt-')
    # for each cell compute fraction of counts in mito genes vs. all genes
    # the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
    adata.obs['percent_mito'] = adata[:, mito_genes].X.sum(axis=1).A1 / adata.X.sum(axis=1).A1
    # add the total counts per cell as observations-annotation to adata
    adata.obs['n_counts'] = adata.X.sum(axis=1).A1
    adata.obs['n_genes'] = np.sum(adata.X > 0, axis=1).A1

    # QC plots
    sc.pl.highest_expr_genes(adata, n_top=20, save=True)
    sc.pl.violin(adata, ['n_genes', 'n_counts', 'percent_mito'], multi_panel=True, save='_QC_basic.pdf')
    sc.pl.violin(adata, 'n_counts', log=True, cut=0, save='_QC_counts.pdf')
    sc.pl.scatter(adata, 'n_counts', 'n_genes', color='percent_mito', save='_QC_countsvgenes.pdf')
    sc.pl.violin(adata, 'percent_mito')
    plt.axhline(0.055, color='orange')
    plt.savefig(fig_dir + 'violin_percentmito.pdf'), plt.close()

    plt.figure()
    sb.distplot(adata.obs['n_counts'], kde=False, bins=60)
    plt.axvline(filt_param[sample]['min_counts'])
    plt.axvline(filt_param[sample]['max_counts'])
    plt.savefig(fig_dir + 'distribution_n_counts.pdf'), plt.close()

    plt.figure()
    sb.distplot(adata.obs['n_counts'][adata.obs['n_counts'] < 30000], kde=False, bins=60)
    plt.axvline(filt_param[sample]['min_counts'])
    plt.savefig(fig_dir + 'distribution_zoom_n_counts.pdf'), plt.close()

    plt.figure()
    p3 = sb.distplot(adata.obs['n_genes'], kde=False, bins=60)
    plt.axvline(filt_param[sample]['min_genes'])
    plt.axvline(filt_param[sample]['max_genes'])
    plt.savefig(fig_dir + 'distribution_n_genes.pdf'), plt.close()

    plt.figure()
    sb.distplot(adata.obs['n_genes'][adata.obs['n_genes'] < 4000], kde=False, bins=60)
    plt.axvline(filt_param[sample]['min_genes'])
    plt.savefig(fig_dir + 'distribution_zoom_n_genes.pdf'), plt.close()

    # Filter cells
    print('Filter sample: {}'.format(sample))
    print('Number of cells before filters: {:d}'.format(adata.n_obs))
    sc.pp.filter_cells(adata, min_counts=filt_param[sample]['min_counts'])
    sc.pp.filter_cells(adata, max_counts=filt_param[sample]['max_counts'])
    adata = adata[adata.obs['percent_mito'] < 0.055, :]
    sc.pp.filter_cells(adata, min_genes=filt_param[sample]['min_genes'])
    sc.pp.filter_cells(adata, max_genes=filt_param[sample]['max_genes'])
    print('Number of cells after filters: {:d}'.format(adata.n_obs))
    adata.var_names_make_unique()

    # ####################### Normalization ################
    # Perform a clustering for scran normalization in clusters
    print('Normalizing using scran size factors...')
    adata_pp = adata.copy()
    sc.pp.normalize_per_cell(adata_pp, counts_per_cell_after=1e6)
    sc.pp.log1p(adata_pp)
    sc.pp.pca(adata_pp, n_comps=15)
    sc.pp.neighbors(adata_pp)
    sc.tl.louvain(adata_pp, key_added='groups', resolution=0.5)
    input_groups = adata_pp.obs['groups']
    data_mat = adata.X.T.toarray()

    # Run normalization using R package scran
    scran = importr('scran')
    size_factors = scran.computeSumFactors(data_mat, clusters=input_groups, **{"min.mean": 0.1})

    # Save results and normalize the data
    del adata_pp
    adata.obs['size_factors'] = size_factors
    # Plot size factors against n_counts and number of genes
    sc.pl.scatter(adata, 'size_factors', 'n_counts', save='_QCsizefactorscounts.pdf')
    sc.pl.scatter(adata, 'size_factors', 'n_genes', save='_QCsizefactorsgenes.pdf')
    # Keep copy of raw, non-normalized counts
    adata.layers["counts"] = adata.X.copy()
    adata.X /= adata.obs['size_factors'].values[:, None]
    adata.X = scipy.sparse.csr_matrix(adata.X, dtype='float32')  # make sparse again

    # Replot QC parameters after normalization
    adata.obs['n_counts'] = adata.X.sum(axis=1).A1
    adata.obs['n_genes'] = np.sum(adata.X > 0, axis=1).A1
    sc.pl.scatter(adata, 'n_counts', 'n_genes', color='percent_mito', save='_QC_postnorm_countsvsgenes.pdf')
    sc.pl.scatter(adata, 'size_factors', 'n_counts', save='_QC_postnorm_sizefactorscounts.pdf')
    sc.pl.scatter(adata, 'size_factors', 'n_genes', save='_QC_postnorm_sizefactorsgenes.pdf')

    adatas.append(adata.copy())

del adata
adata = adatas[0].concatenate(adatas[1:], batch_key='sample', batch_categories=samples)
del adatas

# Filter genes after concatenation
print('Number of genes before filter: {:d}'.format(adata.n_vars))
sc.pp.filter_genes(adata, min_cells=10)
print('Number of genes after filter: {:d}'.format(adata.n_vars))

print('Final dataset:')
print(adata.obs['sample'].value_counts())

# Re-set figure directory to plot all samples together
fig_dir = 'figures/QC/'
sc.settings.figdir = fig_dir

# Plot post-normalization plots for all samples
sc.pl.scatter(adata, 'n_counts', 'n_genes', color='sample', save='_QC_postnorm_countsvsgenes.pdf')
sc.pl.scatter(adata, 'size_factors', 'n_counts', color='sample', save='_QC_postnorm_sizefactorscounts.pdf')
sc.pl.scatter(adata, 'size_factors', 'n_genes', color='sample', save='_QC_postnorm_sizefactorsgenes.pdf')

# Log-normalize the data
sc.pp.log1p(adata)

########################################################################################################################
# General metrics of data quality
########################################################################################################################
X = adata.layers['counts']

# Get mean read count per cell
# X1 = np.expm1(X)
print(X.sum(axis=1).A1.mean())

# Get mean number of detected genes
X2 = X > 0
print(X2.sum(axis=1).A1.mean())

# Write results to file
adata.write(results_path + results_file, compression='gzip')
