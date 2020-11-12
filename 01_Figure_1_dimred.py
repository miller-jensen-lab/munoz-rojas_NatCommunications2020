import os
import numpy as np
import scanpy as sc
import single_cell_functions.sc_plots as scplts

""" Dimensionality reduction/UMAP analysis"""

# Parameters
fig_dir_o = 'figures/'
sc.settings.figdir = fig_dir_o

# Set settings
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(scanpy=True, dpi=80, dpi_save=600, color_map='viridis', vector_friendly=False)
sc.settings.autoshow = False

#Grey Yellow Orange Red color map
new_cmap = scplts.utils.sc_cmap

results_path = 'write/'
results_file = 'bmdm_object.h5ad'


########################################################################################################################
# Read files
########################################################################################################################
adata = sc.read(results_path + 'bmdm_object_raw.h5ad')

# Re-assign colors
vega20 = sc.pl.palettes.default_20
new_colors = ['dimgray', vega20[1], vega20[0], vega20[2]]
adata.uns['sample2_colors'] = new_colors

# Re-name categories for proper plotting
newnames = ['Control', 'LPS+IFNg', 'IL-4', 'LPS+IFN-g\n+IL-4']
adata.obs['sample2'] = adata.obs['sample'].copy()
adata.rename_categories('sample2', newnames)

########################################################################################################################
# Figure 1
########################################################################################################################
fig_dir = fig_dir_o + "Figure1/"
sc.settings.figdir = fig_dir

if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

# Store the dataset as raw layer for differential expression analysis and plotting
adata.raw = adata
# ####################### Find Highly Variable Genes ################
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
sc.pl.highly_variable_genes(adata, save='variable_genes.pdf')
print("Number of highly variable genes: {}".format(adata.var['highly_variable'].sum()))

adata = adata[:, adata.var['highly_variable']]
sc.pp.scale(adata)

# Calculate the visualizations
sc.pp.pca(adata, n_comps=50, use_highly_variable=True, svd_solver='arpack')
sc.pl.pca_loadings(adata, save='_bmdms.pdf')
sc.pl.pca_variance_ratio(adata, save='_bmdms.pdf')
sc.pl.pca_scatter(adata, color='sample', save='_bmdms.pdf')
sc.pp.neighbors(adata, n_pcs=20)

sc.tl.umap(adata)

# UMAP of all samples
sc.pl.umap(adata, color='sample2', save="_Fig1c.pdf")

# Zoom into M1+M2 original UMAP
adata2 = adata[adata.obs['sample'] == 'M1+M2'].copy()
sc.pl.umap(adata2, color='sample2', save = "_Fig1d_M1+M2only_origspace.pdf")
genes = ['Nos2', 'Il12b', 'Arg1', 'Mrc1']
for gene in genes:
    sc.pl.umap(adata2, color=gene, cmap=new_cmap, vmin=np.log1p(2), save=f"_Fig1d_{gene}.pdf")

adata.write(results_path + results_file, compression='gzip')