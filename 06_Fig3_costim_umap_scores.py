import scanpy as sc
import pickle
import matplotlib.pyplot as plt
import os

""" UMAP analysis and UCG scoring on co-stimulated cells only"""

# Parameters
fig_dir_o = 'figures/'
sc.settings.figdir = fig_dir_o

# Set settings
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(scanpy=True, dpi=80, dpi_save=600, color_map='viridis', vector_friendly=False)
sc.settings.autoshow = False

results_path = 'write/'
results_file2 = 'bmdm_object_raw.h5ad'
data_dir = fig_dir_o + 'gene_programs/'

# Plotting params
plt.rcParams['axes.grid'] = False

########################################################################################################################
# Read files
########################################################################################################################
adata = sc.read(results_path + results_file2) #read non-scaled data

########################################################################################################################
# Do dim_red on M1+M2 only
########################################################################################################################
fig_dir = fig_dir_o + 'Figure3/'

if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

sc.settings.figdir = fig_dir

adata = adata[adata.obs['sample'] == 'M1+M2'].copy()
# ####################### Find Highly Variable Genes ################
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
sc.pl.highly_variable_genes(adata, save='variable_genes.pdf')
print("Number of highly variable genes: {}".format(adata.var['highly_variable'].sum()))


# ####################### Dimensionality Reduction and Visualization ################
# #Store the dataset as raw layer for differential expression analysis and plotting
adata.raw = adata

# Scale the data
adata = adata[:, adata.var['highly_variable']]
sc.pp.scale(adata)

# Calculate the visualizations
sc.pp.pca(adata, n_comps=50, use_highly_variable=True, svd_solver='arpack')
sc.pl.pca_loadings(adata, save='_costim_only.pdf')
sc.pl.pca_variance_ratio(adata, save='_costim_only.pdf')
sc.pl.pca_scatter(adata, color='sample', save='_costim_only.pdf')

sc.pp.neighbors(adata, n_pcs=20)
sc.tl.umap(adata)

# ####################### Score with M1 and M2 core genes ################
geneset_file = data_dir + '/unique_programs.pickle'

with open(geneset_file, 'rb') as f:
    unique_genes = pickle.load(f)

# Print number of M1 and M2 genes
print("Number of M1 genes: {}\n"
      "Number of M2 genes: {}".format(unique_genes['M1'].shape[0], unique_genes['M2'].shape[0]))

final_geneset = {"M1": unique_genes['M1']['names'].to_list(),
                 "M2": unique_genes['M2']['names'].to_list()}

sc.tl.score_genes(adata, final_geneset['M1'], score_name='UCG_M1_score', use_raw=True)
sc.tl.score_genes(adata, final_geneset['M2'], score_name='UCG_M2_score', use_raw=True)

# Show scores in umap and in violin plots by leiden
sc.pl.umap(adata, color=['UCG_M1_score', 'UCG_M2_score'],
           cmap='plasma', vmin='p1.5', vmax='p98.5',
           save="_Fig3a_UCG_scores.pdf")

adata.write(results_path + "reclustered_costimulated_cells.h5ad", compression='gzip')