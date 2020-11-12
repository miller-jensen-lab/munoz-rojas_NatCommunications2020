import scanpy as sc
import single_cell_functions.sc_plots as scplts
import matplotlib.pyplot as plt

""" Plot violin plots for chosen targets in Figure 2"""

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
adata = sc.read(results_path + results_file)

# Re-assign colors
vega20 = sc.pl.palettes.default_20
new_colors = ['dimgray', vega20[1], vega20[0], vega20[2]]
adata.uns['sample2_colors'] = new_colors

# Re-name categories for proper plotting
newnames = ['Control', 'LPS+IFNg', 'IL-4', 'LPS+IFN-g\n+IL-4']
adata.obs['sample2'] = adata.obs['sample'].copy()
adata.rename_categories('sample2', newnames)

########################################################################################################################
# Figure 2 violin plots
########################################################################################################################
fig_dir = fig_dir_o + "Figure2/"
sc.settings.figdir = fig_dir

scplts.violin_percent(adata, genes=['Nos2', 'Tnf', 'Il12b', 'Il6', 'Arg1', 'Chil3'], groupby='sample2',
                      cpalette=adata.uns['sample2_colors'])
plt.savefig(fig_dir + "Fig2c_violinplots.pdf", bbox_inches='tight')