import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import single_cell_functions.sc_plots as scplts
import single_cell_functions.scstats as scstats

""" Runs odds ratio analysis on co-stimulated cells, and plots the relevant density plots and violin plots for Figure 4
"""

# Parameters
fig_dir_o = 'figures/'
sc.settings.figdir = fig_dir_o
data_dir = fig_dir_o + 'gene_programs/'

# Set settings
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(scanpy=True, dpi=80, dpi_save=600, color_map='viridis', vector_friendly=False)
sc.settings.autoshow = False

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
newnames = ['Control', 'LPS+IFNg', 'IL-4', 'LPS+IFNg\n+IL-4']
adata.obs['sample2'] = adata.obs['sample'].copy()
adata.rename_categories('sample2', newnames)

########################################################################################################################
# Odds ratio analysis
########################################################################################################################

# Run with 50 downregulated genes
geneset_file2 = data_dir + '/down_in_mixed_unique.pickle'

with open(geneset_file2, 'rb') as f:
    downregulated_genes = pickle.load(f)

fig_dir = fig_dir_o + 'Figure4/'
sc.settings.figdir = fig_dir

if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

th = 2
thresh = np.log1p(th) # transform count threshold
n_genes = 50
all_unique_genes = np.concatenate((downregulated_genes['M1']['names'][0:n_genes],
                                   downregulated_genes['M2']['names'][0:n_genes]))

# Get only co-stimulated cells
adata2 = adata.raw[adata.obs['sample'] == 'M1+M2', :].copy()
df = pd.DataFrame(adata2[:, all_unique_genes].X.todense(), columns=all_unique_genes)
print(df.shape)
# Odds ratio
print('Beginning odds ratio analysis')
scstats.odds_ratio(df, th=thresh, alpha=0.05)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=4)
for i, j in zip(ax.get_xticklabels()[0:n_genes], ax.get_yticklabels()[0:n_genes]):
    i.set_color("orange"), j.set_color("orange")
for i, j in zip(ax.get_xticklabels()[n_genes:], ax.get_yticklabels()[n_genes:]):
    i.set_color("blue"), j.set_color("blue")

plt.savefig(fig_dir + f"Fig4a_Odds_ratio_th_{round(th, 1)}.pdf", bbox_inches='tight')
plt.close()

########################################################################################################################
# Violin and density plots - Figure 4
########################################################################################################################
# Density scatter plots
# 2D Density Plots
x_vars = ['Il6', 'Il12b']
y_vars = ['Arg1']
th = np.log1p(2)

for x in x_vars:
    for y in y_vars:
        scplts.density_plot_sc(adata, x, y, x_thresh=th, y_thresh=th, groupby='sample2')
        save_name = fig_dir + f'Fig4b_{x}vs{y}.pdf'
        print('Saving to {}'.format(save_name))
        plt.savefig(save_name), plt.close()

# Violin plots
scplts.violin_percent(adata, genes=['Nfkbiz' , 'Klf4'], groupby='sample2', cpalette=adata.uns['sample2_colors'])
plt.savefig(fig_dir + "Fig4d-e_violinplotsKlf4_Nfkbiz.pdf", bbox_inches='tight'), plt.close()


########################################################################################################################
# Extra density plots - Figure S4
########################################################################################################################
# 2D Density Plots
x_vars = ['Il6', 'Il12b']
y_vars = ['Chil3']
th = np.log1p(2)

for x in x_vars:
    for y in y_vars:
        scplts.density_plot_sc(adata, x, y, x_thresh=th, y_thresh=th, groupby='sample2')
        save_name = fig_dir + f'FigS4a_{x}vs{y}.pdf'
        print('Saving to {}'.format(save_name))
        plt.savefig(save_name), plt.close()

x_vars = ['Tnf', 'Ccl5']
y_vars = ['Arg1']

for x in x_vars:
    for y in y_vars:
        scplts.density_plot_sc(adata, x, y, x_thresh=th, y_thresh=th, groupby='sample2')
        save_name = fig_dir + f'FigS4b_{x}vs{y}.pdf'
        print('Saving to {}'.format(save_name))
        plt.savefig(save_name), plt.close()