import numpy as np
import pandas as pd
import scanpy as sc
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pickle
import single_cell_functions as scf

"""This is the differential gene analysis that is done to make the volcano plots in Fig 1 and 2. It first finds the 
unique M1 and M2 core genes. The finds which of these are regulated (up or down) by co-stimulation. Filtered parameter
will filter the set of core genes to only include those that are expressed at least by 15% of cells.
Note: all analysis is done on these filtered, unique core genes. """

# Parameters
filtered = True
fig_dir_o = 'figures/'
sc.settings.figdir = fig_dir_o
# Data set directory
data_dir = fig_dir_o + 'gene_programs/'
fig1_dir = fig_dir_o + 'Figure1/'
fig2_dir = fig_dir_o + 'Figure2/'

if not os.path.isdir(fig1_dir):
    os.makedirs(fig1_dir)

if not os.path.isdir(fig2_dir):
    os.makedirs(fig2_dir)

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

# Set settings
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(scanpy=True, dpi=80, dpi_save=600, color_map='viridis', vector_friendly=False)
sc.settings.autoshow = False

results_path = 'write/'
results_file = 'bmdm_object.h5ad'


########################################################################################################################
########################################################################################################################

def volcano_plot2(x, y, *,
                  fc_thresh=None,
                  pval_thresh=None,
                  program_unique=None,
                  program_shared=None,
                  program_up_genes=None,
                  program_down_genes=None,
                  single_stimuli=None,
                  zoom=False):
    x = x.copy()
    y = y.copy()

    if np.any(y == 0):
        y.loc[y == 0] = 1e-300

    if fc_thresh and pval_thresh:
        program_regulated1 = np.any((program_down_genes, program_up_genes), axis=0)
        above_FC = x >= fc_thresh
        below_FC = x <= -fc_thresh
        true_FC = np.any((above_FC, below_FC), axis=0)
        true_pval = y <= pval_thresh
        above_thresh = np.all((true_FC, true_pval), axis=0)

        c = np.full(np.shape(x)[0], 'grey', np.dtype(('U', 10)))
        z = np.full(np.shape(x)[0], 0)
        c[program_unique] = 'blue'
        c[program_shared] = 'cyan'
        c[program_regulated1] = 'orange'
        z[program_unique] = 1
        z[program_shared] = 1
        z[program_regulated1] = 2

        z_ind = np.argsort(z)
        plt.figure()
        plt.scatter(x[z_ind], -np.log10(y)[z_ind], c=c[z_ind], s=1)
        plt.axvline(fc_thresh)
        plt.axvline(-fc_thresh)
        plt.axhline(-np.log10(pval_thresh))
        ax = plt.gca()
        up1 = np.sum(program_up_genes)
        down1 = np.sum(program_down_genes)
        plt.text(0.9, 0.9, up1, color='orange',
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.text(0.1, 0.9, down1, color='orange',
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_axisbelow(True)

        # Axis
        if zoom:
            xmin = np.min(x[above_thresh]) * 1.1
            xmax = np.max(x[above_thresh]) * 1.1
            ax.set_xlim(xmin, xmax)

        # Legend
        marker_orange = mlines.Line2D([], [], color='orange', marker='o', linestyle='', markersize=3,
                                      label='Regulated by costimulation')
        marker_blue = mlines.Line2D([], [], color='blue', marker='o', linestyle='', markersize=3,
                                    label='{} induced'.format(single_stimuli))
        marker_cyan = mlines.Line2D([], [], color='cyan', marker='o', linestyle='', markersize=3,
                                    label='Commonly induced'.format(single_stimuli))
        plt.legend(handles=[marker_orange, marker_blue, marker_cyan], fontsize=10, bbox_to_anchor=(1.02, 1),
                   loc='upper left')

    else:
        c = np.full(np.shape(x)[0], 'black')
        plt.figure()
        plt.scatter(x, -np.log10(y), c=c)

    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10 Adjusted P-value')
    return ax


########################################################################################################################
########################################################################################################################

# Read results file
adata = sc.read(results_path + results_file)
# Rename samples
newnames = ['Control', 'LPS+IFNg', 'IL-4', 'LPS+IFN-g\n+IL-4']
adata.obs['sample2'] = adata.obs['sample'].copy()
adata.rename_categories('sample2', newnames)

########################################################################################################################
########################################################################################################################

# Find M1, M2, and M1+M2 genes up-regulated over control
stims = ['M1', 'M2', 'M1+M2']
pval_t = 0.05
FC_t = np.log2(1.5)
min_pct = 0.15
gene_program = {}
diff_results = {}
diff_results2 = {}
M0_filter = {}
d = {}
sc.tl.rank_genes_groups(adata, 'sample', method='wilcoxon', use_raw=True,
                        groups=stims, reference='M0', n_genes=0, key_added='rank_genes_control')

namedict = {'M0': 'Control', 'M1': 'LPS+IFNg', 'M2': 'IL-4', 'M1+M2': 'LPS+IFNg\n+IL-4'}

if filtered:
    # Filter results to only include genes that are expressed in 15% of that stimulation condition
    sc.tl.filter_rank_genes_groups(adata, key='rank_genes_control', min_in_group_fraction=min_pct,
                                   max_out_group_fraction=1.1,
                                   min_fold_change=0, key_added='rank_genes_control_filtered')

    result1 = adata.uns['rank_genes_control_filtered']
    result2 = adata.uns['rank_genes_control']
    # Get genes that are more than 15% in M0 to calculate proper downregulated filter
    tmp = adata[adata.obs['sample'].isin(['M0']), :]
    percentM0 = tmp.raw.X.todense() > 0
    percentM0 = percentM0.mean(axis=0).A1
    percentM0 = pd.Series(percentM0, index=tmp.raw.var_names)

else:
    result1 = adata.uns['rank_genes_control']

for ss in stims:
    # Export to dictionary of dataframes
    if filtered:
        # Get dataframes for volcano plots
        diff_results[ss] = sc.get.rank_genes_groups_df(adata, group=ss, key='rank_genes_control_filtered')
        diff_results2[ss] = sc.get.rank_genes_groups_df(adata, group=ss, key='rank_genes_control')
        diff_results[ss].loc[diff_results[ss]['names'].isna(), :] = np.NaN
        # Find upregulated genes, and filter the upregulated genes thar are not above 15% in ss condition
        upreg_filtered_out = np.all((result1['pvals_adj'][ss] <= pval_t, result1['logfoldchanges'][ss] >= FC_t,
                                     pd.isna(result1['names'][ss])), axis=0)

        upregulated = np.all((result1['pvals_adj'][ss] <= pval_t, result1['logfoldchanges'][ss] >= FC_t,
                              ~pd.isna(result1['names'][ss])), axis=0)

        # Find downregualted genes, and filter out downreg genes that are not above 15% in M0 (control)
        M0_filter[ss] = percentM0.loc[result2['names'][ss]] > min_pct
        downregulated = np.all((result2['pvals_adj'][ss] <= pval_t, result2['logfoldchanges'][ss] <= -FC_t,
                                M0_filter[ss]), axis=0)
        downreg_filter_out = np.all((result2['pvals_adj'][ss] <= pval_t, result2['logfoldchanges'][ss] <= -FC_t,
                                     ~M0_filter[ss]), axis=0)

        # Update dataframes
        genes_filtered = np.any((upreg_filtered_out, downreg_filter_out), axis=0)
        diff_results2[ss] = diff_results2[ss][~genes_filtered]
    else:
        diff_results[ss] = sc.get.rank_genes_groups_df(adata, group=ss, key='rank_genes_control')
        upregulated = np.all((result1['pvals_adj'][ss] <= pval_t, result1['logfoldchanges'][ss] >= FC_t), axis=0)
        downregulated = np.all((result1['pvals_adj'][ss] <= pval_t, result1['logfoldchanges'][ss] <= -FC_t), axis=0)

    d[ss] = pd.DataFrame(
        {key: result2[key][ss]
         for key in ['names', 'scores', 'logfoldchanges', 'pvals_adj']})

    gene_program[ss] = d[ss][upregulated].reset_index(drop=True)
    gene_program[ss].to_excel(data_dir + "{}_core_genes.xlsx".format(ss))

    downreg = d[ss][downregulated].reset_index(drop=True)
    downreg.to_excel(data_dir + "{}_down_core_genes.xlsx".format(ss))

# ####### Find shared genes in M1 and M2 programs ############
M1_shared = np.in1d(gene_program["M1"]['names'], gene_program["M2"]["names"])
M1_unique = ~M1_shared

M2_shared = np.in1d(gene_program["M2"]['names'], gene_program["M1"]["names"])
M2_unique = ~M2_shared

# M1+M2 upregulated genes
s1 = set(gene_program["M1"]["names"])
s2 = set(gene_program["M2"]["names"])
M1andM2 = pd.Series(list(s1.union(s2)))
M12_shared = np.in1d(gene_program["M1+M2"]['names'], M1andM2)
M12_unique = ~M12_shared

# Bar plots
plt.figure()
unique = [M1_unique.sum(), M2_unique.sum()]
shared = [M1_shared.sum(), M2_shared.sum()]
p1 = plt.bar([0, 1], unique)
p2 = plt.bar([0, 1], shared, bottom=unique)
plt.ylabel('Number of upregulated genes')
plt.xticks([0, 1], (namedict["M1"], namedict["M2"]))
plt.grid(False)
plt.legend(["Unique", "Shared"])
plt.savefig(fig2_dir + "Fig2a_Summary of M1 and M2 genes_barplot.pdf", bbox_inches='tight')

# Volcano plots
for ss in stims:
    scf.pl.volcano_plot(diff_results2[ss]['logfoldchanges'], diff_results2[ss]['pvals_adj'], FC_thresh=FC_t,
                        pval_thresh=pval_t)
    plt.xlabel('Log2 Fold Change {} / control'.format(namedict[ss]))
    if ss == 'M1' or ss == 'M1+M2':
        x_limit = 15
    elif ss == 'M2':
        x_limit = 10

    v_axis = plt.axis()
    new_axis = [-x_limit, x_limit, v_axis[2], v_axis[3]]
    plt.axis(new_axis)
    plt.savefig(fig1_dir + 'Fig1b_Volcano_{}_over_control.pdf'.format(ss), bbox_inches='tight'), plt.close()

########################################################################################################################
########################################################################################################################

# Save unique and shared to excel
summary_program = pd.DataFrame({"Unique": unique,
                                "Shared": shared}, index=["M1", "M2"])

summary_program.to_excel(data_dir + "Summary of M1 and M2 genes.xlsx")

# Concatenate unique genes for all stims
unique_genes = {"M1": gene_program["M1"][M1_unique],
                "M2": gene_program["M2"][M2_unique],
                "M1+M2": gene_program["M1+M2"][M12_unique]}

shared_genes = gene_program["M1"][M1_shared]  # Same for M2

# Save genesets for further analysis
with open(data_dir + 'gene_programs.pickle', 'wb') as f:
    pickle.dump(gene_program, f)

with open(data_dir + 'unique_programs.pickle', 'wb') as f:
    pickle.dump(unique_genes, f)
########################################################################################################################
# ####### Compare co-stim with single-stim ############
########################################################################################################################
pval_t = 0.05
FC_t = np.log2(1.5)
d_costim = {}
costim_ss_down = {}
costim_ss_up = {}
d_summary = {}
result = {}
down = {}
up = {}
single_stims = ["M1", "M2"]
group = 'M1+M2'
for ss in single_stims:
    key2add = 'rank_genes_mixed_vs_{}'.format(ss)
    sc.tl.rank_genes_groups(adata, 'sample', method='wilcoxon', use_raw=True,
                            groups=['M1+M2'], reference=ss, n_genes=0, key_added=key2add)

    # Find genes diff regulated with costim
    result[ss] = adata.uns[key2add]
    down[ss] = np.all((result[ss]['pvals_adj'][group] <= pval_t, result[ss]['logfoldchanges'][group] <= -FC_t), axis=0)
    up[ss] = np.all((result[ss]['pvals_adj'][group] <= pval_t, result[ss]['logfoldchanges'][group] >= FC_t), axis=0)

    # Export to dictionary of dataframes
    toexport = np.any((down[ss], up[ss]), axis=0)
    print("Significant genes co-stim versus {} single-stim: {}".format(ss, toexport.sum()))
    d_costim[ss] = pd.DataFrame(
        {key: result[ss][key][group]
         for key in ['names', 'scores', 'logfoldchanges', 'pvals_adj']})

    # Find the boolean indices of the genes that belong to the gene program of the single stimuli
    ss_genes = np.in1d(d_costim[ss]["names"], gene_program[ss]["names"])  # shared and unique genes
    ss_unique = np.in1d(d_costim[ss]["names"], unique_genes[ss]["names"])
    ss_shared = np.in1d(d_costim[ss]["names"], shared_genes["names"])

    # Find which genes are up and down regulated that belong to single-stim stimuli

    program_down = np.all((down[ss], ss_unique), axis=0)
    program_up = np.all((up[ss], ss_unique), axis=0)
    ####################################################################################################################
    ####################################################################################################################
    program_regulated = np.any((program_down, program_up), axis=0)  # either up or down regulated
    costim_ss_down[ss] = d_costim[ss][program_down].sort_values('scores', ascending=True).reset_index(
        drop=True)  # Actual dataframe with info
    costim_ss_up[ss] = d_costim[ss][program_up].sort_values('scores', ascending=False).reset_index(drop=True)

    # Volcano plots
    volcano_plot2(d_costim[ss].loc[:, 'logfoldchanges'], d_costim[ss].loc[:, 'pvals_adj'], pval_thresh=pval_t,
                  fc_thresh=FC_t,
                  program_unique=ss_unique, program_shared=ss_shared, program_up_genes=program_up,
                  program_down_genes=program_down,
                  single_stimuli=namedict[ss], zoom=True)
    plt.xlabel('Log2 Fold Change Co-stim / {}'.format(namedict[ss]))

    if ss == 'M1':
        x_limit = 8
    elif ss == 'M2':
        x_limit = 12
    v_axis = plt.axis()
    new_axis = [-x_limit, x_limit, v_axis[2], v_axis[3]]
    plt.axis(new_axis)
    plt.savefig(fig2_dir + 'Fig2b_Volcano_M1+M2_over_{}.pdf'.format(ss), bbox_inches='tight'), plt.close()

    # Export to excel
    costim_ss_down[ss].to_excel(data_dir + "{}_genes downregulated with costim.xlsx".format(ss))
    costim_ss_up[ss].to_excel(data_dir + "{}_genes upregulated with costim.xlsx".format(ss))

    # Enumerate regulated genes that are either unique or shared for each single stim
    if ss == "M1":
        other_ss = "M2"
    elif ss == "M2":
        other_ss = "M1"

    ss_unique_other = np.in1d(d_costim[ss]["names"], unique_genes[other_ss]["names"])

    # Upregulated by costim
    up_unique = np.all((up[ss], ss_unique), axis=0)
    up_unique_other = np.all((up[ss], ss_unique_other), axis=0)
    up_shared = np.all((up[ss], ss_shared), axis=0)
    all_programs_up = up_unique + up_unique_other + up_shared
    up_new = np.all((up[ss], ~all_programs_up), axis=0)
    d_summary[ss] = pd.DataFrame({"Unique": up_unique.sum(),
                                  "Unique_other": up_unique_other.sum(),
                                  "Shared": up_shared.sum(),
                                  "New": up_new.sum()}, index=['Upregulated'])

    # Downregulated by costim
    down_unique = np.all((down[ss], ss_unique), axis=0)
    down_unique_other = np.all((down[ss], ss_unique_other), axis=0)
    down_shared = np.all((down[ss], ss_shared), axis=0)
    all_programs_down = down_unique + down_unique_other + down_shared
    down_new = np.all((down[ss], ~all_programs_down), axis=0)
    d_summary[ss].loc["Downregulated"] = [down_unique.sum(),
                                          down_unique_other.sum(),
                                          down_shared.sum(),
                                          down_new.sum()]

    # Export to excel
    d_summary[ss].to_csv(data_dir + "Summary_costim_vs_{}.csv".format(ss), sep='\t')

# Save down-regulated genes for all programs for noise analysis
with open(data_dir + 'down_in_mixed_unique.pickle', 'wb') as f:
    pickle.dump(costim_ss_down, f)

with open(data_dir + 'up_in_mixed_unique.pickle', 'wb') as f:
    pickle.dump(costim_ss_up, f)
