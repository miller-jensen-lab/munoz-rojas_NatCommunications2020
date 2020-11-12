import multi_neural_classifier_fn2 as mnc

""" This scripts runs the neural networks analysis for Figure 3 """
fig_dir_o = 'figures/'
fig_dir = fig_dir_o + 'Figure3/'

# All genes
mnc.M1M2_neuralnet2(geneset='all_genes', fig_dir=fig_dir)

# Downregulated genes
mnc.M1M2_neuralnet2(geneset='down_regulated', n_genes='all', fig_dir=fig_dir)
