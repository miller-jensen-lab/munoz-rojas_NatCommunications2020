import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import time
from multi_neural_classifier_UMAP_results import neural_results_on_UMAP

""" Contains function for neural network. Uses multi-NN to predict whether a co-stim cell is
- M1-dominant,
- M2-dominant,
- Mixed,
- Unclassified.
Outputs:
Full output of classification report
Full output of predicted labels for each cell
Plots predicted labels onto UMAP space """



def M1M2_neuralnet2(*, geneset=None, n_genes=None, fig_dir=None):

    """
    This analysis will use M1 and M2 data to predict the classes in M1+M2.
    Uses multiple neural nets to predict (independent of each other):
    - M1, and nothing else
    - M2, and nothing else
    - A mix of M1 and M2, or
    - No classification, termed unclassified.
     Use top n_genes genes from the M1M2 volcano analysis (or the downregulated genes), specified by parameter

    :param geneset: {unique_core, down_regulated, all}. Specifies which geneset to use, unique_core signature genes, or those genes
     downregulated by co-stim. All specifies using all the genes in the dataset to build the classifier (not supported
     in this function yet, will implement soon).
    :param n_genes: specifies number of genes to use for neural network.
    :param fig_dir: folder to save results
    :return:
    """

    # Set settings
    sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
    sc.settings.set_figure_params(scanpy=True, dpi=80, dpi_save=600, color_map='viridis', vector_friendly=False)
    sc.settings.autoshow = False

    fig_dir_o = 'figures/'
    results_path = 'write/'
    results_file = 'bmdm_object.h5ad'
    data_dir = fig_dir_o + 'gene_programs/'

    # ######################################################################################################################

    # ######################################################################################################################

    # Set n_genes if empty
    if not n_genes:
        n_genes = 'all'

    # Read genesets based on parameter
    if geneset == 'down_regulated':
        geneset_file = data_dir + '/down_in_mixed_unique.pickle'
        print("***\nRunning Neural network analysis with {} {} genes\n***".format(n_genes, geneset))

    elif geneset == 'all_genes':
        print("***\nRunning Neural network analysis with {}\n***".format(geneset))

    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    sc.settings.figdir = fig_dir


    print("*** Starting neural network analysis ***")

    adata = sc.read(results_path + results_file)
    samples = ['M0', 'M1', 'M2', 'M1+M2']

    # Build data sets to train classifier
    M0_mask = np.array(adata.obs['sample'] == 'M0')
    N0 = np.sum(M0_mask)
    M1_mask = np.array(adata.obs['sample'] == 'M1')
    N1 = np.sum(M1_mask)
    M2_mask = np.array(adata.obs['sample'] == 'M2')
    N2 = np.sum(M2_mask)
    M12_mask = np.array(adata.obs['sample'] == 'M1+M2')
    N12 = np.sum(M12_mask)

    single_mask = M1_mask | M2_mask
    cell_state = np.array(adata[single_mask].obs['sample'])


    # Reduce to desired geneset
    if geneset is not 'all_genes':
        with open(geneset_file, 'rb') as f:
            unique_genes = pickle.load(f)
        # Print number of M1 and M2 genes
        print("Number of M1 genes: {}\n"
              "Number of M2 genes: {}".format(unique_genes['M1'].shape[0], unique_genes['M2'].shape[0]))

        # Reduce to desired genelist
        if n_genes == 'all': # This is scenario when you want all the genes in a specific unique or downreg geneset
            genelist = np.concatenate((unique_genes['M1']['names'],
                                       unique_genes['M2']['names']))
        else:
            genelist = np.concatenate((unique_genes['M1']['names'][0:n_genes],
                                       unique_genes['M2']['names'][0:n_genes]))
        adata2 = adata.raw[:, genelist].copy()

    else:
        # Use all genes
        adata2 = adata.raw.copy()
        print("Total number of genes: {}".format(adata2.shape[1]))

    M1orM2 = pd.DataFrame(data=adata2[single_mask, :].X.todense(), columns=adata2.var_names)
    M12 = pd.DataFrame(data=adata2[M12_mask, :].X.todense(), columns=adata2.var_names)
    M12_obs_names = adata[M12_mask, :].obs_names

    del adata, adata2

    #Make Multilabel format
    y = MultiLabelBinarizer().fit_transform(cell_state)
    y = y[:,:-1]

    #Split into test-train

    X_train, X_test, y_train, y_test = train_test_split(M1orM2, y, test_size=0.10)

    #Scale the data
    scaler = StandardScaler()
    # scaler.fit(X_train)
    scaler.fit(M1orM2)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    M12 = scaler.transform(M12)

    #Build classifier with tnree hidden layers, and multi-label one vs all approach
    size_layer = round(X_train.shape[1]*1/2)
    clf = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(size_layer, size_layer, size_layer),
                                            activation='tanh', max_iter=500, verbose=False)) # three hidden layer

    #Train classifier
    print('---- Started training neural network ---')
    tic = time.time()
    clf.fit(X_train, y_train)
    toc = time.time()
    print('---- Elapsed time = {} minutes ----'.format((toc-tic)/60))

    # Check accuracy
    predicted = clf.predict(X_test)
    print(classification_report(y_test, predicted))
    report = pd.DataFrame(classification_report(y_test, predicted, output_dict=True)).T
    report.to_excel(fig_dir + f"{geneset}_{size_layer}_neurons_multiclass_classification_report_full.xlsx")

    print('---- Accuracy = {:.2%} ----'.format(accuracy_score(y_test, predicted)))
    with open(fig_dir + f"{geneset}_{size_layer}_neurons_multiclass_classification_report.txt", "w") as text_file:
        text_file.write('---- Accuracy = {:.2%} ----'.format(accuracy_score(y_test, predicted)))

    # Predict M1 + M2 states
    print('---- Started predicting M1+M2 with neural network ---')
    M12_prediction = clf.predict(M12)

    # Quantify predictions
    print('---- Quantifying and saving results ---')
    c_M1 = np.all((np.sum(M12_prediction, axis=1) == 1, M12_prediction[:,0] == 1), axis=0)
    c_M2 = np.all((np.sum(M12_prediction, axis=1) == 1, M12_prediction[:,1] == 1), axis=0)
    c_mixed = np.all((np.sum(M12_prediction, axis=1) == 2, M12_prediction[:,0] == 1, M12_prediction[:,1] == 1), axis=0)
    c_unclass = np.sum(M12_prediction, axis=1) == 0

    n_M1 = np.sum(c_M1)
    n_M2 = np.sum(c_M2)
    n_mixed = np.sum(c_mixed)
    n_unclass = np.sum(c_unclass)

    # Make label vector
    # predicted_label = np.zeros(M12.shape[0], dtype='object')
    predicted_label = pd.Series(index=M12_obs_names, dtype='object')

    predicted_label[c_M1] = "M1_class"
    predicted_label[c_M2] = "M2_class"
    predicted_label[c_mixed] = "Mixed_class"
    predicted_label[c_unclass] = "Unclassified_class"


    print('{:.2%} classified as M1\n'
          '{:.2%} classified as M2\n'
          '{:.2%} classified as M1+M2\n'
          '{:.2%} unclassified'.format(n_M1/N12, n_M2/N12, n_mixed/N12, n_unclass/N12))

    print('{} classified as M1\n'
          '{} classified as M2\n'
          '{} classified as M1+M2\n'
          '{} unclassified'.format(n_M1, n_M2, n_mixed, n_unclass))

    d = {'Num': [n_M1, n_M2, n_mixed, n_unclass], 'Percent': [n_M1, n_M2, n_mixed, n_unclass]/N12}
    out = pd.DataFrame(data=d, index=["M1-dominant", "M2-dominant", "Mixed", "Unclassified"])
    out.to_excel(fig_dir + f"Fig3c_{geneset}_{size_layer}_neurons_multiclass_classifier_results.xlsx")

    ####################################################################################################################
    # Plot on umap space
    print('---- Overlaying classifications on UMAP space ---')
    neural_results_on_UMAP(predicted_label=predicted_label, dataset='M1M2only', geneset_name=geneset, fig_dir_net=fig_dir)
    ####################################################################################################################