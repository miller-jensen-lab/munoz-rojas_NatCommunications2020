import scanpy as sc
import matplotlib.pyplot as plt


def neural_results_on_UMAP(predicted_label, *, dataset='M1M2only', geneset_name=None, fig_dir_net=None):
    """
        This function overlays the predicted labels from the neural network results into the UMAP space of the orignal
    dataset.

    :param predicted_label: predicted labels from neural network
    :param dataset: which dataset to project onto. "full" is the full dataset (all stims), "M1M2only" is just the
    co-stimulated cells
    :param geneset_name: either 'all genes' or 'downregulated'
    :param fig_dir_net: original figure folder to save images to
    :return: Saves figures to the original folder
    """

    # Set settings
    sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
    sc.settings.set_figure_params(scanpy=True, dpi=200, dpi_save=600, color_map='viridis', vector_friendly=True)
    sc.settings.autoshow = False

    sc.settings.figdir = fig_dir_net
    # Define new colors
    vega20 = sc.pl.palettes.default_20

    results_path = 'write/'
    if dataset == "full":
        results_file = 'bmdm_object.h5ad'
        new_colors = [vega20[1], vega20[0], vega20[2], 'dimgray', 'gray']
    elif dataset == "M1M2only":
        results_file = 'reclustered_costimulated_cells.h5ad'
        new_colors = [vega20[1], vega20[0], vega20[2], 'dimgray']
    else:
        raise ValueError("Parameter dataset must be either 'full' or 'M1M2only'")


    # Plotting params
    plt.rcParams['axes.grid'] = False

    ########################################################################################################################
    # Read files
    ########################################################################################################################
    adata = sc.read(results_path + results_file)

    ########################################################################################################################
    # Visialize in UMAP space
    ########################################################################################################################
    adata.obs['predicted_label'] = predicted_label
    sc._utils.sanitize_anndata(adata)
    adata.uns['predicted_label_colors'] = new_colors
    sc.pl.umap(adata, color='predicted_label', save=f'_FigS3_predicted_label_{geneset_name}_{dataset}.pdf')
