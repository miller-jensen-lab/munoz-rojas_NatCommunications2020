import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

def compositional_plot(adata, *, base_group=None, composition=None):
    """
    Plot compositional bar plot to quantify the proprotion of ``composition`` clusters in each ``base_group``.
    :param adata: anndata object
    :param base_group: Base grouping to quantify (e.g. In each group in base_group, how many cells belong to each
    cluster in 'composition)
    :param composition: Clusters to quantify
    :return:
    """
    bases = adata.obs[base_group].cat.categories
    clusters = adata.obs[composition].cat.categories
    df = pd.DataFrame(index=bases, columns=clusters)
    n_per_base = adata.obs[base_group].value_counts()
    for base in bases:
        adata_b = adata[adata.obs[base_group] == base].copy()
        df.loc[base, :] = adata_b.obs[composition].value_counts() / n_per_base[base]

    df.plot(kind='bar', stacked=True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    return df

