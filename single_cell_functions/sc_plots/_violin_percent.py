import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt


def violin_percent(adata,
                   *,
                   genes=None,
                   th=0,
                   groupby=None,
                   tick_fontsize=12,
                   cpalette=None,
                   use_raw=True,
                   **kwargs):
    """

    Parameters
    ----------
    adata
    genes
    th
    groupby
    tick_fontsize
    cpalette
    use_raw
    kwargs

    Returns
    -------

    """
    # Check if autoshow is on. If it is, it immediately plots the first image and doens't plot the rest correctly, so
    # turn it off and turn it back on at the end of the function.
    return_authoshow = False
    if sc.settings.autoshow:
        sc.settings.autoshow = False
        return_authoshow = True

    if genes == None:
        raise ValueError("Pass a list of genes")

    n_genes = len(genes)
    nrows = int(np.ceil(n_genes / 4))
    ncols = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(15,nrows*4))
    axes = axs.flatten(order='C')
    for gene, ax in zip(genes, axes):
        if cpalette:
            ax1 = sc.pl.violin(adata, gene, groupby=groupby, use_raw=use_raw, palette=cpalette, ax=ax, **kwargs)
        else:
            ax1 = sc.pl.violin(adata, gene, groupby=groupby, use_raw=use_raw, ax=ax, **kwargs)
        labels = ax.get_xticklabels()
        # labels_text = [label.get_text() for label in labels]
        # Add threshold line
        ax.axhline(y=th, color='black', linewidth=0.8, alpha=0.5, linestyle='--')

        for i, label in enumerate(labels):
            group = label.get_text()
            if use_raw:
                adata2 = adata[adata.obs[groupby] == group].raw.copy()
            else:
                adata2 = adata[adata.obs[groupby] == group].copy()
            # adata2 = adata2[:, gene].X > th
            adata2 = adata2.obs_vector(gene) > th
            percent = adata2.mean(axis=0)
            labels[i].set_text(group + "\n{:.1f}%".format(percent * 100))

        ax.set_xticklabels(labels, fontsize=tick_fontsize)
        ax.set_xlabel('') # Remove repeated x label

    # Delete unused axes, if any
    for idx, ax in enumerate(axes[n_genes:]):
        fig.delaxes(ax)

    fig.tight_layout()

    if return_authoshow:
        sc.settings.autoshow = True

    return fig