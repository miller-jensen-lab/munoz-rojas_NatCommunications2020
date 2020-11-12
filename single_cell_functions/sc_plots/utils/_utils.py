import numpy as np
import matplotlib.pyplot as plt
import matplotlib as plt
from matplotlib import colors

#Grey Yellow Orange Red color map
cmap_p = plt.cm.get_cmap('YlOrRd')
new_colors = cmap_p(np.linspace(0.1, 1, cmap_p.N))
new_colors[0] = (0.8, 0.8, 0.8, 1)
sc_cmap = colors.LinearSegmentedColormap.from_list('Oranges_grey', new_colors, cmap_p.N)

def equal_stacked_axis(axes):
    """
    Function to make all y_axes in stacked_violin_plot equal
    :param axes: axes returned from sc.pl.stacked_violin_plot

    :return: axes: same axes object
    """
    lims = [ax.get_ylim() for ax in axes]
    miny = np.min(lims)
    maxy = np.max(lims)
    for ax in axes:
        ax.set_ylim([miny, maxy])

    return axes