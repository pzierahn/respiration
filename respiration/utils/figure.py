import os
import matplotlib.pyplot as plt


def savefig(fig: plt.Figure, dir: str, filename: str) -> None:
    """
    Store a figure in the reports/figures directory.
    :param fig: The figure to store.
    :param dir: The directory to store the figure in.
    :param filename: The filename of the figure.
    """

    path_png = os.path.join(dir, f'{filename}.png')
    fig.savefig(path_png, format='png')

    path_svg = os.path.join(dir, f'{filename}.svg')
    fig.savefig(path_svg, format='svg')
