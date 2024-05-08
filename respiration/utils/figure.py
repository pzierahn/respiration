import os
import matplotlib.pyplot as plt


def store_figure(fig: plt.Figure, dirname: str, name: str) -> None:
    """
    Store a figure in the reports/figures directory.
    :param fig: The figure to store.
    :param dirname: The directory name to store the figure in.
    :param name: The name of the figure.
    """
    path = os.path.join(os.getcwd(), '..', '..', 'outputs', 'figures', dirname)
    os.makedirs(path, exist_ok=True)

    path_png = os.path.join(path, f'{name}.png')
    fig.savefig(path_png, format='png')

    path_svg = os.path.join(path, f'{name}.svg')
    fig.savefig(path_svg, format='svg')
