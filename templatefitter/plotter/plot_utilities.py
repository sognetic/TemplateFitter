"""
Provides some utility functions for matplotlib plots.
"""
import os
import logging
import tikzplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.colors as mpl_colors

from typing import Tuple, List, Optional
from matplotlib import figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from templatefitter.utility import PathType
from templatefitter.plotter.plot_style import KITColors
from templatefitter.binned_distributions.binning import Binning

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AxesType",
    "FigureType",
    "export",
    "save_figure_as_tikz_tex_file",
    "get_white_or_black_from_background",
    "color_fader",
]

AxesType = axes.Axes
FigureType = figure.Figure


def export(
    fig: plt.Figure,
    filename: PathType,
    target_dir: PathType = "plots/",
    file_formats: Tuple[str, ...] = (".pdf", ".png"),
    save_as_tikz: bool = False,
    close_figure: bool = False,
) -> None:
    """
    Convenience function for saving a matplotlib figure.

    :param fig: A matplotlib figure.
    :param filename: Filename of the plot without .pdf suffix.
    :param file_formats: Tuple of file formats specifying the format
                         figure will be saved as.
    :param target_dir: Directory where the plot will be saved in.
                       Default is './plots/'.
    :param save_as_tikz: Save the plot also as raw tikz tex document.
    :param close_figure: Whether to close the figure after saving it.
                         Default is False
    :return: None
    """
    os.makedirs(target_dir, exist_ok=True)

    for file_format in file_formats:
        fig.savefig(os.path.join(target_dir, f"{filename}{file_format}"), bbox_inches="tight")

    if save_as_tikz:
        save_figure_as_tikz_tex_file(fig=fig, target_path=os.path.join(target_dir, f"{filename}_tikz.tex"))

    if close_figure:
        plt.close(fig)
        fig.clf()


def save_figure_as_tikz_tex_file(
    fig: plt.Figure,
    target_path: PathType,
) -> None:
    try:
        tikzplotlib.clean_figure(fig=fig)
        tikzplotlib.save(figure=fig, filepath=target_path, strict=True)
    except Exception as e:
        logging.error(
            f"Exception ({e.__class__.__name__}) occurred in attempt to export plot in tikz raw text format!\n"
            f"The following tikz tex file was not produced.\n\t{target_path}\n"
            f"The following lines show additional information on the {e.__class__.__name__}",
            exc_info=e,
        )


def get_white_or_black_from_background(
    bkg_color: str,
) -> str:
    # See https://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color
    luminance = 0.0  # type: float
    color_factors = (0.2126, 0.7152, 0.0722)  # type: Tuple[float, float, float]
    for color_value, color_factor in zip(mpl_colors.to_rgb(bkg_color), color_factors):
        c_value = color_value / 12.92 if color_value <= 0.03928 else ((color_value + 0.055) / 1.055) ** 2.4  # type: float

        luminance += color_factor * c_value

    return KITColors.kit_black if luminance > 0.179 else KITColors.white


def color_fader(
    color_1: str,
    color_2: str,
    mix: float = 0.0,
) -> str:
    c1 = np.array(mpl_colors.to_rgb(color_1))
    c2 = np.array(mpl_colors.to_rgb(color_2))
    return mpl_colors.to_hex((1 - mix) * c1 + mix * c2)


def _convert_flat_binning_to_axis_hierarchy_and_labels(flat_binning: List[Tuple[float, ...]]):
    """
    Helper function to convert tuple-wise "flat" binning to format needed for hierarchical axes
    """

    right_bin_edge = [len(flat_binning)]
    flat_binning = np.array(flat_binning)

    labels_and_ranges = []
    factor = 1
    for binlevel in range(flat_binning.shape[1] - 1, -1, -1):
        a = flat_binning[:, binlevel]
        ranges, labels = zip(*enumerate(a[np.insert(np.diff(a).astype(np.bool), 0, True)]))

        if labels_and_ranges:
            prev = len(labels_and_ranges[-1][1])
            factor *= int(prev / len(labels))

        labels_and_ranges.append(([r * factor for r in ranges] + right_bin_edge, labels))

    return labels_and_ranges


def add_hierarchical_axes_to_plot(ax, x_binning: Binning, y_binning: Optional[Binning] = None, colors = None):
    """
    This is a somewhat hacky way to add multiple axis levels with color-coding to plots.
    Inspired by this solution:
     https://stackoverflow.com/questions/46507472/matplotlib-correlation-matrix-heatmap-with-grouped-colors-as-labels
    """

    if colors is None:
        colors = KITColors.default_colors

    # create axes next to plot
    divider = make_axes_locatable(ax)

    barkw = dict(linewidth=0.72, ec="k", clip_on=False, align='edge')
    textkw = dict(ha="center", va="center", fontsize="small")

    for xranges, xlabels in _convert_flat_binning_to_axis_hierarchy_and_labels(x_binning.get_flat_list_of_bins()):

        binmids = xranges[:-1] + np.diff(xranges)/2
        xax = divider.append_axes("bottom", "10%", pad=0.06, sharex=ax)
        xax.invert_yaxis()
        xax.axis("off")
        xax.bar(xranges[:-1], np.ones(len(xlabels)),
                width=np.diff(xranges), color=colors[:len(np.unique(xlabels))], **barkw)

        for binmid, xlabel in zip(binmids, xlabels):
            xax.text(binmid, 0.5 , xlabel, **textkw)

    if y_binning is not None:
        for yranges, ylabels in _convert_flat_binning_to_axis_hierarchy_and_labels(y_binning.get_flat_list_of_bins()):

            binmids = yranges[:-1] + np.diff(yranges)/2
            yax = divider.append_axes("left", "10%", pad=0.06, sharey=ax)
            yax.invert_yaxis()
            yax.axis("off")
            yax.barh(yranges[:-1] ,np.ones(len(ylabels)),
                     height=np.diff(yranges), **barkw)

            for binmid, ylabel in zip(binmids, ylabels):
                yax.text(0.5, binmid, ylabel,rotation=-90, **textkw)

    ax.margins(0)
    ax.tick_params(axis="both", bottom=0, left=0, labelbottom=0, labelleft=0)

    return ax


