"""
Contains abstract base class for histogram plots --- HistogramPlot.
"""
import logging
import numpy as np

from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from typing import Optional, Union, Any, Tuple
import copy

from templatefitter.utility import PathType
from templatefitter.binned_distributions.binning import Binning, BinsInputType
from templatefitter.binned_distributions.weights import WeightsInputType
from templatefitter.binned_distributions.systematics import SystematicsInputType
from templatefitter.binned_distributions.binned_distribution import DataInputType, DataColumnNamesInput

from templatefitter.plotter import plot_style
from templatefitter.plotter.histogram_variable import HistVariable
from templatefitter.plotter.plot_utilities import AxesType, FigureType
from templatefitter.plotter.histogram import Histogram, HistogramContainer

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "HistogramPlot",
]

plot_style.set_matplotlibrc_params()


class HistogramPlot(ABC):

    legend_cols_default = 1  # type: int
    legend_loc_default = plt.rcParams["legend.loc"]  # type: Union[int, str]
    legend_font_size_default = plt.rcParams["legend.fontsize"]  # type: Union[int, str]

    def __init__(
        self,
        variable: HistVariable,
    ) -> None:
        self._variable = variable  # type: HistVariable
        self._histograms = HistogramContainer()  # type: HistogramContainer

        self._last_figure = None  # type: Optional[FigureType]
        self.additional_text = None

    @abstractmethod
    def plot_on(self, *args, **kwargs) -> Union[AxesType, Tuple[FigureType, Tuple[AxesType, AxesType]], Any]:
        raise NotImplementedError(f"The 'plot_on' method is not implemented for the class {self.__class__.__name__}!")

    @abstractmethod
    def add_component(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"The 'add_component' method is not implemented for the class {self.__class__.__name__}!"
        )

    def _add_component(
        self,
        label: str,
        histogram_key: str,
        data: DataInputType,
        special_binning: Union[None, BinsInputType, Binning] = None,
        weights: WeightsInputType = None,
        systematics: SystematicsInputType = None,
        hist_type: Optional[str] = None,
        color: Optional[str] = None,
        alpha: float = 1.0,
    ) -> None:
        if histogram_key not in self._histograms.histogram_keys:
            new_histogram = Histogram(variable=self.variable, hist_type=hist_type, special_binning=special_binning)
            self._histograms.add_histogram(key=histogram_key, histogram=new_histogram)

        self._histograms[histogram_key].add_histogram_component(
            label=label,
            data=data,
            weights=weights,
            systematics=systematics,
            data_column_names=self.variable.df_label,
            color=color,
            alpha=alpha,
        )

    def _add_prebinned_component(
        self,
        label: str,
        histogram_key: str,
        bin_counts: np.ndarray,
        original_binning: Binning,
        bin_errors_squared: np.ndarray = None,
        data_column_names: DataColumnNamesInput = None,
        hist_type: Optional[str] = None,
        color: Optional[str] = None,
        alpha: float = 1.0,
    ) -> None:
        if histogram_key not in self._histograms.histogram_keys:
            new_histogram = Histogram(variable=self.variable, hist_type=hist_type, special_binning=original_binning)
            self._histograms.add_histogram(key=histogram_key, histogram=new_histogram)

        if data_column_names is not None:
            if isinstance(data_column_names, str):
                assert data_column_names == self.variable.df_label, (data_column_names, self.variable.df_label)
            elif isinstance(data_column_names, list):
                assert len(data_column_names) == 1, (len(data_column_names), data_column_names)
                assert data_column_names[0] == self.variable.df_label, (data_column_names[0], self.variable.df_label)
            else:
                raise RuntimeError(
                    f"Unexpected type for argument 'data_column_names':\n"
                    f"Should be None, str or List[str], but is {type(data_column_names)}!"
                )

        self._histograms[histogram_key].add_histogram_component(
            label=label,
            bin_counts=bin_counts,
            original_binning=original_binning,
            bin_errors_squared=bin_errors_squared,
            data_column_names=self.variable.df_label,
            color=color,
            alpha=alpha,
        )

    @property
    def binning(self) -> Binning:
        return self._histograms.common_binning

    @property
    def bin_edges(self) -> Tuple[float, ...]:
        assert len(self.binning.bin_edges) == 1, self.binning.bin_edges
        return self.binning.bin_edges[0]

    @property
    def bin_widths(self) -> np.ndarray:
        assert len(self.binning.bin_widths) == 1, self.binning.bin_widths
        return np.array(self.binning.bin_widths[0])

    @property
    def bin_mids(self) -> Tuple[float, ...]:
        assert len(self.binning.bin_mids) == 1, self.binning.bin_mids
        return self.binning.bin_mids[0]

    @property
    def number_of_bins(self) -> int:
        assert len(self.binning.num_bins) == 1, self.binning.num_bins
        return self.binning.num_bins[0]

    @property
    def minimal_bin_width(self) -> float:
        return min(self.bin_widths)

    @property
    def variable(self) -> HistVariable:
        return self._variable

    @property
    def number_of_histograms(self) -> int:
        return self._histograms.number_of_histograms

    def reset_binning_to_use_raw_data_range(self) -> None:
        self._histograms.reset_binning_to_use_raw_data_range_of_all()

    def reset_binning_to_use_raw_data_range_of_histogram(
        self,
        histogram_key: str,
    ) -> None:
        self._histograms.reset_binning_to_use_raw_data_range_of_key(key=histogram_key)

    def apply_adaptive_binning_based_on_histogram(
        self,
        histogram_key: str,
        minimal_bin_count: int = 5,
        minimal_number_of_bins: int = 7,
    ) -> None:
        self._histograms.apply_adaptive_binning_based_on_key(
            key=histogram_key,
            minimal_bin_count=minimal_bin_count,
            minimal_number_of_bins=minimal_number_of_bins,
        )

    def _get_y_label(
        self,
        normed: bool,
        evts_or_cands: str = "Events",
    ) -> str:
        if normed:
            return "Normalized in arb. units"
        elif self._variable.use_log_scale:
            return f"{evts_or_cands} / Bin"
        else:
            return "{e} / {bo}{b:.4g}{v}{bc}".format(
                e=evts_or_cands,
                b=self.minimal_bin_width,
                v=" " + self._variable.unit if self._variable.unit else "",
                bo="(" if self._variable.unit else "",
                bc=")" if self._variable.unit else "",
            )

    def get_bin_info_for_component(
        self,
        component_key: Optional[str] = None,
        data_key: Optional[str] = None,
        normalize_to_data: bool = False,
        include_sys: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if component_key is None:
            component_key = self.mc_key
        if component_key not in self._histograms.histogram_keys:
            raise KeyError(
                f"Histogram key '{component_key}' was not added to the {self.__class__.__name__} "
                f"instance!\nAvailable histogram keys are: {self._histograms.histogram_keys}"
            )

        if normalize_to_data:
            if data_key is None:
                data_key = self.data_key
            if data_key not in self._histograms.histogram_keys:
                raise KeyError(
                    f"Histogram key '{data_key}' was not added to the {self.__class__.__name__} "
                    f"instance!\nAvailable histogram keys are: {self._histograms.histogram_keys}"
                )

        # Making mypy happy with extra np array call
        component_bin_count = np.array(np.sum(np.array(self._histograms[component_key].get_bin_counts()), axis=0))

        if not normalize_to_data:
            norm_factor = None  # type: None
        else:
            norm_factor = (
                self._histograms[data_key].raw_data_size / self._histograms[component_key].raw_weight_sum
            )  # type: float
            component_bin_count *= norm_factor

        component_stat_uncert_sq = self._histograms[component_key].get_statistical_uncertainty_per_bin(
            normalization_factor=norm_factor
        )
        component_uncert_sq = copy.deepcopy(component_stat_uncert_sq)

        if include_sys:
            sys_uncertainty_squared = self._histograms[component_key].get_systematic_uncertainty_per_bin()
            if sys_uncertainty_squared is not None:
                component_uncert_sq += sys_uncertainty_squared

        assert len(component_bin_count.shape) == 1, component_bin_count.shape
        assert component_bin_count.shape[0] == self.number_of_bins, (component_bin_count.shape, self.number_of_bins)
        assert component_bin_count.shape == component_uncert_sq.shape, (
            component_bin_count.shape,
            component_uncert_sq.shape,
        )

        return component_bin_count, component_uncert_sq, component_stat_uncert_sq, norm_factor

    def draw_legend(
        self,
        axis: AxesType,
        inside: bool,
        loc: Optional[Union[int, str]] = None,
        ncols: Optional[int] = None,
        y_axis_scale: Optional[float] = None,
        font_size: Optional[Union[int, float, str]] = None,
        bbox_to_anchor_tuple: Tuple[float, float] = None,
    ) -> None:
        if loc is None:
            loc = self.legend_loc_default
        if ncols is None:
            ncols = self.legend_cols_default
        if font_size is None:
            font_size = self.legend_font_size_default

        if inside:
            axis.legend(frameon=False, loc=loc, ncol=ncols, fontsize=font_size)

            if y_axis_scale is not None:
                y_limits = axis.get_ylim()
                axis.set_ylim(bottom=y_limits[0], top=y_axis_scale * y_limits[1])
        else:
            if bbox_to_anchor_tuple is None:
                bbox_to_anchor_tuple = (1.0, 1.0)

            axis.legend(frameon=False, loc=loc, ncol=ncols, bbox_to_anchor=bbox_to_anchor_tuple)

    def get_last_figure(self) -> Optional[FigureType]:
        return self._last_figure

    def write_hist_data_to_file(self, file_path: PathType):
        self._histograms.write_to_file(file_path=file_path)

    def draw_info_text(self, axis: AxesType, fig: FigureType, this_additional_info_str: str):

        draw_info_text(axis=axis, fig=fig, this_additional_info_str=this_additional_info_str)


def draw_info_text(axis: AxesType, fig: FigureType, this_additional_info_str: str) -> None:

    fig.canvas.draw()  # Figure needs to be drawn so that the relative coordinates can be calculated.

    legend_pos = axis.get_legend().get_window_extent()
    legend_left_lower_edge_pos_in_ax_coords = axis.transAxes.inverted().transform(legend_pos.min)
    axis.text(
        x=legend_left_lower_edge_pos_in_ax_coords[0] * 0.95,
        y=legend_left_lower_edge_pos_in_ax_coords[1],
        s=this_additional_info_str,
        transform=axis.transAxes,
        va="top",
        ha="left",
        linespacing=1.5,
    )
