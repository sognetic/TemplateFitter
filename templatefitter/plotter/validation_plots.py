"""
Provides plot classes for the validation of distributions.
"""

import logging
import collections
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib import font_manager as mpl_font_mgr

from typing import Optional, Union, Tuple, List, Dict

from templatefitter.plotter import plot_style
from templatefitter.plotter.plot_utilities import AxesType
from templatefitter.binned_distributions.binning import Binning
from templatefitter.plotter.histogram_variable import HistVariable

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "BinMigrationPlot",
    "BinCompositionPlot",
]

plot_style.set_matplotlibrc_params()


class BinMigrationPlot:

    def __init__(
            self,
            from_variable: HistVariable,
            to_variable: HistVariable,
            weight_column: Optional[str] = None,
            color_map: Optional[Union[str, mpl_colors.Colormap]] = None,
            from_to_label_appendix: Optional[Tuple[Optional[str], Optional[str]]] = None,
    ) -> None:

        self.from_hist_var = from_variable  # type: HistVariable
        self.to_hist_var = to_variable  # type: HistVariable

        self.weight_column = weight_column  # type: Optional[str]
        self._binning = None  # type: Optional[Binning]

        self.label_appendix_tuple = from_to_label_appendix  # type: Optional[Tuple[Optional[str], Optional[str]]]
        self._color_map = color_map  # type: Optional[Union[str, mpl_colors.Colormap]]

    def plot_on(
            self,
            df: pd.DataFrame,
            ax: Optional[AxesType] = None,
            normalize_to_origin: bool = True,
            show_color_bar: bool = False
    ) -> AxesType:
        if ax is None:
            _, ax = plt.subplots()

        assert self.from_hist_var.df_label in df.columns, self.from_hist_var.df_label
        assert self.to_hist_var.df_label in df.columns, self.to_hist_var.df_label
        assert self.weight_column in df.columns or self.weight_column is None, self.weight_column

        plot_style.set_matplotlibrc_params()

        migration_matrix = self._calculate_bin_migration(
            df=df,
            normalize_to_origin=normalize_to_origin,
        )  # type: np.ndarray

        heatmap = ax.imshow(X=migration_matrix, cmap=self._color_map, aspect="auto")
        if show_color_bar:
            plt.colorbar(heatmap)

        for i in range(self.from_hist_var.n_bins):
            for j in range(self.from_hist_var.n_bins):
                ax.text(j, i, round(migration_matrix[i, j], 2), ha="center", va="center", color="w")

        ax.set_xlabel(xlabel=self.get_axis_label(hist_var=self.to_hist_var), **plot_style.xlabel_pos)
        ax.set_ylabel(ylabel=self.get_axis_label(hist_var=self.from_hist_var), **plot_style.ylabel_pos)

        self._set_axis_tick_labels(ax=ax, migration_matrix_shape=migration_matrix.shape)

        return ax

    def _calculate_bin_migration(
            self,
            df: pd.DataFrame,
            normalize_to_origin: bool = True,
    ) -> np.ndarray:

        from_bins = np.digitize(x=df[self.from_hist_var.df_label].values, bins=self.bin_edges)  # type: np.array
        to_bins = np.digitize(x=df[self.to_hist_var.df_label].values, bins=self.bin_edges)  # type: np.array

        if self.weight_column is None:
            migration_matrix = pd.crosstab(
                index=from_bins,
                columns=to_bins,
                normalize="index" if normalize_to_origin else "columns",
                dropna=False,
            ).values  # type: np.ndarray
        else:
            info_df = pd.DataFrame(data={
                "bin_a": from_bins,
                "bin_b": to_bins,
                "weight": df[self.weight_column].values,
            })  # type: pd.DataFrame
            weight_sum = info_df.groupby(["bin_a", "bin_b"])["weight"].sum().unstack().values  # type: np.ndarray
            np.nan_to_num(x=weight_sum, copy=False)

            norm_denominator = np.sum(a=weight_sum, axis=1 if normalize_to_origin else 0)  # type: np.ndarray
            if normalize_to_origin:
                migration_matrix = weight_sum / norm_denominator[:, np.newaxis]
            else:
                migration_matrix = weight_sum / norm_denominator[np.newaxis, :]

        assert migration_matrix.shape == (self.binning.num_bins_total + 2, self.binning.num_bins_total + 2), (
            migration_matrix.shape,
            self.binning.num_bins_total,
        )
        assert np.isfinite(migration_matrix).all(), np.sum(~np.isfinite(migration_matrix))
        assert np.all(migration_matrix <= 1.0), np.max(migration_matrix)
        assert np.all(migration_matrix >= 0.0), np.min(migration_matrix)

        return migration_matrix

    @property
    def binning(self) -> Binning:
        if self._binning is None:
            assert self.from_hist_var.n_bins == self.to_hist_var.n_bins, (
                self.from_hist_var.n_bins,
                self.to_hist_var.n_bins
            )
            assert self.from_hist_var.scope == self.to_hist_var.scope, (self.from_hist_var.scope, self.to_hist_var.scope)
            assert self.from_hist_var.use_log_scale == self.to_hist_var.use_log_scale, (
                self.from_hist_var.use_log_scale,
                self.to_hist_var.use_log_scale
            )

            self._binning = Binning(
                bins=self.from_hist_var.n_bins,
                dimensions=1,
                scope=self.from_hist_var.scope,
                log_scale=self.from_hist_var.use_log_scale
            )
        return self._binning

    @property
    def bin_edges(self) -> Tuple[float, ...]:
        assert len(self.binning.bin_edges) == 1, self.binning.bin_edges
        return self.binning.bin_edges[0]

    def _set_axis_tick_labels(self, ax: AxesType, migration_matrix_shape: Tuple[int, ...]) -> None:
        tick_positions = np.arange(0, len(self.bin_edges) + 2, 1) - 0.5  # type: np.array
        assert len(tick_positions) - 1 == migration_matrix_shape[0] == migration_matrix_shape[1], (
            len(tick_positions),
            (tick_positions[0], tick_positions[-1]),
            (self.bin_edges[0], self.bin_edges[-1]),
            migration_matrix_shape,
        )
        ax.set_xticks(ticks=tick_positions)
        ax.set_yticks(ticks=tick_positions)
        ax.set_xticklabels([f"{self.bin_edges[int(x + 0.5)]}" for x in ax.get_xticks()[1:-1]], rotation=45)
        ax.set_yticklabels([f"{self.bin_edges[int(y + 0.5)]}" for y in ax.get_yticks()[1:-1]])

    def get_axis_label(self, hist_var: HistVariable) -> str:
        if self.label_appendix_tuple is None:
            appendix = ""  # type: str
        else:
            assert isinstance(self.label_appendix_tuple, tuple), self.label_appendix_tuple
            assert len(self.label_appendix_tuple) == 2, self.label_appendix_tuple
            assert all(isinstance(a, str) or a is None for a in self.label_appendix_tuple), self.label_appendix_tuple
            if hist_var is self.from_hist_var:
                appendix = f" {self.label_appendix_tuple[0]}" if self.label_appendix_tuple[0] is not None else ""
            elif hist_var is self.to_hist_var:
                appendix = f" {self.label_appendix_tuple[1]}" if self.label_appendix_tuple[1] is not None else ""
            else:
                raise RuntimeError(
                    f"Expected to receive either the from_hist_var\n\t{self.from_hist_var.df_label}\n, "
                    f"or the to_hist_var\n\t{self.to_hist_var.df_label}, "
                    f"but received a hist variable based on\n\t{hist_var.df_label}"
                )

        unit = f" in {hist_var.unit}" if hist_var.unit else ""  # type: str
        return f"{hist_var.variable_name}{appendix}{unit}"


class BinCompositionPlot:

    def __init__(
            self,
            variable: HistVariable,
            secondary_variable: HistVariable,
            secondary_variable_binning: Binning,
            weight_column: Optional[str] = None,
            secondary_tick_label_mapping: Optional[Dict[int, str]] = None
    ) -> None:

        self.primary_hist_var = variable  # type: HistVariable
        self.secondary_hist_var, self.secondary_binning = self._init_secondary_var(
            secondary_hist_variable=secondary_variable,
            secondary_variable_binning=secondary_variable_binning,
        )  # type: HistVariable, Binning
        self._secondary_tick_label_mapping = secondary_tick_label_mapping  # type: Optional[Dict[int, str]]

        self.weight_column = weight_column  # type: Optional[str]

        self._primary_binning = None  # type: Optional[Binning]

        self._auto_color_index = 0  # type: int

    def plot_on(
            self,
            df: pd.DataFrame,
            ax: Optional[AxesType] = None,
            draw_legend: bool = True,
    ) -> AxesType:
        if ax is None:
            _, ax = plt.subplots()

        assert self.primary_hist_var.df_label in df.columns, self.primary_hist_var.df_label
        assert self.secondary_hist_var.df_label in df.columns, self.secondary_hist_var.df_label
        assert self.weight_column in df.columns or self.weight_column is None, self.weight_column

        plot_style.set_matplotlibrc_params()

        normed_weights = self._calculate_bin_weights(df=df)  # type: List[np.ndarray]

        ax.hist(
            x=[self.primary_binning.bin_mids[0] for _ in range(self.secondary_binning.num_bins[0])],
            bins=self.primary_bin_edges,
            weights=normed_weights,
            stacked=True,
            edgecolor="black",
            lw=0.3,
            color=[self._get_auto_color() for _ in range(self.secondary_binning.num_bins[0])],
            label=self.secondary_labels,
            histtype="stepfilled"
        )

        ax.set_xlabel(xlabel=self.primary_hist_var.x_label, **plot_style.xlabel_pos)
        ax.set_ylabel(ylabel="Bin Composition", **plot_style.ylabel_pos)

        if draw_legend:
            bbox_to_anchor_tuple = (1.02, 1.)
            legend_font_size = (
                "medium"
                if self.secondary_binning.num_bins_total < 10
                else mpl_font_mgr.FontProperties(size=plt.rcParams["legend.fontsize"]).get_size_in_points() * 0.8
            )

            ax.legend(
                frameon=False,
                loc="upper left",
                ncol=1,
                bbox_to_anchor=bbox_to_anchor_tuple,
                fontsize=legend_font_size,
            )

        return ax

    def _calculate_bin_weights(self, df: pd.DataFrame) -> List[np.ndarray]:
        binned_weights, _, _ = np.histogram2d(
            x=df[self.primary_hist_var.df_label].values,
            y=df[self.secondary_hist_var.df_label].values,
            bins=(self.primary_bin_edges, self.secondary_bin_edges),
            weights=df[self.weight_column].values if self.weight_column is not None else None,
        )

        assert binned_weights.shape == (self.primary_binning.num_bins_total, self.secondary_binning.num_bins_total), (
            binned_weights.shape,
            self.primary_binning.num_bins_total,
            self.secondary_binning.num_bins_total
        )

        weight_sums = np.sum(a=binned_weights, axis=1)  # type: np.ndarray
        normed_weights = binned_weights / weight_sums[:, np.newaxis]  # type: List[np.ndarray]
        normed_weight_list = list(np.transpose(a=normed_weights))  # type: List[np.ndarray]

        assert all(len(w) == self.primary_binning.num_bins_total for w in normed_weight_list), [
            len(w) for w in normed_weight_list
        ]
        assert len(normed_weight_list) == self.secondary_binning.num_bins[0], (
            len(normed_weight_list),
            self.secondary_binning.num_bins
        )

        return normed_weight_list

    @property
    def primary_binning(self) -> Binning:
        if self._primary_binning is None:
            self._primary_binning = Binning(
                bins=self.primary_hist_var.n_bins,
                dimensions=1,
                scope=self.primary_hist_var.scope,
                log_scale=self.primary_hist_var.use_log_scale
            )
        return self._primary_binning

    @property
    def primary_bin_edges(self) -> Tuple[float, ...]:
        assert len(self.primary_binning.bin_edges) == 1, self.primary_binning.bin_edges
        return self.primary_binning.bin_edges[0]

    @property
    def secondary_bin_edge_pairs(self) -> List[Tuple[float, float]]:
        bin_edges = self.secondary_binning.bin_edges[0]
        return [(left, right) for left, right in zip(bin_edges[:-1], bin_edges[1:])]

    @property
    def secondary_bin_edges(self) -> Tuple[float, ...]:
        assert len(self.secondary_binning.bin_edges) == 1, self.secondary_binning.bin_edges
        return self.secondary_binning.bin_edges[0]

    @property
    def secondary_labels(self) -> Union[List[str], Optional[str]]:  # Optional[str] only added for ax.hist requirement.
        if self._secondary_tick_label_mapping is not None:
            sorted_keys = sorted(self._secondary_tick_label_mapping.keys())  # type: List[int]
            sorted_map = [(k, self._secondary_tick_label_mapping[k]) for k in sorted_keys]  # type: List[Tuple[int, str]]
            return list(collections.OrderedDict(sorted_map).values())
        else:
            return [
                rf"{self.secondary_hist_var.variable_name} " + r"$\in (" + f"{b[0]:.4f}" + ", " + f"{b[1]:.4f}" + r")$"
                for b in self.secondary_bin_edge_pairs
            ]

    def _get_auto_color(self):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color = colors[self._auto_color_index % len(colors)]
        self._auto_color_index += 1
        return color

    @staticmethod
    def _init_secondary_var(
            secondary_hist_variable: HistVariable,
            secondary_variable_binning: Optional[Binning],
    ) -> Tuple[HistVariable, Binning]:
        if secondary_variable_binning is None:
            binning = Binning(
                bins=secondary_hist_variable.n_bins,
                dimensions=1,
                scope=secondary_hist_variable.scope,
                log_scale=False,
            )  # type: Binning
        else:
            assert isinstance(secondary_hist_variable, HistVariable), type(secondary_hist_variable).__name__
            binning = secondary_variable_binning

        assert isinstance(binning, Binning), type(binning).__name__

        assert binning.dimensions == 1, binning.dimensions
        assert isinstance(binning.range, tuple), type(binning.range)
        assert len(binning.range) == 1, (binning.range, len(binning.range))
        assert isinstance(binning.range[0], tuple), (type(binning.range[0]), binning.range[0])

        assert secondary_hist_variable.scope == binning.range[0], (secondary_hist_variable.scope, binning.range)
        return secondary_hist_variable, binning
