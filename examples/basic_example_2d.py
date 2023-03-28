#!/usr/bin/env python
# coding: utf-8

"""
This is a basic example of a template fit with two templates in a single fit dimension and channel.
"""

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Tuple, Dict

from templatefitter.fit_model.template import Template
from templatefitter.fit_model.model_builder import FitModel
from templatefitter.fit_model.parameter_handler import ParameterHandler

from templatefitter.fitter import TemplateFitter
from templatefitter.minimizer import MinimizeResult

from templatefitter.plotter.plot_style import KITColors
from templatefitter.plotter.histogram_variable import HistVariable
from templatefitter.plotter.fit_result_plots import FitResultPlotter, BinNuisancePullPlotter
from templatefitter.plotter.fit_template_plots import FitTemplatesPlotter


def gen_2d_sig(
    size: int, voi: Tuple[str, str], voi_limits: Tuple[Tuple[float, float], Tuple[float, float]]
) -> pd.DataFrame:

    loc1 = (voi_limits[0][1] + voi_limits[0][0]) / 2
    loc2 = (voi_limits[1][1] + voi_limits[1][0]) / 2

    variable_values = np.random.normal([loc1, loc2], 1, [size, 2])
    df = pd.DataFrame(
        {voi[0]: variable_values[:, 0], voi[1]: variable_values[:, 1], "weight": np.ones_like(variable_values[:, 0])}
    )
    df.name = "signal"
    return df


def gen_2d_bkg(
    size: int, voi: Tuple[str, str], voi_limits: Tuple[Tuple[float, float], Tuple[float, float]], offset=3
) -> pd.DataFrame:

    loc1 = (voi_limits[0][1] + voi_limits[0][0]) / 2 + offset
    loc2 = (voi_limits[1][1] + voi_limits[1][0]) / 2 + offset

    variable_values = np.random.normal([loc1, loc2], 1, [size, 2])
    df = pd.DataFrame(
        {voi[0]: variable_values[:, 0], voi[1]: variable_values[:, 1], "weight": np.ones_like(variable_values[:, 0])}
    )
    df.name = "background"
    return df


def run_basic_example() -> Tuple[MinimizeResult, Dict[str, float]]:

    # Defining a variable and range for which we will generate data
    voi = ("firstVariableOfInterest", "secondVariableOfInterest")  # type: Tuple[str, str]
    voi_physics_limits = ((0, 20), (0, 10))  # type: Tuple[Tuple[float, float], Tuple[float, float]]

    # Defining a HistVariable object for the fit to use.
    # Note: The scope / fit range used here can be (and is) smaller than the one used for generating the dataset.
    # This is done for illustration here but can sometimes be beneficial.
    # The fitter will later correct the yields, so they apply to the entire dataset.
    fit_var_1 = HistVariable(df_label=voi[0], n_bins=20, scope=(0, 25), var_name="Variable of Interest 1")
    fit_var_2 = HistVariable(df_label=voi[1], n_bins=20, scope=(0, 15), var_name="Variable of Interest 2")
    fit_vars = [fit_var_1, fit_var_2]

    # Defining a decay channel and two components from which templates will be generated
    channel_name = "XtoYDecay"
    components = [gen_2d_sig(1000, voi, voi_physics_limits), gen_2d_bkg(3000, voi, voi_physics_limits)]
    component_colors = {"signal": KITColors.kit_red, "background": KITColors.light_grey}

    # How much simulated data is there in each component?
    initial_yield_dict = {component.name: component.loc[:, "weight"].sum() for component in components}

    # Which fraction is within our fit window?
    # This value is later used to extrapolate the fit yield to the entire variable range.
    initial_eff_dict = {
        component.name: component.loc[
            component[fit_var_1.df_label].between(*fit_var_1.scope)
            & component[fit_var_2.df_label].between(*fit_var_2.scope),
            "weight",
        ].sum()
        / initial_yield_dict[component.name]
        for component in components
    }

    # Defining a ParameterHandler which holds all model parameters
    # and a FitModel object which describes the rest of the fit
    param_handler = ParameterHandler()
    model = FitModel(parameter_handler=param_handler)

    # Giving the fit model some yield and efficiency parameters. We'll not fit the efficiency parameter in this example,
    # to do this we would need additional channels
    for component in components:
        model.add_model_parameter(
            name=f"yield_{component.name}",
            parameter_type=ParameterHandler.yield_parameter_type,
            floating=True,
            initial_value=initial_yield_dict[component.name],
        )

        model.add_model_parameter(
            name=f"eff_{channel_name}_{component.name}",
            parameter_type=ParameterHandler.efficiency_parameter_type,
            floating=False,
            initial_value=initial_eff_dict[component.name],
        )

    # First creating some templates, then adding them to the template
    templates = []
    for component_number, component in enumerate(components):
        template = Template(
            name=f"{channel_name}_{component.name}",
            latex_label=component.name,
            process_name=component.name,
            color=component_colors[component.name],
            dimensions=2,
            bins=tuple([fv.n_bins for fv in fit_vars]),
            scope=tuple([fv.scope for fv in fit_vars]),
            params=param_handler,
            data_column_names=[fv.df_label for fv in fit_vars],
            data=component,
            weights="weight",
        )

        templates.append(template)

        model.add_template(template=template, yield_parameter=f"yield_{component.name}")

    # Now we have to tell the fit model how to relate templates and efficiency parameters (together as a channel)
    model.add_channel(
        efficiency_parameters=[f"eff_{channel_name}_{component.name}" for component in components],
        name=channel_name,
        components=templates,
    )

    # Instead of real data we'll add some Asimov data to the fitter. This is equivalent to the sum of all templates.
    model.add_asimov_data_from_templates()

    # Lastly we'll finalize the model. This finalized model could now be saved with Python's pickle serializer.
    model.finalize_model()

    # Now that we're done creating the model, we can see how data and MC agreed before the fit.
    # Agreement should be perfect in a fit to Asimov data.
    # To do this, we first have to set up a FitResultPlotter.

    output_folder = Path("./plots")
    output_folder.mkdir(exist_ok=True)
    fit_result_plotter_m2 = FitResultPlotter(reference_dimension=0, variables_by_channel=tuple(fit_vars), fit_model=model)

    # This also returns the path of the plot files that are created
    fit_result_plotter_m2.plot_fit_result(use_initial_values=True, output_dir_path=output_folder, output_name_tag="")

    fit_template_plotter_m2 = FitTemplatesPlotter(variables_by_channel=tuple(fit_vars), fit_model=model)
    fit_template_plotter_m2.plot_2d_templates(use_initial_values=True, output_dir_path=output_folder, output_name_tag="")
    fit_template_plotter_m2.plot_2d_contours(use_initial_values=True, output_dir_path=output_folder, output_name_tag="")
    quit()

    # Here we'll now set up a TemplateFitter object and perform the fit itself.
    # We'll use nuisance parameter for each bin and each template in the model.
    fitter = TemplateFitter(fit_model=model, minimizer_id="iminuit")
    result = fitter.do_fit(update_templates=True, get_hesse=True, verbose=False, fix_nui_params=False)

    # Now let's plot the fit results
    fit_result_plotter_m2.plot_fit_result(output_dir_path=output_folder, output_name_tag="")

    # We can now also do a plot of the bin nuisance parameter pulls
    pull_plotter = BinNuisancePullPlotter(fit_model=model, minimize_result=result, variables_by_channel=tuple(fit_vars))
    pull_plotter.plot_bin_nuisance_parameter(
        output_dir_path=output_folder, output_name_tag="", separate_plots_for_components=True
    )

    # Lastly, we can have a look at the significance of the signal + background hypothesis
    # vs. the background only hypothesis.
    # Note: You might get a warning from numpy which you can safely ignore here.
    significance_dict = {}
    for yield_parameter in param_handler.get_parameter_names_for_type(ParameterHandler.yield_parameter_type):
        significance_dict[yield_parameter] = fitter.get_significance(
            yield_parameter=yield_parameter,
            fix_nui_params=False,
            verbose=False,
            catch_exception=True,
        )

    return result, significance_dict


if __name__ == "__main__":
    fit_result, sign_dict = run_basic_example()

    print(f"The result of the fit is: \n {str(fit_result.params)} \n")

    print(
        f"The significance of the signal + background hypothesis vs. only background "
        f"is {sign_dict['yield_signal']:.1f} sigma."
    )
