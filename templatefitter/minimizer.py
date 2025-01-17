"""
Implementation of a minimizer class based on the scipy.optimize.minimize function.
"""

import os
import json
import logging
import tabulate
import functools
import numpy as np
import numdifftools as ndt

from iminuit import Minuit
from abc import ABC, abstractmethod
from scipy.optimize import minimize as scipy_minimize

from typing import Union, Optional, Tuple, List, Dict, Any, Callable, NamedTuple

from templatefitter.utility import cov2corr, PathType

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "MinimizerParameters",
    "AbstractMinimizer",
    "IMinuitMinimizer",
    "ScipyMinimizer",
    "minimizer_factory",
    "MinimizeResult",
    "BoundType",
    "available_template_fitter_minimizer",
]

BoundType = Tuple[Optional[float], Optional[float]]


class MinimizerParameters:
    """
    Container for parameters used by the Minimizer class.
    Maps parameters described as arrays to names and indices.
    Values for parameter values, errors, covariance and correlation
    matrices are only available after they've been set by the
    minimizer.

    Parameters
    ----------
    names : list of str
        List of parameter names.
    """

    def __init__(
        self,
        names: Tuple[str, ...],
        param_types: Tuple[str, ...],
    ) -> None:
        self._names = names  # type: Tuple[str, ...]
        self._param_types = param_types  # type: Tuple[str, ...]
        self._number_of_params = len(names)  # type: int

        assert len(self._param_types) == self._number_of_params, (len(self._param_types), self._number_of_params)

        self._fixed_params = [False for _ in self._names]

        self._values = np.zeros(self._number_of_params)
        self._errors = np.zeros(self._number_of_params)
        self._covariance = np.zeros((self._number_of_params, self._number_of_params))
        self._correlation = np.zeros((self._number_of_params, self._number_of_params))

    def __str__(self) -> str:
        data = {
            "No": list(range(self.num_params)),
            "Name": self._names,
            "Value": self._values,
            "Sym. Err": self.errors,
        }  # type: Dict[str, Any]
        return tabulate.tabulate(data, headers="keys")

    def get_param_value(self, param_id: Union[int, str, np.integer]) -> float:
        """
        Returns value of parameter specified by `param_id`.

        Parameters
        ----------
        param_id : int or str
            Name or index in list of names of wanted parameter

        Returns
        -------
        float
        """
        param_index = self.param_id_to_index(param_id=param_id)
        return self.values[param_index]

    def get_param_error(self, param_id: Union[int, str, np.integer]) -> float:
        """
        Returns error of parameter specified by `param_id`.

        Parameters
        ----------
        param_id : int or str
            Name or index in list of names of wanted parameter

        Returns
        -------
        float
        """
        param_index = self.param_id_to_index(param_id=param_id)
        return self.errors[param_index]

    def __getitem__(
        self,
        param_id: Union[int, str, np.integer],
    ) -> Tuple[float, float]:
        """
        Gets the value and error of the specified parameter.

        Parameters
        ----------
        param_id : int or str
            Parameter index or name.

        Returns
        -------
        float
            Parameter value.
        float
            Parameter error.
        """
        param_index = self.param_id_to_index(param_id=param_id)
        return self.values[param_index], self.errors[param_index]

    def param_id_to_index(self, param_id: Union[int, str, np.integer]) -> int:
        """
        Returns the index of the parameter specified by `param_id`.

        Parameters
        ----------
        param_id : int or str
            Parameter index or name.

        Returns
        -------
        int
        """
        if isinstance(param_id, (int, np.integer)) and (param_id in range(len(self.names))):
            return param_id
        elif isinstance(param_id, str) and (param_id in self.names):
            return self.names.index(param_id)
        else:
            raise ValueError(
                f"Specify the parameter either by its name (as str) or by its index (as int). "
                f"The provided value {param_id} of type {type(param_id)} is not valid!"
            )

    def set_param_fixed(
        self,
        param_id: Union[int, str, np.integer],
    ) -> None:
        param_index = self.param_id_to_index(param_id=param_id)
        self._fixed_params[param_index] = True

    def release_params(self) -> None:
        self._fixed_params = [False for _ in self.names]

    @property
    def names(self) -> Tuple[str, ...]:
        return self._names

    @property
    def parameter_types(self) -> Tuple[str, ...]:
        return self._param_types

    @property
    def num_params(self) -> int:
        """int: Number of parameters."""
        return self._number_of_params

    @property
    def num_params_not_fixed(self) -> int:
        """int: Number of parameters."""
        return self.num_params - sum(self.fixed_params)

    @property
    def values(self) -> np.ndarray:
        """np.ndarray: Parameter values. Shape is (`num_params`,)."""
        return self._values

    @values.setter
    def values(
        self,
        new_values: np.ndarray,
    ) -> None:
        if not len(new_values) == self.num_params:
            raise ValueError(
                f"Number of parameter values must be equal to number of parameters:\n"
                f"\toriginal number of parameters: {self.num_params}\n"
                f"\tnew number of parameters: {len(new_values)}"
            )
        self._values = new_values

    @property
    def errors(self) -> np.ndarray:
        """np.ndarray: Parameter errors. Shape is (`num_params`,)."""
        return self._errors

    @errors.setter
    def errors(
        self,
        new_errors: np.ndarray,
    ) -> None:
        if not len(new_errors) == self.num_params:
            raise ValueError("Number of parameter errors must be equal to number of parameters")
        self._errors = new_errors

    @property
    def covariance(self) -> np.ndarray:
        """ np.ndarray: Parameter covariance matrix. Shape is (`num_params`, `num_params`)."""
        return self._covariance

    @covariance.setter
    def covariance(
        self,
        new_covariance: np.ndarray,
    ) -> None:
        if not new_covariance.shape == (self.num_params_not_fixed, self.num_params_not_fixed):
            raise ValueError(
                f"Shape of new covariance matrix {new_covariance.shape} must match "
                f"the number of not fixed parameters {self.num_params_not_fixed}"
            )
        self._covariance = new_covariance

    @property
    def correlation(self) -> np.ndarray:
        """np.ndarray: Parameter correlation matrix. Shape is  (`num_params`, `num_params`)."""
        return self._correlation

    @correlation.setter
    def correlation(
        self,
        new_correlation: np.ndarray,
    ) -> None:
        if not new_correlation.shape == (self.num_params_not_fixed, self.num_params_not_fixed):
            raise ValueError(
                f"Shape of new correlation matrix {new_correlation.shape} must match "
                f"the number of not fixed parameters {self.num_params_not_fixed}"
            )
        self._correlation = new_correlation

    @property
    def fixed_params(self) -> List[bool]:
        return self._fixed_params

    @staticmethod
    def _dict_key_mapping() -> Dict[str, str]:
        return {
            "names": "names",
            "param_types": "parameter_types",
            "num_params": "number_of_parameters",
            "num_params_not_fixed": "number_of_parameters_not_fixed",
            "fixed_params": "fixed_parameters_map",
            "values": "parameter_values",
            "errors": "parameter_errors",
            "covariance": "covariance_matrix",
            "correlation": "correlation_matrix",
        }

    @property
    def as_dict(self) -> Dict[Any, Any]:
        dict_key_map = self._dict_key_mapping()
        self_as_dict = {
            dict_key_map["names"]: list(self.names),
            dict_key_map["param_types"]: list(self.parameter_types),
            dict_key_map["num_params"]: self.num_params,
            dict_key_map["num_params_not_fixed"]: self.num_params_not_fixed,
            dict_key_map["fixed_params"]: self.fixed_params,
            dict_key_map["values"]: self.values.tolist(),
            dict_key_map["errors"]: self.errors.tolist(),
            dict_key_map["covariance"]: self.covariance.tolist(),
            dict_key_map["correlation"]: self.correlation.tolist(),
        }
        return self_as_dict

    @classmethod
    def initialize_from_dict(
        cls,
        dictionary: Dict[Any, Any],
    ) -> "MinimizerParameters":
        km = MinimizerParameters._dict_key_mapping()
        assert all(key in dictionary.keys() for key in km.values())

        assert len(dictionary[km["names"]]) == dictionary[km["num_params"]]
        instance = cls(
            names=tuple(dictionary[km["names"]]),
            param_types=tuple(dictionary[km["param_types"]]),
        )

        instance._fixed_params = dictionary[km["fixed_params"]]
        assert instance.num_params_not_fixed == dictionary[km["num_params_not_fixed"]]

        assert len(dictionary[km["values"]]) == dictionary[km["num_params"]]
        instance.values = np.array(dictionary[km["values"]])
        assert len(dictionary[km["errors"]]) == dictionary[km["num_params"]]
        instance.errors = np.array(dictionary[km["errors"]])

        assert len(dictionary[km["covariance"]]) == dictionary[km["num_params_not_fixed"]]
        assert all(isinstance(cov_row, list) for cov_row in dictionary[km["covariance"]])
        assert all(len(cov_row) == dictionary[km["num_params_not_fixed"]] for cov_row in dictionary[km["covariance"]])
        instance.covariance = np.array(dictionary[km["covariance"]])

        assert len(dictionary[km["correlation"]]) == dictionary[km["num_params_not_fixed"]]
        assert all(isinstance(cor_row, list) for cor_row in dictionary[km["correlation"]])
        assert all(len(cor_row) == dictionary[km["num_params_not_fixed"]] for cor_row in dictionary[km["correlation"]])
        instance.correlation = np.array(dictionary[km["correlation"]])

        return instance


class MinimizeResult(NamedTuple):
    """NamedTuple storing the minimization results."""

    fcn_min_val: float
    params: MinimizerParameters
    success: bool

    @property
    def as_dict(self) -> Dict[Any, Any]:
        return {
            "fcn_min_val": self.fcn_min_val,
            "params": self.params.as_dict,
            "success": self.success,
        }

    def to_dict(self, drop_matrices: bool = False) -> Dict[Any, Any]:

        if drop_matrices:
            to_drop = ["correlation_matrix", "covariance_matrix"]
        else:
            to_drop = []

        return {
            "fcn_min_val": self.fcn_min_val,
            "params": {k: v for k, v in self.params.as_dict.items() if k not in to_drop},
            "success": self.success,
        }

    def save_to(
        self,
        path: PathType,
        overwrite: bool = False,
    ) -> None:
        if not overwrite and os.path.exists(path=path):
            raise RuntimeError(f"Trying to overwrite existing file {path}, but overwrite is set to False!")
        with open(file=path, mode="w") as fp:
            json.dump(obj=self.as_dict, fp=fp, indent=2)

    @classmethod
    def from_dict(
        cls,
        result_dict: Dict[Any, Any],
    ) -> "MinimizeResult":
        assert isinstance(result_dict, dict), type(result_dict)
        assert all(k in result_dict.keys() for k in ["fcn_min_val", "params", "success"]), result_dict.keys()
        params = MinimizerParameters.initialize_from_dict(dictionary=result_dict["params"])

        instance = cls(
            fcn_min_val=result_dict["fcn_min_val"],
            params=params,
            success=result_dict["success"],
        )
        return instance

    @classmethod
    def load_from(
        cls,
        path: PathType,
    ) -> "MinimizeResult":
        assert os.path.exists(path=path), path
        with open(file=path, mode="r") as fp:
            restored_dict = json.load(fp)

        return cls.from_dict(result_dict=restored_dict)


MinimizeResult.fcn_min_val.__doc__ = """float: Estimated minimum of the objective function."""
MinimizeResult.params.__doc__ = """MinimizerParameters: An instance of the parameters class."""
MinimizeResult.success.__doc__ = """bool: Whether or not the optimizer exited successfully."""


class AbstractMinimizer(ABC):
    def __init__(
        self,
        fcn: Callable,
        param_names: Tuple[str, ...],
        param_types: Tuple[str, ...],
    ) -> None:
        self._fcn = fcn
        self._params = MinimizerParameters(
            names=param_names,
            param_types=param_types,
        )

        # this lists can be different for different minimizer implementations
        self._param_bounds = [(None, None) for _ in self._params.names]  # type: List[BoundType]

        self._fcn_min_val = None

        self._success = None  # type: Optional[bool]
        self._status = None  # type: Optional[str]
        self._message = None  # type: Optional[str]

    @abstractmethod
    def minimize(
        self,
        initial_param_values: np.ndarray,
        verbose: bool = False,
        error_def: float = Minuit.LIKELIHOOD,
        additional_args: Optional[Tuple[Any, ...]] = None,
        get_hesse: bool = True,
        check_success: bool = True,
    ) -> MinimizeResult:
        pass

    def set_param_fixed(self, param_id: Union[int, str]) -> None:
        """
        Fixes specified parameter to it's initial value given in
        `initial_param_values`.

        Parameters
        ----------
        param_id : int or str
            Parameter identifier, which can be it's name or its index
            in `param_names`.
        """
        self.params.set_param_fixed(param_id=param_id)

    def release_params(self) -> None:
        """
        Removes all constraint specified.
        """
        self.params.release_params()

    def set_param_bounds(
        self,
        param_id: Union[int, str],
        bounds: BoundType,
    ) -> None:
        """
        Sets parameter boundaries which constrain the minimization.

        Parameters
        ----------
        param_id : int or str
            Parameter identifier, which can be it's name or its index
            in `param_names`.
        bounds : tuple of float or None
            A tuple specifying the lower and upper boundaries for the
            given parameter. A value of `None` corresponds to no
            boundary.
        """
        param_index = self.params.param_id_to_index(param_id=param_id)
        self._param_bounds[param_index] = bounds

    @property
    def fcn_min_val(self) -> Optional[str]:
        """
        str: Value of the objective function at it's estimated minimum.
        """
        return self._fcn_min_val

    @property
    def params(self) -> MinimizerParameters:
        """
        MinimizerParameters: Instance of the MinimizerParameters class. Stores the parameter values,
        errors, covariance and correlation matrix.
        """
        return self._params

    @property
    def param_values(self) -> np.ndarray:
        """
        np.ndarray: Estimated parameter values at the minimum of fcn.
        Shape is (`num_params`).
        """
        return self._params.values

    @property
    def param_errors(self) -> np.ndarray:
        """
        np.ndarray: Estimated parameter values at the minimum of fcn.
        Shape is (`num_params`).
        """
        return self._params.errors

    @property
    def param_covariance(self) -> np.ndarray:
        """
        np.ndarray: Estimated covariance matrix of the parameters.
        Calculated from the inverse of the Hesse matrix of fcn evaluated
        at it's minimum. Shape is (`num_params`, `num_params`).
        """
        return self._params.covariance

    @property
    def param_correlation(self) -> np.ndarray:
        """
        np.ndarray: Estimated correlation matrix of the parameters.
        Shape is (`num_params`, `num_params`).
        """
        return self._params.correlation


class IMinuitMinimizer(AbstractMinimizer):
    def __init__(
        self,
        fcn: Callable,
        param_names: Tuple[str, ...],
        param_types: Tuple[str, ...],
    ) -> None:
        super().__init__(
            fcn=fcn,
            param_names=param_names,
            param_types=param_types,
        )

        self._minuit_obj = None  # type: Optional[Minuit]

    def reset(self, initial_param_values):
        self._minuit_obj.reset()
        self._minuit_obj.values = initial_param_values
        for i in range(len(self._minuit_obj.params)):
            self._minuit_obj.fixed[i] = self._get_fixed_params()[i]

    def _create_minuit_obj(
        self,
        initial_param_values: np.ndarray,
        verbose: bool = False,
        error_def: float = Minuit.LIKELIHOOD,
    ):

        self._minuit_obj = Minuit(
            self._fcn,
            initial_param_values,
            name=self.params.names,
        )

        self._minuit_obj.strategy = 2
        self._minuit_obj.errors = 0.05 * initial_param_values
        self._minuit_obj.errordef = error_def
        self._minuit_obj.limits = self._param_bounds

        for i in range(len(self._minuit_obj.params)):
            self._minuit_obj.fixed[i] = self._get_fixed_params()[i]

        self._minuit_obj.print_level = 1 if verbose else 0

    def minimize(
        self,
        initial_param_values: np.ndarray,
        verbose: bool = False,
        error_def: float = Minuit.LIKELIHOOD,
        additional_args: Optional[Tuple[Any, ...]] = None,
        get_hesse: bool = True,
        check_success: bool = True,
    ) -> MinimizeResult:

        if error_def not in [Minuit.LIKELIHOOD, Minuit.LEAST_SQUARES]:
            raise RuntimeWarning(
                f"Parameter error_def, which is set to {error_def}, "
                f"should be either 'Minuit.LIKELIHOOD' = {Minuit.LIKELIHOOD} "
                f"or 'Minuit.LEAST_SQUARES' = {Minuit.LEAST_SQUARES}!"
            )

        if self._minuit_obj is None:
            self._create_minuit_obj(initial_param_values=initial_param_values, verbose=verbose, error_def=error_def)
        else:
            self.reset(initial_param_values=initial_param_values)

        m = self._minuit_obj  # type: Minuit

        success = False
        for attempt in range(1, 5):
            # perform minimization at least twice!
            fmin = m.migrad(ncall=600_000 * attempt, iterate=2 + attempt).fmin
            if get_hesse:
                m.hesse()
                success = fmin.is_valid and fmin.has_valid_parameters and fmin.has_covariance and fmin.has_accurate_covar
            else:
                success = fmin.is_valid and fmin.has_valid_parameters and fmin.has_covariance

            if success or not check_success:
                if attempt > 1:
                    logging.warning(f"Fit successful after retrying {attempt} times.")
                break
            else:
                logging.warning(f"Minimum is inadequate, trying again w/ more iterations/calls (attempt {attempt + 1})")
                logging.warning(
                    f"fmin.is_valid={fmin.is_valid} "
                    f"and fmin.has_valid_parameters={fmin.has_valid_parameters} "
                    f"and fmin.has_covariance={fmin.has_covariance} "
                    f"and fmin.has_accurate_covar={fmin.has_accurate_covar}"
                )

        self._fcn_min_val = m.fval
        self._params.values = np.array(m.values)
        self._params.errors = np.array(m.errors)

        if fmin.has_covariance and m.covariance is not None and get_hesse:
            fixed_params = tuple(~np.array(self._get_fixed_params()))  # type: Tuple[bool, ...]
            self._params.covariance = np.array(m.covariance)[fixed_params, :][:, fixed_params]
            self._params.correlation = np.array(m.covariance.correlation())[fixed_params, :][:, fixed_params]

        self._success = success

        success_text = (
            f"valid minimum: {fmin.is_valid}\n"
            f"valid parameters: {fmin.has_valid_parameters}\n"
            f"covariance exists: {fmin.has_covariance}\n"
            f"covariance is accurate: {fmin.has_accurate_covar}\n"
        )

        if check_success and not self._success:
            raise RuntimeError(f"Minimization was not successful.\n" f"{fmin}\n" + success_text)

        assert self._success is not None
        return MinimizeResult(fcn_min_val=m.fval, params=self._params, success=self._success)

    def _get_fixed_params(self) -> List[bool]:
        return self.params.fixed_params


class ScipyMinimizer(AbstractMinimizer):
    """
    General wrapper class around scipy.optimize.minimize
    function. Allows mapping of parameter names to the array
    entries used by scipy's `minimize` function.

    Parameters
    ----------
    fcn : callable
        Objective function to be minimized of type ``fun(x, *args)``
        where `x` is an np.ndarray of shape (`n`,) and args is a tuple
        of fixed parameters.
    param_names : list of str
        A list of parameter names. This maps the entries from the `x`
        argument of `fcn` to strings.
    """

    def __init__(
        self,
        fcn: Callable,
        param_names: Tuple[str, ...],
        param_types: Tuple[str, ...],
    ) -> None:
        super().__init__(
            fcn=fcn,
            param_names=param_names,
            param_types=param_types,
        )

    def minimize(
        self,
        initial_param_values: np.ndarray,
        verbose: bool = False,
        error_def: float = 0.5,
        additional_args: Optional[Tuple[Any, ...]] = None,
        get_hesse: bool = True,
        check_success: bool = True,
    ) -> MinimizeResult:
        """
        Performs minimization of given objective function.

        Parameters
        ----------
        initial_param_values : np.ndarray or list of floats
            Initial values for the parameters used as starting values.
            Shape is (`num_params`,).
        error_def : Not used for this implementation
        additional_args : tuple
            Tuple of additional arguments for the objective function.
        get_hesse : bool
            If True, the Hesse matrix is estimated at the minimum
            of the objective function. This allows the calculation
            of parameter errors. Default is True.
        verbose: bool
            If True, a convergence message is printed. Default is False.
        check_success: bool
            Check if fit was successful and raise if not! Default: True

        Returns
        -------
        MinimizeResult
        """
        constraints = self._create_constraints(initial_param_values)

        _additional_args = tuple([])  # type: Tuple[Any, ...]
        if additional_args is not None:
            _additional_args = additional_args

        logging.info(
            f"Starting SciPy minimization with {sum(~np.array(self.params.fixed_params))} floating and {sum(self.params.fixed_params)} fixed parameters."
        )
        opt_result = scipy_minimize(
            fun=self._fcn,
            x0=initial_param_values,
            args=additional_args,
            method="SLSQP",
            bounds=self._param_bounds,
            constraints=constraints,  # type: ignore
            options={"disp": verbose},
        )

        logging.info(
            f"Finished SciPy minimization with success = {opt_result.success}, status {opt_result.status} and message {opt_result.message}."
        )

        self._success = opt_result.success
        self._status = opt_result.status
        self._message = opt_result.message

        if not opt_result.success:
            raise RuntimeError(
                "Minimization was not successful.\n",
                f"Status: {opt_result.status}\n",
                f"Message: {opt_result.message}",
            )

        self._params.values = opt_result.x
        self._fcn_min_val = opt_result.fun

        if get_hesse:
            logging.info("Calculating Hesse Matrix for SciPy minimization.")
            hesse = ndt.Hessian(self._fcn)(self._params.values, *_additional_args)
            self._params.covariance = np.linalg.inv(hesse)
            self._params.correlation = cov2corr(self._params.covariance)
            self._params.errors = np.sqrt(np.diag(self._params.covariance))
            logging.info("Finished calculation of Hesse Matrix for SciPy minimization.")

        if verbose:
            print(self._params)

        if check_success:
            pass  # TODO

        assert self._success is not None
        result = MinimizeResult(fcn_min_val=opt_result.fun, params=self._params, success=self._success)

        return result

    def _get_fixed_params(self) -> List[int]:
        return [p_id for p_id, fixed in enumerate(self.params.fixed_params) if fixed]

    def _create_constraints(
        self,
        initial_param_values: np.ndarray,
    ) -> List[Dict[str, Union[str, Callable]]]:
        """
        Creates the dictionary used by scipy's minimize function
        to constrain parameters. The dictionary is used to fix
        parameters specified set in `fixed_param`.

        Parameters
        ----------
        initial_param_values : np.ndarray or list of floats
            Initial parameter values.

        Returns
        -------
        list of dict
            A list of dictionaries, which is passed to scipy's
            minimize function.
        """
        constraints = list()  # type: List[Dict[str, Union[str, Callable]]]

        for fixed_param in self._get_fixed_params():
            _constraint_info = {
                "type": "eq",
                "fun": lambda x: x[fixed_param] - initial_param_values[fixed_param],
            }  # type: Dict[str, Union[str, Callable]]
            constraints.append(_constraint_info)

        return constraints

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def calculate_hesse_matrix(
        fcn: Callable,
        x: np.ndarray,
        args: Tuple[Any],
    ) -> np.ndarray:
        """
        Calculates the Hesse matrix of callable `fcn` numerically.

        Parameters
        ----------
        fcn : callable
            Objective function of type ``fun(x, *args)``.
        x : np.ndarray
            Parameters of `fcn` as np.ndarray of shape (`num_params`,).
        args : tuple
            Additional arguments for `fcn`.

        Returns
        -------
        np.ndarray
            Hesse matrix of `fcn` at point x. Shape is (`num_params`, `num_params`).
        """
        return ndt.Hessian(fcn)(x, *args)


available_template_fitter_minimizer = {
    "scipy": ScipyMinimizer,
    "iminuit": IMinuitMinimizer,
}


def minimizer_factory(
    minimizer_id: str,
    fcn: Callable,
    names: Tuple[str, ...],
    param_types: Tuple[str, ...],
) -> AbstractMinimizer:
    return available_template_fitter_minimizer[minimizer_id.lower()](
        fcn=fcn,
        param_names=names,
        param_types=param_types,
    )
