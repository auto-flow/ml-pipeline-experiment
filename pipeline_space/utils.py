#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-22
# @Contact    : qichun.tang@bupt.edu.cn
import hashlib
import json
from collections import deque
from copy import deepcopy
from typing import Optional, Dict, List, Tuple

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter, Constant, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter


def get_chunks(seq, chunks=1):
    N = len(seq)
    item_size = N // chunks
    remainder = N % chunks
    itv_list = [item_size] * chunks
    for i in range(remainder):
        itv_list[i] += 1
    ans = []
    idx = 0
    end = 0
    for itv in itv_list:
        end += itv
        ans.append(seq[idx:end])
        if end >= N:
            break
        idx = end
    if len(ans) < chunks:
        ans.extend([[] for _ in range(chunks - len(ans))])
    return ans


def generate_grid(configuration_space: ConfigurationSpace,
                  num_steps_dict: Optional[Dict[str, int]] = None,
                  ) -> List[Configuration]:
    """
    Generates a grid of Configurations for a given ConfigurationSpace.
    Can be used, for example, for grid search.

    Parameters
    ----------
    configuration_space: :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
        The Configuration space over which to create a grid of HyperParameter Configuration values.
        It knows the types for all parameter values.

    num_steps_dict: dict
        A dict containing the number of points to divide the grid side formed by Hyperparameters
        which are either of type UniformFloatHyperparameter or type UniformIntegerHyperparameter.
        The keys in the dict should be the names of the corresponding Hyperparameters and the values
        should be the number of points to divide the grid side formed by the corresponding
        Hyperparameter in to.

    Returns
    -------
    list
        List containing Configurations. It is a cartesian product of tuples of
        HyperParameter values.
        Each tuple lists the possible values taken by the corresponding HyperParameter.
        Within the cartesian product, in each element, the ordering of HyperParameters is the same
        for the OrderedDict within the ConfigurationSpace.
    """

    def get_value_set(num_steps_dict: Optional[Dict[str, int]], hp_name: str):
        '''
        Gets values along the grid for a particular hyperparameter.
        Uses the num_steps_dict to determine number of grid values for UniformFloatHyperparameter
        and UniformIntegerHyperparameter. If these values are not present in num_steps_dict, the
        quantization factor, q, of these classes will be used to divide the grid. NOTE: When q
        is used if it is None, a ValueError is raised.
        Parameters
        ----------
        num_steps_dict: dict
            Same description as above
        hp_name: str
            Hyperparameter name
        Returns
        -------
        tuple
            Holds grid values for the given hyperparameter
        '''
        param = configuration_space.get_hyperparameter(hp_name)
        if isinstance(param, (CategoricalHyperparameter)):
            return param.choices

        elif isinstance(param, (OrdinalHyperparameter)):
            return param.sequence

        elif isinstance(param, Constant):
            return tuple([param.value, ])

        elif isinstance(param, UniformFloatHyperparameter):
            if param.log:
                lower, upper = np.log([param.lower, param.upper])
            else:
                lower, upper = param.lower, param.upper

            if num_steps_dict is not None and param.name in num_steps_dict:
                num_steps = num_steps_dict[param.name]
                grid_points = np.linspace(lower, upper, num_steps)
            else:
                # check for log and for rounding issues
                if param.q is not None:
                    grid_points = np.arange(lower, upper + param.q, param.q)
                else:
                    raise ValueError(
                        "num_steps_dict is None or doesn't contain the number of points"
                        f" to divide {param.name} into. And its quantization factor "
                        "is None. Please provide/set one of these values."
                    )

            if param.log:
                grid_points = np.exp(grid_points)

            # Avoiding rounding off issues
            if grid_points[0] < param.lower:
                grid_points[0] = param.lower
            if grid_points[-1] > param.upper:
                grid_points[-1] = param.upper

            return tuple(grid_points)

        elif isinstance(param, UniformIntegerHyperparameter):
            if param.log:
                lower, upper = np.log([param.lower, param.upper])
            else:
                lower, upper = param.lower, param.upper

            if num_steps_dict is not None and param.name in num_steps_dict:
                num_steps = num_steps_dict[param.name]
                grid_points = np.linspace(lower, upper, num_steps)
            else:
                # check for log and for rounding issues
                if param.q is not None:
                    grid_points = np.arange(lower, upper + param.q, param.q)
                else:
                    raise ValueError(
                        "num_steps_dict is None or doesn't contain the number of points "
                        f"to divide {param.name} into. And its quantization factor "
                        "is None. Please provide/set one of these values."
                    )

            if param.log:
                grid_points = np.exp(grid_points)
            grid_points = grid_points.astype(int)

            # Avoiding rounding off issues
            if grid_points[0] < param.lower:
                grid_points[0] = param.lower
            if grid_points[-1] > param.upper:
                grid_points[-1] = param.upper

            return tuple(grid_points)

        else:
            raise TypeError("Unknown hyperparameter type %s" % type(param))

    def get_cartesian_product(value_sets: List[Tuple], hp_names: List[str]):
        '''
        Returns a grid for a subspace of the configuration with given hyperparameters
        and their grid values.
        Takes a list of tuples of grid values of the hyperparameters and list of
        hyperparameter names. The outer list iterates over the hyperparameters corresponding
        to the order in the list of hyperparameter names.
        The inner tuples contain grid values of the hyperparameters for each hyperparameter.
        Parameters
        ----------
        value_sets: list of tuples
            Same description as return value of get_value_set()
        hp_names: list of strs
            List of hyperparameter names
        Returns
        -------
        list of dicts
            List of configuration dicts
        '''
        grid = []
        import itertools
        if len(value_sets) == 0:
            # Edge case
            pass
        else:
            for element in itertools.product(*value_sets):
                config_dict = {}
                for j, hp_name in enumerate(hp_names):
                    config_dict[hp_name] = element[j]
                grid.append(config_dict)

        return grid

    # list of tuples: each tuple within is the grid values to be taken on by a Hyperparameter
    value_sets = []
    hp_names = []

    # Get HP names and allowed grid values they can take for the HPs at the top
    # level of ConfigSpace tree
    for hp_name in configuration_space._children['__HPOlib_configuration_space_root__']:
        value_sets.append(get_value_set(num_steps_dict, hp_name))
        hp_names.append(hp_name)

    # Create a Cartesian product of above allowed values for the HPs. Hold them in an
    # "unchecked" deque because some of the conditionally dependent HPs may become active
    # for some of the elements of the Cartesian product and in these cases creating a
    # Configuration would throw an Error (see below).
    # Creates a deque of Configuration dicts
    unchecked_grid_pts = deque(get_cartesian_product(value_sets, hp_names))
    checked_grid_pts = []

    while len(unchecked_grid_pts) > 0:
        try:
            grid_point = Configuration(configuration_space, unchecked_grid_pts[0])
            checked_grid_pts.append(grid_point)
        except ValueError as e:
            assert (str(e)[:23] == "Active hyperparameter '" and
                    str(e)[-16:] == "' not specified!"), \
                "Caught exception contains unexpected message."
            value_sets = []
            hp_names = []
            new_active_hp_names = []

            # "for" loop over currently active HP names
            for hp_name in unchecked_grid_pts[0]:
                value_sets.append(tuple([unchecked_grid_pts[0][hp_name], ]))
                hp_names.append(hp_name)
                # Checks if the conditionally dependent children of already active
                # HPs are now active
                for new_hp_name in configuration_space._children[hp_name]:
                    if (
                            new_hp_name not in new_active_hp_names and
                            new_hp_name not in unchecked_grid_pts[0]
                    ):
                        all_cond_ = True
                        for cond in configuration_space._parent_conditions_of[new_hp_name]:
                            if not cond.evaluate(unchecked_grid_pts[0]):
                                all_cond_ = False
                        if all_cond_:
                            new_active_hp_names.append(new_hp_name)

            for hp_name in new_active_hp_names:
                value_sets.append(get_value_set(num_steps_dict, hp_name))
                hp_names.append(hp_name)
            # this check might not be needed, as there is always going to be a new
            # active HP when in this except block?
            if len(new_active_hp_names) > 0:
                new_conditonal_grid = get_cartesian_product(value_sets, hp_names)
                unchecked_grid_pts += new_conditonal_grid
            else:
                raise RuntimeError(
                    "Unexpected error: There should have been a newly activated hyperparameter"
                    f" for the current configuration values: {str(unchecked_grid_pts[0])}. "
                    "Please contact the developers with the code you ran and the stack trace."
                )
        unchecked_grid_pts.popleft()

    return checked_grid_pts


def generate_grid_yield(configuration_space: ConfigurationSpace,
                        num_steps_dict: Optional[Dict[str, int]] = None,
                        ) -> List[Configuration]:
    """
    Generates a grid of Configurations for a given ConfigurationSpace.
    Can be used, for example, for grid search.

    Parameters
    ----------
    configuration_space: :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
        The Configuration space over which to create a grid of HyperParameter Configuration values.
        It knows the types for all parameter values.

    num_steps_dict: dict
        A dict containing the number of points to divide the grid side formed by Hyperparameters
        which are either of type UniformFloatHyperparameter or type UniformIntegerHyperparameter.
        The keys in the dict should be the names of the corresponding Hyperparameters and the values
        should be the number of points to divide the grid side formed by the corresponding
        Hyperparameter in to.

    Returns
    -------
    list
        List containing Configurations. It is a cartesian product of tuples of
        HyperParameter values.
        Each tuple lists the possible values taken by the corresponding HyperParameter.
        Within the cartesian product, in each element, the ordering of HyperParameters is the same
        for the OrderedDict within the ConfigurationSpace.
    """

    def get_value_set(num_steps_dict: Optional[Dict[str, int]], hp_name: str):
        '''
        Gets values along the grid for a particular hyperparameter.
        Uses the num_steps_dict to determine number of grid values for UniformFloatHyperparameter
        and UniformIntegerHyperparameter. If these values are not present in num_steps_dict, the
        quantization factor, q, of these classes will be used to divide the grid. NOTE: When q
        is used if it is None, a ValueError is raised.
        Parameters
        ----------
        num_steps_dict: dict
            Same description as above
        hp_name: str
            Hyperparameter name
        Returns
        -------
        tuple
            Holds grid values for the given hyperparameter
        '''
        param = configuration_space.get_hyperparameter(hp_name)
        if isinstance(param, (CategoricalHyperparameter)):
            return param.choices

        elif isinstance(param, (OrdinalHyperparameter)):
            return param.sequence

        elif isinstance(param, Constant):
            return tuple([param.value, ])

        elif isinstance(param, UniformFloatHyperparameter):
            if param.log:
                lower, upper = np.log([param.lower, param.upper])
            else:
                lower, upper = param.lower, param.upper

            if num_steps_dict is not None and param.name in num_steps_dict:
                num_steps = num_steps_dict[param.name]
                grid_points = np.linspace(lower, upper, num_steps)
            else:
                # check for log and for rounding issues
                if param.q is not None:
                    grid_points = np.arange(lower, upper + param.q, param.q)
                else:
                    raise ValueError(
                        "num_steps_dict is None or doesn't contain the number of points"
                        f" to divide {param.name} into. And its quantization factor "
                        "is None. Please provide/set one of these values."
                    )

            if param.log:
                grid_points = np.exp(grid_points)

            # Avoiding rounding off issues
            if grid_points[0] < param.lower:
                grid_points[0] = param.lower
            if grid_points[-1] > param.upper:
                grid_points[-1] = param.upper

            return tuple(grid_points)

        elif isinstance(param, UniformIntegerHyperparameter):
            if param.log:
                lower, upper = np.log([param.lower, param.upper])
            else:
                lower, upper = param.lower, param.upper

            if num_steps_dict is not None and param.name in num_steps_dict:
                num_steps = num_steps_dict[param.name]
                grid_points = np.linspace(lower, upper, num_steps)
            else:
                # check for log and for rounding issues
                if param.q is not None:
                    grid_points = np.arange(lower, upper + param.q, param.q)
                else:
                    raise ValueError(
                        "num_steps_dict is None or doesn't contain the number of points "
                        f"to divide {param.name} into. And its quantization factor "
                        "is None. Please provide/set one of these values."
                    )

            if param.log:
                grid_points = np.exp(grid_points)
            grid_points = grid_points.astype(int)

            # Avoiding rounding off issues
            if grid_points[0] < param.lower:
                grid_points[0] = param.lower
            if grid_points[-1] > param.upper:
                grid_points[-1] = param.upper

            return tuple(grid_points)

        else:
            raise TypeError("Unknown hyperparameter type %s" % type(param))

    def get_cartesian_product(value_sets: List[Tuple], hp_names: List[str]):
        '''
        Returns a grid for a subspace of the configuration with given hyperparameters
        and their grid values.
        Takes a list of tuples of grid values of the hyperparameters and list of
        hyperparameter names. The outer list iterates over the hyperparameters corresponding
        to the order in the list of hyperparameter names.
        The inner tuples contain grid values of the hyperparameters for each hyperparameter.
        Parameters
        ----------
        value_sets: list of tuples
            Same description as return value of get_value_set()
        hp_names: list of strs
            List of hyperparameter names
        Returns
        -------
        list of dicts
            List of configuration dicts
        '''
        grid = []
        import itertools
        if len(value_sets) == 0:
            # Edge case
            pass
        else:
            for element in itertools.product(*value_sets):
                config_dict = {}
                for j, hp_name in enumerate(hp_names):
                    config_dict[hp_name] = element[j]
                grid.append(config_dict)

        return grid

    # list of tuples: each tuple within is the grid values to be taken on by a Hyperparameter
    value_sets = []
    hp_names = []

    # Get HP names and allowed grid values they can take for the HPs at the top
    # level of ConfigSpace tree
    for hp_name in configuration_space._children['__HPOlib_configuration_space_root__']:
        value_sets.append(get_value_set(num_steps_dict, hp_name))
        hp_names.append(hp_name)

    # Create a Cartesian product of above allowed values for the HPs. Hold them in an
    # "unchecked" deque because some of the conditionally dependent HPs may become active
    # for some of the elements of the Cartesian product and in these cases creating a
    # Configuration would throw an Error (see below).
    # Creates a deque of Configuration dicts
    unchecked_grid_pts = deque(get_cartesian_product(value_sets, hp_names))

    while len(unchecked_grid_pts) > 0:
        try:
            yield Configuration(configuration_space, unchecked_grid_pts[0])
        except ValueError as e:
            assert (str(e)[:23] == "Active hyperparameter '" and
                    str(e)[-16:] == "' not specified!"), \
                "Caught exception contains unexpected message."
            value_sets = []
            hp_names = []
            new_active_hp_names = []

            # "for" loop over currently active HP names
            for hp_name in unchecked_grid_pts[0]:
                value_sets.append(tuple([unchecked_grid_pts[0][hp_name], ]))
                hp_names.append(hp_name)
                # Checks if the conditionally dependent children of already active
                # HPs are now active
                for new_hp_name in configuration_space._children[hp_name]:
                    if (
                            new_hp_name not in new_active_hp_names and
                            new_hp_name not in unchecked_grid_pts[0]
                    ):
                        all_cond_ = True
                        for cond in configuration_space._parent_conditions_of[new_hp_name]:
                            if not cond.evaluate(unchecked_grid_pts[0]):
                                all_cond_ = False
                        if all_cond_:
                            new_active_hp_names.append(new_hp_name)

            for hp_name in new_active_hp_names:
                value_sets.append(get_value_set(num_steps_dict, hp_name))
                hp_names.append(hp_name)
            # this check might not be needed, as there is always going to be a new
            # active HP when in this except block?
            if len(new_active_hp_names) > 0:
                new_conditonal_grid = get_cartesian_product(value_sets, hp_names)
                unchecked_grid_pts += new_conditonal_grid
            else:
                raise RuntimeError(
                    "Unexpected error: There should have been a newly activated hyperparameter"
                    f" for the current configuration values: {str(unchecked_grid_pts[0])}. "
                    "Please contact the developers with the code you ran and the stack trace."
                )
        unchecked_grid_pts.popleft()


def softmax(df):
    if len(df.shape) == 1:
        df[df > 20] = 20
        df[df < -20] = -20
        ppositive = 1 / (1 + np.exp(-df))
        ppositive[ppositive > 0.999999] = 1
        ppositive[ppositive < 0.0000001] = 0
        return np.transpose(np.array((1 - ppositive, ppositive)))
    else:
        # Compute the Softmax like it is described here:
        # http://www.iro.umontreal.ca/~bengioy/dlbook/numerical.html
        tmp = df - np.max(df, axis=1).reshape((-1, 1))
        tmp = np.exp(tmp)
        return tmp / np.sum(tmp, axis=1).reshape((-1, 1))


def sanitize_array(array):
    """
    Replace NaN and Inf (there should not be any!)
    :param array:
    :return:
    """
    a = np.ravel(array)
    maxi = np.nanmax(a[np.isfinite(a)])
    mini = np.nanmin(a[np.isfinite(a)])
    array[array == float('inf')] = maxi
    array[array == float('-inf')] = mini
    mid = (maxi + mini) / 2
    array[np.isnan(array)] = mid
    return array


def sort_dict(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = sort_dict(v)
        return dict(sorted(obj.items(), key=lambda x: str(x[0])))
    elif isinstance(obj, list):
        for i, elem in enumerate(obj):
            obj[i] = sort_dict(elem)
        return list(sorted(obj, key=str))
    else:
        return obj


def get_hash_of_dict(dict_, m=None, sort=True):
    if m is None:
        m = hashlib.md5()
    if sort:
        dict_ = sort_dict(deepcopy(dict_))
    m.update(str(dict_).encode("utf-8"))
    return m.hexdigest()


def get_hash_of_str(s, m=None):
    if m is None:
        m = hashlib.md5()
    if isinstance(s, str):
        s = s.encode("utf-8")
    m.update(s)
    return m.hexdigest()


def dict_to_csv_str(obj):
    return "\"" + json.dumps(obj).replace("\"", "\"\"") + "\""