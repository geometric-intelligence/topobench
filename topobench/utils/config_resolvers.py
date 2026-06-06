"""Configuration resolvers for the topobench package."""

import os
from collections import defaultdict

import numpy as np
import omegaconf
import torch
from omegaconf import OmegaConf


def register_all_resolvers():
    """Register all custom OmegaConf resolvers.

    This centralizes resolver registration to avoid duplication across modules. Should be called
    before Hydra initialization in any script that uses configs.
    """
    OmegaConf.register_new_resolver(
        "define_task_level", define_task_level, replace=True
    )
    OmegaConf.register_new_resolver(
        "get_default_metrics", get_default_metrics, replace=True
    )
    OmegaConf.register_new_resolver(
        "get_default_trainer", get_default_trainer, replace=True
    )
    OmegaConf.register_new_resolver(
        "get_default_transform", get_default_transform, replace=True
    )
    OmegaConf.register_new_resolver(
        "get_flattened_channels",
        get_flattened_channels,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "get_required_lifting", get_required_lifting, replace=True
    )
    OmegaConf.register_new_resolver(
        "get_monitor_metric", get_monitor_metric, replace=True
    )
    OmegaConf.register_new_resolver(
        "get_monitor_mode", get_monitor_mode, replace=True
    )
    OmegaConf.register_new_resolver(
        "get_non_relational_out_channels",
        get_non_relational_out_channels,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "infer_in_channels", infer_in_channels, replace=True
    )
    OmegaConf.register_new_resolver(
        "infer_num_cell_dimensions", infer_num_cell_dimensions, replace=True
    )
    OmegaConf.register_new_resolver(
        "infer_topotune_num_cell_dimensions",
        infer_topotune_num_cell_dimensions,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "parameter_multiplication",
        lambda x, y: int(int(x) * int(y)),
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "get_default_metrics", get_default_metrics, replace=True
    )
    OmegaConf.register_new_resolver(
        "get_default_trainer", get_default_trainer, replace=True
    )
    OmegaConf.register_new_resolver(
        "get_default_transform", get_default_transform, replace=True
    )
    OmegaConf.register_new_resolver(
        "get_list_element", get_list_element, replace=True
    )
    OmegaConf.register_new_resolver(
        "get_flattened_channels",
        get_flattened_channels,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "get_required_lifting", get_required_lifting, replace=True
    )
    OmegaConf.register_new_resolver(
        "get_monitor_metric", get_monitor_metric, replace=True
    )
    OmegaConf.register_new_resolver(
        "get_monitor_mode", get_monitor_mode, replace=True
    )
    OmegaConf.register_new_resolver(
        "get_non_relational_out_channels",
        get_non_relational_out_channels,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "infer_in_channels", infer_in_channels, replace=True
    )
    OmegaConf.register_new_resolver(
        "infer_num_cell_dimensions", infer_num_cell_dimensions, replace=True
    )
    OmegaConf.register_new_resolver(
        "infer_topotune_num_cell_dimensions",
        infer_topotune_num_cell_dimensions,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "parameter_multiplication",
        lambda x, y: int(int(x) * int(y)),
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "infer_in_khop_feature_dim",
        infer_in_khop_feature_dim,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "infer_in_hasse_graph_agg_dim",
        infer_in_hasse_graph_agg_dim,
        replace=True,
    )

    OmegaConf.register_new_resolver(
        "get_pse_dimensions",
        get_pse_dimensions,
        replace=True,
    )

    OmegaConf.register_new_resolver(
        "get_fes_dimensions",
        get_fes_dimensions,
        replace=True,
    )

    OmegaConf.register_new_resolver(
        "get_all_encoding_dimensions",
        get_all_encoding_dimensions,
        replace=True,
    )

    OmegaConf.register_new_resolver(
        "get_hop_num_gpse",
        lambda x: (
            int(x) + 1
        ),  # 2-hop if copy_initial = True else 1-hop (only GPSE info)
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "get_hop_num_pses", lambda x, y: len(x) + int(y), replace=True
    )
    OmegaConf.register_new_resolver(
        "set_preserve_edge_attr",
        set_preserve_edge_attr,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "infer_list_length",
        infer_list_length,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "infer_list_length_plus_one",
        infer_list_length_plus_one,
        replace=True,
    )
    OmegaConf.register_new_resolver("pid", lambda: os.getpid(), replace=True)


def define_task_level(dataset_task_level, learning_setting):
    r"""Define the task level for a given dataset task level and learning setting.

    Parameters
    ----------
    dataset_task_level : str
        Task level defined in the dataset configuration file.
    learning_setting : str
        Learning setting defined in the dataset split parameters.

    Returns
    -------
    str
        Task level for the model.

    Raises
    ------
    ValueError
        If the dataset task level or learning setting is invalid.
    """
    if dataset_task_level == "node" and learning_setting == "inductive":
        return "node_inductive"
    else:
        return dataset_task_level


def get_flattened_channels(num_nodes, channels):
    r"""Get the output dimension of flattening a feature matrix.

    Parameters
    ----------
    num_nodes : int
        Hidden dimension for the first layer.
    channels : int
        Channel dimension.

    Returns
    -------
    int
        Flatenned cchannels dimension.
    """
    return num_nodes * channels


def get_non_relational_out_channels(num_nodes, channels, task_level):
    r"""Get the output dimension for a non-relational model.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the input graph.
    channels : int
        Channel dimension.
    task_level : int
        Task level for the model.

    Returns
    -------
    int
        Output dimension.
    """
    if task_level == "node":  # node-level task
        return num_nodes * channels
    elif task_level == "graph":  # graph-level task
        return channels
    else:
        raise ValueError(f"Invalid task level {task_level}")


def get_default_trainer():
    r"""Get default trainer configuration.

    Returns
    -------
    str
        Default trainer configuration file name.
    """
    return "gpu" if torch.cuda.is_available() else "cpu"


def get_routes_from_neighborhoods(neighborhoods):
    """Get the routes from the neighborhoods.

    Combination of src_rank, dst_rank. ex: [[0, 0], [1, 0], [1, 1], [1, 1], [2, 1]].

    Parameters
    ----------
    neighborhoods : list
        List of neighborhoods of interest.

    Returns
    -------
    list
        List of routes.
    """
    routes = []
    for neighborhood in neighborhoods:
        split = neighborhood.split("-")
        src_rank = int(split[-1])
        r = int(split[0]) if len(split) == 3 else 1
        if "incidence" in neighborhood:
            route = (
                [src_rank, src_rank - r]
                if "down" in neighborhood
                else [src_rank, src_rank + r]
            )
        elif "adjacency" in neighborhood:
            route = [src_rank, src_rank]
        else:
            raise Exception(f"Invalid neighborhood {neighborhood}")

        routes.append(route)
    return routes


def get_default_transform(dataset, model):
    r"""Get default transform for a given data domain and model.

    Parameters
    ----------
    dataset : str
        Dataset name. Should be in the format "data_domain/name".
    model : str
        Model name. Should be in the format "model_domain/name".

    Returns
    -------
    str
        Default transform.
    """
    data_domain, dataset = dataset.split("/")
    model_domain, model = model.split("/")
    # TODO: improve logic for pointcloud models
    if model_domain == "non_relational" or model_domain == "pointcloud":
        model_domain = "graph"
    # Check if there is a default transform for the dataset at ./configs/transforms/dataset_defaults/
    # If not, use the default lifting transform for the dataset to be compatible with the model
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    dataset_configs_dir = os.path.join(
        base_dir, "configs", "transforms", "dataset_defaults"
    )
    model_configs_dir = os.path.join(
        base_dir, "configs", "transforms", "model_defaults"
    )
    datasets_with_defaults = [
        f.split(".")[0] for f in os.listdir(dataset_configs_dir)
    ]
    model_with_defaults = [
        f.split(".")[0] for f in os.listdir(model_configs_dir)
    ]
    model_dataset_configs_dir = os.path.join(
        base_dir, "configs", "transforms", "model_dataset_defaults"
    )
    model_dataset_defaults = [
        f.split(".")[0] for f in os.listdir(model_dataset_configs_dir)
    ]
    if f"{model}_{dataset}" in model_dataset_defaults:
        # TODO: Work in progress, check logic here
        return f"model_dataset_defaults/{model}_{dataset}"
    elif dataset in datasets_with_defaults:
        return f"dataset_defaults/{dataset}"
    elif model in model_with_defaults:
        return f"model_defaults/{model}"
    else:
        if data_domain == model_domain:
            return "no_transform"
        else:
            return f"liftings/{data_domain}2{model_domain}_default"


def get_required_lifting(data_domain, model):
    r"""Get required transform for a given data domain and model.

    Parameters
    ----------
    data_domain : str
        Dataset domain.
    model : str
        Model name. Should be in the format "model_domain/name".

    Returns
    -------
    str
        Required transform.
    """
    data_domain = data_domain.split("/")[0]
    model_domain = model.split("/")[0]
    if data_domain == model_domain:
        return "no_lifting"
    else:
        return f"{data_domain}2{model_domain}_default"


def get_monitor_metric(task, metric):
    r"""Get monitor metric for a given task.

    Parameters
    ----------
    task : str
        Task, either "classification" or "regression".
    metric : str
        Name of the metric function.

    Returns
    -------
    str
        Monitor metric.

    Raises
    ------
    ValueError
        If the task is invalid.
    """
    if (
        task == "classification"
        or task == "regression"
        or task == "multioutput classification"
        or task == "multilabel classification"
    ):
        return f"val/{metric}"
    else:
        raise ValueError(f"Invalid task {task}")


def get_monitor_mode(task):
    r"""Get monitor mode for a given task.

    Parameters
    ----------
    task : str
        Task, either "classification" or "regression".

    Returns
    -------
    str
        Monitor mode, either "max" or "min".

    Raises
    ------
    ValueError
        If the task is invalid.
    """
    if task == "classification" or task == "multilabel classification":
        return "max"

    elif task == "regression" or task == "multioutput classification":
        return "min"

    else:
        raise ValueError(f"Invalid task {task}")


def get_pse_dimensions(encodings, parameters):
    r"""Get dimensions of positional or structural encodings.

    Parameters
    ----------
    encodings : list
        List of positional or structural encodings.
    parameters : dict
        Dictionary of parameters for the positional or structural encodings, which should
        contain the key "parameters" with the parameters for each encoding.

    Returns
    -------
    list
        List with dimensions of the positional or structural encodings.
    """
    dimensions = []
    for pse in encodings:
        if pse == "LapPE":
            if parameters[pse].get("include_eigenvalues"):
                dimensions.append(parameters[pse].get("max_pe_dim") * 2)
            else:
                dimensions.append(parameters[pse].get("max_pe_dim"))
        elif pse == "RWSE":
            dimensions.append(parameters[pse].get("max_pe_dim"))
        elif pse == "ElectrostaticPE":
            dimensions.append(7)
        elif pse == "HKdiagSE":
            kernel_param = parameters[pse].get("kernel_param_HKdiagSE")
            # Handle both OmegaConf ListConfig and regular lists/tuples
            if (
                isinstance(kernel_param, (list, tuple))
                or type(kernel_param) is omegaconf.listconfig.ListConfig
            ):
                dimensions.append(kernel_param[1] - kernel_param[0])
            else:
                dimensions.append(kernel_param)
    return dimensions


def get_fes_dimensions(encodings, parameters):
    r"""Get dimensions of feature encodings.

    Parameters
    ----------
    encodings : list
        List of feature encodings.
    parameters : dict
        Dictionary of parameters for the feature encodings.

    Returns
    -------
    list
        List with dimensions of the feature encodings.
    """
    dimensions = []
    for fe in encodings:
        if fe == "HKFE":
            kernel_param = parameters[fe].get("kernel_param_HKFE")
            # Handle both OmegaConf ListConfig and regular lists/tuples
            if (
                isinstance(kernel_param, (list, tuple))
                or type(kernel_param) is omegaconf.listconfig.ListConfig
            ):
                dimensions.append(kernel_param[1] - kernel_param[0])
            else:
                dimensions.append(kernel_param)
        elif fe == "KHopFE":
            # max_hop - 1 because the 0th hop is the features themselves
            dimensions.append(parameters[fe].get("max_hop") - 1)
        elif fe == "PPRFE":
            fe_params = parameters.get(fe, {})
            alpha_param = fe_params.get("alpha_param_PPRFE", [0.1, 10])

            if (
                isinstance(alpha_param, (list, tuple))
                or type(alpha_param) is omegaconf.listconfig.ListConfig
            ):
                dimensions.append(alpha_param[1])
            else:
                dimensions.append(alpha_param)
        elif fe == "SheafConnLapPE":
            dimensions.append(parameters[fe].get("max_pe_dim"))
    return dimensions


def get_all_encoding_dimensions(encodings, parameters):
    r"""Get dimensions of all encodings (PSEs and FEs) in order.

    Parameters
    ----------
    encodings : list
        List of all encodings (both PSEs and FEs).
    parameters : dict
        Dictionary of parameters for all encodings.

    Returns
    -------
    list
        List with dimensions of all encodings in the same order as input.
    """
    dimensions = []
    for enc in encodings:
        # PSE encodings
        if enc == "LapPE":
            if parameters[enc].get("include_eigenvalues"):
                dimensions.append(parameters[enc].get("max_pe_dim") * 2)
            else:
                dimensions.append(parameters[enc].get("max_pe_dim"))
        elif enc == "RWSE":
            dimensions.append(parameters[enc].get("max_pe_dim"))
        elif enc == "ElectrostaticPE":
            dimensions.append(7)
        elif enc == "HKdiagSE":
            kernel_param = parameters[enc].get("kernel_param_HKdiagSE")
            # Handle both OmegaConf ListConfig and regular lists/tuples
            if (
                isinstance(kernel_param, (list, tuple))
                or type(kernel_param) is omegaconf.listconfig.ListConfig
            ):
                dimensions.append(kernel_param[1] - kernel_param[0])
            else:
                dimensions.append(kernel_param)
        # FE encodings
        elif enc == "HKFE":
            kernel_param = parameters[enc].get("kernel_param_HKFE")
            # Handle both OmegaConf ListConfig and regular lists/tuples
            if (
                isinstance(kernel_param, (list, tuple))
                or type(kernel_param) is omegaconf.listconfig.ListConfig
            ):
                dimensions.append(kernel_param[1] - kernel_param[0])
            else:
                dimensions.append(kernel_param)
        elif enc == "KHopFE":
            # max_hop - 1 because the 0th hop is the features themselves
            dimensions.append(parameters[enc].get("max_hop") - 1)
        elif enc == "PPRFE":
            # Safely get parameters, defaulting to empty dict if missing
            enc_params = parameters.get(enc, {})
            # Safely get alpha_param, defaulting to [0.1, 10]
            alpha_param = enc_params.get("alpha_param_PPRFE", [0.1, 10])

            if (
                isinstance(alpha_param, (list, tuple))
                or type(alpha_param) is omegaconf.listconfig.ListConfig
            ):
                dimensions.append(alpha_param[1])
            else:
                dimensions.append(alpha_param)
        elif enc == "SheafConnLapPE":
            dimensions.append(parameters[enc].get("max_pe_dim"))
    return dimensions


def check_pses_in_transforms(transforms):
    r"""Check if there are positional or structural encodings in the transforms.

    Parameters
    ----------
    transforms : DictConfig
        Configuration parameters for the transforms.

    Returns
    -------
    int
       Count of the number of features added by the encodings.
    """
    added_features = 0
    # Single transform
    transform = transforms.get("transform_name", None)
    if transform is not None:
        if transform == "LapPE":
            if transforms.get("include_eigenvalues"):
                added_features += transforms.get("max_pe_dim") * 2
            else:
                added_features += transforms.get("max_pe_dim")
        elif transform == "RWSE" or transform == "SheafConnLapPE":
            added_features += transforms.get("max_pe_dim")
    # Potentially multiple transforms
    for key in transforms:
        if "CombinedPSEs" in key or "encodings" in key:
            for pse in transforms[key].get("encodings", []):
                if pse == "LapPE":
                    if (
                        transforms[key]
                        .get("parameters")
                        .get(pse)
                        .get("include_eigenvalues")
                    ):
                        added_features += (
                            transforms[key]
                            .get("parameters")
                            .get(pse)
                            .get("max_pe_dim")
                            * 2
                        )
                    else:
                        added_features += (
                            transforms[key]
                            .get("parameters")
                            .get(pse)
                            .get("max_pe_dim")
                        )
                elif pse == "RWSE":
                    added_features += (
                        transforms[key]
                        .get("parameters")
                        .get(pse)
                        .get("max_pe_dim")
                    )
                elif pse == "ElectrostaticPE":
                    added_features += 7
                elif pse == "HKdiagSE":
                    kernel_param = (
                        transforms[key]
                        .get("parameters")
                        .get(pse)
                        .get("kernel_param_HKdiagSE")
                    )
                    added_features += (
                        (kernel_param[1] - kernel_param[0])
                        if type(kernel_param)
                        is omegaconf.listconfig.ListConfig
                        else kernel_param
                    )
        elif "LapPE" in key and omegaconf.OmegaConf.is_dict(transforms[key]):
            if transforms[key].get("include_eigenvalues"):
                added_features += transforms[key].get("max_pe_dim") * 2
            else:
                added_features += transforms[key].get("max_pe_dim")
        elif (
            "RWSE" in key or "SheafConnLapPE" in key
        ) and omegaconf.OmegaConf.is_dict(transforms[key]):
            added_features += transforms[key].get("max_pe_dim")
        elif "ElectrostaticPE" in key and omegaconf.OmegaConf.is_dict(
            transforms[key]
        ):
            added_features += 7
        elif "HKdiagSE" in key and omegaconf.OmegaConf.is_dict(
            transforms[key]
        ):
            kernel_param = transforms[key].get("kernel_param_HKdiagSE")
            added_features += (
                (kernel_param[1] - kernel_param[0])
                if type(kernel_param) is omegaconf.listconfig.ListConfig
                else kernel_param
            )

    return added_features


def check_fes_in_transforms(transforms):
    r"""Check if there are feature encodings in the transforms.

    Parameters
    ----------
    transforms : DictConfig
        Configuration parameters for the transforms.

    Returns
    -------
    int
        Count of the number of features added by the encodings.
    """
    added_features = 0
    # Single transform
    transform = transforms.get("transform_name", None)
    if transform is not None:
        if transform == "HKFE":
            kernel_param = transforms.get("kernel_param_HKFE")
            added_features += (
                (kernel_param[1] - kernel_param[0])
                if type(kernel_param) is omegaconf.listconfig.ListConfig
                else kernel_param
            )
        elif transform == "KHopFE":
            # max_hop - 1 because the 0th hop is the features themselves
            added_features += transforms.get("max_hop") - 1
        elif transform == "PPRFE":
            alpha_param = transforms.get("alpha_param_PPRFE")
            if (
                isinstance(alpha_param, (list, tuple))
                or type(alpha_param) is omegaconf.listconfig.ListConfig
            ):
                added_features += alpha_param[1]
            else:
                added_features += alpha_param
        elif transform == "SheafConnLapPE":
            added_features += transforms.get("max_pe_dim")
    # Potentially multiple transforms
    for key in transforms:
        if "CombinedFEs" in key:
            for fe in transforms[key].get("encodings", []):
                if fe == "HKFE":
                    kernel_param = (
                        transforms[key]
                        .get("parameters")
                        .get(fe)
                        .get("kernel_param_HKFE")
                    )
                    added_features += (
                        (kernel_param[1] - kernel_param[0])
                        if type(kernel_param)
                        is omegaconf.listconfig.ListConfig
                        else kernel_param
                    )
                elif fe == "KHopFE":
                    # max_hop - 1 because the 0th hop is the features themselves
                    added_features += (
                        transforms[key]
                        .get("parameters")
                        .get(fe)
                        .get("max_hop")
                        - 1
                    )
                elif fe == "PPRFE":
                    # Safely chain the gets so it never throws an error
                    fe_params = (
                        transforms[key].get("parameters", {}).get(fe, {})
                    )
                    alpha_param = fe_params.get("alpha_param_PPRFE", [0.1, 10])

                    if (
                        isinstance(alpha_param, (list, tuple))
                        or type(alpha_param) is omegaconf.listconfig.ListConfig
                    ):
                        added_features += alpha_param[1]
                    else:
                        added_features += alpha_param
                elif fe == "SheafConnLapPE":
                    added_features += (
                        transforms[key]
                        .get("parameters")
                        .get(fe)
                        .get("max_pe_dim")
                    )
        elif "HKFE" in key and omegaconf.OmegaConf.is_dict(transforms[key]):
            kernel_param = transforms[key].get("kernel_param_HKFE")
            added_features += (
                (kernel_param[1] - kernel_param[0])
                if type(kernel_param) is omegaconf.listconfig.ListConfig
                else kernel_param
            )
        elif "KHopFE" in key and omegaconf.OmegaConf.is_dict(transforms[key]):
            # max_hop - 1 because the 0th hop is the features themselves
            added_features += transforms[key].get("max_hop") - 1
        elif "PPRFE" in key and omegaconf.OmegaConf.is_dict(transforms[key]):
            alpha_param = transforms[key].get("alpha_param_PPRFE")
            if (
                isinstance(alpha_param, (list, tuple))
                or type(alpha_param) is omegaconf.listconfig.ListConfig
            ):
                added_features += alpha_param[1]
            else:
                added_features += alpha_param
        elif "SheafConnLapPE" in key and omegaconf.OmegaConf.is_dict(
            transforms[key]
        ):
            added_features += transforms[key].get("max_pe_dim")
    return added_features


def infer_in_channels(dataset, transforms):
    r"""Infer the number of input channels for a given dataset.

    Parameters
    ----------
    dataset : DictConfig
        Configuration parameters for the dataset.
    transforms : DictConfig
        Configuration parameters for the transforms.

    Returns
    -------
    list
        List with dimensions of the input channels.
    """
    num_features = dataset.parameters.num_features
    if isinstance(num_features, int) and transforms is not None:
        num_features = (
            num_features
            + check_pses_in_transforms(transforms)
            + check_fes_in_transforms(transforms)
        )

    # Make it possible to pass lifting configuration as file path
    if transforms is not None and transforms.keys() == {"liftings"}:
        transforms = transforms.liftings

    def find_complex_lifting(transforms):
        r"""Find if there is a complex lifting in the complex_transforms.

        Parameters
        ----------
        transforms : List[str]
            List of transforms.

        Returns
        -------
        bool
            True if there is a complex lifting, False otherwise.
        str
            Name of the complex lifting, if it exists.
        """

        if transforms is None:
            return False, None
        complex_transforms = [
            # Default liftig configurations
            "graph2cell_lifting",
            "graph2simplicial_lifting",
            "graph2combinatorial_lifting",
            "graph2hypergraph_lifting",
            "pointcloud2graph_lifting",
            "pointcloud2simplicial_lifting",
            "pointcloud2combinatorial_lifting",
            "pointcloud2hypergraph_lifting",
            "pointcloud2cell_lifting",
            "hypergraph2combinatorial_lifting",
            # Make it possible to run directly from the folder
            "graph2cell",
            "graph2simplicial",
            "graph2combinatorial",
            "graph2hypergraph",
            "pointcloud2graph",
            "pointcloud2simplicial",
            "pointcloud2combinatorial",
            "pointcloud2hypergraph",
            "pointcloud2cell",
            "hypergraph2combinatorial",
        ]
        for t in complex_transforms:
            if t in transforms:
                return True, t
        return False, None

    def check_for_type_feature_lifting(transforms, lifting):
        r"""Check the type of feature lifting in the dataset.

        Parameters
        ----------
        transforms : DictConfig
            Configuration parameters for the transforms.
        lifting : str
            Name of the complex lifting.

        Returns
        -------
        str
            Type of feature lifting.
        """
        lifting_params_keys = transforms[lifting].keys()
        if "feature_lifting" in lifting_params_keys:
            feature_lifting = transforms[lifting]["feature_lifting"]
        else:
            feature_lifting = "ProjectionSum"

        return feature_lifting

    there_is_complex_lifting, lifting = find_complex_lifting(transforms)
    if there_is_complex_lifting:
        # Get type of feature lifting
        feature_lifting = check_for_type_feature_lifting(transforms, lifting)

        # Check if the num_features defines a single value or a list
        if isinstance(num_features, int):
            # Case when the dataset has no edge attributes
            if feature_lifting == "Concatenation":
                return_value = [num_features]
                for i in range(2, transforms[lifting].complex_dim + 2):
                    return_value += [int(return_value[-1]) * i]

                return return_value

            else:
                # ProjectionSum feature lifting by default
                return [num_features] * (transforms[lifting].complex_dim + 1)
        # Case when the dataset has edge attributes (cells attributes)
        else:
            assert type(num_features) is omegaconf.listconfig.ListConfig, (
                f"num_features should be a list of integers, not {type(num_features)}"
            )
            # If preserve_edge_attr == False
            if not transforms[lifting].preserve_edge_attr:
                if feature_lifting == "Concatenation":
                    return_value = [num_features[0]]
                    for i in range(2, transforms[lifting].complex_dim + 2):
                        return_value += [int(return_value[-1]) * i]

                    return return_value

                else:
                    # ProjectionSum feature lifting by default
                    return [num_features[0]] * (
                        transforms[lifting].complex_dim + 1
                    )
            # If preserve_edge_attr == True
            else:
                return list(num_features) + [num_features[1]] * (
                    transforms[lifting].complex_dim + 1 - len(num_features)
                )

    # Case when there is no lifting
    elif not there_is_complex_lifting:
        # Check if dataset and model are from the same domain and data_domain is higher-order

        # TODO: Does this if statement ever execute? model_domain == data_domain and data_domain in ["simplicial", "cell", "combinatorial", "hypergraph"]
        # BUT get_default_transform() returns "no_transform" when model_domain == data_domain
        if (
            dataset.loader.parameters.get("model_domain", "graph")
            == dataset.loader.parameters.data_domain
            and dataset.loader.parameters.data_domain
            in ["simplicial", "cell", "combinatorial", "hypergraph"]
        ):
            if isinstance(
                num_features,
                omegaconf.listconfig.ListConfig,
            ):
                return list(num_features)
            else:
                raise ValueError(
                    "The dataset and model are from the same domain but the data_domain is not higher-order."
                )

        elif isinstance(num_features, int):
            return [num_features]

        else:
            pe_features = (
                check_pses_in_transforms(transforms)
                if transforms is not None
                else 0
            )
            fe_features = (
                check_fes_in_transforms(transforms)
                if transforms is not None
                else 0
            )
            return [num_features[0] + pe_features + fe_features]

    # This else is never executed
    else:
        raise ValueError(
            "There is a problem with the complex lifting. Please check the configuration file."
        )


def infer_num_cell_dimensions(selected_dimensions, in_channels):
    r"""Infer the length of a list.

    Parameters
    ----------
    selected_dimensions : list
        List of selected dimensions. If not None it will be used to infer the length.
    in_channels : list
        List of input channels. If selected_dimensions is None, this list will be used to infer the length.

    Returns
    -------
    int
        Length of the input list.
    """
    if selected_dimensions is not None:
        return len(selected_dimensions)
    else:
        return len(in_channels)


def infer_list_length(list):
    r"""Infer the length of a list.

    Parameters
    ----------
    list : list
        List.

    Returns
    -------
    int
        Length of the input list.
    """
    return len(list)


def infer_list_length_plus_one(list):
    r"""Infer the length of a list plus one.

    Parameters
    ----------
    list : list
        List.

    Returns
    -------
    int
        Length of the input list plus one.
    """
    return len(list) + 1


def infer_topotune_num_cell_dimensions(neighborhoods):
    r"""Infer the length of a list.

    Parameters
    ----------
    neighborhoods : list
        List of neighborhoods.

    Returns
    -------
    int
        Length of the input list.
    """
    from topobench.data.utils import get_routes_from_neighborhoods

    routes = get_routes_from_neighborhoods(neighborhoods)
    return max([max(route) for route in routes]) + 1


def get_default_metrics(task, num_classes, metrics=None):
    r"""Get default metrics for a given task.

    Parameters
    ----------
    task : str
        Task, either "classification" or "regression".
    num_classes : int
        Number of classes, relevant for multilabel and multioutput tasks.
    metrics : list, optional
        List of metrics to be used. If None, the default metrics will be used.

    Returns
    -------
    list
        List of default metrics.

    Raises
    ------
    ValueError
        If the task is invalid.
    """
    if metrics is not None:
        return metrics
    else:
        if task in ["multioutput classification", "multilabel classification"]:
            metric_names = ["accuracy", "precision", "recall", "f1"]
            metrics = []
            for dim in range(num_classes):
                for name in metric_names:
                    metrics.append(f"{name}-{dim}")
            return metrics
        elif "classification" in task:
            return ["accuracy", "precision", "recall", "auroc", "f1"]
        elif "regression" in task:
            return ["mse", "mae"]
        else:
            raise ValueError(f"Invalid task {task}")


def get_list_element(list, index):
    r"""Get element of a list.

    Parameters
    ----------
    list : list
        List of elements.
    index : int
        Index of the element to get.

    Returns
    -------
    any
        Element of the list.
    """
    return list[index]


def infer_in_khop_feature_dim(dataset_in_channels, max_hop, complex_dim=None):
    r"""Infer the dimension of the feature vector in the SANN k-hop model.

    Parameters
    ----------
    dataset_in_channels : np.ndarray
        1D array of input channels for the dataset.
    max_hop : int
        Maximum hop distance.
    complex_dim : int, optional
        Number of cell ranks processed by the transform. When provided,
        ``dataset_in_channels`` is truncated to this length so the
        recursive formula only considers ranks that actually appear in
        the k-hop feature computation.

    Returns
    -------
    int :
        Dimension of the feature vector in the SANN k-hop model.
    """
    if complex_dim is not None:
        dataset_in_channels = list(dataset_in_channels)[:complex_dim]

    def compute_recursive_sequence(initial_values, time_steps):
        """Compute the sequence D_k^(t) based on the given recursive formula.

        D_k^(t) = 2 * D_k^(t-1) + D_(k-1)^(t-1) + D_(k+1)^(t-1)

        Parameters
        ----------
        initial_values : np.ndarray
            1D array of initial values for D_k^(0), where k = 0, 1, ..., N-1.
        time_steps : int
            Number of time steps to compute the sequence.

        Returns
        -------
        np.ndarray
            2D array where each row corresponds to D_k^(t) for a specific time step.
        """
        # Initialize the result array
        N = len(initial_values)
        results = np.zeros((time_steps + 1, N))
        results[0] = initial_values  # Set the initial values

        # Iterate over time steps
        for t in range(1, time_steps + 1):
            for k in range(N):
                # Use modular arithmetic to handle boundary conditions (e.g., cyclic boundaries)
                D_k = 2 * results[t - 1][k] if k > 0 else results[t - 1][k]
                D_k_minus_1 = results[t - 1][k - 1] if k - 1 >= 0 else 0
                D_k_plus_1 = results[t - 1][k + 1] if k + 1 < N else 0

                results[t][k] = D_k + D_k_minus_1 + D_k_plus_1

        return results

    result = np.transpose(
        compute_recursive_sequence(dataset_in_channels, max_hop)
    )

    return result.astype(np.int32).tolist()


def infer_in_hasse_graph_agg_dim(
    neighborhoods,
    dim_pses,
    complex_dim,
    max_hop,
    dim_in,
    dim_hidden_graph,
    dim_hidden_node,
    copy_initial,
    use_edge_attr,
):
    """Compute which input dimensions need to changed based on if they are the output of a neighborhood.

    Set the list of dimensions as outputs to the hasse graph as a GNN

    Parameters
    ----------
    neighborhoods : List[str]
        List of strings representing the neighborhood.
    dim_pses : List[int]
        List of dimensions of the positional or structural encodings.
    complex_dim : int
        Maximum dimension of the complex.
    max_hop : int
        Maximum number of hops (counting the intial features).
    dim_in : int
        The dataset feature input dimension.
    dim_hidden_graph : int
        The output hidden dimension of the GNN over the Hasse Graph aggregation.
    dim_hidden_node : int
        The output hidden dimension of the GNN over the Hasse Graph for each node.
    copy_initial : bool
        If the initial features should be copied as the 0-th hop.
    use_edge_attr : bool
        If the edge attributes are used as features in the 1-cells and should be considered for channel inference.

    Returns
    -------
    np.ndarray
        A 2D array where.
    """
    # TODO, to my understanding this should never change

    # Count how many times each dimension is a target in the neighborhood routes,
    # to know how many times it will be aggregated over in the Hasse graph and thus
    # how many dimensions it needs to output to be able to be aggregated over.
    neighbor_targets = defaultdict(int)
    routes = get_routes_from_neighborhoods(neighborhoods)
    for _s, t in routes:
        neighbor_targets[t] += 1

    dim_hidden = dim_hidden_graph + dim_hidden_node
    hop_num = max_hop  # If copy_intial max_hop contains one extra hop
    results = np.zeros(shape=(complex_dim + 1, hop_num))
    if copy_initial:
        # First dimension is always the input dimension
        if isinstance(dim_in, int):
            results.fill(dim_in)
        else:
            results.fill(dim_in[0])

        # If edge_attr is used, set those dimensions
        if use_edge_attr:
            for i in range(1, complex_dim + 1):
                results[i][0] = dim_in[1]
        # TODO: If there are face attributes, another condition needs to be added

    else:
        results.fill(dim_hidden)

    # For each complex
    for i in range(complex_dim + 1):
        for j in range(1, hop_num):
            results[i][j] = max(1, neighbor_targets[i]) * dim_pses[j - 1]

    return results.astype(np.int32).tolist()


def set_preserve_edge_attr(model_name, default=True):
    r"""Set the preserve_edge_attr parameter of datasets depending on the model.

    Parameters
    ----------
    model_name : str
        Model name.
    default : bool, optional
        Default value for the parameter. Defaults to True.

    Returns
    -------
    bool
        Default if the model can preserve edge attributes, False otherwise.
    """
    if model_name in ["hopse_m", "hopse_g"]:
        return True
    elif model_name in ["sann"]:
        return False
    else:
        return default
