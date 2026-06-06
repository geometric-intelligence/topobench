"""Combined Positional and Structural Encodings Transform."""

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

# Supported Positional and Structural Encodings
PSE_ENCODINGS = {"LapPE", "RWSE", "ElectrostaticPE", "HKdiagSE"}


class CombinedPSEs(BaseTransform):
    r"""
    Combined PSEs transform.

    Applies one or more pre-defined positional or structural encoding transforms
    (LapPE, RWSE) to a graph, storing their outputs and optionally
    concatenating them to `data.x`.

    Parameters
    ----------
    encodings : list of str
        List of structural encodings to apply. Supported values are
        "LapPE", "RWSE", "ElectrostaticPE", and "HKdiagSE".
    parameters : dict, optional
        Additional parameters for the encoding transforms.
    preprocessor_device : str, optional
        The overarching device to use for the combined transforms (e.g., 'cpu', 'cuda').
        If a specific encoding specifies its own device in `parameters`, that will
        take precedence. Default is None.
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(
        self,
        encodings: list[str],
        parameters: dict | None = None,
        preprocessor_device: str | None = None,
        **kwargs,
    ):
        self.encodings = encodings
        self.parameters = parameters if parameters is not None else {}
        self.device = preprocessor_device

    def forward(self, data: Data) -> Data:
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data with added structural encodings.
        """
        from topobench.transforms.data_manipulations import (
            RWSE,
            ElectrostaticPE,
            HKdiagSE,
            LapPE,
        )

        encoding_classes = {
            "LapPE": LapPE,
            "RWSE": RWSE,
            "ElectrostaticPE": ElectrostaticPE,
            "HKdiagSE": HKdiagSE,
        }

        # Validate encoding_classes matches PSE_ENCODINGS
        if set(encoding_classes.keys()) != PSE_ENCODINGS:
            missing_in_classes = PSE_ENCODINGS - set(encoding_classes.keys())
            missing_in_set = set(encoding_classes.keys()) - PSE_ENCODINGS
            raise RuntimeError(
                f"encoding_classes and PSE_ENCODINGS are out of sync. "
                f"Missing in encoding_classes: {missing_in_classes}. "
                f"Missing in PSE_ENCODINGS: {missing_in_set}."
            )

        if hasattr(data, "edge_index") and data.edge_index is not None:
            baseline_device = data.edge_index.device
        elif hasattr(data, "x") and data.x is not None:
            baseline_device = data.x.device
        else:
            baseline_device = torch.device("cpu")

        current_device = baseline_device

        for enc in self.encodings:
            if enc not in encoding_classes:
                raise ValueError(f"Unsupported encoding type: {enc}")

            enc_params = self.parameters.get(enc, {}).copy()

            # Determine the target device for this specific transform
            # Priority: 1. PE-specific device, 2. CombinedPSEs overarching device, 3. Baseline
            req_device = enc_params.pop("device", self.device)
            target_device = (
                torch.device(req_device) if req_device else baseline_device
            )

            # Fallback to CPU if CUDA is requested but physically unavailable
            if target_device.type == "cuda" and not torch.cuda.is_available():
                target_device = torch.device("cpu")

            if current_device != target_device:
                data = data.to(target_device)
                current_device = target_device

            # Instantiate and apply the encoder
            # The encoder naturally uses `current_device` because it reads `data.edge_index.device`
            encoder = encoding_classes[enc](**enc_params)
            data = encoder(data)

        # Ensure the graph is returned to its original device before exiting
        if current_device != baseline_device:
            data = data.to(baseline_device)

        return data
