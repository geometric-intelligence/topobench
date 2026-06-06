"""Combined Feature Encodings Transform."""

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

# Supported Feature Encodings
FE_ENCODINGS = {"HKFE", "KHopFE", "SheafConnLapPE", "PPRFE"}


class CombinedFEs(BaseTransform):
    r"""
    Combined FEs transform.

    Applies one or more pre-defined feature encoding transforms
    (KHopFE, HKFE, SheafConnLapPE, PPRFE) to a graph, storing their outputs
    and optionally concatenating them to `data.x`.

    Parameters
    ----------
    encodings : list of str
        List of feature encodings to apply. Supported values are
        "KHopFE" for K-hop Feature Encoding, "HKFE" for Heat Kernel
        Feature Encoding, "SheafConnLapPE" for Sheaf Connection
        Laplacian Positional Encoding, and "PPRFE" for Personalized
        Page Rank Feature Encoding.
    parameters : dict, optional
        Additional parameters for the encoding transforms.
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(
        self,
        encodings: list[str],
        parameters: dict | None = None,
        **kwargs,
    ):
        self.encodings = encodings
        self.parameters = parameters if parameters is not None else {}

    def forward(self, data: Data) -> Data:
        r"""Apply the transform to the input data.

        All encodings are computed on the original features first, then
        those with concat_to_x=True are concatenated at the end. This
        ensures each encoding sees the original features, not features
        modified by previous encodings.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data with added feature encodings.
        """
        import torch

        from topobench.transforms.data_manipulations import (
            HKFE,
            PPRFE,
            KHopFE,
            SheafConnLapPE,
        )

        encoding_classes = {
            "HKFE": HKFE,
            "KHopFE": KHopFE,
            "SheafConnLapPE": SheafConnLapPE,
            "PPRFE": PPRFE,
        }

        # Validate encoding_classes matches FE_ENCODINGS
        if set(encoding_classes.keys()) != FE_ENCODINGS:
            missing_in_classes = FE_ENCODINGS - set(encoding_classes.keys())
            missing_in_set = set(encoding_classes.keys()) - FE_ENCODINGS
            raise RuntimeError(
                f"encoding_classes and FE_ENCODINGS are out of sync. "
                f"Missing in encoding_classes: {missing_in_classes}. "
                f"Missing in FE_ENCODINGS: {missing_in_set}."
            )

        # Track encoding results: list of (enc_name, tensor, should_concat)
        encoding_results = []

        for enc in self.encodings:
            if enc not in encoding_classes:
                raise ValueError(f"Unsupported encoding type: {enc}")

            # Get user params and check if they want concat_to_x
            user_params = self.parameters.get(enc, {}).copy()
            should_concat = user_params.pop("concat_to_x", True)

            # Force separate storage during computation
            user_params["concat_to_x"] = False

            # Apply encoding
            encoder = encoding_classes[enc](**user_params)
            data = encoder(data)

            # Get the encoding result and remove temporary attribute
            encoding_tensor = getattr(data, enc)
            delattr(data, enc)

            encoding_results.append((enc, encoding_tensor, should_concat))

        # Store results and build concatenation list
        tensors_to_concat = [data.x] if data.x is not None else []

        for enc_name, tensor, should_concat in encoding_results:
            if should_concat:
                tensors_to_concat.append(tensor)
            else:
                setattr(data, enc_name, tensor)

        # Concatenate all encodings that had concat_to_x=True
        if len(tensors_to_concat) > 1:
            data.x = torch.cat(tensors_to_concat, dim=-1)

        return data


class SelectDestinationFEs(BaseTransform):
    r"""
    Select Destination Feature Encodings (FEs) transform.

    Selects and retains only the FEs corresponding to the destination nodes
    of edges in `data.edge_index`.

    Parameters
    ----------
    encodings : list of str
        List of encoding keys in `data` where the FEs are stored (e.g., 'HKFE', 'KHopFE').
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(self, encodings, **kwargs):
        self.encodings = encodings

    def forward(self, data: Data, n_dst_nodes: int) -> Data:
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.
        n_dst_nodes : int
            Number of destination nodes.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data with selected FEs.
        """
        new_data = {}
        new_data["x"] = data.x[:n_dst_nodes, :] if data.x is not None else None
        for encoding_key in self.encodings:
            if hasattr(data, encoding_key):
                fe = getattr(data, encoding_key)
                selected_fe = fe[:n_dst_nodes, :]
                new_data[encoding_key] = selected_fe
            else:
                raise ValueError(
                    f"Encoding key '{encoding_key}' not found in data."
                )
        return Data(**new_data)

    def __call__(self, data: Data, n_dst_nodes: int) -> Data:
        r"""Apply the transform to ``data``.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data containing the combined feature encodings to slice.
        n_dst_nodes : int
            Number of destination nodes to retain in each encoding.

        Returns
        -------
        torch_geometric.data.Data
            Data restricted to the first ``n_dst_nodes`` rows of each encoding.
        """
        return self.forward(data, n_dst_nodes)
