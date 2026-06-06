"""Combined Encodings Transform (FEs + PSEs)."""

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

# Import encoding sets from their respective modules (single source of truth)
from topobench.transforms.data_manipulations.combined_feature_encodings import (
    FE_ENCODINGS,
)
from topobench.transforms.data_manipulations.combined_positional_and_structural_encodings import (
    PSE_ENCODINGS,
)

ALL_ENCODINGS = PSE_ENCODINGS | FE_ENCODINGS


class CombinedEncodings(BaseTransform):
    r"""Combined Encodings transform.

    Applies both Feature Encodings (FEs) and Positional/Structural Encodings
    (PSEs) to a graph. FEs are applied first since they use ``data.x`` as
    input, while PSEs only use graph structure.

    Supported Feature Encodings (FEs):
        - "HKFE": Heat Kernel Feature Encoding
        - "KHopFE": K-hop Feature Encoding
        - "SheafConnLapPE": Sheaf Connection Laplacian Positional Encoding

    Supported Positional/Structural Encodings (PSEs):
        - "LapPE": Laplacian Positional Encoding
        - "RWSE": Random Walk Structural Encoding
        - "ElectrostaticPE": Electrostatic Positional Encoding
        - "HKdiagSE": Heat Kernel Diagonal Structural Encoding

    Parameters
    ----------
    encodings : list of str
        List of encodings to apply. Can include any mix of FEs and PSEs.
        FEs will always be applied before PSEs regardless of order in list.
    parameters : dict, optional
        Parameters for each encoding, keyed by encoding name.
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

        # Validate encodings
        all_supported = PSE_ENCODINGS | FE_ENCODINGS
        for enc in encodings:
            if enc not in all_supported:
                raise ValueError(
                    f"Unsupported encoding: {enc}. "
                    f"Supported FEs: {FE_ENCODINGS}. "
                    f"Supported PSEs: {PSE_ENCODINGS}."
                )

        # Separate into FEs and PSEs (preserving order within each group)
        self.fe_encodings = [e for e in encodings if e in FE_ENCODINGS]
        self.pse_encodings = [e for e in encodings if e in PSE_ENCODINGS]

    def forward(self, data: Data) -> Data:
        r"""Apply the transform to the input data.

        FEs are applied first (they use data.x as input), then PSEs
        (they only use graph structure).

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data with added encodings.
        """
        from topobench.transforms.data_manipulations.combined_feature_encodings import (
            CombinedFEs,
        )
        from topobench.transforms.data_manipulations.combined_positional_and_structural_encodings import (
            CombinedPSEs,
        )

        # Apply FEs first (they use data.x as input)
        if self.fe_encodings:
            fe_params = {
                enc: self.parameters.get(enc, {}) for enc in self.fe_encodings
            }
            fe_transform = CombinedFEs(
                encodings=self.fe_encodings,
                parameters=fe_params,
            )
            data = fe_transform(data)

        # Apply PSEs second (they only use graph structure)
        if self.pse_encodings:
            pse_params = {
                enc: self.parameters.get(enc, {}) for enc in self.pse_encodings
            }
            pse_transform = CombinedPSEs(
                encodings=self.pse_encodings,
                parameters=pse_params,
            )
            data = pse_transform(data)

        return data


class SelectDestinationEncodings(BaseTransform):
    r"""Select destination node encodings from expanded graph data.

    Used in interrank message passing where we expand the graph to include
    both source and destination nodes, compute encodings, then select only
    the encodings for destination nodes.

    Parameters
    ----------
    encodings : list of str
        List of encoding names to select (e.g., ['HKFE', 'LapPE']).
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(self, encodings: list[str], **kwargs):
        self.encodings = encodings

    def forward(self, data: Data, n_dst_nodes: int) -> Data:
        r"""Select encodings for destination nodes only.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data with encodings computed on expanded graph.
        n_dst_nodes : int
            Number of destination nodes (first n_dst_nodes rows to keep).

        Returns
        -------
        torch_geometric.data.Data
            Data with encodings selected for destination nodes only.
        """
        new_data = {}
        new_data["x"] = data.x[:n_dst_nodes, :] if data.x is not None else None

        for enc in self.encodings:
            if hasattr(data, enc):
                encoding_tensor = getattr(data, enc)
                new_data[enc] = encoding_tensor[:n_dst_nodes, :]
            else:
                raise ValueError(f"Encoding '{enc}' not found in data.")

        return Data(**new_data)

    def __call__(self, data: Data, n_dst_nodes: int) -> Data:
        r"""Apply the transform to ``data``.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data containing the encodings to slice.
        n_dst_nodes : int
            Number of destination nodes to retain in each encoding.

        Returns
        -------
        torch_geometric.data.Data
            Data restricted to the first ``n_dst_nodes`` rows of each encoding.
        """
        return self.forward(data, n_dst_nodes)
