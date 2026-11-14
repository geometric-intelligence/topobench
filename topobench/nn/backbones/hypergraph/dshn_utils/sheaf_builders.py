r"""Restriction-map predictors for Directional Sheaf Hypergraph Networks.

DSHN learns, for every node-hyperedge incidence :math:`(v, e)`, a **real**
restriction map :math:`F_{v \lhd e} \in \mathbb{R}^{d \times d}`. Only the
scalar charge :math:`S^{(q)}_{v \lhd e}` is complex (see :mod:`.laplacian`),
so these builders are real-valued networks fed real-valued inputs.

Following §C.1 item 3 (p. 8) of

    E. Mule et al., *"Directional Sheaf Hypergraph Networks"*, ICLR 2026,
    https://arxiv.org/abs/2510.04727

the predictor is

.. math::
    F_{v \lhd e} = \Phi(x_v \,\|\, x_e), \qquad
    \Phi(x_v, x_e) = \sigma\big(V (x_v \,\|\, x_e)\big),

with :math:`V` learnable and :math:`\sigma` a nonlinearity. Because the node
and hyperedge features are complex (Eq. 9), the paper applies ``unwind`` to
both before concatenating:

    "Given that the node and the hyperedge features :math:`x_v` and
    :math:`x_e` are complex-valued due to Eq. (9), we employ the same
    ``unwind`` operation to map them into a form suitable for input to
    :math:`\Phi`."

so :math:`\Phi` sees :math:`4f` real inputs -- :math:`2f` from the unwound
node feature and :math:`2f` from the unwound hyperedge feature.

.. note::
    The reference implementation
    (``models/sheaf_utils/directed_sheaf_builders.py``) offers three
    predictors, ``MLP_var1``/``var2``/``var3``, selected by
    ``--sheaf_pred_block``. Only ``var1`` -- the default, and the one
    described above -- corresponds to what the paper documents, so it is the
    only one implemented here. ``var2``/``var3`` are undocumented variants;
    the authors' published best configuration happens to use ``var3``, but
    since accuracy is not what we are reproducing, we prefer the predictor
    the paper actually specifies.

.. note::
    Also following the reference, :math:`\Phi` is a single linear map rather
    than a multi-layer perceptron, despite the ``MLP_*`` naming, and the
    per-node feature handed to it is the mean over the :math:`d` stalk
    dimensions (``x.view(n, d, f).mean(1)``).
"""

import torch
import torch.nn.functional as F
from torch import nn

from topobench.nn.backbones.graph.nsd_utils.orthogonal import Orthogonal

from .complex_ops import unwind

SHEAF_ACTIVATIONS = ("sigmoid", "tanh", "relu", "none")


def _activate(raw: torch.Tensor, act: str) -> torch.Tensor:
    """Apply the sheaf activation :math:`\\sigma` to raw predictions.

    Parameters
    ----------
    raw : torch.Tensor
        Unactivated predictions of any shape.
    act : str
        One of :data:`SHEAF_ACTIVATIONS`.

    Returns
    -------
    torch.Tensor
        Tensor of the same shape as ``raw``.

    Raises
    ------
    ValueError
        If ``act`` is not one of :data:`SHEAF_ACTIVATIONS`.
    """
    if act == "sigmoid":
        return torch.sigmoid(raw)
    if act == "tanh":
        return torch.tanh(raw)
    if act == "relu":
        return F.relu(raw)
    if act == "none":
        return raw
    raise ValueError(
        f"Unknown sheaf activation {act!r}; expected one of "
        f"{SHEAF_ACTIVATIONS}."
    )


class SheafBuilder(nn.Module):
    r"""Predict restriction maps from node and hyperedge features.

    Base class implementing :math:`\Phi(x_v \| x_e)` up to the point where the
    raw parameter vector is reshaped into a ``d x d`` block; subclasses supply
    the parameter count and that reshape.

    Parameters
    ----------
    d : int
        Stalk dimension :math:`d`.
    hidden_channels : int
        Feature width :math:`f` of the complex features.
    out_dim : int
        Number of scalars :math:`\Phi` predicts per incidence.
    sheaf_act : str, optional
        Nonlinearity :math:`\sigma`, one of :data:`SHEAF_ACTIVATIONS`
        (default: ``"sigmoid"``).
    sheaf_dropout : bool, optional
        Whether to drop predicted parameters (default: ``False``).
    dropout : float, optional
        Dropout probability used when ``sheaf_dropout`` is set (default:
        ``0.0``).
    input_norm : bool, optional
        Whether to layer-normalize the :math:`4f` inputs of :math:`\Phi`
        (default: ``True``). Corresponds to the reference's
        ``--AllSet_input_norm``.

    Attributes
    ----------
    lin : torch.nn.Linear
        The learnable map :math:`V`, from ``4 * hidden_channels`` to
        ``out_dim``. Bias-free, following the reference.
    norm : torch.nn.LayerNorm or torch.nn.Identity
        Optional input normalization.
    """

    def __init__(
        self,
        d: int,
        hidden_channels: int,
        out_dim: int,
        sheaf_act: str = "sigmoid",
        sheaf_dropout: bool = False,
        dropout: float = 0.0,
        input_norm: bool = True,
    ) -> None:
        super().__init__()
        self.d = d
        self.hidden_channels = hidden_channels
        self.out_dim = out_dim
        self.sheaf_act = sheaf_act
        self.sheaf_dropout = sheaf_dropout
        self.dropout = dropout
        in_dim = 4 * hidden_channels
        self.norm = nn.LayerNorm(in_dim) if input_norm else nn.Identity()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the learnable parameters of the predictor."""
        self.lin.reset_parameters()
        if isinstance(self.norm, nn.LayerNorm):
            self.norm.reset_parameters()

    def _stalk_mean(self, x: torch.Tensor, count: int) -> torch.Tensor:
        r"""Average complex features over the stalk dimension and unwind.

        Parameters
        ----------
        x : torch.Tensor
            Complex tensor of shape ``[count * d, hidden_channels]``, with the
            :math:`d` rows of each node or hyperedge contiguous.
        count : int
            Number of nodes or hyperedges.

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``[count, 2 * hidden_channels]``.
        """
        reduced = x.view(count, self.d, self.hidden_channels).mean(dim=1)
        return unwind(reduced)

    def predict(
        self,
        x: torch.Tensor,
        hyperedge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_hyperedges: int,
    ) -> torch.Tensor:
        r"""Evaluate :math:`\Phi(x_v \| x_e)` on every incidence.

        Parameters
        ----------
        x : torch.Tensor
            Complex node features of shape ``[num_nodes * d, f]``.
        hyperedge_attr : torch.Tensor
            Complex hyperedge features of shape ``[num_hyperedges * d, f]``.
        edge_index : torch.Tensor
            Incidence index of shape ``[2, nnz]``.
        num_nodes : int
            Number of nodes :math:`n`.
        num_hyperedges : int
            Number of hyperedges :math:`m`.

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``[nnz, out_dim]``.
        """
        xs = self._stalk_mean(x, num_nodes)[edge_index[0]]
        es = self._stalk_mean(hyperedge_attr, num_hyperedges)[edge_index[1]]
        raw = self.lin(self.norm(torch.cat((xs, es), dim=-1)))
        raw = _activate(raw, self.sheaf_act)
        if self.sheaf_dropout:
            raw = F.dropout(raw, p=self.dropout, training=self.training)
        return raw

    def to_blocks(self, raw: torch.Tensor) -> torch.Tensor:
        """Reshape raw predictions into ``d x d`` restriction maps.

        Parameters
        ----------
        raw : torch.Tensor
            Real tensor of shape ``[nnz, out_dim]``.

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``[nnz, d, d]``.

        Raises
        ------
        NotImplementedError
            Always; subclasses must override this method.
        """
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_hyperedges: int,
    ) -> torch.Tensor:
        r"""Predict one ``d x d`` restriction map per incidence.

        Parameters
        ----------
        x : torch.Tensor
            Complex node features of shape ``[num_nodes * d, f]``.
        hyperedge_attr : torch.Tensor
            Complex hyperedge features of shape ``[num_hyperedges * d, f]``.
        edge_index : torch.Tensor
            Incidence index of shape ``[2, nnz]``.
        num_nodes : int
            Number of nodes :math:`n`.
        num_hyperedges : int
            Number of hyperedges :math:`m`.

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``[nnz, d, d]``.
        """
        raw = self.predict(
            x, hyperedge_attr, edge_index, num_nodes, num_hyperedges
        )
        return self.to_blocks(raw)


class DiagSheafBuilder(SheafBuilder):
    r"""Predict diagonal restriction maps.

    :math:`\Phi` emits :math:`d` scalars per incidence, placed on the
    diagonal of :math:`F_{v \lhd e}`. This is the variant used for **every**
    DSHN and DSHNLight result reported in the paper
    (``--method=SheafGeDiDiag``).

    Parameters
    ----------
    d : int
        Stalk dimension :math:`d`.
    hidden_channels : int
        Feature width :math:`f`.
    **kwargs : dict, optional
        Forwarded to :class:`SheafBuilder`.
    """

    def __init__(self, d: int, hidden_channels: int, **kwargs) -> None:
        super().__init__(d, hidden_channels, out_dim=d, **kwargs)

    def to_blocks(self, raw: torch.Tensor) -> torch.Tensor:
        """Place each predicted vector on a diagonal.

        Parameters
        ----------
        raw : torch.Tensor
            Real tensor of shape ``[nnz, d]``.

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``[nnz, d, d]``.
        """
        return torch.diag_embed(raw)


class GeneralSheafBuilder(SheafBuilder):
    r"""Predict unconstrained restriction maps.

    :math:`\Phi` emits :math:`d^2` scalars per incidence, reshaped directly
    into :math:`F_{v \lhd e}`.

    Parameters
    ----------
    d : int
        Stalk dimension :math:`d`.
    hidden_channels : int
        Feature width :math:`f`.
    **kwargs : dict, optional
        Forwarded to :class:`SheafBuilder`.
    """

    def __init__(self, d: int, hidden_channels: int, **kwargs) -> None:
        super().__init__(d, hidden_channels, out_dim=d * d, **kwargs)

    def to_blocks(self, raw: torch.Tensor) -> torch.Tensor:
        """Reshape each predicted vector into a full matrix.

        Parameters
        ----------
        raw : torch.Tensor
            Real tensor of shape ``[nnz, d * d]``.

        Returns
        -------
        torch.Tensor
            Real tensor of shape ``[nnz, d, d]``.
        """
        return raw.view(-1, self.d, self.d)


class OrthoSheafBuilder(SheafBuilder):
    r"""Predict orthogonal restriction maps.

    :math:`\Phi` emits the :math:`d(d-1)/2` strictly-lower-triangular entries
    of a skew-symmetric matrix :math:`A`, which is mapped to an orthogonal
    :math:`Q` by the Cayley or matrix-exponential transform. Reuses
    TopoBench's existing
    :class:`~topobench.nn.backbones.graph.nsd_utils.orthogonal.Orthogonal`.

    Two departures from the reference:

    * That class expects :math:`d(d+1)/2` parameters, but forms
      :math:`A = P - P^\top`, which annihilates the diagonal -- so :math:`d`
      of those parameters have identically zero gradient. We predict only the
      :math:`d(d-1)/2` that do anything and zero-fill the diagonal slots.
    * The reference defaults to a Householder parameterization, which needs
      the external ``torch_householder`` package. Cayley and ``matrix_exp``
      both produce matrices in :math:`SO(d)`, so the reflection component of
      :math:`O(d)` is unreachable here.

    Parameters
    ----------
    d : int
        Stalk dimension :math:`d`; must be greater than 1.
    hidden_channels : int
        Feature width :math:`f`.
    orthogonal_map : str, optional
        Either ``"cayley"`` (default) or ``"matrix_exp"``.
    **kwargs : dict, optional
        Forwarded to :class:`SheafBuilder`.

    Raises
    ------
    ValueError
        If ``d`` is not greater than 1, since a ``1 x 1`` orthogonal map
        carries no free parameters.
    """

    def __init__(
        self,
        d: int,
        hidden_channels: int,
        orthogonal_map: str = "cayley",
        **kwargs,
    ) -> None:
        if d <= 1:
            raise ValueError(
                "Orthogonal restriction maps need d > 1 (a 1x1 orthogonal "
                f"matrix has no free parameters); got d={d}."
            )
        super().__init__(
            d, hidden_channels, out_dim=d * (d - 1) // 2, **kwargs
        )
        self.orth_transform = Orthogonal(d=d, orthogonal_map=orthogonal_map)
        tril = torch.tril_indices(row=d, col=d, offset=0)
        # Which of Orthogonal's d(d+1)/2 slots are strictly below the
        # diagonal, i.e. the ones that survive A = P - P^T.
        self.register_buffer(
            "offdiag_slots", tril[0] != tril[1], persistent=False
        )

    def to_blocks(self, raw: torch.Tensor) -> torch.Tensor:
        """Map skew-symmetric parameters to orthogonal matrices.

        Parameters
        ----------
        raw : torch.Tensor
            Real tensor of shape ``[nnz, d * (d - 1) / 2]``.

        Returns
        -------
        torch.Tensor
            Orthogonal matrices of shape ``[nnz, d, d]``.
        """
        full = raw.new_zeros((raw.size(0), self.offdiag_slots.numel()))
        full[:, self.offdiag_slots] = raw
        return self.orth_transform(full)


SHEAF_BUILDERS = {
    "DiagSheafs": DiagSheafBuilder,
    "OrthoSheafs": OrthoSheafBuilder,
    "GeneralSheafs": GeneralSheafBuilder,
}
