"""Continuous Simplicial Neural Network backbone."""

import math

import torch
import torch.nn.functional as F
from torch import nn


class COSIMO(nn.Module):
    r"""Continuous Simplicial Neural Network.

    This module implements the COSIMO continuous Hodge diffusion idea from
    "Continuous Simplicial Neural Networks" (Einizade et al., NeurIPS 2025)
    for 0-, 1-, and 2-simplicial signals. Each layer evolves lower, upper, and
    cross-dimensional projected signals with heat kernels of the form
    ``exp(-t L) X`` as in the integrated dynamics of Eq. (8), and then learns a
    channel mixing map analogous to the multi-dimensional COSIMO layer.

    Parameters
    ----------
    in_channels_all : tuple[int, int, int]
        Input dimensions for node, edge, and face features.
    hidden_channels_all : tuple[int, int, int]
        Hidden dimensions for node, edge, and face features.
    n_layers : int, optional
        Number of COSIMO layers.
    t_init : float, optional
        Initial positive diffusion time for every branch.
    num_branches : int, optional
        Number of parallel continuous diffusion branches. This corresponds to
        the ``M`` filters aggregated in the COSIMO layer definition.
    diffusion_method : str, optional
        ``"taylor"`` uses a sparse Taylor approximation to avoid densifying
        Laplacians. ``"exact"`` uses ``torch.matrix_exp`` and is intended for
        small complexes and tests.
    taylor_order : int, optional
        Number of Taylor terms for sparse exponential diffusion.
    update_func : str, optional
        Nonlinearity applied after each layer. One of ``"relu"``,
        ``"sigmoid"``, ``"tanh"``, ``"leaky_relu"``, or ``None``.
    stabilize : bool, optional
        Whether to replace non-finite diffusion responses by finite values.
    normalize_laplacian : bool, optional
        Whether to divide Taylor diffusion operators by their maximum absolute
        row sum.
    max_diffusion_time : float, optional
        Upper bound for each learned diffusion time.
    max_abs_value : float, optional
        Absolute value used to clip stabilized diffusion responses.
    residual : bool, optional
        Whether to add an inter-layer residual connection at every rank.
    layer_norm : bool, optional
        Whether to apply per-rank layer normalization after each layer.
    """

    def __init__(
        self,
        in_channels_all,
        hidden_channels_all,
        n_layers=2,
        t_init=1.0,
        num_branches=2,
        diffusion_method="taylor",
        taylor_order=6,
        update_func="relu",
        stabilize=True,
        normalize_laplacian=True,
        max_diffusion_time=2.0,
        max_abs_value=1.0e4,
        residual=True,
        layer_norm=True,
    ):
        super().__init__()
        if len(in_channels_all) != 3 or len(hidden_channels_all) != 3:
            raise ValueError(
                "COSIMO expects channel tuples for 0-, 1-, and 2-simplices."
            )
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1.")
        if num_branches < 1:
            raise ValueError("num_branches must be at least 1.")

        self.residual = residual
        self.layer_norm = layer_norm

        self.in_linear_0 = nn.Linear(
            in_channels_all[0], hidden_channels_all[0]
        )
        self.in_linear_1 = nn.Linear(
            in_channels_all[1], hidden_channels_all[1]
        )
        self.in_linear_2 = nn.Linear(
            in_channels_all[2], hidden_channels_all[2]
        )

        if layer_norm:
            self.norms = nn.ModuleList(
                nn.ModuleList(
                    nn.LayerNorm(hidden_channels_all[rank])
                    for rank in range(3)
                )
                for _ in range(n_layers)
            )
        else:
            self.norms = None

        self.layers = nn.ModuleList(
            COSIMOLayer(
                in_channels=hidden_channels_all,
                out_channels=hidden_channels_all,
                t_init=t_init,
                num_branches=num_branches,
                diffusion_method=diffusion_method,
                taylor_order=taylor_order,
                update_func=update_func,
                stabilize=stabilize,
                normalize_laplacian=normalize_laplacian,
                max_diffusion_time=max_diffusion_time,
                max_abs_value=max_abs_value,
            )
            for _ in range(n_layers)
        )

    def forward(self, x_all, laplacian_all, incidence_all):
        r"""Forward computation.

        Parameters
        ----------
        x_all : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Feature tensors for 0-, 1-, and 2-simplices.
        laplacian_all : tuple[torch.Tensor, ...]
            ``(L0, L1_down, L1_up, L2_down, L2_up)`` for a 2-complex with
            available upper 2-Laplacian, or ``(L0, L1_down, L1_up, L2)``.
        incidence_all : tuple[torch.Tensor, torch.Tensor]
            Incidence matrices ``(B1, B2)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated embeddings for 0-, 1-, and 2-simplices.
        """
        x_0, x_1, x_2 = x_all
        x_all = (
            self.in_linear_0(x_0),
            self.in_linear_1(x_1),
            self.in_linear_2(x_2),
        )
        for depth, layer in enumerate(self.layers):
            out_all = layer(x_all, laplacian_all, incidence_all)
            if self.residual:
                out_all = tuple(
                    out + res for out, res in zip(out_all, x_all, strict=True)
                )
            if self.norms is not None:
                out_all = tuple(
                    norm(out)
                    for norm, out in zip(
                        self.norms[depth], out_all, strict=True
                    )
                )
            x_all = out_all
        return x_all


class COSIMOLayer(nn.Module):
    r"""One COSIMO layer with continuous Hodge diffusion branches.

    The lower and upper Hodge Laplacians are treated as independent diffusion
    generators. For each simplicial level, the layer combines diffused same-rank
    signals and diffused projections from adjacent ranks. The responses from
    ``num_branches`` independently parameterized receptive fields are
    concatenated and passed through a learnable linear aggregation, matching the
    branch aggregation in Eq. (11) of the COSIMO paper.

    Parameters
    ----------
    in_channels : tuple[int, int, int]
        Input dimensions for node, edge, and face embeddings.
    out_channels : tuple[int, int, int]
        Output dimensions for node, edge, and face embeddings.
    t_init : float, optional
        Initial positive diffusion time for every branch.
    num_branches : int, optional
        Number of parallel continuous diffusion branches.
    diffusion_method : str, optional
        Diffusion backend. ``"taylor"`` uses a sparse Taylor approximation,
        while ``"exact"`` uses a dense matrix exponential.
    taylor_order : int, optional
        Number of Taylor terms for sparse exponential diffusion.
    update_func : str, optional
        Nonlinearity applied after channel mixing.
    stabilize : bool, optional
        Whether to replace non-finite diffusion responses by finite values.
    normalize_laplacian : bool, optional
        Whether to divide Taylor diffusion operators by their maximum absolute
        row sum.
    max_diffusion_time : float, optional
        Upper bound for each learned diffusion time.
    max_abs_value : float, optional
        Absolute value used to clip stabilized diffusion responses.
    """

    _VALID_METHODS = {"exact", "taylor"}
    _VALID_UPDATES = {None, "relu", "sigmoid", "tanh", "leaky_relu"}

    def __init__(
        self,
        in_channels,
        out_channels,
        t_init=1.0,
        num_branches=2,
        diffusion_method="taylor",
        taylor_order=6,
        update_func="relu",
        stabilize=True,
        normalize_laplacian=True,
        max_diffusion_time=2.0,
        max_abs_value=1.0e4,
    ):
        super().__init__()
        if diffusion_method not in self._VALID_METHODS:
            raise ValueError(
                f"diffusion_method must be one of {self._VALID_METHODS}."
            )
        if update_func not in self._VALID_UPDATES:
            raise ValueError(
                f"update_func must be one of {self._VALID_UPDATES}."
            )
        if t_init <= 0:
            raise ValueError("t_init must be positive.")
        if num_branches < 1:
            raise ValueError("num_branches must be at least 1.")
        if taylor_order < 1:
            raise ValueError("taylor_order must be at least 1.")
        if max_diffusion_time <= 0:
            raise ValueError("max_diffusion_time must be positive.")
        if max_abs_value <= 0:
            raise ValueError("max_abs_value must be positive.")

        self.diffusion_method = diffusion_method
        self.num_branches = num_branches
        self.taylor_order = taylor_order
        self.update_func = update_func
        self.stabilize = stabilize
        self.normalize_laplacian = normalize_laplacian
        self.max_diffusion_time = max_diffusion_time
        self.max_abs_value = max_abs_value

        # The raw (un-diffused) same-rank signal is concatenated alongside the
        # diffused branch responses. Because ``exp(-t L)`` is a low-pass
        # (smoothing) filter, giving the linear mixer direct access to the raw
        # signal lets it realize high-pass responses of the form
        # ``a * x - b * exp(-t L) x``. Those are exactly the filters needed on
        # heterophilic complexes, where pure diffusion oversmooths and destroys
        # the class signal.
        self.mix_0 = nn.Linear(
            num_branches * (in_channels[0] + in_channels[1]) + in_channels[0],
            out_channels[0],
        )
        self.mix_1 = nn.Linear(
            num_branches
            * (in_channels[0] + 2 * in_channels[1] + in_channels[2])
            + in_channels[1],
            out_channels[1],
        )
        self.mix_2 = nn.Linear(
            num_branches * (in_channels[1] + 2 * in_channels[2])
            + in_channels[2],
            out_channels[2],
        )

        raw_t = math.log(math.expm1(t_init))
        self.raw_times = nn.ParameterDict(
            {
                f"{branch}_{name}": nn.Parameter(
                    torch.tensor(raw_t, dtype=torch.float32)
                )
                for branch in range(num_branches)
                for name in self.branch_names()
            }
        )

    def forward(self, x_all, laplacian_all, incidence_all):
        r"""Apply one continuous simplicial convolution layer.

        Parameters
        ----------
        x_all : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Feature tensors for 0-, 1-, and 2-simplices.
        laplacian_all : tuple[torch.Tensor, ...]
            Hodge Laplacian tensors for the available simplicial ranks.
        incidence_all : tuple[torch.Tensor, torch.Tensor]
            Incidence matrices ``(B1, B2)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated embeddings for 0-, 1-, and 2-simplices.
        """
        x_0, x_1, x_2 = x_all
        laplacian_0, laplacian_down_1, laplacian_up_1, *laplacian_2 = (
            laplacian_all
        )
        if len(laplacian_2) == 1:
            laplacian_down_2 = laplacian_2[0]
            laplacian_up_2 = self._zero_operator_like(laplacian_down_2)
        else:
            laplacian_down_2, laplacian_up_2 = laplacian_2
        incidence_1, incidence_2 = incidence_all

        # Seed each rank with its raw same-rank signal so the mixer can build
        # high-pass (heterophily-friendly) filters, not only low-pass diffusion.
        x_0_branches = [x_0]
        x_1_branches = [x_1]
        x_2_branches = [x_2]
        for branch in range(self.num_branches):
            x_0_branches.extend(
                [
                    self.diffuse(laplacian_0, x_0, branch, "x0_self"),
                    self.diffuse(
                        laplacian_0,
                        self.matmul(incidence_1, x_1),
                        branch,
                        "x1_to_x0",
                    ),
                ]
            )

            x_1_branches.extend(
                [
                    self.diffuse(laplacian_down_1, x_1, branch, "x1_lower"),
                    self.diffuse(laplacian_up_1, x_1, branch, "x1_upper"),
                    self.diffuse(
                        laplacian_down_1,
                        self.matmul(incidence_1.T, x_0),
                        branch,
                        "x0_to_x1",
                    ),
                    self.diffuse(
                        laplacian_up_1,
                        self.matmul(incidence_2, x_2),
                        branch,
                        "x2_to_x1",
                    ),
                ]
            )

            x_2_branches.extend(
                [
                    self.diffuse(laplacian_down_2, x_2, branch, "x2_lower"),
                    self.diffuse(laplacian_up_2, x_2, branch, "x2_upper"),
                    self.diffuse(
                        laplacian_down_2,
                        self.matmul(incidence_2.T, x_1),
                        branch,
                        "x1_to_x2",
                    ),
                ]
            )

        out = (
            self.mix_0(torch.cat(x_0_branches, dim=-1)),
            self.mix_1(torch.cat(x_1_branches, dim=-1)),
            self.mix_2(torch.cat(x_2_branches, dim=-1)),
        )
        return tuple(self.update(x) for x in out)

    @staticmethod
    def branch_names():
        """Return receptive-field names used by each COSIMO branch.

        Returns
        -------
        tuple[str, ...]
            Names of the diffusion paths parameterized in every branch.
        """
        return (
            "x0_self",
            "x1_to_x0",
            "x1_lower",
            "x1_upper",
            "x0_to_x1",
            "x2_to_x1",
            "x2_lower",
            "x2_upper",
            "x1_to_x2",
        )

    def diffuse(self, laplacian, x, branch, time_name):
        r"""Compute ``exp(-t L) X`` using the selected diffusion method.

        Parameters
        ----------
        laplacian : torch.Tensor
            Sparse or dense Laplacian operator.
        x : torch.Tensor
            Input feature matrix.
        branch : int
            Branch index selecting the learnable diffusion time.
        time_name : str
            Diffusion path name selecting the learnable diffusion time.

        Returns
        -------
        torch.Tensor
            Diffused feature matrix.
        """
        time = self.diffusion_time(branch, time_name, x)
        laplacian = laplacian.to(device=x.device, dtype=x.dtype)
        if self.diffusion_method == "exact":
            return self.ensure_finite(
                torch.matrix_exp(-time * laplacian.to_dense()) @ x
            )
        if self.normalize_laplacian:
            laplacian = self.normalize_operator(laplacian)
        return self.taylor_diffusion(laplacian, x, time)

    def diffusion_time(self, branch, time_name, x):
        """Return a positive, bounded diffusion time for a branch.

        Parameters
        ----------
        branch : int
            Branch index selecting the learnable diffusion time.
        time_name : str
            Diffusion path name selecting the learnable diffusion time.
        x : torch.Tensor
            Reference tensor whose device and dtype are reused.

        Returns
        -------
        torch.Tensor
            Positive scalar diffusion time clipped by ``max_diffusion_time``.
        """
        time = F.softplus(self.raw_times[f"{branch}_{time_name}"]).to(
            device=x.device, dtype=x.dtype
        )
        return time.clamp(max=self.max_diffusion_time)

    def taylor_diffusion(self, laplacian, x, time):
        r"""Sparse Taylor approximation of the heat kernel action.

        Parameters
        ----------
        laplacian : torch.Tensor
            Sparse Laplacian operator.
        x : torch.Tensor
            Input feature matrix.
        time : torch.Tensor
            Positive diffusion time.

        Returns
        -------
        torch.Tensor
            Approximate heat-kernel response.
        """
        result = x
        term = x
        for order in range(1, self.taylor_order + 1):
            term = (-time / order) * self.matmul(laplacian, term)
            term = self.ensure_finite(term)
            result = result + term
            result = self.ensure_finite(result)
        return self.ensure_finite(result)

    def matmul(self, operator, x):
        """Multiply a sparse or dense operator by a feature matrix.

        Parameters
        ----------
        operator : torch.Tensor
            Sparse or dense linear operator.
        x : torch.Tensor
            Feature matrix.

        Returns
        -------
        torch.Tensor
            Matrix product ``operator @ x``.
        """
        if operator.is_sparse:
            op = operator if operator.is_coalesced() else operator.coalesce()
            return torch.sparse.mm(op, x)
        return torch.mm(operator, x)

    def normalize_operator(self, operator):
        """Normalize an operator by its maximum absolute row sum.

        Parameters
        ----------
        operator : torch.Tensor
            Sparse or dense linear operator.

        Returns
        -------
        torch.Tensor
            Operator divided by a scale of at least one.
        """
        if operator.is_sparse:
            op = operator.coalesce()
            if op.shape[0] == 0:
                return op
            row_sum = torch.zeros(
                op.shape[0], device=op.device, dtype=op.dtype
            )
            row_sum.scatter_add_(0, op.indices()[0], op.values().abs())
            scale = row_sum.max().clamp_min(1.0)
            return torch.sparse_coo_tensor(
                op.indices(),
                op.values() / scale,
                op.shape,
                device=op.device,
                dtype=op.dtype,
            ).coalesce()
        if operator.shape[0] == 0:
            return operator
        scale = operator.abs().sum(dim=1).max().clamp_min(1.0)
        return operator / scale

    def ensure_finite(self, x):
        """Replace non-finite values when stabilization is enabled.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to sanitize.

        Returns
        -------
        torch.Tensor
            Finite tensor when stabilization is enabled, otherwise ``x``.
        """
        if not self.stabilize:
            return x
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.clamp(min=-self.max_abs_value, max=self.max_abs_value)

    def update(self, x):
        """Apply the configured pointwise nonlinearity.

        Parameters
        ----------
        x : torch.Tensor
            Input feature matrix.

        Returns
        -------
        torch.Tensor
            Activated feature matrix.
        """
        if self.update_func == "relu":
            return F.relu(x)
        if self.update_func == "sigmoid":
            return torch.sigmoid(x)
        if self.update_func == "tanh":
            return torch.tanh(x)
        if self.update_func == "leaky_relu":
            return F.leaky_relu(x)
        return x

    def _zero_operator_like(self, laplacian):
        """Return a sparse zero operator with the same shape and device.

        Parameters
        ----------
        laplacian : torch.Tensor
            Reference Laplacian whose shape, device, and dtype are reused.

        Returns
        -------
        torch.Tensor
            Zero operator matching the reference Laplacian.
        """
        if laplacian.is_sparse:
            indices = torch.empty(
                (2, 0), dtype=torch.long, device=laplacian.device
            )
            values = torch.empty(
                (0,), dtype=laplacian.dtype, device=laplacian.device
            )
            return torch.sparse_coo_tensor(
                indices, values, laplacian.shape, device=laplacian.device
            )
        return torch.zeros_like(laplacian)
