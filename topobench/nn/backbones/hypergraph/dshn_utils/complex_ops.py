r"""Complex-valued primitives for Directional Sheaf Hypergraph Networks.

DSHN operates on complex features :math:`X \in \mathbb{C}^{nd \times f}`
(Eq. 9), so every layer component has to be lifted to the complex domain.
This module collects those primitives, following Appendix C.2 / C.3 of

    E. Mule, S. Fiorini, A. Purificato, F. Siciliano, S. Coniglio and
    F. Silvestri, *"Directional Sheaf Hypergraph Networks: Unifying Learning
    on Directed and Undirected Hypergraphs"*, ICLR 2026,
    https://arxiv.org/abs/2510.04727

and the authors' reference implementation
(https://github.com/EmaMule/DirectionalSheafHypergraphs), where the same
pieces live in ``models/complex_utils/complex_operators.py`` and
``models/complex_utils/complex_linear.py``.

Note that all of these operators derive one mask or weight from the real part
and apply it to both components, matching the reference. That is deliberate in
both codebases: it keeps :math:`\Re` and :math:`\Im` on the same computational
footing, which is what makes the undirected collapse described in
:mod:`.laplacian` exact.
"""

import torch
import torch.nn.functional as F
from torch import nn


def unwind(x: torch.Tensor) -> torch.Tensor:
    r"""Map complex features to real ones by concatenating the components.

    Implements the ``unwind`` operator of the paper (p. 8),

    .. math::
        \mathrm{unwind}(X) = \Re(X) \,\|\, \Im(X),

    used both to feed the final classifier and to feed the restriction-map
    predictor :math:`\Phi` (§C.1, item 3).

    .. note::
        The paper states the codomain as :math:`\mathbb{R}^{n \times 2f}`,
        which is imprecise: the operand of Eq. 9 lives in
        :math:`\mathbb{C}^{nd \times f}`. The reference implementation
        (``models/sheafgedi.py``, ``x.view(num_nodes, MLP_hidden * d)``)
        reshapes to ``[n, d * f]`` first, so the true codomain is
        :math:`\mathbb{R}^{n \times 2df}`. We follow the implementation.

    Parameters
    ----------
    x : torch.Tensor
        Complex tensor of any shape ``[..., k]``.

    Returns
    -------
    torch.Tensor
        Real tensor of shape ``[..., 2 * k]``.
    """
    return torch.cat((x.real, x.imag), dim=-1)


def complex_relu(x: torch.Tensor) -> torch.Tensor:
    r"""Apply the complex ReLU of Zhang et al. (2021).

    Defined in Appendix C.2 (p. 25) as

    .. math::
        \mathrm{ReLU}(x) = \begin{cases}
            x & \text{if } \Re(x) > 0 \\
            0 & \text{otherwise.}
        \end{cases}

    The gate depends only on :math:`\Re(x)` and is applied to both
    components, so a signal whose imaginary part equals its real part keeps
    that property (see :mod:`.laplacian` for why this matters).

    Parameters
    ----------
    x : torch.Tensor
        Complex input tensor.

    Returns
    -------
    torch.Tensor
        Complex tensor of the same shape as ``x``.
    """
    # The reference uses `>= 0` (complex_operators.py:16); the paper writes
    # `> 0`. The two differ only on the null set Re(x) == 0.
    return x * (x.real >= 0)


def complex_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    r"""Drop whole complex entries.

    One Bernoulli mask is drawn on the real part and applied to both
    components, so an entry is either kept in full or zeroed in full. This
    matches the reference (``complex_operators.py:119-120``) and differs from
    dropping :math:`\Re` and :math:`\Im` independently.

    Parameters
    ----------
    x : torch.Tensor
        Complex input tensor.
    p : float
        Probability of zeroing an entry.
    training : bool
        Whether to apply dropout. When ``False`` the input is returned
        unchanged.

    Returns
    -------
    torch.Tensor
        Complex tensor of the same shape as ``x``.
    """
    if not training or p == 0.0:
        return x
    mask = F.dropout(torch.ones_like(x.real), p=p, training=True)
    return x * mask


class RealLinear(nn.Module):
    r"""Apply one real weight matrix to both components of a complex input.

    The learnable maps of Eq. 9 (:math:`W_1 \in \mathbb{R}^{d \times d}` and
    :math:`W_2 \in \mathbb{R}^{f \times f}`) are **real**, so a complex input
    is transformed component-wise by the same weights,

    .. math::
        \mathrm{RealLinear}(x) = W\Re(x) + i\,W\Im(x).

    This is not the same as a complex-valued linear layer (which would need
    two independent weight matrices and mix the components). Reference:
    ``models/complex_utils/complex_linear.py:135-138``.

    Parameters
    ----------
    in_channels : int
        Size of each input sample.
    out_channels : int
        Size of each output sample.
    bias : bool, optional
        Whether to learn an additive bias (default: ``True``). The bias is
        real and, following the reference, is added to both components.

    Attributes
    ----------
    lin : torch.nn.Linear
        The shared real-valued linear map.
    """

    def __init__(
        self, in_channels: int, out_channels: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def reset_parameters(self) -> None:
        """Reset the parameters of the underlying linear map."""
        self.lin.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared real map to both components.

        Parameters
        ----------
        x : torch.Tensor
            Complex tensor of shape ``[..., in_channels]``.

        Returns
        -------
        torch.Tensor
            Complex tensor of shape ``[..., out_channels]``.
        """
        return torch.complex(self.lin(x.real), self.lin(x.imag))


class ComplexLayerNorm(nn.Module):
    r"""Whiten complex features as two-dimensional real vectors.

    Implements the complex layer normalization of Appendix C.3 (pp. 24-25),
    after Trabelsi et al. (2018) and Barrachina et al. (2023). Treating
    :math:`x` as the real vector :math:`(\Re(x), \Im(x))` with mean
    :math:`\mu` and covariance

    .. math::
        \Sigma = \begin{bmatrix}
            \sigma_{rr} & \sigma_{ri} \\ \sigma_{ri} & \sigma_{ii}
        \end{bmatrix},

    the layer applies

    .. math::
        \tilde{x} = \Sigma^{-1/2}(x - \mu), \qquad
        x_o = \gamma \tilde{x} + \beta,

    with :math:`\gamma \in \mathbb{R}^{2 \times 2}` and
    :math:`\beta \in \mathbb{R}^2` learnable. Following the paper, they are
    initialized to :math:`\gamma = \tfrac{1}{\sqrt{2}} I_2` and
    :math:`\beta = 0`, "preserving the norm of unit-modulus inputs while
    maintaining the identity mapping at initialization".

    Because :math:`\Sigma` is a symmetric positive-definite ``2 x 2`` matrix,
    its inverse square root is available in closed form. With
    :math:`s = \sqrt{\det \Sigma}` and :math:`t = \operatorname{tr} \Sigma`,

    .. math::
        \Sigma^{-1/2} = \frac{1}{s\sqrt{t + 2s}}
            \begin{bmatrix}
                \sigma_{ii} + s & -\sigma_{ri} \\
                -\sigma_{ri} & \sigma_{rr} + s
            \end{bmatrix},

    which avoids an eigendecomposition in the forward pass.

    Parameters
    ----------
    num_features : int
        Size of the trailing (feature) dimension. Retained for interface
        symmetry with :class:`torch.nn.LayerNorm`; the statistics are always
        taken over that dimension.
    eps : float, optional
        Value added to the covariance diagonal for numerical stability
        (default: ``1e-5``).
    elementwise_affine : bool, optional
        Whether to learn :math:`\gamma` and :math:`\beta` (default:
        ``True``).

    Attributes
    ----------
    gamma : torch.nn.Parameter or None
        The ``2 x 2`` affine weight, initialized to ``I / sqrt(2)``.
    beta : torch.nn.Parameter or None
        The length-2 affine bias, initialized to zero.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.gamma = nn.Parameter(torch.empty(2, 2))
            self.beta = nn.Parameter(torch.empty(2))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Reset ``gamma`` to :math:`I_2/\sqrt{2}` and ``beta`` to zero."""
        if self.elementwise_affine:
            with torch.no_grad():
                self.gamma.copy_(torch.eye(2) / (2.0**0.5))
                self.beta.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Whiten and optionally affinely transform a complex tensor.

        Parameters
        ----------
        x : torch.Tensor
            Complex tensor of shape ``[..., num_features]``.

        Returns
        -------
        torch.Tensor
            Complex tensor of the same shape as ``x``.
        """
        xr, xi = x.real, x.imag
        xr = xr - xr.mean(dim=-1, keepdim=True)
        xi = xi - xi.mean(dim=-1, keepdim=True)

        s_rr = (xr * xr).mean(dim=-1, keepdim=True) + self.eps
        s_ii = (xi * xi).mean(dim=-1, keepdim=True) + self.eps
        s_ri = (xr * xi).mean(dim=-1, keepdim=True)

        det = (s_rr * s_ii - s_ri * s_ri).clamp_min(self.eps)
        s = det.sqrt()
        t = s_rr + s_ii
        scale = 1.0 / (s * (t + 2.0 * s).clamp_min(self.eps).sqrt())

        # Sigma^{-1/2} applied to (xr, xi); see the closed form above.
        wr = scale * ((s_ii + s) * xr - s_ri * xi)
        wi = scale * (-s_ri * xr + (s_rr + s) * xi)

        if self.elementwise_affine:
            out_r = self.gamma[0, 0] * wr + self.gamma[0, 1] * wi
            out_i = self.gamma[1, 0] * wr + self.gamma[1, 1] * wi
            out_r = out_r + self.beta[0]
            out_i = out_i + self.beta[1]
        else:
            out_r, out_i = wr, wi

        return torch.complex(out_r, out_i)
