r"""Complex arithmetic for directed sheaf diffusion, in real arithmetic.

Appendix D of the paper reduces complex products to real ones: for
:math:`Y = AX` with :math:`A = A_R + iA_I` and :math:`X = X_R + iX_I`,

.. math::

    \hat{X} = \begin{bmatrix} X_R \\ X_I \end{bmatrix}, \qquad
    \hat{A} = \begin{bmatrix} A_R & -A_I \\ A_I & A_R \end{bmatrix},
    \qquad \hat{Y} = \hat{A}\hat{X},

"so that complex-valued operations can be reduced to real-valued operations
with a constant factor overhead". This module implements exactly that, and the
rest of the backbone never materializes a complex tensor.

Three reasons this is the right representation here, beyond following the paper:

* the compiled sparse helpers used elsewhere in TopoBench document no support
  for complex dtypes, so a complex operator would rely on an implementation
  accident;
* the nonlinearity of Sec. 3 gates on :math:`\Re(z)`, and the lifting supplies
  the real/imaginary split it needs for free;
* everything stays in a real dtype, so autograd, scatter/gather and reduced
  precision all behave conventionally.

Layout convention: a lifted feature matrix of shape ``[2 * nd, f]`` holds
:math:`\Re(X)` in its first ``nd`` rows and :math:`\Im(X)` in the last ``nd``.
Within each half, node ``u`` owns rows ``u*d`` through ``u*d + d - 1``.
"""

import torch


def lifted_index(real_index, imag_index, size: int):
    r"""Assemble the sparse indices of the real lifting :math:`\hat{A}`.

    The four blocks are emitted in the order ``A_R`` (top left), ``A_R``
    (bottom right), ``-A_I`` (top right), ``+A_I`` (bottom left);
    :func:`lifted_value` must be called with the matching value order.

    Parameters
    ----------
    real_index : torch.Tensor
        Indices of :math:`A_R`, shape ``[2, nnz_real]``.
    imag_index : torch.Tensor
        Indices of :math:`A_I`, shape ``[2, nnz_imag]``.
    size : int
        Side length of the complex operator, i.e. ``num_nodes * d``.

    Returns
    -------
    torch.Tensor
        Indices of shape ``[2, 2 * nnz_real + 2 * nnz_imag]`` addressing a
        ``2 * size`` square matrix.
    """
    top_left = real_index
    bottom_right = real_index + size
    top_right = torch.stack([imag_index[0], imag_index[1] + size])
    bottom_left = torch.stack([imag_index[0] + size, imag_index[1]])
    return torch.cat([top_left, bottom_right, top_right, bottom_left], dim=1)


def lifted_value(real_value, imag_value):
    r"""Assemble the sparse values of the real lifting :math:`\hat{A}`.

    Parameters
    ----------
    real_value : torch.Tensor
        Values of :math:`A_R`, shape ``[nnz_real]``.
    imag_value : torch.Tensor
        Values of :math:`A_I`, shape ``[nnz_imag]``.

    Returns
    -------
    torch.Tensor
        Values ordered to match :func:`lifted_index`, of shape
        ``[2 * nnz_real + 2 * nnz_imag]``.
    """
    return torch.cat([real_value, real_value, -imag_value, imag_value])


def stack_real(x):
    r"""Lift a real feature matrix to a complex one with zero imaginary part.

    Input features are real, and the paper does not specify an imaginary part
    for :math:`X^{0}`; we use zeros so that the readout's real and imaginary
    halves are not duplicates at the first layer. The reference implementation
    defaults to the opposite (``complex_copy_values``, which sets
    :math:`\Im(X^{0}) = \Re(X^{0})`).

    Parameters
    ----------
    x : torch.Tensor
        Real features of shape ``[nd, f]``.

    Returns
    -------
    torch.Tensor
        Lifted features of shape ``[2 * nd, f]``.
    """
    return torch.cat([x, torch.zeros_like(x)], dim=0)


def split_parts(x, size: int):
    """Split a lifted matrix into its real and imaginary halves.

    Parameters
    ----------
    x : torch.Tensor
        Lifted features of shape ``[2 * size, f]``.
    size : int
        Number of rows of the complex matrix, i.e. ``num_nodes * d``.

    Returns
    -------
    real : torch.Tensor
        Real part, shape ``[size, f]``.
    imag : torch.Tensor
        Imaginary part, shape ``[size, f]``.
    """
    return x[:size], x[size:]


def complex_relu_split(x, size: int):
    r"""Apply the paper's complex ReLU to a lifted matrix.

    Section 3 defines

    .. math::

        \sigma(z) = \begin{cases} z & \Re(z) \ge 0 \\ 0 &
        \text{otherwise,}\end{cases}

    so the gate is decided by the real part and zeroes **both** components.
    This is not a componentwise ReLU: :math:`\sigma(-1 + 2i) = 0`, not
    :math:`2i`.

    Parameters
    ----------
    x : torch.Tensor
        Lifted features of shape ``[2 * size, f]``.
    size : int
        Number of rows of the complex matrix.

    Returns
    -------
    torch.Tensor
        Gated features of shape ``[2 * size, f]``.
    """
    real, _ = split_parts(x, size)
    keep = (real >= 0).to(x.dtype)
    return x * keep.repeat(2, 1)


def complex_dropout_split(x, size: int, p: float, training: bool):
    """Drop whole complex entries from a lifted matrix.

    A single Bernoulli mask is drawn per complex entry so that the real and
    imaginary parts of an entry are dropped together, keeping the argument of
    the surviving entries intact.

    Dropout is not part of Eq. 8; its placement follows the Neural Sheaf
    Diffusion implementation the paper's hyperparameter names come from. The
    merged mask is our choice: that implementation is real-valued and so has
    no such decision to make, and the reference implementation of this paper
    defaults to dropping the two parts independently
    (``complex_separate_dropout``), which does not preserve the argument.

    Parameters
    ----------
    x : torch.Tensor
        Lifted features of shape ``[2 * size, f]``.
    size : int
        Number of rows of the complex matrix.
    p : float
        Probability of dropping an entry.
    training : bool
        Whether the module is in training mode; if false this is the identity.

    Returns
    -------
    torch.Tensor
        Features of shape ``[2 * size, f]``.
    """
    if not training or p <= 0.0:
        return x
    real, _ = split_parts(x, size)
    keep = torch.rand_like(real) >= p
    scale = keep.to(x.dtype) / (1.0 - p)
    return x * scale.repeat(2, 1)


def unwind_split(x, num_nodes: int):
    r"""Concatenate the real and imaginary parts into real node features.

    Implements the paper's readout
    :math:`\mathrm{unwind}(X) = (\Re(X) \Vert \Im(X)) \in
    \mathbb{R}^{n \times 2c}`.

    Parameters
    ----------
    x : torch.Tensor
        Lifted features of shape ``[2 * num_nodes * d, f]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    torch.Tensor
        Real features of shape ``[num_nodes, 2 * d * f]``.
    """
    size = x.size(0) // 2
    real, imag = split_parts(x, size)
    return torch.cat(
        [real.reshape(num_nodes, -1), imag.reshape(num_nodes, -1)], dim=1
    )


def hermitian_to_dense(
    real_index, real_value, imag_index, imag_value, size: int
):
    """Materialize the complex operator densely, for tests and diagnostics.

    Uses ``O(size^2)`` memory, so it is not used on the training path.

    Parameters
    ----------
    real_index : torch.Tensor
        Indices of the real part, shape ``[2, nnz_real]``.
    real_value : torch.Tensor
        Values of the real part, shape ``[nnz_real]``.
    imag_index : torch.Tensor
        Indices of the imaginary part, shape ``[2, nnz_imag]``.
    imag_value : torch.Tensor
        Values of the imaginary part, shape ``[nnz_imag]``.
    size : int
        Side length of the operator.

    Returns
    -------
    torch.Tensor
        Complex matrix of shape ``[size, size]``.
    """
    real = torch.zeros(
        size, size, dtype=real_value.dtype, device=real_value.device
    )
    real.index_put_(
        (real_index[0], real_index[1]), real_value, accumulate=True
    )
    imag = torch.zeros_like(real)
    if imag_index.numel():
        imag.index_put_(
            (imag_index[0], imag_index[1]), imag_value, accumulate=True
        )
    return torch.complex(real, imag)
