"""TopoU-Net: a U-Net architecture for topological domains.

Implementation of "TopoU-Net: A U-Net Architecture for Topological Domains"
(arXiv:2605.10091). No official implementation has been released; this module
follows the equations of the paper directly:

- Incidence-convolution rank transports (Section 3.2, Section 3.5):
  :math:`T^{\\uparrow}(H) = \\sigma(\\bar{B}^{\\top}_{s_i,s_{i+1}} H W^{\\uparrow}_i)` and
  :math:`T^{\\downarrow}(G) = \\sigma(\\bar{B}_{s_i,s_{i+1}} G W^{\\downarrow}_i)`.
- Encoder/decoder recursion with within-rank refinements and matched-rank
  skip merges (Definition 3.3).
- Additive skip merge (Section 3.5):
  :math:`D_{s_i} = \\sigma((E_{s_i} + \\tilde{D}_{s_i}) W^m_i)`.
"""

import torch


class TopoUNet(torch.nn.Module):
    r"""TopoU-Net backbone (arXiv:2605.10091).

    A U-shaped encoder-decoder over the ranked cells of a combinatorial
    complex. Given an increasing encoder rank path
    :math:`S = (s_0 < s_1 < \dots < s_L)` (Section 3.2 of the paper), the
    encoder transports cochains upward along incidence matrices, a bottleneck
    map is applied at rank :math:`s_L`, and the decoder transports features
    back down, merging encoder and decoder states at matched ranks through
    skip connections (Definition 3.3).

    The canonical instantiation of Section 3.5 is used:

    - Upward transport: :math:`E_{s_{i+1}} = \Phi_{s_{i+1}}(\sigma(
      \bar{B}^{\top}_{s_i, s_{i+1}} E_{s_i} W^{\uparrow}_i))`.
    - Bottleneck: :math:`D_{s_L} = \Omega_{s_L}(E_{s_L})`.
    - Downward transport: :math:`\tilde{D}_{s_i} = \Psi_{s_i}(\sigma(
      \bar{B}_{s_i, s_{i+1}} D_{s_{i+1}} W^{\downarrow}_i))`.
    - Additive skip merge: :math:`D_{s_i} = \sigma((E_{s_i} +
      \tilde{D}_{s_i}) W^m_i)` when ``use_skip`` is True, otherwise
      :math:`D_{s_i} = \tilde{D}_{s_i}` (no-skip ablation, Section 4.6.3).

    Within-rank refinements :math:`\Phi` and :math:`\Psi` are pointwise MLPs
    (Section 3.5 lists this as the canonical choice). All maps act on cochain
    spaces :math:`\mathcal{C}^{r}(\mathcal{X}; \mathbb{R}^{d})` with a shared
    feature dimension ``in_channels`` across ranks, following the
    hyperparameter settings of Appendix B.4.

    If two consecutive ranks of the path are not consecutive integers
    (:math:`s_{i+1} - s_i > 1`), the transport uses the direct incidence
    matrix :math:`B_{s_i, s_{i+1}}` (Section 3.5), computed here as the
    binarized product of the consecutive incidence matrices
    :math:`B_{s_i, s_i+1} \cdots B_{s_{i+1}-1, s_{i+1}}`.

    Parameters
    ----------
    in_channels : int
        Feature dimension shared by all ranks of the path.
    encoder_rank_path : list[int]
        Strictly increasing encoder rank path :math:`S = (s_0 < \dots <
        s_L)`. The full U-shaped traversal reverses it, e.g. ``[0, 1, 2]``
        realizes the path :math:`0 \to 1 \to 2 \to 1 \to 0`.
    use_skip : bool, optional
        Whether to apply the additive skip merges at matched ranks. Setting
        it to False reproduces the no-skip ablation of Section 4.6.3
        (default: True).
    aggr_norm : bool, optional
        If True, incidence matrices are degree-normalized so that transports
        average (rather than sum) over incident cells, as allowed by
        Section 3.2 (default: True).
    dropout : float, optional
        Dropout rate applied inside the within-rank refinements, following
        Appendix B.4 (default: 0.0).
    **kwargs : dict
        Additional keyword arguments accepted for API compatibility.
    """

    def __init__(
        self,
        in_channels: int,
        encoder_rank_path: list[int],
        use_skip: bool = True,
        aggr_norm: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        encoder_rank_path = list(encoder_rank_path)
        if len(encoder_rank_path) < 2:
            raise ValueError(
                "encoder_rank_path must contain at least two ranks, got "
                f"{encoder_rank_path}"
            )
        if any(
            s_next <= s
            for s, s_next in zip(
                encoder_rank_path, encoder_rank_path[1:], strict=False
            )
        ):
            raise ValueError(
                "encoder_rank_path must be strictly increasing, got "
                f"{encoder_rank_path}"
            )

        self.in_channels = in_channels
        self.encoder_rank_path = encoder_rank_path
        self.use_skip = use_skip
        self.aggr_norm = aggr_norm
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

        n_steps = len(encoder_rank_path) - 1
        d = in_channels
        # Per-step transport weights W^{up}_i, W^{down}_i (Section 3.5).
        self.w_up = torch.nn.ModuleList(
            torch.nn.Linear(d, d) for _ in range(n_steps)
        )
        self.w_down = torch.nn.ModuleList(
            torch.nn.Linear(d, d) for _ in range(n_steps)
        )
        # Within-rank refinements Phi_{s_{i+1}} (encoder) and Psi_{s_i}
        # (decoder), implemented as pointwise MLPs (Section 3.5).
        self.phi = torch.nn.ModuleList(
            torch.nn.Linear(d, d) for _ in range(n_steps)
        )
        self.psi = torch.nn.ModuleList(
            torch.nn.Linear(d, d) for _ in range(n_steps)
        )
        # Skip-merge weights W^m_i (Section 3.5).
        self.w_merge = torch.nn.ModuleList(
            torch.nn.Linear(d, d) for _ in range(n_steps)
        )
        # Bottleneck map Omega_{s_L} (Definition 3.3).
        self.omega = torch.nn.Linear(d, d)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, "
            f"encoder_rank_path={self.encoder_rank_path}, "
            f"use_skip={self.use_skip}, aggr_norm={self.aggr_norm})"
        )

    def _step_incidence(
        self, incidence_all: dict[int, torch.Tensor], src: int, dst: int
    ) -> torch.Tensor:
        r"""Return the incidence matrix :math:`B_{src, dst}` for a path step.

        ``incidence_all[r]`` holds the consecutive incidence
        :math:`B_{r-1, r}` of shape :math:`n_{r-1} \times n_r`. For a step
        skipping ranks (:math:`dst - src > 1`), the direct incidence of
        Section 3.5 is computed as the binarized product of consecutive
        incidence matrices.

        Parameters
        ----------
        incidence_all : dict[int, torch.Tensor]
            Sparse consecutive incidence matrices keyed by their upper rank.
        src : int
            Lower rank of the step.
        dst : int
            Upper rank of the step.

        Returns
        -------
        torch.Tensor
            Sparse incidence matrix of shape :math:`n_{src} \times n_{dst}`.
        """
        incidence = incidence_all[src + 1]
        for rank in range(src + 2, dst + 1):
            incidence = torch.sparse.mm(incidence, incidence_all[rank])
        if incidence._nnz() > 0:
            incidence = torch.sparse_coo_tensor(
                incidence._indices(),
                torch.ones_like(incidence._values()),
                incidence.size(),
            ).coalesce()
        return incidence

    def _transport(
        self, incidence: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        r"""Aggregate features through a sparse incidence matrix.

        Computes :math:`\bar{B} H` where :math:`\bar{B}` is either the raw
        incidence matrix or, when ``aggr_norm`` is True, its degree-normalized
        version (mean over incident cells), as permitted by Section 3.2.

        Parameters
        ----------
        incidence : torch.Tensor
            Sparse matrix of shape :math:`n_{dst} \times n_{src}` (already
            oriented in the transport direction).
        x : torch.Tensor
            Cochain of shape :math:`n_{src} \times d`.

        Returns
        -------
        torch.Tensor
            Transported cochain of shape :math:`n_{dst} \times d`.
        """
        out = torch.sparse.mm(incidence, x)
        if self.aggr_norm:
            degree = torch.sparse.sum(incidence, dim=1).to_dense()
            out = out / degree.clamp(min=1.0).unsqueeze(-1)
        return out

    def forward(
        self,
        x_all: dict[int, torch.Tensor],
        incidence_all: dict[int, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        r"""Forward pass through the U-shaped rank traversal.

        Implements the encoder/decoder recursion of Definition 3.3 with the
        canonical maps of Section 3.5.

        Parameters
        ----------
        x_all : dict[int, torch.Tensor]
            Input cochains ``{rank: tensor of shape (n_rank, in_channels)}``.
            Only the entry at the input rank :math:`s_0` is consumed; the
            model reconstructs all higher-rank states through transport.
        incidence_all : dict[int, torch.Tensor]
            Sparse consecutive incidence matrices ``{r: B_{r-1, r}}`` for all
            ranks :math:`r` up to the bottleneck rank :math:`s_L`.

        Returns
        -------
        dict[int, torch.Tensor]
            Decoder states ``{rank: tensor of shape (n_rank, in_channels)}``
            for every rank of the encoder path; the entry at :math:`s_0` is
            the output cochain :math:`D_{s_0}`.
        """
        path = self.encoder_rank_path

        # Encoder: E_{s_{i+1}} = Phi(sigma(B^T E_{s_i} W^up_i))
        # (Definition 3.3).
        encoder_states = {path[0]: x_all[path[0]]}
        for i, (src, dst) in enumerate(zip(path, path[1:], strict=False)):
            incidence = self._step_incidence(incidence_all, src, dst)
            up = self.act(
                self._transport(
                    incidence.t(), self.w_up[i](encoder_states[src])
                )
            )
            encoder_states[dst] = self.dropout(self.act(self.phi[i](up)))

        # Bottleneck: D_{s_L} = Omega(E_{s_L}) (Definition 3.3).
        decoder_states = {
            path[-1]: self.act(self.omega(encoder_states[path[-1]]))
        }

        # Decoder: D~_{s_i} = Psi(sigma(B D_{s_{i+1}} W^down_i)), then
        # D_{s_i} = sigma((E_{s_i} + D~_{s_i}) W^m_i) at matched ranks
        # (Definition 3.3, Section 3.5).
        for i in reversed(range(len(path) - 1)):
            src, dst = path[i], path[i + 1]
            incidence = self._step_incidence(incidence_all, src, dst)
            down = self.act(
                self._transport(incidence, self.w_down[i](decoder_states[dst]))
            )
            d_tilde = self.dropout(self.act(self.psi[i](down)))
            if self.use_skip:
                decoder_states[src] = self.act(
                    self.w_merge[i](encoder_states[src] + d_tilde)
                )
            else:
                decoder_states[src] = d_tilde

        return decoder_states
