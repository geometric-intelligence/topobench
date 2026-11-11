"""OGFormer Neighborhood Maximum Homogeneity loss function.

Implements Eqs. (13)-(16) of Zhang et al. "A graph transformer with
optimized attention scores for node classification" (Scientific Reports,
2025).
"""

import torch
import torch.nn.functional as F
import torch_geometric

from topobench.loss.base import AbstractLoss


class OGFormerLoss(AbstractLoss):
    r"""Neighborhood Maximum Homogeneity loss for OGFormer.

    Combines, for every OGFormer layer (Eq. (16)):

    - a KL-divergence term :math:`\mathcal{L}_{KL} = \sum_{i,j}
      \hat{R}_{ij} D_{KL}(Q_i \| Q_j)` (Eqs. (13)-(14)) that suppresses
      attention between nodes with dissimilar query distributions, and
    - a neighborhood homogeneity term :math:`\mathcal{L}_h =
      \|1 - h(\hat{R}, Y_{train} \| Y_{pred})\|_2` (Eq. (15)) that
      increases the attention mass placed on same-label nodes.

    The homogeneity term uses ground-truth labels for training nodes and
    model predictions for the remaining nodes (to prevent label leakage in
    transductive settings), and is only applied to node-level
    classification tasks. Both terms are computed only during training.

    Parameters
    ----------
    lambda_kl : float, optional
        Weight of the KL-divergence term; the paper recommends searching
        in {0, 1e-3, 1e-4, 1e-5, 1e-7} (default: 1e-4).
    lambda_h : float, optional
        Weight of the neighborhood homogeneity term (default: 1e-4).
    kl_norm_p : float, optional
        Norm used to project queries onto probability distributions
        before the KL divergence (default: 1.0).
    """

    def __init__(
        self,
        lambda_kl: float = 1e-4,
        lambda_h: float = 1e-4,
        kl_norm_p: float = 1.0,
    ):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_h = lambda_h
        self.kl_norm_p = kl_norm_p

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(lambda_kl={self.lambda_kl}, "
            f"lambda_h={self.lambda_h}, kl_norm_p={self.kl_norm_p})"
        )

    def kl_divergence_matrix(self, queries: torch.Tensor) -> torch.Tensor:
        r"""Pairwise KL divergence between node query distributions.

        Queries are normalized to probability distributions and
        :math:`D_{KL}(Q_i \| Q_j) = \sum_d Q_i(d) \log(Q_i(d) / Q_j(d))`
        is computed for every node pair (Eq. (13)).

        Parameters
        ----------
        queries : torch.Tensor
            Query embeddings of shape [num_nodes, hidden_channels] with
            positive entries (sigmoid outputs).

        Returns
        -------
        torch.Tensor
            KL divergence matrix of shape [num_nodes, num_nodes].
        """
        q = F.normalize(queries, p=self.kl_norm_p, dim=1)
        log_q = torch.log(q + 1e-10)
        self_term = (q * log_q).sum(dim=1)
        cross_term = q @ log_q.T
        return self_term.unsqueeze(1) - cross_term

    @staticmethod
    def weighted_homophily(
        attention: torch.Tensor, y: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        r"""Attention-weighted neighborhood homogeneity rate per node.

        Implements :math:`h_i = \sum_{j: Y_i = Y_j} \hat{R}_{ij} /
        \sum_j \hat{R}_{ij}` (Eq. (15)).

        Parameters
        ----------
        attention : torch.Tensor
            Row-normalized attention scores of shape
            [num_nodes, num_nodes].
        y : torch.Tensor
            Node labels of shape [num_nodes].
        eps : float, optional
            Small value preventing division by zero for nodes without
            attention mass (default: 1e-8).

        Returns
        -------
        torch.Tensor
            Homogeneity rates of shape [num_nodes] in [0, 1].
        """
        same_label = (y.unsqueeze(0) == y.unsqueeze(1)).to(attention.dtype)
        return (attention * same_label).sum(dim=1) / attention.sum(
            dim=1
        ).clamp_min(eps)

    @staticmethod
    def _effective_labels(
        model_out: dict, batch: torch_geometric.data.Data
    ) -> torch.Tensor | None:
        """Assemble per-node labels for the homogeneity term.

        Uses ground-truth labels for training nodes and model predictions
        for the remaining nodes when a training mask is available
        (transductive setting), otherwise all node labels are known
        (inductive setting) and used directly. Returns None when the task
        provides no per-node integer labels.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        torch.Tensor or None
            Effective node labels of shape [num_nodes], or None.
        """
        y = batch.y
        num_nodes = batch.x_0.shape[0]
        if y is None or y.dim() != 1 or y.shape[0] != num_nodes:
            return None
        if torch.is_floating_point(y):
            return None

        train_mask = batch.get("train_mask", None)
        node_logits = model_out.get("node_logits")
        if (
            train_mask is not None
            and train_mask.dtype == torch.bool
            and train_mask.shape[0] == num_nodes
            and not bool(train_mask.all())
            and node_logits is not None
            and node_logits.shape[0] == num_nodes
        ):
            y = y.clone()
            y[~train_mask] = node_logits.argmax(dim=1)[~train_mask]
        return y

    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> torch.Tensor:
        r"""Forward pass of the loss function.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output, including the
            per-layer ``ogformer_queries`` and ``ogformer_attention``
            auxiliary outputs of the OGFormer backbone.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        torch.Tensor
            The Neighborhood Maximum Homogeneity loss (Eq. (16)), zero outside training.
        """
        queries = model_out.get("ogformer_queries")
        attention_scores = model_out.get("ogformer_attention")
        if not queries or not attention_scores:  # Validation and test
            return torch.tensor(0.0, device=batch.x_0.device)

        loss = torch.tensor(0.0, device=batch.x_0.device)

        if self.lambda_kl > 0:
            for q, attention in zip(queries, attention_scores, strict=True):
                kl = self.kl_divergence_matrix(q)
                loss = loss + self.lambda_kl * (kl * attention).sum()

        if self.lambda_h > 0:
            y = self._effective_labels(model_out, batch)
            if y is not None:
                for attention in attention_scores:
                    h = self.weighted_homophily(attention, y)
                    loss = loss + self.lambda_h * torch.norm(1.0 - h)

        return loss
