"""Sheaf Connection Laplacian Positional Encoding (ConnLap) Transform."""

import warnings

import numpy as np
import torch
from scipy.linalg import svd
from scipy.sparse import coo_matrix
from scipy.sparse import diags as sp_diags
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, remove_self_loops, to_undirected


class SheafConnLapPE(BaseTransform):
    r"""Sheaf Connection Laplacian Positional Encoding (SheafConnLapPE) transform.

    Based on "Sheaf-based Positional Encodings for Graph Neural Networks"
    by He, Bodnar & Liò (NeurIPS 2023 Workshop / PMLR 2024).
    https://openreview.net/pdf?id=ZtAabWUPu3

    The Connection Laplacian generalises the standard graph Laplacian by
    replacing each scalar off-diagonal entry (-1) with a d×d orthogonal
    *restriction map* — a rotation encoding the geometric alignment between
    the local node-feature neighbourhoods of the two endpoints.

    For each edge (v, u), the algorithm:

    1. Runs local PCA on the 1-hop feature neighbourhood of v and u separately,
       yielding orthonormal bases B_v, B_u ∈ R^{p×d} that approximate the
       local tangent spaces T_{x_v}M and T_{x_u}M under the manifold assumption.
    2. Solves the orthogonal Procrustes problem to find the rotation O_{vu} ∈ O(d)
       that best maps B_v onto B_u (closed form: SVD of B_v^T B_u).
    3. Sets the off-diagonal block L_F[v, u] = -O_{vu}.

    The resulting nd×nd block matrix L_F is symmetric positive semi-definite.
    Its k smallest non-trivial eigenvectors (each reshaped from nd to n×d) are
    concatenated column-wise to form a PE of total dimension k×d per node.

    On homophilic edges (similar features) O_{vu} ≈ I and the Connection
    Laplacian closely resembles the standard Laplacian. On heterophilic edges
    O_{vu} is a non-trivial rotation, introducing cross-dimensional coupling
    that encodes semantic disagreement — information the standard Laplacian
    cannot represent.

    .. note::
        **Feature dimension requirement :** ``data.x`` must be present
        and ``data.x.shape[1] >= stalk_dim``. The method assumes that node
        features lie near a ``stalk_dim``-dimensional manifold; if feature_dim
        < stalk_dim this assumption is violated and the PCA basis would contain
        zero columns, making the Procrustes rotation degenerate and breaking
        the PSD property of L_F. A ``ValueError`` is raised in this case.

        **Isolated nodes :** For isolated nodes (degree 0), the
        diagonal block of L_F is the zero matrix, making D^{-1/2} undefined.
        The normalisation substitutes 1.0 for these zero diagonal entries,
        which is equivalent to adding a unit self-loop for numerical purposes.
        The eigenvector components of isolated nodes are still well-defined
        (the corresponding rows of L_F remain all-zero), but their PE values
        will reflect their position in the global spectrum rather than local
        connectivity.

    Parameters
    ----------
    max_pe_dim : int
        Total output PE dimension. Must be divisible by ``stalk_dim``.
        Internally, the number of eigenvectors used is
        ``k = max_pe_dim // stalk_dim``, so the output shape is always
        ``[num_nodes, max_pe_dim]`` (zero-padded if fewer eigenvectors
        are available).
    stalk_dim : int, optional
        Dimension d of each stalk / restriction map. Controls the rank of
        the local tangent-space approximation. Default is 3, as used in
        the paper experiments. Must be <= feature_dim of ``data.x``.
    include_first : bool, optional
        If False (default), discards eigenvectors whose eigenvalue is below
        ``eps`` (the trivial zero-eigenvectors / global sections of the sheaf).
    concat_to_x : bool, optional
        If True (default), concatenates the PE with ``data.x``.
        If False, stores it as ``data.SheafConnLapPE`` instead.
    eps : float, optional
        Threshold below which eigenvalues are considered trivial. Default 1e-6.
    **kwargs
        Additional keyword arguments (unused; reserved for future extensions).
    """

    def __init__(
        self,
        max_pe_dim: int,
        stalk_dim: int = 3,
        include_first: bool = False,
        concat_to_x: bool = True,
        eps: float = 1e-6,
        **kwargs,
    ):
        if max_pe_dim % stalk_dim != 0:
            raise ValueError(
                f"max_pe_dim ({max_pe_dim}) must be divisible by "
                f"stalk_dim ({stalk_dim}). "
                f"The number of eigenvectors used is k = max_pe_dim // stalk_dim."
            )
        self.max_pe_dim = max_pe_dim
        self.stalk_dim = stalk_dim
        self.k = max_pe_dim // stalk_dim  # number of eigenvectors
        self.include_first = include_first
        self.concat_to_x = concat_to_x
        self.eps = eps

    def forward(self, data: Data) -> Data:
        """Compute and attach the ConnLap PE to a graph data object.

        Parameters
        ----------
        data : Data
            Input graph. ``data.x`` must be set and
            ``data.x.shape[1] >= stalk_dim``.

        Returns
        -------
        Data
            Graph with PE concatenated to ``data.x`` (``concat_to_x=True``)
            or stored in ``data.SheafConnLapPE`` (``concat_to_x=False``).

        Raises
        ------
        ValueError
            If ``data.x`` is None, or if ``feature_dim < stalk_dim``.
        """
        if data.x is None:
            raise ValueError(
                "SheafConnLapPE requires node features (data.x) to compute "
                "local PCA tangent spaces. data.x is None."
            )

        # Snapshot raw features *before* any concatenation — PCA always
        # operates on the original input features, not any augmented version.
        x_np = data.x.detach().cpu().numpy().astype(np.float64)

        feature_dim = x_np.shape[1]
        if feature_dim < self.stalk_dim:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be >= stalk_dim "
                f"({self.stalk_dim}).  The Connection Laplacian assumes node "
                f"features lie near a {self.stalk_dim}-dimensional manifold; "
                f"this is impossible when the ambient feature space has fewer "
                f"than {self.stalk_dim} dimensions.  Either reduce stalk_dim "
                f"or use higher-dimensional features."
            )

        pe = self._compute_sheaf_pe(data.edge_index, data.num_nodes, x_np)

        if self.concat_to_x:
            data.x = torch.cat([data.x, pe], dim=-1)
        else:
            data.SheafConnLapPE = pe

        return data

    def _compute_sheaf_pe(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        x_np: np.ndarray,
    ) -> torch.Tensor:
        """Full ConnLap PE pipeline.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge index tensor of shape (2, num_edges).
        num_nodes : int
            Number of nodes in the graph.
        x_np : np.ndarray
            Node feature matrix of shape (num_nodes, feature_dim).

        Returns
        -------
        torch.Tensor
            PE matrix of shape (num_nodes, max_pe_dim).
        """
        device = edge_index.device
        d = self.stalk_dim

        # Degenerate graph: return zero PE
        if edge_index.size(1) == 0 or num_nodes <= 1:
            return torch.zeros(num_nodes, self.max_pe_dim, device=device)

        # Warn when the dense nd×nd matrix will be very large.
        nd = num_nodes * d
        if nd > 10_000:
            warnings.warn(
                f"SheafConnLapPE: the dense eigendecomposition will allocate a "
                f"{nd}×{nd} matrix (~{nd**2 * 8 / 1e9:.1f} GB float64). "
                f"Consider reducing stalk_dim (currently {d}) or the graph size "
                f"(currently {num_nodes} nodes). For very large graphs a sparse "
                f"solver would be more appropriate.",
                ResourceWarning,
                stacklevel=3,
            )

        ei_sym, _ = remove_self_loops(to_undirected(edge_index))
        ei = ei_sym.cpu().numpy()

        # Node degrees from the symmetrised edge_index (used for diagonal blocks)
        deg_np = (
            degree(ei_sym[0], num_nodes=num_nodes)
            .to("cpu")
            .numpy()
            .astype(np.int64)
        )

        # Build adjacency
        src_sorted_idx = np.argsort(ei[0], kind="stable")
        dst_sorted = ei[1][src_sorted_idx]
        src_sorted = ei[0][src_sorted_idx]
        bounds = np.searchsorted(src_sorted, np.arange(num_nodes + 1))
        adjacency = [
            dst_sorted[bounds[v] : bounds[v + 1]].tolist()
            for v in range(num_nodes)
        ]

        # Local PCA basis per node
        # B_v ∈ R^{p×d}: orthonormal basis for the tangent space at node v,
        # estimated from the 1-hop feature neighbourhood.
        bases = [
            self._local_pca_basis(v, x_np, adjacency[v])
            for v in range(num_nodes)
        ]

        # ── Steps 2 & 3: build the sparse nd×nd connection Laplacian ──────
        # For each edge (v,u): O_{vu} = Procrustes(B_v, B_u), then
        # L_F[v,u] = -O_{vu}, L_F[v,v] += I_d per incident edge.
        L_F = self._build_connection_laplacian(num_nodes, ei, deg_np, bases)

        # ── Step 4: k smallest eigenvectors of the sheaf Laplacian ────────

        # np.linalg.eigh: dense, exact, deterministic, returns eigenvalues in
        # ascending order.  This is the correct default for graphs up to ~3 k
        # nodes (nd ≤ 9 k) — well within the target scale.
        # eigsh (ARPACK) would be faster for very large graphs but introduces
        # non-deterministic sign flips; the dense path avoids this entirely.
        evals, evecs = np.linalg.eigh(L_F.toarray())
        # eigh guarantees ascending order — no sort needed.

        # Drop trivial eigenvectors (global sections with eigenvalue ≈ 0)
        if not self.include_first:
            mask = evals > self.eps
            evals, evecs = evals[mask], evecs[:, mask]  # noqa: F841

        # Take the k smallest remaining
        k_use = min(self.k, evecs.shape[1])
        evecs = evecs[:, :k_use]  # (nd, k_use)

        max_abs_idx = np.argmax(np.abs(evecs), axis=0)  # (k_use,)
        signs = np.sign(evecs[max_abs_idx, np.arange(k_use)])  # (k_use,)
        signs[signs == 0] = 1.0  # guard against the (pathological) zero case
        evecs = evecs * signs[np.newaxis, :]  # broadcast

        # Reshape eigenvectors → PE matrix
        # Each eigenvector is (nd,) = (n*d,); reshape to (n, d).
        # Pack k_use such matrices side-by-side → (n, k_use * d).
        pe_np = np.zeros((num_nodes, k_use * d), dtype=np.float64)
        for i in range(k_use):
            vec = evecs[:, i].reshape(num_nodes, d)  # (n, d)
            pe_np[:, i * d : (i + 1) * d] = vec

        # Zero-pad to max_pe_dim if fewer than k eigenvectors were available
        if k_use * d < self.max_pe_dim:
            pad = np.zeros((num_nodes, self.max_pe_dim - k_use * d))
            pe_np = np.concatenate([pe_np, pad], axis=1)

        return torch.from_numpy(pe_np).to(dtype=torch.float32, device=device)

    def _local_pca_basis(
        self,
        node_idx: int,
        x: np.ndarray,
        neighbors: list,
    ) -> np.ndarray:
        """Orthonormal tangent-space basis B_v ∈ R^{p×d} via local PCA.

        Gathers features of node v and its 1-hop neighbours, centres them,
        and returns the top-d right singular vectors. These span the dominant
        directions of local feature variation — an approximation of T_{x_v}M
        under the manifold assumption.

        If the neighbourhood has fewer than d linearly independent directions
        (e.g. very small degree or duplicate features), the basis is padded
        with orthogonal complement vectors from the standard basis.

        Note: the guard in ``forward`` ensures p >= d before this method is
        called, so the Gram–Schmidt padding loop can always find d - n_comp
        additional orthogonal vectors in R^p.

        Parameters
        ----------
        node_idx : int
            Index of the target node.
        x : np.ndarray
            Node feature matrix of shape (n, p); guaranteed p >= stalk_dim.
        neighbors : list[int]
            Indices of 1-hop neighbours of node_idx.

        Returns
        -------
        np.ndarray
            Orthonormal basis matrix of shape (p, stalk_dim).
        """
        p = x.shape[1]
        d = self.stalk_dim
        local_idx = [node_idx] + list(neighbors)
        local_x = x[local_idx]
        local_x = local_x - local_x.mean(axis=0)

        if len(local_idx) < 2:
            # Isolated node: fall back to the first d standard basis vectors.
            return np.eye(p, d)

        try:
            _, _, Vt = svd(local_x, full_matrices=False)
            # Vt shape: (min(n_local, p), p). Rows = principal directions.
            n_comp = Vt.shape[0]
        except np.linalg.LinAlgError:
            return np.eye(p, d)

        if n_comp >= d:
            return Vt[:d].T  # (p, d) — top-d principal directions as columns

        # Fewer PCA components than d: pad with orthogonal extras via
        # Gram–Schmidt.
        basis = np.zeros((p, d))
        basis[:, :n_comp] = Vt.T
        for target_col in range(n_comp, d):
            for j in range(p):
                candidate = np.eye(1, p, j).flatten()
                for filled in range(target_col):
                    candidate -= (candidate @ basis[:, filled]) * basis[
                        :, filled
                    ]
                norm = np.linalg.norm(candidate)
                if norm > 1e-10:
                    basis[:, target_col] = candidate / norm
                    break
        return basis  # (p, d)

    @staticmethod
    def _orthogonal_procrustes(B_v: np.ndarray, B_u: np.ndarray) -> np.ndarray:
        """Find O* ∈ O(d) minimising ||B_u - B_v O||_F.

        Closed-form solution: SVD of B_v^T B_u = U S V^T → O* = U V^T.

        This is the parallel transport approximation from T_{x_v}M to T_{x_u}M,
        used as the restriction map for edge (v, u).

        Parameters
        ----------
        B_v : np.ndarray
            Orthonormal basis of shape (p, d) for the source node.
        B_u : np.ndarray
            Orthonormal basis of shape (p, d) for the target node.

        Returns
        -------
        np.ndarray
            Orthogonal rotation matrix of shape (d, d).
        """
        M = B_v.T @ B_u  # (d, d)
        U, _, Vt = svd(M, full_matrices=False)
        return U @ Vt  # (d, d) orthogonal

    def _build_connection_laplacian(
        self,
        num_nodes: int,
        ei: np.ndarray,
        deg_np: np.ndarray,
        bases: list,
    ):
        """Assemble the sparse normalised connection Laplacian.

        Block structure (nd × nd, where nd = num_nodes × stalk_dim):

            Diagonal block [v, v] :  deg(v) · I_d
            Off-diagonal   [v, u] : -O_{vu}   for each directed edge (v→u)

        After assembly, symmetrically normalises:
            Δ_F = D^{-1/2} L_F D^{-1/2}
        where D is the scalar diagonal of L_F (each entry = deg(v)).

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph.
        ei : np.ndarray
            Symmetrised, self-loop-free edge index of shape (2, num_edges).
        deg_np : np.ndarray
            Node degree array of shape (num_nodes,), dtype int64.
        bases : list[np.ndarray]
            PCA bases where bases[v] is B_v of shape (p, stalk_dim).

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse normalised connection Laplacian of shape (nd, nd).
        """
        d = self.stalk_dim
        nd = num_nodes * d
        num_edges = ei.shape[1]

        # Pre-compute block row/col offset patterns for a d×d block
        bi, bj = np.divmod(np.arange(d * d), d)  # both shape (d²,)

        # ── Off-diagonal COO entries ──────────────────────────────────────
        # For each directed edge (v, u): d² entries at rows v*d+bi, cols u*d+bj
        off_row = (
            np.repeat(ei[0], d * d) * d + np.tile(bi, num_edges)
        ).astype(np.int32)
        off_col = (
            np.repeat(ei[1], d * d) * d + np.tile(bj, num_edges)
        ).astype(np.int32)
        off_data = np.empty(num_edges * d * d, dtype=np.float64)

        for e in range(num_edges):
            v, u = int(ei[0, e]), int(ei[1, e])
            off_data[
                e * d * d : (e + 1) * d * d
            ] = -self._orthogonal_procrustes(bases[v], bases[u]).ravel()

        # ── Diagonal COO entries ──────────────────────────────────────────
        # Each node v contributes deg(v) to d consecutive diagonal positions
        diag_idx = np.arange(nd, dtype=np.int32)
        diag_data = np.repeat(deg_np.astype(np.float64), d)

        # ── Assemble and convert ──────────────────────────────────────────
        L_F = coo_matrix(
            (
                np.concatenate([diag_data, off_data]),
                (
                    np.concatenate([diag_idx, off_row]),
                    np.concatenate([diag_idx, off_col]),
                ),
            ),
            shape=(nd, nd),
        ).tocsr()

        # Symmetric normalisation: Δ_F = D^{-1/2} L_F D^{-1/2}
        diag = np.array(L_F.diagonal(), dtype=np.float64)
        diag_safe = np.where(np.abs(diag) > 1e-10, diag, 1.0)
        inv_sqrt = 1.0 / np.sqrt(diag_safe)
        D_inv_sqrt = sp_diags(inv_sqrt)
        return D_inv_sqrt @ L_F @ D_inv_sqrt
