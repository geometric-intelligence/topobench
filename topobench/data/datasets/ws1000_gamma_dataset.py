import os
import os.path as osp
import random
from collections import deque

import torch
from torch_geometric.data import Data, InMemoryDataset


class WS1000GammaDataset(InMemoryDataset):
    """
    WS1000-Gamma Synthetic Dataset
    ==============================

    This module implements the WS1000-Gamma dataset introduced in:

        Katsman, I., Lou, E., & Gilbert, A. (2024).
        *Revisiting the Necessity of Graph Learning and Common Graph Benchmarks*.
        arXiv:2412.06173
        https://arxiv.org/abs/2412.06173

    The dataset is a synthetic Watts–Strogatz small-world graph with
    BFS-dependent Gaussian node features. It is designed as a principled
    benchmark that requires graph structure to perform EDGE-level tasks (see Note c).

    Notes
    -----
    a.- This implementation follows the Watts & Strogatz (1998) construction:
      1. Create a regular ring lattice with mean degree ``K``.
      2. Rewire each oriented ring edge ``(i, i+j)`` with probability ``beta``.

    b.- Node features are generated via **BFS parental dependence**:
      ``x_child = gamma * x_parent + noise_scale * z``, where ``z ~ N(0, I_d)``.

    c.- The current implementation evaluates NODE-level distance classification
      (predict BFS distance to the root).
      EDGE prediction is NOT yet implemented.

    Dataset Structure
    -----------------
    The output is a single :class:`torch_geometric.data.Data` object with:

    - ``x`` : ``[num_nodes, feature_dim]`` float tensor
    - ``edge_index`` : ``[2, 2 * num_edges]`` long tensor (undirected)
    - ``y`` : ``[num_nodes]`` long tensor of BFS distances from the root node
    - metadata fields: ``gamma``, ``beta``, ``mean_degree``, ``feature_dim``, ``seed``

    Configuration Parameters
    ------------------------
    The dataset accepts the following Hydra parameters:

    - ``num_nodes`` : int
    - ``feature_dim`` : int
    - ``mean_degree`` : int (must be even)
    - ``beta`` : float
    - ``gamma`` : float
    - ``noise_scale`` : float
    - ``seed`` : int

    These are typically defined in:

        ``configs/dataset/graph/WS1000-gamma.yaml``

    """
    
    def __init__(
        self,
        root: str,
        name: str = "WS1000-gamma",
        parameters=None,
        transform=None,
        pre_transform=None,
    ) -> None:
        self.name = name
        self.parameters = parameters

        # Defaults, can be overridden from Hydra DictConfig
        self.num_nodes = 1000
        self.feature_dim = 1000
        self.mean_degree = 4      # K in WS model
        self.beta = 0.5           # rewiring probability
        self.gamma = 0.0          # parental coefficient
        self.noise_scale = 1.0
        self.seed = 0

        if parameters is not None:
            if "num_nodes" in parameters:
                self.num_nodes = int(parameters.num_nodes)
            if "feature_dim" in parameters:
                self.feature_dim = int(parameters.feature_dim)
            if "mean_degree" in parameters:
                self.mean_degree = int(parameters.mean_degree)
            if "beta" in parameters:
                self.beta = float(parameters.beta)
            if "gamma" in parameters:
                self.gamma = float(parameters.gamma)
            if "noise_scale" in parameters:
                self.noise_scale = float(parameters.noise_scale)
            if "seed" in parameters:
                self.seed = int(parameters.seed)

        super().__init__(root=root, transform=transform, pre_transform=pre_transform)

        # Load processed data (super() will call process() the first time)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # ---------------------------------------------------------------------
    # Required PyG properties
    # ---------------------------------------------------------------------
    @property
    def raw_file_names(self) -> list[str]:
        # Dummy file to satisfy InMemoryDataset's bookkeeping.
        return ["synthetic.done"]

    @property
    def processed_file_names(self) -> list[str]:
        return ["data_v1.pt"]

    # ---------------------------------------------------------------------
    # Download: here we don't download anything.
    # ---------------------------------------------------------------------
    def download(self) -> None:
        raw_path = osp.join(self.raw_dir, self.raw_file_names[0])
        os.makedirs(self.raw_dir, exist_ok=True)
        with open(raw_path, "w") as f:
            f.write("synthetic ws1000_gamma marker\n")

    # ---------------------------------------------------------------------
    # Process: generate WS graph + WS1000_gamma features and save.
    # ---------------------------------------------------------------------
    def process(self) -> None:
        data = self._generate_ws1000_gamma()
        data_list = [data]
        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])

    # ---------------------------------------------------------------------
    # Helper: Watts–Strogatz graph + gamma-based features
    # ---------------------------------------------------------------------
    def _generate_ws1000_gamma(self) -> Data:
        N = self.num_nodes
        K = self.mean_degree
        beta = self.beta
        d = self.feature_dim
        gamma = self.gamma
        noise_scale = self.noise_scale
        seed = self.seed

        assert K % 2 == 0, "mean_degree K must be even for Watts–Strogatz ring construction."

        # --- Seed everything deterministically
        random.seed(seed)
        torch.manual_seed(seed)

        # --- 1) Build regular ring lattice
        # neighbors: undirected adjacency; edges: undirected edge set
        neighbors = {i: set() for i in range(N)}
        edges = set()

        half_k = K // 2

        ring_edges_oriented = []
        for j in range(1, half_k + 1):      # distance layer outer
            for i in range(N):              # then each vertex
                v = (i + j) % N
                ring_edges_oriented.append((i, v))
                u_min, u_max = (i, v) if i < v else (v, i)
                if (u_min, u_max) not in edges:
                    edges.add((u_min, u_max))
                    neighbors[i].add(v)
                    neighbors[v].add(i)
        # --- 2) Rewire edges in Watts–Strogatz style (exactly as in the paper)
        # For each original ring edge (i, i+j) in clockwise sense, with probability beta,
        # rewire the endpoint i+j to a new node w chosen uniformly at random
        for (i, v) in ring_edges_oriented:
            if random.random() < beta:
                # Candidates: all nodes except i and current neighbours of i
                possible_nodes = [w for w in range(N)
                                  if w != i and w not in neighbors[i]]
                if not possible_nodes:
                    # No valid candidate; skip rewiring for this edge
                    continue

                w = random.choice(possible_nodes)

                # Remove old edge (i, v) if it still exists
                if v in neighbors[i]:
                    neighbors[i].remove(v)
                    neighbors[v].remove(i)
                    edges.discard((i, v) if i < v else (v, i))

                # Add new edge (i, w)
                neighbors[i].add(w)
                neighbors[w].add(i)
                edges.add((i, w) if i < w else (w, i))


        # --- 3) Convert to undirected edge_index with both directions
        edge_list = []
        for (u, v) in edges:
            edge_list.append((u, v))
            edge_list.append((v, u))
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # --- 4) Generate features with BFS parental dependence
        # Use neighbors directly as adjacency (adj = neighbors)
        x = torch.empty((N, d), dtype=torch.float)

        root = 0
        queue = deque([root])

        # root feature
        x[root] = torch.randn(d)
        dist = torch.full((N,), -1, dtype=torch.long)
        dist[root] = 0

        while queue:
            u = queue.popleft()
            for v in neighbors[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                    noise = torch.randn(d)
                    x[v] = gamma * x[u] + noise_scale * noise

        # For unvisited (disconnected) nodes:
        for i in range(N):
            if dist[i] == -1:
                x[i] = torch.randn(d)
                

        data = Data(
            x=x,
            edge_index=edge_index,
            y=dist,
        )
        # Metadata
        data.num_nodes = N
        data.gamma = gamma
        data.beta = beta
        data.mean_degree = K
        data.feature_dim = d
        data.seed = seed

        return data
