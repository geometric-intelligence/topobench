import torch
from topobench.nn.backbones.simplicial.dirsnn import DirSNNLayer, DirSNN

class DummyBatch:
    """Mock TopoBench batch structure for testing."""
    def __init__(self, edge_x, B1, B2):
        self.edge_x = edge_x
        self.B1 = B1
        self.B2 = B2

def test_dirsnn_layer():
    """Test the core message passing layer."""
    in_features = 16
    out_features = 32
    num_edges = 5

    layer = DirSNNLayer(in_features, out_features)

    # Dummy features and directed boundary matrices (sparse)
    x = torch.randn(num_edges, in_features)

    # Create random sparse matrices to simulate L_down and L_up
    indices = torch.randint(0, num_edges, (2, 10))
    values = torch.randn(10)
    boundary_lower = torch.sparse_coo_tensor(indices, values, (num_edges, num_edges))

    indices_up = torch.randint(0, num_edges, (2, 10))
    values_up = torch.randn(10)
    boundary_upper = torch.sparse_coo_tensor(indices_up, values_up, (num_edges, num_edges))

    out = layer(x, boundary_lower, boundary_upper)

    # Assert output dimensions and type
    assert out.shape == (num_edges, out_features)
    assert not torch.isnan(out).any(), "Output contains NaNs"

def test_dirsnn_backbone():
    """Test the full DirSNN wrapper integration."""
    in_channels = 16
    hidden_channels = 32
    out_channels = 8
    num_nodes = 6
    num_edges = 10
    num_faces = 4

    backbone = DirSNN(in_channels, hidden_channels, out_channels, num_layers=2)

    edge_x = torch.randn(num_edges, in_channels)

    # Simulate B1 (nodes to edges) -> Shape: (num_nodes, num_edges)
    B1_indices = torch.randint(0, num_nodes, (2, num_edges))
    B1_indices[1, :] = torch.arange(num_edges)  # Ensure valid column indices
    B1_values = torch.ones(num_edges)
    B1 = torch.sparse_coo_tensor(B1_indices, B1_values, (num_nodes, num_edges))

    # Simulate B2 (edges to faces) -> Shape: (num_edges, num_faces)
    B2_indices = torch.randint(0, num_edges, (2, num_faces))
    B2_indices[1, :] = torch.arange(num_faces)  # Ensure valid column indices
    B2_values = torch.ones(num_faces)
    B2 = torch.sparse_coo_tensor(B2_indices, B2_values, (num_edges, num_faces))

    batch = DummyBatch(edge_x, B1, B2)

    out = backbone(batch)

    # Final output should map the out_channels back to the edges (1-simplices)
    assert out.shape == (num_edges, out_channels)

if __name__ == "__main__":
    print("Testing DirSNN Layer...")
    test_dirsnn_layer()
    print("Testing DirSNN Backbone...")
    test_dirsnn_backbone()
    print("🟢 ALL DIRSNN TESTS PASSED SUCCESSFULLY!")
