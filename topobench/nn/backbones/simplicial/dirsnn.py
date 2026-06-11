import torch
import torch.nn as nn

class DirSNNLayer(nn.Module):
    """
    Directed Simplicial Neural Network Layer.
    Computes message passing over directed simplices using oriented boundary matrices.
    """
    def __init__(self, in_features, out_features, dropout=0.0):
        super(DirSNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout)
        
        # Learnable weight matrices for lower and upper directed neighborhoods
        self.W_lower = nn.Linear(in_features, out_features, bias=False)
        self.W_upper = nn.Linear(in_features, out_features, bias=False)
        self.W_self = nn.Linear(in_features, out_features, bias=True)
        
        self.activation = nn.ReLU()

    def forward(self, x, boundary_lower, boundary_upper):
        """
        x: Simplex features (shape: [num_simplices, in_features])
        boundary_lower: Directed lower incidence matrix
        boundary_upper: Directed upper incidence matrix
        """
        # Dropout for regularization
        x = self.dropout(x)
        
        # Compute message from lower adjacent simplices
        msg_lower = torch.sparse.mm(boundary_lower, x)
        out_lower = self.W_lower(msg_lower)
        
        # Compute message from upper adjacent simplices
        msg_upper = torch.sparse.mm(boundary_upper, x)
        out_upper = self.W_upper(msg_upper)
        
        # Self-loop update
        out_self = self.W_self(x)
        
        # Aggregate directed signals
        out = self.activation(out_lower + out_upper + out_self)
        return out


class DirSNN(nn.Module):
    """
    The main DirSNN Backbone that integrates into TopoBench.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, **kwargs):
        super(DirSNN, self).__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(DirSNNLayer(in_channels, hidden_channels, dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(DirSNNLayer(hidden_channels, hidden_channels, dropout))
            
        # Output layer
        self.layers.append(DirSNNLayer(hidden_channels, out_channels, dropout))

    def forward(self, batch):
        """
        Expected batch inputs from TopoBench's Simplicial wrapper.
        """
        # Extract edge features (1-simplices) as the primary representation
        x = batch.edge_x 
        
        # Extract incidence matrices for the directed flow
        # In TopoBench, B1 is node-to-edge, B2 is edge-to-face
        B1 = batch.B1
        B2 = batch.B2
        
        # Compute the directed Hodge Laplacians (lower and upper)
        # L_down = B1^T * B1 (flow from lower dimensions)
        L_down = torch.sparse.mm(B1.t(), B1)
        # L_up = B2 * B2^T (flow from higher dimensions)
        L_up = torch.sparse.mm(B2, B2.t())
        
        # Pass messages through layers
        for layer in self.layers:
            x = layer(x, L_down, L_up)
            
        return x