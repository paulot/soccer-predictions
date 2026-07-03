import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class OutcomeNN(nn.Module):
    """
    Neural Network to predict pass outcome (Success vs Turnover).
    Uses a learnable Entity Embedding for the player role and Dropout.
    """
    def __init__(self, input_dim: int, role_idx: int):
        super(OutcomeNN, self).__init__()
        self.role_idx: int = role_idx
        self.role_embedding: nn.Embedding = nn.Embedding(4, 2)
        
        # Input dimension: input_dim - 1 (role) + 2 (role embedding)
        self.fc1: nn.Linear = nn.Linear(input_dim + 1, 64)
        self.fc2: nn.Linear = nn.Linear(64, 32)
        self.fc3: nn.Linear = nn.Linear(32, 1)
        self.dropout: nn.Dropout = nn.Dropout(0.2) # Regularization
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract and embed role
        role: torch.Tensor = x[:, self.role_idx].long()
        role_emb: torch.Tensor = self.role_embedding(role)
        
        # Remove the raw role column and concatenate the embedding
        x_no_role: torch.Tensor = torch.cat([x[:, :self.role_idx], x[:, self.role_idx+1:]], dim=1)
        x_combined: torch.Tensor = torch.cat([x_no_role, role_emb], dim=1)
        
        h1: torch.Tensor = F.relu(self.fc1(x_combined))
        h1 = self.dropout(h1)
        h2: torch.Tensor = F.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        return torch.sigmoid(self.fc3(h2))

class DestinationNN(nn.Module):
    """
    Hybrid Neural Network to predict pass destination zone (0 to 29).
    Combines:
    1) A Bilinear Spatial Attention pathway (physical/spatial constraints).
    2) A Tactical MLP pathway (team-specific and tactical preferences).
    Optimized with precomputed spatial relations and TorchScript compatibility.
    """
    def __init__(self, input_dim: int, role_idx: int, def_density_start_idx: int, output_dim: int = 30):
        super(DestinationNN, self).__init__()
        self.role_idx: int = role_idx
        self.def_density_start_idx: int = def_density_start_idx
        self.output_dim: int = output_dim
        
        self.role_embedding: nn.Embedding = nn.Embedding(4, 2)
        
        # Precompute context indices to keep (excludes role_idx and the 30 defensive densities)
        context_indices: List[int] = [
            idx for idx in range(input_dim) 
            if idx != role_idx and not (def_density_start_idx <= idx < def_density_start_idx + 30)
        ]
        self.register_buffer('context_indices', torch.LongTensor(context_indices))
        
        # Precompute static spatial relations between all 30 starting and target zones
        distances: torch.Tensor = torch.zeros(30, 30)
        angles: torch.Tensor = torch.zeros(30, 30)
        target_xs: torch.Tensor = torch.zeros(30)
        target_ys: torch.Tensor = torch.zeros(30)
        
        for i in range(30):
            sx: int = i // 5
            sy: int = i % 5
            start_cx: float = sx * 20.0 + 10.0
            start_cy: float = sy * 16.0 + 8.0
            
            for j in range(30):
                tx: int = j // 5
                ty: int = j % 5
                target_cx: float = tx * 20.0 + 10.0
                target_cy: float = ty * 16.0 + 8.0
                
                dx: float = target_cx - start_cx
                dy: float = target_cy - start_cy
                distances[i, j] = torch.sqrt(torch.tensor(dx**2 + dy**2)) / 100.0
                angles[i, j] = torch.atan2(torch.tensor(dy), torch.tensor(dx)) / 3.14159265
                
        for j in range(30):
            target_xs[j] = (j // 5) / 5.0
            target_ys[j] = (j % 5) / 4.0
            
        self.register_buffer('static_distances', distances)
        self.register_buffer('static_angles', angles)
        self.register_buffer('static_target_xs', target_xs)
        self.register_buffer('static_target_ys', target_ys)
        
        # --- Pathway 1: Spatial Attention ---
        # Input dim of context network = len(context_indices) + 2 (role emb)
        context_input_dim: int = len(context_indices) + 2
        self.context_fc1: nn.Linear = nn.Linear(context_input_dim, 128)
        self.context_fc2: nn.Linear = nn.Linear(128, 64) # Query size = 64
        self.zone_fc: nn.Linear = nn.Linear(5, 64) # Key size = 64
        
        # --- Pathway 2: Tactical MLP ---
        # Input dim: input_dim - 1 (role) + 2 (role embedding)
        self.mlp_fc1: nn.Linear = nn.Linear(input_dim + 1, 128)
        self.mlp_fc2: nn.Linear = nn.Linear(128, 64)
        self.mlp_fc3: nn.Linear = nn.Linear(64, output_dim)
        
        self.dropout: nn.Dropout = nn.Dropout(0.2) # Regularization
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size: int = x.shape[0]
        
        # Extract and embed role
        role: torch.Tensor = x[:, self.role_idx].long()
        role_emb: torch.Tensor = self.role_embedding(role)
        
        # --- Pathway 1: Spatial Attention ---
        # Vectorized lookup of distances and angles
        start_idx: torch.Tensor = (x[:, 0] * 5 + x[:, 1]).long()
        batch_dist: torch.Tensor = self.static_distances[start_idx]
        batch_angle: torch.Tensor = self.static_angles[start_idx]
        
        # Extract the 30 target defensive densities
        def_densities: torch.Tensor = x[:, self.def_density_start_idx : self.def_density_start_idx + 30]
        
        # Extract context using precomputed indices
        x_context: torch.Tensor = x[:, self.context_indices]
        x_context_combined: torch.Tensor = torch.cat([x_context, role_emb], dim=1)
        
        h_context: torch.Tensor = F.relu(self.context_fc1(x_context_combined))
        h_context = self.dropout(h_context)
        query: torch.Tensor = self.context_fc2(h_context) # Shape: [batch_size, 64]
        
        # Vectorized construction of the 5D spatial features
        batch_tx: torch.Tensor = self.static_target_xs.unsqueeze(0).expand(batch_size, -1)
        batch_ty: torch.Tensor = self.static_target_ys.unsqueeze(0).expand(batch_size, -1)
        
        # Shape: [batch_size, 30, 5]
        zones_spatial: torch.Tensor = torch.stack([
            batch_tx, 
            batch_ty, 
            batch_dist, 
            batch_angle, 
            def_densities
        ], dim=2)
        
        # Project all zones to keys -> Shape: [batch_size, 30, 64]
        keys: torch.Tensor = self.zone_fc(zones_spatial.view(-1, 5)).view(batch_size, 30, 64)
        
        # Compute bilinear attention scores: query^T * key
        spatial_logits: torch.Tensor = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
        
        # --- Pathway 2: Tactical MLP ---
        x_no_role: torch.Tensor = torch.cat([x[:, :self.role_idx], x[:, self.role_idx+1:]], dim=1)
        x_full_emb: torch.Tensor = torch.cat([x_no_role, role_emb], dim=1)
        
        h_mlp: torch.Tensor = F.relu(self.mlp_fc1(x_full_emb))
        h_mlp = self.dropout(h_mlp)
        h_mlp = F.relu(self.mlp_fc2(h_mlp))
        h_mlp = self.dropout(h_mlp)
        mlp_logits: torch.Tensor = self.mlp_fc3(h_mlp)
        
        # --- Combine Logits ---
        return spatial_logits + mlp_logits
