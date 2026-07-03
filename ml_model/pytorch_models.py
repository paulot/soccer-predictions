import torch
import torch.nn as nn
import torch.nn.functional as F

class OutcomeNN(nn.Module):
    """
    Neural Network to predict pass outcome (Success vs Turnover).
    Uses a learnable Entity Embedding for the player role and Dropout.
    """
    def __init__(self, input_dim: int, role_idx: int):
        super(OutcomeNN, self).__init__()
        self.role_idx = role_idx
        self.role_embedding = nn.Embedding(4, 2)
        
        # Input dimension: input_dim - 1 (role) + 2 (role embedding)
        self.fc1 = nn.Linear(input_dim + 1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2) # Regularization
        
    def forward(self, x):
        # Extract and embed role
        role = x[:, self.role_idx].long()
        role_emb = self.role_embedding(role)
        
        # Remove the raw role column and concatenate the embedding
        x_no_role = torch.cat([x[:, :self.role_idx], x[:, self.role_idx+1:]], dim=1)
        x_combined = torch.cat([x_no_role, role_emb], dim=1)
        
        x = F.relu(self.fc1(x_combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))

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
        self.role_idx = role_idx
        self.def_density_start_idx = def_density_start_idx
        self.output_dim = output_dim
        
        self.role_embedding = nn.Embedding(4, 2)
        
        # Precompute context indices to keep (excludes role_idx and the 30 defensive densities)
        context_indices = [
            idx for idx in range(input_dim) 
            if idx != role_idx and not (def_density_start_idx <= idx < def_density_start_idx + 30)
        ]
        self.register_buffer('context_indices', torch.LongTensor(context_indices))
        
        # Precompute static spatial relations between all 30 starting and target zones
        distances = torch.zeros(30, 30)
        angles = torch.zeros(30, 30)
        target_xs = torch.zeros(30)
        target_ys = torch.zeros(30)
        
        for i in range(30):
            sx = i // 5
            sy = i % 5
            start_cx = sx * 20.0 + 10.0
            start_cy = sy * 16.0 + 8.0
            
            for j in range(30):
                tx = j // 5
                ty = j % 5
                target_cx = tx * 20.0 + 10.0
                target_cy = ty * 16.0 + 8.0
                
                dx = target_cx - start_cx
                dy = target_cy - start_cy
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
        context_input_dim = len(context_indices) + 2
        self.context_fc1 = nn.Linear(context_input_dim, 128)
        self.context_fc2 = nn.Linear(128, 64) # Query size = 64
        self.zone_fc = nn.Linear(5, 64) # Key size = 64
        
        # --- Pathway 2: Tactical MLP ---
        # Input dim: input_dim - 1 (role) + 2 (role embedding)
        self.mlp_fc1 = nn.Linear(input_dim + 1, 128)
        self.mlp_fc2 = nn.Linear(128, 64)
        self.mlp_fc3 = nn.Linear(64, output_dim)
        
        self.dropout = nn.Dropout(0.2) # Regularization
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Extract and embed role
        role = x[:, self.role_idx].long()
        role_emb = self.role_embedding(role)
        
        # --- Pathway 1: Spatial Attention ---
        # Vectorized lookup of distances and angles
        start_idx = (x[:, 0] * 5 + x[:, 1]).long()
        batch_dist = self.static_distances[start_idx]
        batch_angle = self.static_angles[start_idx]
        
        # Extract the 30 target defensive densities
        def_densities = x[:, self.def_density_start_idx : self.def_density_start_idx + 30]
        
        # Extract context using precomputed indices
        x_context = x[:, self.context_indices]
        x_context_combined = torch.cat([x_context, role_emb], dim=1)
        
        h_context = F.relu(self.context_fc1(x_context_combined))
        h_context = self.dropout(h_context)
        query = self.context_fc2(h_context) # Shape: [batch_size, 64]
        
        # Vectorized construction of the 5D spatial features
        batch_tx = self.static_target_xs.unsqueeze(0).expand(batch_size, -1)
        batch_ty = self.static_target_ys.unsqueeze(0).expand(batch_size, -1)
        
        # Shape: [batch_size, 30, 5]
        zones_spatial = torch.stack([
            batch_tx, 
            batch_ty, 
            batch_dist, 
            batch_angle, 
            def_densities
        ], dim=2)
        
        # Project all zones to keys -> Shape: [batch_size, 30, 64]
        keys = self.zone_fc(zones_spatial.view(-1, 5)).view(batch_size, 30, 64)
        
        # Compute bilinear attention scores: query^T * key
        spatial_logits = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
        
        # --- Pathway 2: Tactical MLP ---
        x_no_role = torch.cat([x[:, :self.role_idx], x[:, self.role_idx+1:]], dim=1)
        x_full_emb = torch.cat([x_no_role, role_emb], dim=1)
        
        h_mlp = F.relu(self.mlp_fc1(x_full_emb))
        h_mlp = self.dropout(h_mlp)
        h_mlp = F.relu(self.mlp_fc2(h_mlp))
        h_mlp = self.dropout(h_mlp)
        mlp_logits = self.mlp_fc3(h_mlp)
        
        # --- Combine Logits ---
        return spatial_logits + mlp_logits
