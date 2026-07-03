import torch
import torch.nn as nn
import torch.nn.functional as F

class OutcomeNN(nn.Module):
    """
    Neural Network to predict pass outcome (Success vs Turnover).
    Uses a learnable Entity Embedding for the player role and Dropout.
    """
    def __init__(self, input_dim, role_idx):
        super(OutcomeNN, self).__init__()
        self.role_idx = role_idx
        self.role_embedding = nn.Embedding(4, 2)
        
        # Input dimension: input_dim - 1 (role) + 2 (role embedding)
        self.fc1 = nn.Linear(input_dim + 1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3) # Regularization
        
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
    Neural Network to predict pass destination zone (0 to 29).
    Uses Bilinear Spatial Attention: matching a context query vector against
    30 candidate zone key vectors constructed from spatial coordinates,
    distance, angle, and opponent defensive pressure.
    """
    def __init__(self, input_dim, role_idx, def_density_start_idx, output_dim=30):
        super(DestinationNN, self).__init__()
        self.role_idx = role_idx
        self.def_density_start_idx = def_density_start_idx
        self.output_dim = output_dim
        
        self.role_embedding = nn.Embedding(4, 2)
        
        # Context network: projects continuous features + role embedding to a query vector
        # Input dim of context network = input_dim - 1 (role) - 30 (def densities) + 2 (role emb)
        context_input_dim = input_dim - 1 - 30 + 2
        self.context_fc1 = nn.Linear(context_input_dim, 64)
        self.context_fc2 = nn.Linear(64, 16) # Query size = 16
        
        # Zone key network: projects the 5D spatial feature of a zone to a key vector
        # Spatial features: [norm_x, norm_y, norm_dist, norm_angle, def_pressure]
        self.zone_fc = nn.Linear(5, 16) # Key size = 16
        
        self.dropout = nn.Dropout(0.3) # Regularization
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. Extract and embed role
        role = x[:, self.role_idx].long()
        role_emb = self.role_embedding(role)
        
        # 2. Extract starting zone coordinates (first two columns: start_zone_x, start_zone_y)
        start_x = x[:, 0]
        start_y = x[:, 1]
        
        # 3. Extract the 30 target defensive densities
        def_densities = x[:, self.def_density_start_idx : self.def_density_start_idx + 30]
        
        # 4. Build context vector (exclude role and the 30 defensive densities)
        mask = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
        mask[self.role_idx] = False
        mask[self.def_density_start_idx : self.def_density_start_idx + 30] = False
        
        x_context = x[:, mask]
        x_context_combined = torch.cat([x_context, role_emb], dim=1)
        
        h_context = F.relu(self.context_fc1(x_context_combined))
        h_context = self.dropout(h_context)
        query = self.context_fc2(h_context) # Shape: [batch_size, 16]
        
        # 5. Build 5D spatial features for all 30 candidate zones
        zones_spatial = []
        for j in range(30):
            tx = j // 5
            ty = j % 5
            
            # Normalized coordinates
            nx = tx / 5.0
            ny = ty / 4.0
            
            # Start centers
            start_cx = start_x * 20.0 + 10.0
            start_cy = start_y * 16.0 + 8.0
            
            # Target centers
            target_cx = tx * 20.0 + 10.0
            target_cy = ty * 16.0 + 8.0
            
            dx = target_cx - start_cx
            dy = target_cy - start_cy
            dist = torch.sqrt(dx**2 + dy**2) / 100.0
            angle = torch.atan2(dy, dx) / 3.14159265
            
            # Opponent defensive density in this zone
            def_press = def_densities[:, j]
            
            # Stack to get [batch_size, 5]
            nx_tensor = torch.full((batch_size,), nx, device=x.device)
            ny_tensor = torch.full((batch_size,), ny, device=x.device)
            
            zone_feats = torch.stack([nx_tensor, ny_tensor, dist, angle, def_press], dim=1)
            zones_spatial.append(zone_feats)
            
        # Shape: [batch_size, 30, 5]
        zones_spatial = torch.stack(zones_spatial, dim=1)
        
        # 6. Project all zones to keys -> Shape: [batch_size, 30, 16]
        keys = self.zone_fc(zones_spatial.view(-1, 5)).view(batch_size, 30, 16)
        
        # 7. Compute bilinear attention scores: query^T * key
        # [batch_size, 1, 16] x [batch_size, 16, 30] -> [batch_size, 1, 30] -> [batch_size, 30]
        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
        
        return scores
