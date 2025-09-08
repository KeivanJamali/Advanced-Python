import torch
import torch.nn as nn

class PreModel(nn.Module):
    def __init__(self, x1_dim, x2_dim, d_model=64, nhead=4, num_layers=2, mlp_hidden=64, out_dim=32):
        """
        x1_dim: feature dimension of sequential input
        x2_dim: dimension of static features
        d_model: transformer embedding size
        nhead: number of attention heads
        num_layers: number of transformer encoder layers
        mlp_hidden: hidden size of MLP
        out_dim: final output size (z), default 32
        """
        super().__init__()

        # Project input sequence into transformer dimension
        self.input_proj = nn.Linear(x1_dim, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2*d_model,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP after concatenating transformer output and x2
        self.mlp = nn.Sequential(
            nn.Linear(d_model + x2_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, out_dim)
        )

    def forward(self, x1, x2):
        """
        x1: [batch, seq_len, x1_dim]
        x2: [batch, x2_dim]
        """
        # Transformer part
        x1_proj = self.input_proj(x1)                # [batch, seq_len, d_model]
        x1_enc = self.transformer(x1_proj)           # [batch, seq_len, d_model]
        x1_pooled = x1_enc.mean(dim=1)               # simple mean pooling over sequence
        # Merge with x2
        merged = torch.cat([x1_pooled, x2], dim=1)   # [batch, d_model + x2_dim]

        # Final MLP
        z = self.mlp(merged)   
        print(x1[0])                      # [batch, out_dim]
        print(x2[0])                      # [batch, out_dim]
        print(z[0])
        return z

class MainModel(nn.Module):
    def __init__(self, z_dim, x_dim, hidden1, hidden2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + x_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def forward(self, z, x):
        """
        z: [batch, z_dim]
        x: [batch, x_dim]
        """
        merged = torch.cat([z, x], dim=1)
        logits = self.mlp(merged)
        return logits  # pass through sigmoid outside if using BCEWithLogitsLoss
