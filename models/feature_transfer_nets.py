# Alex Costanzino, CVLab
# July 2023

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class FeatureProjectionMLP(torch.nn.Module):
    def __init__(self, in_features = None, out_features = None, act_layer = torch.nn.GELU):
        super().__init__()
        
        self.act_fcn = act_layer()

        self.input = torch.nn.Linear(in_features, (in_features + out_features) // 2)
        self.projection = torch.nn.Linear((in_features + out_features) // 2, (in_features + out_features) // 2)
        self.output = torch.nn.Linear((in_features + out_features) // 2, out_features)

    def forward(self, x):
        x = self.input(x)
        x = self.act_fcn(x)

        x = self.projection(x)
        x = self.act_fcn(x)

        x = self.output(x)

        return x
    
class FeatureProjectionMLP_big(torch.nn.Module):
    def __init__(self, in_features = None, out_features = None, act_layer = torch.nn.GELU):
        super().__init__()
        
        self.act_fcn = act_layer()

        self.input = torch.nn.Linear(in_features, (in_features + out_features) // 2)
        
        self.projection_a = torch.nn.Linear((in_features + out_features) // 2, (in_features + out_features) // 2)
        self.projection_b = torch.nn.Linear((in_features + out_features) // 2, (in_features + out_features) // 2)
        self.projection_c = torch.nn.Linear((in_features + out_features) // 2, (in_features + out_features) // 2)
        self.projection_d = torch.nn.Linear((in_features + out_features) // 2, (in_features + out_features) // 2)
        self.projection_e = torch.nn.Linear((in_features + out_features) // 2, (in_features + out_features) // 2)

        self.output = torch.nn.Linear((in_features + out_features) // 2, out_features)

    def forward(self, x):
        x = self.input(x)
        x = self.act_fcn(x)

        x = self.projection_a(x)
        x = self.act_fcn(x)
        x = self.projection_b(x)
        x = self.act_fcn(x)
        x = self.projection_c(x)
        x = self.act_fcn(x)
        x = self.projection_d(x)
        x = self.act_fcn(x)
        x = self.projection_e(x)
        x = self.act_fcn(x)

        x = self.output(x)

        return x    

# Perhaps you need more traditional ResNet; This is not a standard one.
class ResMLP(nn.Module):
    def __init__(self, in_features=None, out_features=None, act_layer=torch.nn.GELU):
        super().__init__()

        self.act = act_layer()
        mid = (in_features + out_features) // 2

        self.input = torch.nn.Linear(in_features, mid)

        self.blocks = torch.nn.ModuleList([
            torch.nn.Linear(mid, mid) for _ in range(5)
        ])

        self.output = torch.nn.Linear(mid, out_features)

    def forward(self, x):
        x = self.act(self.input(x))

        for layer in self.blocks:
            residual = x
            x = self.act(layer(x))
            x = x + residual 

        x = self.output(x)
        return x


class TokenTransformer(nn.Module):
    def __init__(
        self,
        in_dim=1152,
        out_dim=768,
        num_tokens=12,       # so token_dim = 1152 // 12 = 96
        token_dim=96,
        hidden_dim=384,
        num_layers=2,
        num_heads=4,
    ):
        super().__init__()
        assert in_dim == num_tokens * token_dim

        self.num_tokens = num_tokens
        self.token_dim = token_dim

        # 1. Linear projection into token space
        self.to_tokens = nn.Linear(in_dim, num_tokens * token_dim)

        # 2. Transformer encoder
        encoder = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers)

        # 3. Output projection
        self.fc_out = nn.Linear(token_dim, out_dim)

    def forward(self, x):
        """
        x: (B, 1152) or (B, N, 1152) where N is an arbitrary grouping dim
        return: (B, 768) or (B, N, 768) respectively
        """
        restore_seq = x.dim() == 3

        if restore_seq:
            B, N, D = x.shape
            x = x.reshape(B * N, D)
        else:
            B = x.size(0)
            N = 1

        # (B, 1152) → (B, num_tokens * token_dim)
        x = self.to_tokens(x)

        # → (B, num_tokens, token_dim)
        x = x.view(-1, self.num_tokens, self.token_dim)

        # transformer
        x = self.transformer(x)   # (B, num_tokens, token_dim)

        # mean-pool tokens
        x = x.mean(dim=1)         # (B, token_dim)

        # final projection
        x = self.fc_out(x)        # (B, out_dim)

        if restore_seq:
            x = x.view(B, N, -1)

        return x
    

class FNetBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        # FNet replaces Attention with a Fourier Transform, so there are no 
        # Attention weights (Q, K, V) to initialize here.
        
        # Norm layer 1 (equivalent to the one before attention in Pre-Norm)
        self.norm1 = nn.LayerNorm(dim)
        
        # Norm layer 2 (equivalent to the one before MLP)
        self.norm2 = nn.LayerNorm(dim)

        # Feed Forward Network (same as standard Transformer)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x: (Batch, Seq_Len, Dim)
        """
        # 1. Fourier Mixing Branch (Pre-Norm)
        residual = x
        x = self.norm1(x)
        
        # 2D Fourier Transform
        # FNet performs FFT along both the Sequence dimension (-2) and Hidden dimension (-1)
        # We take the real part of the result as per the paper.
        x = torch.fft.fft2(x, dim=(-2, -1)).real
        
        # Add Residual
        x = x + residual

        # 2. MLP Branch (Pre-Norm)
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        
        # Add Residual
        x = x + residual
        
        return x

class TokenFNet(nn.Module):
    def __init__(
        self,
        in_dim=1152,
        out_dim=768,
        num_tokens=12,       # so token_dim = 1152 // 12 = 96
        token_dim=96,
        hidden_dim=384,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()
        assert in_dim == num_tokens * token_dim

        self.num_tokens = num_tokens
        self.token_dim = token_dim

        # 1. Linear projection into token space
        self.to_tokens = nn.Linear(in_dim, num_tokens * token_dim)

        # 2. FNet Encoder Stack
        # We replace nn.TransformerEncoder with a sequential stack of FNetBlocks
        self.fnet_blocks = nn.Sequential(*[
            FNetBlock(
                dim=token_dim, 
                hidden_dim=hidden_dim, 
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # 3. Output projection
        self.fc_out = nn.Linear(token_dim, out_dim)

    def forward(self, x):
        """
        x: (B, 1152) or (B, N, 1152) where N is an arbitrary grouping dim
        return: (B, 768) or (B, N, 768) respectively
        """
        restore_seq = x.dim() == 3

        if restore_seq:
            B, N, D = x.shape
            x = x.reshape(B * N, D)
        else:
            B = x.size(0)
            N = 1

        # (B, 1152) → (B, num_tokens * token_dim)
        x = self.to_tokens(x)

        # → (B, num_tokens, token_dim)
        # This creates the "Sequence" dimension that FNet will mix
        x = x.view(-1, self.num_tokens, self.token_dim)

        # FNet Processing
        # Input: (B, 12, 96)
        # The FFT will mix information across the 12 tokens AND the 96 dimensions
        x = self.fnet_blocks(x) 

        # mean-pool tokens
        x = x.mean(dim=1)         # (B, token_dim)

        # final projection
        x = self.fc_out(x)        # (B, out_dim)

        if restore_seq:
            x = x.view(B, N, -1)

        return x
    

# To replace a standard MLP while maintaining the same parameter count, 
# you must reduce the hidden dimension by a factor of 2/3
class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit.
    
    Concept:
    Instead of strictly relu(x @ W), it is:
    (x @ W_gate).swish() * (x @ W_value)
    
    To be efficient, we project x -> 2*hidden in one go, then split.
    """
    def __init__(
        self, 
        in_features: int, 
        hidden_features: int = None, 
        out_features: int = None, 
        bias: bool = True
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # We project to 2 * hidden_features to create both the gate and the value
        # in a single kernel launch.
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        # 1. Project to double dimension
        x12 = self.w12(x)
        
        # 2. Split into gate and value
        # chunk(2, dim=-1) splits the last dimension into 2 halves
        gate, value = x12.chunk(2, dim=-1)
        
        # 3. Apply Swish (SiLU) to the gate and multiply by value
        # "The gate decides what information from 'value' passes through"
        hidden = F.silu(gate) * value
        
        # 4. Final projection
        return self.w3(hidden)

# Perhaps Better because GELU is used.
# og Projection uses linear -> GELU -> linear -> GELU -> linear
class GeGLU(nn.Module):
    """
    GELU-Gated Linear Unit.
    Same mechanics as SwiGLU, but uses GELU for the gate.
    """
    def __init__(
        self, 
        in_features: int, 
        hidden_features: int = None, 
        out_features: int = None, 
        bias: bool = True
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        gate, value = x12.chunk(2, dim=-1)
        
        # Main difference is here: GELU instead of SiLU
        hidden = F.gelu(gate) * value
        
        return self.w3(hidden)