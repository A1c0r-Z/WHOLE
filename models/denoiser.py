"""Transformer-based DDPM denoiser for WHOLE.

Architecture (Sec. 3.1, Appendix A):
  - 4-layer Transformer encoder (bidirectional, non-autoregressive)
  - 4 attention heads, hidden dim d_model=512, FFN dim d_ff=2048
  - ~12.35M parameters
  - Input: noisy trajectory x_n (B, T, 73) + diffusion step n
  - Conditioning: noisy hand estimate H̃ (B, T, 62) + object BPS O (B, K)
  - Optional: Ambient Sensor feature (B, T, J*3) for contact reasoning
  - Output: predicted clean trajectory x̂_0 (B, T, 73)

All modalities are projected to d_model=512 and summed before the transformer.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal diffusion timestep → MLP → d_model."""

    def __init__(self, d_model: int, max_steps: int = 1000):
        super().__init__()
        self.d_model = d_model
        # Sinusoidal base dimension (half of d_model)
        half = d_model // 2
        freqs = torch.exp(-math.log(max_steps) *
                          torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer('freqs', freqs)   # (half,)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer diffusion steps in [0, N-1]
        Returns:
            (B, d_model) timestep embeddings
        """
        t = t.float()
        args = t.unsqueeze(1) * self.freqs.unsqueeze(0)   # (B, half)
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, d_model)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Learnable positional encoding
# ---------------------------------------------------------------------------

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model) with positional encoding added
        """
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        return x + self.pe(pos).unsqueeze(0)


# ---------------------------------------------------------------------------
# Main denoiser
# ---------------------------------------------------------------------------

class WHOLEDenoiser(nn.Module):
    """Non-autoregressive Transformer denoiser for WHOLE.

    Predicts the clean trajectory x̂_0 from noisy x_n and conditioning.

    Args:
        x_dim:     diffusion variable dim (73)
        h_dim:     noisy hand conditioning dim (62)
        bps_dim:   object BPS descriptor dim (1024)
        d_model:   transformer hidden dim (512)
        nhead:     attention heads (4)
        d_ff:      feedforward dim (2048)
        num_layers: number of transformer layers (4)
        window_len: sequence length T (120)
        n_joints:  hand joints for ambient sensor (21 per hand × 2 = 42)
        use_ambient: whether to use ambient sensor feature
        dropout:   dropout rate
    """

    def __init__(
        self,
        x_dim:      int   = 73,
        h_dim:      int   = 62,
        bps_dim:    int   = 1024,
        d_model:    int   = 512,
        nhead:      int   = 4,
        d_ff:       int   = 2048,
        num_layers: int   = 4,
        window_len: int   = 120,
        n_joints:   int   = 42,   # 21 per hand × 2 hands
        use_ambient: bool = True,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.x_dim       = x_dim
        self.use_ambient = use_ambient

        # --- Input projections (all → d_model) ---
        self.x_proj       = nn.Linear(x_dim, d_model)
        self.h_proj       = nn.Linear(h_dim,  d_model)
        self.o_proj       = nn.Linear(bps_dim, d_model)
        self.t_emb        = SinusoidalTimestepEmbedding(d_model)
        self.pos_enc      = LearnablePositionalEncoding(window_len, d_model)

        if use_ambient:
            # Ambient sensor: per-joint displacement (J, 3) → flattened → d_model
            self.ambient_proj = nn.Linear(n_joints * 3, d_model)

        # --- Transformer encoder layers ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = d_ff,
            dropout         = dropout,
            batch_first     = True,   # (B, T, D) convention
            norm_first      = True,   # Pre-LN for stable training
        )
        # enable_nested_tensor requires norm_first=False; suppress the warning
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # --- Output projection ---
        self.out_proj = nn.Linear(d_model, x_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        x_n:        torch.Tensor,                    # (B, T, 73)  noisy trajectory
        t:          torch.Tensor,                    # (B,)        diffusion step
        H_tilde:    torch.Tensor,                    # (B, T, 62)  noisy hand est.
        O:          torch.Tensor,                    # (B, K)      object BPS
        ambient:    torch.Tensor | None = None,      # (B, T, J*3) ambient sensor
        key_padding_mask: torch.Tensor | None = None,# (B, T) bool: True=padding
    ) -> torch.Tensor:
        """Predict clean trajectory x̂_0.

        Args:
            x_n:      (B, T, 73) noisy trajectory at step n
            t:        (B,)       integer diffusion step indices
            H_tilde:  (B, T, 62) noisy bimanual hand conditioning
            O:        (B, K)     object BPS descriptor
            ambient:  (B, T, J*3) ambient sensor feature; computed externally
                      or omitted (model still runs without it)
            key_padding_mask: True for padded/invalid frames
        Returns:
            (B, T, 73) predicted clean trajectory x̂_0
        """
        B, T, _ = x_n.shape

        # Project each modality to d_model
        h = self.x_proj(x_n)                                 # (B, T, D)
        h = h + self.h_proj(H_tilde)                          # per-frame hand cond
        h = h + self.o_proj(O).unsqueeze(1).expand(-1, T, -1) # global object cond

        # Timestep embedding broadcast over T
        t_emb = self.t_emb(t).unsqueeze(1).expand(-1, T, -1) # (B, T, D)
        h = h + t_emb

        # Ambient sensor (optional)
        if self.use_ambient and ambient is not None:
            h = h + self.ambient_proj(ambient)                # (B, T, D)

        # Positional encoding
        h = self.pos_enc(h)                                   # (B, T, D)

        # Transformer (bidirectional, non-autoregressive)
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)  # (B, T, D)

        return self.out_proj(h)                               # (B, T, 73)

    @torch.no_grad()
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory from config
# ---------------------------------------------------------------------------

def build_denoiser(cfg: dict) -> WHOLEDenoiser:
    """Build denoiser from config dict (subset of default.yaml model section).

    Args:
        cfg: dict with keys matching WHOLEDenoiser __init__ args
    Returns:
        WHOLEDenoiser instance
    """
    m = cfg.get('model', cfg)
    return WHOLEDenoiser(
        x_dim       = m.get('x_dim',       73),
        h_dim       = m.get('h_dim',       62),
        bps_dim     = m.get('bps_n_points', 1024),
        d_model     = m.get('d_model',     512),
        nhead       = m.get('nhead',         4),
        d_ff        = m.get('d_ff',        2048),
        num_layers  = m.get('num_layers',    4),
        window_len  = m.get('window_len',  120),
        n_joints    = m.get('n_hand_joints', 21) * 2,
        use_ambient = m.get('use_ambient_sensor', True),
        dropout     = m.get('dropout',     0.1),
    )
