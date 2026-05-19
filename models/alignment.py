import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


class CrossModalAlignment(nn.Module):
    """Hierarchical Cross-Modal Alignment Module.

    Resolves structural misalignment between visual event timesteps T_v
    and acoustic timesteps T_a using bidirectional cross-attention.

    Input:  F_v [B, T_v, 256], F_a [B, T_a, 256]
    Output: F_align [B, T_a, 256]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Feature projection
        self.v_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.a_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Visual-to-Audio cross-attention (V->A)
        self.v_to_a_q = nn.Linear(hidden_dim, hidden_dim)
        self.v_to_a_k = nn.Linear(hidden_dim, hidden_dim)
        self.v_to_a_v = nn.Linear(hidden_dim, hidden_dim)
        self.v_to_a_out = nn.Linear(hidden_dim, hidden_dim)

        # Audio-to-Visual cross-attention (A->V)
        self.a_to_v_q = nn.Linear(hidden_dim, hidden_dim)
        self.a_to_v_k = nn.Linear(hidden_dim, hidden_dim)
        self.a_to_v_v = nn.Linear(hidden_dim, hidden_dim)
        self.a_to_v_out = nn.Linear(hidden_dim, hidden_dim)

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def _interpolate_visual(
        self,
        v_features: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """Interpolate visual features to match audio length.

        Uses 1D linear interpolation to temporally align visual
        sequence to acoustic sequence length.
        """
        # [B, T_v, D] -> [B, D, T_v]
        v_t = v_features.transpose(1, 2)

        # Interpolate to target length
        v_interp = F.interpolate(
            v_t,
            size=target_length,
            mode='linear',
            align_corners=False,
        )

        # [B, D, T_a] -> [B, T_a, D]
        return v_interp.transpose(1, 2)

    def _bidirectional_cross_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        out_proj: nn.Linear,
    ) -> torch.Tensor:
        """Compute multi-head cross-attention.

        Args:
            query: Query tensor [B, T_q, D]
            key: Key tensor [B, T_k, D]
            value: Value tensor [B, T_k, D]
            q_proj: Query projection
            k_proj: Key projection
            v_proj: Value projection
            out_proj: Output projection

        Returns:
            Attention output [B, T_q, D]
        """
        B, T_q, D = query.shape
        T_k = key.shape[1]

        # Project Q, K, V
        Q = q_proj(query).reshape(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = k_proj(key).reshape(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = v_proj(value).reshape(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, H, T_q, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(B, T_q, D)

        return out_proj(attn_output)

    def forward(
        self,
        F_v: torch.Tensor,
        F_a: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            F_v: Visual features [B, T_v, 256]
            F_a: Acoustic features [B, T_a, 256]

        Returns:
            Aligned features F_align [B, T_a, 256]
        """
        B, T_v, D = F_v.shape
        T_a = F_a.shape[1]

        # Project features
        V_proj = self.v_proj(F_v)  # [B, T_v, 256]
        A_proj = self.a_proj(F_a)  # [B, T_a, 256]

        # Interpolate visual to match audio length
        V_proj_interp = self._interpolate_visual(V_proj, T_a)  # [B, T_a, 256]

        # Visual-to-Audio cross-attention
        # Query: Audio, Key/Value: Visual
        F_v_to_a = self._bidirectional_cross_attention(
            query=A_proj,
            key=V_proj_interp,
            value=V_proj_interp,
            q_proj=self.v_to_a_q,
            k_proj=self.v_to_a_k,
            v_proj=self.v_to_a_v,
            out_proj=self.v_to_a_out,
        )  # [B, T_a, 256]

        # Audio-to-Visual cross-attention
        # Query: Visual, Key/Value: Audio
        F_a_to_v = self._bidirectional_cross_attention(
            query=V_proj_interp,
            key=A_proj,
            value=A_proj,
            q_proj=self.a_to_v_q,
            k_proj=self.a_to_v_k,
            v_proj=self.a_to_v_v,
            out_proj=self.a_to_v_out,
        )  # [B, T_a, 256]

        # Concatenate and fuse
        F_concat = torch.cat([F_v_to_a, F_a_to_v], dim=-1)  # [B, T_a, 512]
        F_align = self.fusion(F_concat)  # [B, T_a, 256]

        return F_align
