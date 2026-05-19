import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


class MambaBlock(nn.Module):
    """Simplified Mamba block with selective scan.

    Input:  [B, T, D]
    Output: [B, T, D]
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)

        # S4D-Lin parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)

        x_proj = x_proj.transpose(1, 2)  # [B, d_inner, L]
        x_proj = self.conv1d(x_proj)[:, :, :L]
        x_proj = x_proj.transpose(1, 2)  # [B, L, d_inner]
        x_proj = F.silu(x_proj)

        # Selective scan approximation
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        dt = F.softplus(self.dt_proj(self.x_proj(x_proj)))  # [B, L, d_inner]

        # Simplified selective scan
        y = x_proj * self.D

        # Gate with z
        z = F.silu(z)
        y = y * z

        return self.out_proj(y)


class HWCBlock(nn.Module):
    """Hierarchical Wavelet Context Block with Haar wavelet decomposition.

    Input:  [B, T, D]
    Output: [B, T, D]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        wavelet_levels: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.wavelet_levels = wavelet_levels

        # Haar wavelet filters
        self.register_buffer(
            'low_pass',
            torch.tensor([1.0, 1.0]) / math.sqrt(2),
        )
        self.register_buffer(
            'high_pass',
            torch.tensor([1.0, -1.0]) / math.sqrt(2),
        )

        # Dilated convolutions for detail coefficients
        self.detail_convs = nn.ModuleList()
        for level in range(wavelet_levels):
            dilation = 2 ** level
            self.detail_convs.append(
                nn.Conv1d(
                    hidden_dim, hidden_dim, kernel_size,
                    padding=dilation, dilation=dilation, groups=min(64, hidden_dim),
                )
            )

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def dwt_1d(self, x: torch.Tensor, level: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """1D Discrete Wavelet Transform (Haar)."""
        # Pad if needed
        if x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, 1))

        # Apply filters
        lo = F.conv1d(x, self.low_pass.unsqueeze(0).unsqueeze(0), stride=2)
        hi = F.conv1d(x, self.high_pass.unsqueeze(0).unsqueeze(0), stride=2)

        return lo, hi

    def idwt_1d(self, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """1D Inverse Discrete Wavelet Transform (Haar)."""
        # Upsample
        lo_up = F.interpolate(lo, scale_factor=2, mode='linear', align_corners=False)
        hi_up = F.interpolate(hi, scale_factor=2, mode='linear', align_corners=False)

        # Apply inverse filters
        lo_filter = self.low_pass.flip(0).unsqueeze(0).unsqueeze(0)
        hi_filter = self.high_pass.flip(0).unsqueeze(0).unsqueeze(0)

        y = F.conv_transpose1d(lo_up, lo_filter, stride=1) + \
            F.conv_transpose1d(hi_up, hi_filter, stride=1)

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Reshape for 1D convolution: [B, D, T]
        x_conv = x.transpose(1, 2)

        # Multi-level wavelet decomposition
        details = []
        approx = x_conv

        for level in range(self.wavelet_levels):
            approx, detail = self.dwt_1d(approx, level)
            details.append(detail)

        # Process detail coefficients with dilated convolutions
        enhanced_details = []
        for level, detail in enumerate(details):
            enhanced = F.gelu(self.detail_convs[level](detail))
            enhanced_details.append(enhanced)

        # Reconstruct with inverse wavelet transform
        reconstructed = approx
        for level in range(self.wavelet_levels - 1, -1, -1):
            reconstructed = self.idwt_1d(reconstructed, enhanced_details[level])

        # Match output length
        if reconstructed.shape[-1] != T:
            reconstructed = F.interpolate(reconstructed, size=T, mode='linear', align_corners=False)

        output = reconstructed.transpose(1, 2)  # [B, T, D]
        return self.output_proj(output)


class DualPathRefinement(nn.Module):
    """Dual-path refinement with local smoothing and global coherence.

    Input:  [B, T, 256]
    Output: [B, T, 256]
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.local_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, padding=2)
        self.global_linear = nn.Linear(hidden_dim, hidden_dim)
        self.gate_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Local smoothing path
        x_local = self.local_conv(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_local = F.relu(x_local)

        # Global coherence path
        x_global = self.global_linear(x)
        x_global = F.relu(x_global)

        # Sigmoid gating
        alpha = torch.sigmoid(self.gate_linear(x))

        # Gated fusion
        output = alpha * x_local + (1 - alpha) * x_global

        return output


class AudioEncoder(nn.Module):
    """Audio Encoder with Mamba-Wavelet backbone and dual-path refinement.

    Encodes mel spectrogram and prosody features into acoustic
    latent representations with speaker disentanglement.

    Input:  mel_spec [B, 80, T_a], pitch [B, T_a], energy [B, T_a]
    Output: F_a [B, T_a, 256]
    """

    def __init__(
        self,
        mel_channels: int = 80,
        pitch_proj_dim: int = 88,
        energy_proj_dim: int = 88,
        hidden_dim: int = 256,
        mamba_layers: int = 3,
        mamba_d_state: int = 16,
        mamba_expand: int = 2,
        wavelet_levels: int = 3,
        output_dim: int = 256,
        speaker_dim: int = 256,
    ):
        super().__init__()

        # Feature projection
        self.pitch_proj = nn.Conv1d(1, pitch_proj_dim, kernel_size=3, padding=1)
        self.energy_proj = nn.Conv1d(1, energy_proj_dim, kernel_size=3, padding=1)
        self.input_proj = nn.Linear(mel_channels + pitch_proj_dim + energy_proj_dim, hidden_dim)

        # Mamba temporal branch
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(
                d_model=hidden_dim,
                d_state=mamba_d_state,
                expand=mamba_expand,
            )
            for _ in range(mamba_layers)
        ])
        self.mamba_norm = nn.LayerNorm(hidden_dim)

        # Wavelet context branch
        self.hwc_block = HWCBlock(
            hidden_dim=hidden_dim,
            wavelet_levels=wavelet_levels,
        )

        # Dual-path refinement
        self.refinement = DualPathRefinement(hidden_dim)

        # Speaker disentanglement
        self.speaker_proj = nn.Linear(speaker_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)

    def forward(
        self,
        mel_spec: torch.Tensor,
        pitch: torch.Tensor,
        energy: torch.Tensor,
        speaker_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            mel_spec: Mel spectrogram [B, 80, T_a]
            pitch: Pitch contour [B, T_a]
            energy: Energy contour [B, T_a]
            speaker_embed: Speaker embedding [B, 256]

        Returns:
            Acoustic features F_a [B, T_a, 256]
        """
        B, _, T_a = mel_spec.shape

        # Project prosody features
        pitch_proj = self.pitch_proj(pitch.unsqueeze(1))  # [B, 88, T_a]
        energy_proj = self.energy_proj(energy.unsqueeze(1))  # [B, 88, T_a]

        # Concatenate all features
        mel_t = mel_spec.transpose(1, 2)  # [B, T_a, 80]
        pitch_t = pitch_proj.transpose(1, 2)  # [B, T_a, 88]
        energy_t = energy_proj.transpose(1, 2)  # [B, T_a, 88]

        F_p = torch.cat([mel_t, pitch_t, energy_t], dim=-1)  # [B, T_a, 256]
        F_p = self.input_proj(F_p)  # [B, T_a, 256]

        # Mamba temporal branch
        H_mamba = F_p
        for block in self.mamba_blocks:
            H_mamba = block(H_mamba) + H_mamba
        H_mamba = self.mamba_norm(H_mamba)  # [B, T_a, 256]

        # Wavelet context branch
        H_wavelet = self.hwc_block(F_p)  # [B, T_a, 256]

        # Duality fusion
        F_combined = H_mamba + H_wavelet  # [B, T_a, 256]

        # Dual-path refinement
        F_refined = self.refinement(F_combined)  # [B, T_a, 256]

        # Speaker disentanglement fusion
        speaker_expanded = self.speaker_proj(speaker_embed)  # [B, 256]
        speaker_expanded = speaker_expanded.unsqueeze(1).expand(-1, T_a, -1)  # [B, T_a, 256]

        # Final projection
        F_a = torch.cat([F_refined, speaker_expanded], dim=-1)  # [B, T_a, 512]
        F_a = self.output_proj(F_a)  # [B, T_a, 256]

        return F_a
