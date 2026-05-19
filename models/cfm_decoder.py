import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps.

    Input:  t [B, 1]
    Output: embed [B, embed_dim]
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device).float() * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class DilatedResidualBlock(nn.Module):
    """Dilated residual block with conditional bias injection.

    Input:  x_in [B, C, T]
    Output: x_out [B, C, T], skip [B, C, T]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        kernel_size: int = 3,
        dilation: int = 1,
        condition_dim: int = 256,
    ):
        super().__init__()
        self.dilation = dilation
        padding = dilation * (kernel_size - 1) // 2

        # Dilated convolution
        self.dilated_conv = nn.Conv1d(
            hidden_dim, hidden_dim * 2,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        # Condition projection
        self.condition_proj = nn.Conv1d(condition_dim, hidden_dim * 2, kernel_size=1)

        # Time embedding projection
        self.time_proj = nn.Linear(128, hidden_dim * 2)

        # Skip connection
        self.skip_proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

        # Residual connection
        self.residual_proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(
        self,
        x_in: torch.Tensor,
        t_embed: torch.Tensor,
        cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x_in: Input features [B, C, T]
            t_embed: Time embedding [B, 128]
            cond: Condition features [B, T, C] (transposed internally)

        Returns:
            Tuple of (residual_output, skip_output)
        """
        # Dilated convolution
        x_conv = self.dilated_conv(x_in)  # [B, 2*C, T]

        # Add time embedding bias
        t_bias = self.time_proj(t_embed).unsqueeze(-1)  # [B, 2*C, 1]
        x_conv = x_conv + t_bias

        # Add condition bias
        cond_t = cond.transpose(1, 2)  # [B, C, T]
        cond_bias = self.condition_proj(cond_t)  # [B, 2*C, T]
        x_conv = x_conv + cond_bias

        # Gated activation
        filter_gate, gate = x_conv.chunk(2, dim=1)
        x_gated = torch.tanh(filter_gate) * torch.sigmoid(gate)  # [B, C, T]

        # Skip connection
        skip = self.skip_proj(x_gated)

        # Residual connection
        residual = self.residual_proj(x_gated)
        x_out = x_in + residual

        return x_out, skip


class CFMDecoder(nn.Module):
    """Conditional Flow Matching Decoder.

    Estimates velocity field for ODE-based mel spectrogram generation.
    Uses 20-layer dilated residual network with gated activation.

    Input:  x_t [B, 80, T_a], t [B, 1], cond [B, T_a, 256]
    Output: v_theta [B, 80, T_a]
    """

    def __init__(
        self,
        mel_channels: int = 80,
        hidden_dim: int = 256,
        time_embed_dim: int = 128,
        num_dilation_blocks: int = 20,
        dilation_schedule: list = [1, 2, 4, 8],
        kernel_size: int = 3,
        condition_dim: int = 256,
    ):
        super().__init__()
        self.mel_channels = mel_channels
        self.hidden_dim = hidden_dim
        self.num_blocks = num_dilation_blocks
        self.dilation_schedule = dilation_schedule

        # Input projection
        self.input_proj = nn.Conv1d(mel_channels, hidden_dim, kernel_size=1)

        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Dilated residual blocks with cyclic dilation schedule
        self.blocks = nn.ModuleList()
        for i in range(num_dilation_blocks):
            dilation = dilation_schedule[i % len(dilation_schedule)]
            self.blocks.append(
                DilatedResidualBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    condition_dim=condition_dim,
                )
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, mel_channels, kernel_size=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_t: Noisy mel spectrogram [B, 80, T_a]
            t: Timestep [B, 1]
            cond: Condition features [B, T_a, 256]

        Returns:
            Velocity field v_theta [B, 80, T_a]
        """
        B, C, T = x_t.shape

        # Project input
        x = self.input_proj(x_t)  # [B, 256, T]

        # Time embedding
        t_embed = self.time_embedding(t)  # [B, 256]

        # Process through dilated residual blocks
        skip_sum = torch.zeros_like(x)
        for block in self.blocks:
            x, skip = block(x, t_embed, cond)
            skip_sum = skip_sum + skip

        # Output projection
        v_theta = self.output_proj(skip_sum)  # [B, 80, T]

        return v_theta

    @torch.no_grad()
    def sample_euler(
        self,
        cond: torch.Tensor,
        num_steps: int = 20,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample using Euler ODE solver.

        Args:
            cond: Condition features [B, T_a, 256]
            num_steps: Number of Euler steps
            initial_noise: Initial noise [B, 80, T_a] (optional)

        Returns:
            Generated mel spectrogram [B, 80, T_a]
        """
        B = cond.shape[0]
        T = cond.shape[1]
        device = cond.device

        # Start from Gaussian noise
        if initial_noise is None:
            x = torch.randn(B, self.mel_channels, T, device=device)
        else:
            x = initial_noise

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B, 1), i * dt, device=device)
            v = self.forward(x, t, cond)
            x = x + v * dt

        return x

    @torch.no_grad()
    def sample_rk4(
        self,
        cond: torch.Tensor,
        num_steps: int = 20,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample using 4th-order Runge-Kutta ODE solver.

        Args:
            cond: Condition features [B, T_a, 256]
            num_steps: Number of RK4 steps
            initial_noise: Initial noise [B, 80, T_a] (optional)

        Returns:
            Generated mel spectrogram [B, 80, T_a]
        """
        B = cond.shape[0]
        T = cond.shape[1]
        device = cond.device

        # Start from Gaussian noise
        if initial_noise is None:
            x = torch.randn(B, self.mel_channels, T, device=device)
        else:
            x = initial_noise

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B, 1), i * dt, device=device)

            k1 = self.forward(x, t, cond)
            k2 = self.forward(x + 0.5 * dt * k1, t + 0.5 * dt, cond)
            k3 = self.forward(x + 0.5 * dt * k2, t + 0.5 * dt, cond)
            k4 = self.forward(x + dt * k3, t + dt, cond)

            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return x
