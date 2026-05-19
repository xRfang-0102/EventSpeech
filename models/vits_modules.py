import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """Transformer-based text encoder for phoneme sequences.

    Input:  text_indices [B, T_text]
    Output: F_t [B, T_a, 256]
    """

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_length, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, hidden_dim)

    def forward(
        self,
        text: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            text: Text/phoneme indices [B, T_text]
            lengths: Sequence lengths [B] (optional)

        Returns:
            Text features F_t [B, T_text, 256]
        """
        B, T = text.shape
        x = self.embedding(text) + self.pos_encoding[:, :T, :]

        if lengths is not None:
            # Create padding mask
            mask = torch.arange(T, device=text.device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            mask = None

        x = self.transformer(x, src_key_padding_mask=mask)
        return self.output_proj(x)


class PriorEncoder(nn.Module):
    """Prior encoder for VITS architecture.

    Predicts prior distribution parameters from text,
    emotion, and speaker features.

    Input:  F_t [B, T_a, 256], emotion_id [B], speaker_id [B]
    Output: mu_prior [B, T_a, 128], log_sigma_prior [B, T_a, 128]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_emotions: int = 8,
        num_speakers: int = 100,
        emotion_dim: int = 256,
        speaker_dim: int = 256,
    ):
        super().__init__()
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_dim)
        self.speaker_embedding = nn.Embedding(num_speakers, speaker_dim)

        self.input_proj = nn.Linear(hidden_dim + emotion_dim + speaker_dim, hidden_dim)

        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.log_sigma_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        F_t: torch.Tensor,
        emotion_id: torch.Tensor,
        speaker_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            F_t: Text features [B, T_a, 256]
            emotion_id: Emotion class ID [B]
            speaker_id: Speaker class ID [B]

        Returns:
            Tuple of (mu_prior, log_sigma_prior) each [B, T_a, 128]
        """
        B, T, _ = F_t.shape

        # Get embeddings
        emotion_emb = self.emotion_embedding(emotion_id).unsqueeze(1).expand(-1, T, -1)
        speaker_emb = self.speaker_embedding(speaker_id).unsqueeze(1).expand(-1, T, -1)

        # Concatenate all features
        x = torch.cat([F_t, emotion_emb, speaker_emb], dim=-1)
        x = self.input_proj(x)

        # Process through network
        x = self.network(x)

        # Predict distribution parameters
        mu = self.mu_proj(x)
        log_sigma = self.log_sigma_proj(x)

        return mu, log_sigma


class PosteriorEncoder(nn.Module):
    """Posterior encoder for VITS architecture.

    Encodes ground-truth mel spectrogram into posterior distribution
    during training.

    Input:  mel_spec [B, 80, T_a]
    Output: mu_post [B, T_a, 128], log_sigma_post [B, T_a, 128]
    """

    def __init__(
        self,
        mel_channels: int = 80,
        hidden_dim: int = 256,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.input_proj = nn.Linear(mel_channels, hidden_dim)

        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.log_sigma_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        mel_spec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            mel_spec: Mel spectrogram [B, 80, T_a]

        Returns:
            Tuple of (mu_post, log_sigma_post) each [B, T_a, 128]
        """
        # [B, 80, T_a] -> [B, T_a, 80]
        x = mel_spec.transpose(1, 2)
        x = self.input_proj(x)

        x = self.network(x)

        mu = self.mu_proj(x)
        log_sigma = self.log_sigma_proj(x)

        return mu, log_sigma

    def sample(
        self,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from posterior using reparameterization trick."""
        sigma = torch.exp(0.5 * log_sigma)
        epsilon = torch.randn_like(sigma)
        return mu + epsilon * sigma


class KnowledgeBridge(nn.Module):
    """Knowledge Bridge module.

    Maps low-resolution latent variables to high-capacity
    hidden representations for CFM decoder conditioning.

    Input:  z [B, T_a, 128]
    Output: hidden [B, T_a, 256]
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 256,
        num_layers: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()
        layers = []

        # First layer
        layers.extend([
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
        ])

        # Residual layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(inplace=True),
            ])

        self.network = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Latent variable [B, T_a, 128]

        Returns:
            Hidden representation [B, T_a, 256]
        """
        # [B, T_a, 128] -> [B, 128, T_a]
        x = z.transpose(1, 2)
        x = self.network(x)
        # [B, 256, T_a] -> [B, T_a, 256]
        x = x.transpose(1, 2)
        return self.norm(x)


def kl_divergence_loss(
    mu_post: torch.Tensor,
    log_sigma_post: torch.Tensor,
    mu_prior: torch.Tensor,
    log_sigma_prior: torch.Tensor,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Compute KL divergence between posterior and prior distributions.

    Args:
        mu_post: Posterior mean [B, T, D]
        log_sigma_post: Posterior log variance [B, T, D]
        mu_prior: Prior mean [B, T, D]
        log_sigma_prior: Prior log variance [B, T, D]
        epsilon: Safety constant to prevent log(0)

    Returns:
        KL divergence loss scalar
    """
    sigma_post = torch.exp(log_sigma_post) + epsilon
    sigma_prior = torch.exp(log_sigma_prior) + epsilon

    kl = 0.5 * (
        log_sigma_prior - log_sigma_post +
        (sigma_post ** 2 + (mu_post - mu_prior) ** 2) / (sigma_prior ** 2 + epsilon) -
        1.0
    )

    return kl.mean()
