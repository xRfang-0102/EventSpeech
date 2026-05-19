import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ResBlock(nn.Module):
    """Residual block with optional downsampling.

    Input:  [B, C_in, H, W]
    Output: [B, C_out, H//stride, W//stride]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class SpatialCNN(nn.Module):
    """Spatial downsampling backbone for event encoder.

    Input:  [B*T_v, 3, 346, 260]
    Output: [B*T_v, 512]
    """

    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer2 = ResBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.layer3 = ResBlock(256, 512, kernel_size=3, stride=2, padding=1)
        self.layer4 = ResBlock(512, 512, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.flatten(1)


class MHFELayer(nn.Module):
    """Multi-Head Feature Extraction Layer.

    Decomposes features into 5 parallel trait heads:
    Lip Motion, Facial AU, Head Pose, Speaking Rhythm, Visual Prosody.

    Input:  [B, T_v, 512]
    Output: 5 x [B, T_v, 256]
    """

    def __init__(self, input_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.lip_motion_head = nn.Linear(input_dim, output_dim)
        self.facial_au_head = nn.Linear(input_dim, output_dim)
        self.head_pose_head = nn.Linear(input_dim, output_dim)
        self.speaking_rhythm_head = nn.Linear(input_dim, output_dim)
        self.visual_prosody_head = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_lip = self.lip_motion_head(x)
        h_au = self.facial_au_head(x)
        h_pose = self.head_pose_head(x)
        h_rhythm = self.speaking_rhythm_head(x)
        h_prosody = self.visual_prosody_head(x)
        return h_lip, h_au, h_pose, h_rhythm, h_prosody


class EventEncoder(nn.Module):
    """Event Encoder with Spatial CNN, BiGRU, and MHFE.

    Encodes neuromorphic event voxels into visual latent features
    with multi-head disentanglement.

    Input:  x_event [B, T_v, 3, 346, 260]
    Output: F_v [B, T_v, 256]
    """

    def __init__(
        self,
        spatial_channels: list = [64, 128, 256, 512, 512],
        bigru_hidden: int = 512,
        bigru_layers: int = 1,
        mhfe_dim: int = 256,
        output_dim: int = 256,
        emotion_dim: int = 256,
        speaker_dim: int = 256,
    ):
        super().__init__()

        self.spatial_cnn = SpatialCNN()

        self.bilstm = nn.GRU(
            input_size=512,
            hidden_size=bigru_hidden,
            num_layers=bigru_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.temporal_proj = nn.Linear(bigru_hidden * 2, 512)

        self.mhfe = MHFELayer(input_dim=512, output_dim=mhfe_dim)

        self.fusion = nn.Sequential(
            nn.Linear(mhfe_dim + emotion_dim + speaker_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        x_event: torch.Tensor,
        emotion_embed: torch.Tensor,
        speaker_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_event: Event voxels [B, T_v, 3, H, W]
            emotion_embed: Emotion embedding [B, 256]
            speaker_embed: Speaker embedding [B, 256]

        Returns:
            Visual features F_v [B, T_v, 256]
        """
        B, T_v, C, H, W = x_event.shape

        # Spatial CNN: flatten batch and time
        x = x_event.reshape(B * T_v, C, H, W)
        x = self.spatial_cnn(x)  # [B*T_v, 512]
        x = x.reshape(B, T_v, -1)  # [B, T_v, 512]

        # Temporal BiGRU
        x, _ = self.bilstm(x)  # [B, T_v, 1024]
        x = self.temporal_proj(x)  # [B, T_v, 512]

        # MHFE disentanglement
        h_lip, h_au, h_pose, h_rhythm, h_prosody = self.mhfe(x)

        # Hierarchical fusion: sum all heads
        F_motion = h_lip + h_au + h_pose + h_rhythm + h_prosody  # [B, T_v, 256]

        # Fuse with global emotion and speaker embeddings
        emotion_expanded = emotion_embed.unsqueeze(1).expand(-1, T_v, -1)
        speaker_expanded = speaker_embed.unsqueeze(1).expand(-1, T_v, -1)

        fused = torch.cat([F_motion, emotion_expanded, speaker_expanded], dim=-1)
        F_v = self.fusion(fused)  # [B, T_v, 256]

        return F_v
