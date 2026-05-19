import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ReconstructionLoss(nn.Module):
    """L1 reconstruction loss between generated and ground-truth mel spectrogram.

    Input: pred_mel [B, 80, T_a], target_mel [B, 80, T_a]
    Output: scalar loss
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_mel: torch.Tensor,
        target_mel: torch.Tensor,
    ) -> torch.Tensor:
        return F.l1_loss(pred_mel, target_mel)


class FlowMatchingLoss(nn.Module):
    """Flow matching velocity field loss.

    Computes MSE between estimated velocity field and
    optimal transport straight-line trajectory velocity.

    Input: v_pred [B, 80, T_a], v_target [B, 80, T_a]
    Output: scalar loss
    """

    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(v_pred, v_target)


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for emotion alignment.

    Aligns audio and visual features in emotion embedding space
    through contrastive learning.

    Input: audio_features [B, D], visual_features [B, D]
    Output: scalar loss
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        # Normalize features
        audio_features = F.normalize(audio_features, dim=-1)
        visual_features = F.normalize(visual_features, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(audio_features, visual_features.T) / self.temperature

        # Labels are diagonal (matching pairs)
        labels = torch.arange(logits.shape[0], device=logits.device)

        # Cross-entropy loss in both directions
        loss_a2v = F.cross_entropy(logits, labels)
        loss_v2a = F.cross_entropy(logits.T, labels)

        return (loss_a2v + loss_v2a) / 2.0


class AdversarialLoss(nn.Module):
    """WGAN-GP adversarial loss for spectrogram refinement.

    Input: real_mel [B, 80, T_a], fake_mel [B, 80, T_a]
    Output: dict with generator and discriminator losses
    """

    def __init__(self, lambda_gp: float = 10.0):
        super().__init__()
        self.lambda_gp = lambda_gp

    def generator_loss(self, fake_score: torch.Tensor) -> torch.Tensor:
        return -fake_score.mean()

    def discriminator_loss(
        self,
        real_score: torch.Tensor,
        fake_score: torch.Tensor,
    ) -> torch.Tensor:
        return fake_score.mean() - real_score.mean()

    def gradient_penalty(
        self,
        real_mel: torch.Tensor,
        fake_mel: torch.Tensor,
        discriminator: nn.Module,
    ) -> torch.Tensor:
        batch_size = real_mel.shape[0]
        alpha = torch.rand(batch_size, 1, 1, device=real_mel.device)

        interpolated = (alpha * real_mel + (1 - alpha) * fake_mel).requires_grad_(True)
        interpolated_score = discriminator(interpolated)

        gradients = torch.autograd.grad(
            outputs=interpolated_score,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_score),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.reshape(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()

        return self.lambda_gp * penalty


class DisentanglementLoss(nn.Module):
    """Disentanglement loss for MHFE module.

    Includes:
    - Physical landmark regression for Lip Motion and Facial AU heads
    - Orthogonality penalty for Speaking Rhythm and Visual Prosody heads

    Input: h_lip, h_au, h_pose, h_rhythm, h_prosody [B, T_v, 256]
    Output: scalar loss
    """

    def __init__(self):
        super().__init__()

    def orthogonality_penalty(
        self,
        h_rhythm: torch.Tensor,
        h_prosody: torch.Tensor,
    ) -> torch.Tensor:
        """Compute orthogonality penalty between two feature heads."""
        # Flatten features
        h_r = h_rhythm.reshape(-1, h_rhythm.shape[-1])
        h_p = h_prosody.reshape(-1, h_prosody.shape[-1])

        # Normalize
        h_r = F.normalize(h_r, dim=-1)
        h_p = F.normalize(h_p, dim=-1)

        # Cosine similarity
        cos_sim = (h_r * h_p).sum(dim=-1)

        # Minimize cosine similarity (maximize orthogonality)
        return (cos_sim ** 2).mean()

    def forward(
        self,
        h_lip: torch.Tensor,
        h_au: torch.Tensor,
        h_rhythm: torch.Tensor,
        h_prosody: torch.Tensor,
        lip_targets: Optional[torch.Tensor] = None,
        au_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=h_lip.device)

        # Physical landmark regression (if targets available)
        if lip_targets is not None:
            total_loss = total_loss + F.mse_loss(h_lip, lip_targets)
        if au_targets is not None:
            total_loss = total_loss + F.mse_loss(h_au, au_targets)

        # Orthogonality penalty
        ortho_loss = self.orthogonality_penalty(h_rhythm, h_prosody)
        total_loss = total_loss + ortho_loss

        return total_loss


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining all loss components.

    Implements the composite loss with dynamic weight scheduling:
    - L_rec: Reconstruction loss (weight 1.0)
    - L_KL: KL divergence with linear annealing
    - L_align: Alignment loss (weight 0.3)
    - L_flow: Flow matching loss (weight 0.2)
    - L_adv: Adversarial loss (weight 0.1, activated after epoch 90)
    - L_disentangle: Disentanglement loss
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Loss weights
        self.lambda_rec = config.get('loss', {}).get('reconstruction', {}).get('weight', 1.0)
        self.lambda_align = config.get('loss', {}).get('alignment', {}).get('weight', 0.3)
        self.lambda_flow = config.get('loss', {}).get('flow_matching', {}).get('weight', 0.2)
        self.lambda_adv = config.get('loss', {}).get('adversarial', {}).get('weight', 0.1)

        # KL annealing
        self.kl_anneal_epochs = config.get('loss', {}).get('kl_divergence', {}).get('weight_anneal_epochs', 60)
        self.kl_weight_final = config.get('loss', {}).get('kl_divergence', {}).get('weight_final', 0.8)

        # Adversarial activation
        self.adv_activation_epoch = config.get('loss', {}).get('adversarial', {}).get('activation_epoch', 90)

        # Individual losses
        self.rec_loss = ReconstructionLoss()
        self.flow_loss = FlowMatchingLoss()
        self.info_nce_loss = InfoNCELoss()
        self.adv_loss = AdversarialLoss()
        self.disentangle_loss = DisentanglementLoss()

    def get_kl_weight(self, epoch: int) -> float:
        """Get KL divergence weight with linear annealing."""
        if epoch < self.kl_anneal_epochs:
            return self.kl_weight_final * (epoch / self.kl_anneal_epochs)
        return self.kl_weight_final

    def get_adv_weight(self, epoch: int) -> float:
        """Get adversarial loss weight (activated after threshold)."""
        if epoch >= self.adv_activation_epoch:
            return self.lambda_adv
        return 0.0

    def forward(
        self,
        pred_mel: torch.Tensor,
        target_mel: torch.Tensor,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        mu_post: torch.Tensor,
        log_sigma_post: torch.Tensor,
        mu_prior: torch.Tensor,
        log_sigma_prior: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        h_lip: torch.Tensor,
        h_au: torch.Tensor,
        h_rhythm: torch.Tensor,
        h_prosody: torch.Tensor,
        epoch: int,
        real_mel: Optional[torch.Tensor] = None,
        discriminator: Optional[nn.Module] = None,
        lip_targets: Optional[torch.Tensor] = None,
        au_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses.

        Returns:
            Dictionary with individual and total losses
        """
        losses = {}

        # Reconstruction loss
        L_rec = self.rec_loss(pred_mel, target_mel)
        losses['reconstruction'] = L_rec

        # KL divergence loss with annealing
        lambda_kl = self.get_kl_weight(epoch)
        sigma_post = torch.exp(0.5 * log_sigma_post)
        sigma_prior = torch.exp(0.5 * log_sigma_prior)
        epsilon = 1e-7

        L_KL = 0.5 * (
            log_sigma_prior - log_sigma_post +
            (sigma_post ** 2 + (mu_post - mu_prior) ** 2) / (sigma_prior ** 2 + epsilon) -
            1.0
        ).mean()
        losses['kl_divergence'] = L_KL

        # Alignment loss (InfoNCE)
        L_align = self.info_nce_loss(
            audio_features.mean(dim=1),
            visual_features.mean(dim=1),
        )
        losses['alignment'] = L_align

        # Flow matching loss
        L_flow = self.flow_loss(v_pred, v_target)
        losses['flow_matching'] = L_flow

        # Adversarial loss (if activated)
        adv_weight = self.get_adv_weight(epoch)
        if adv_weight > 0 and discriminator is not None:
            with torch.no_grad():
                real_score = discriminator(real_mel)
            fake_score = discriminator(pred_mel)
            L_adv = self.adv_loss.generator_loss(fake_score)
            losses['adversarial'] = L_adv
        else:
            losses['adversarial'] = torch.tensor(0.0, device=pred_mel.device)

        # Disentanglement loss
        L_dis = self.disentangle_loss(
            h_lip, h_au, h_rhythm, h_prosody,
            lip_targets, au_targets,
        )
        losses['disentanglement'] = L_dis

        # Total loss
        total_loss = (
            self.lambda_rec * L_rec +
            lambda_kl * L_KL +
            self.lambda_align * L_align +
            self.lambda_flow * L_flow +
            adv_weight * losses['adversarial'] +
            L_dis
        )
        losses['total'] = total_loss

        return losses
