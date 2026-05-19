import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import random
import logging
from pathlib import Path
from typing import Dict, Optional
from torch.cuda.amp import autocast, GradScaler

from models import EventEncoder, AudioEncoder, CrossModalAlignment
from models import PriorEncoder, PosteriorEncoder, KnowledgeBridge
from models import CFMDecoder
from datasets import EVTSPKDataset, create_dataloaders
from losses import MultiTaskLoss
from utils.logger import ExperimentLogger
from utils.ddp_init import DistributedManager, get_device
from utils.evaluator import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class EventSpeechModel(nn.Module):
    """Main EventSpeech model combining all components."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        model_cfg = config.get('model', {})

        # Embedding layers
        num_emotions = 8
        num_speakers = 100
        emotion_dim = model_cfg.get('event_encoder', {}).get('mhfe_dim', 256)
        speaker_dim = emotion_dim

        self.emotion_embedding = nn.Embedding(num_emotions, emotion_dim)
        self.speaker_embedding = nn.Embedding(num_speakers, speaker_dim)

        # Event Encoder
        self.event_encoder = EventEncoder(
            spatial_channels=model_cfg.get('event_encoder', {}).get('spatial_channels', [64, 128, 256, 512, 512]),
            bigru_hidden=model_cfg.get('event_encoder', {}).get('bigru_hidden', 512),
            bigru_layers=model_cfg.get('event_encoder', {}).get('bigru_layers', 1),
            mhfe_dim=model_cfg.get('event_encoder', {}).get('mhfe_dim', 256),
            output_dim=model_cfg.get('event_encoder', {}).get('output_dim', 256),
            emotion_dim=emotion_dim,
            speaker_dim=speaker_dim,
        )

        # Audio Encoder
        self.audio_encoder = AudioEncoder(
            mel_channels=model_cfg.get('audio_encoder', {}).get('mel_channels', 80),
            pitch_proj_dim=model_cfg.get('audio_encoder', {}).get('pitch_proj_dim', 88),
            energy_proj_dim=model_cfg.get('audio_encoder', {}).get('energy_proj_dim', 88),
            hidden_dim=model_cfg.get('audio_encoder', {}).get('hidden_dim', 256),
            mamba_layers=model_cfg.get('audio_encoder', {}).get('mamba_layers', 3),
            mamba_d_state=model_cfg.get('audio_encoder', {}).get('mamba_d_state', 16),
            mamba_expand=model_cfg.get('audio_encoder', {}).get('mamba_expand', 2),
            wavelet_levels=model_cfg.get('audio_encoder', {}).get('wavelet_levels', 3),
            output_dim=model_cfg.get('audio_encoder', {}).get('output_dim', 256),
            speaker_dim=speaker_dim,
        )

        # Cross-Modal Alignment
        self.alignment = CrossModalAlignment(
            hidden_dim=model_cfg.get('alignment', {}).get('hidden_dim', 256),
            num_heads=model_cfg.get('alignment', {}).get('num_heads', 4),
        )

        # Text Encoder
        self.text_encoder = nn.Sequential(
            nn.Embedding(256, 256),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True),
                num_layers=4,
            ),
            nn.Linear(256, 256),
        )

        # VITS Modules
        self.prior_encoder = PriorEncoder(
            hidden_dim=model_cfg.get('vits', {}).get('prior_encoder', {}).get('hidden_dim', 256),
            latent_dim=model_cfg.get('vits', {}).get('prior_encoder', {}).get('latent_dim', 128),
            num_emotions=num_emotions,
            num_speakers=num_speakers,
            emotion_dim=emotion_dim,
            speaker_dim=speaker_dim,
        )

        self.posterior_encoder = PosteriorEncoder(
            mel_channels=model_cfg.get('audio_encoder', {}).get('mel_channels', 80),
            hidden_dim=model_cfg.get('vits', {}).get('posterior_encoder', {}).get('hidden_dim', 256),
            latent_dim=model_cfg.get('vits', {}).get('posterior_encoder', {}).get('latent_dim', 128),
        )

        self.knowledge_bridge = KnowledgeBridge(
            in_channels=model_cfg.get('vits', {}).get('knowledge_bridge', {}).get('in_channels', 128),
            out_channels=model_cfg.get('vits', {}).get('knowledge_bridge', {}).get('out_channels', 256),
            num_layers=model_cfg.get('vits', {}).get('knowledge_bridge', {}).get('num_layers', 3),
        )

        # CFM Decoder
        self.cfm_decoder = CFMDecoder(
            mel_channels=model_cfg.get('cfm_decoder', {}).get('mel_channels', 80),
            hidden_dim=model_cfg.get('cfm_decoder', {}).get('hidden_dim', 256),
            time_embed_dim=model_cfg.get('cfm_decoder', {}).get('time_embed_dim', 128),
            num_dilation_blocks=model_cfg.get('cfm_decoder', {}).get('num_dilation_blocks', 20),
            dilation_schedule=model_cfg.get('cfm_decoder', {}).get('dilation_schedule', [1, 2, 4, 8]),
            kernel_size=model_cfg.get('cfm_decoder', {}).get('kernel_size', 3),
        )

    def forward(
        self,
        events: torch.Tensor,
        mel_spec: torch.Tensor,
        pitch: torch.Tensor,
        energy: torch.Tensor,
        text: torch.Tensor,
        emotion_id: torch.Tensor,
        speaker_id: torch.Tensor,
        is_training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            events: Event voxels [B, T_v, 3, H, W]
            mel_spec: Mel spectrogram [B, 80, T_a]
            pitch: Pitch contour [B, T_a]
            energy: Energy contour [B, T_a]
            text: Text sequence [B, T_text]
            emotion_id: Emotion class ID [B]
            speaker_id: Speaker class ID [B]
            is_training: Whether in training mode

        Returns:
            Dictionary with model outputs
        """
        B = events.shape[0]
        T_a = mel_spec.shape[2]

        # Get embeddings
        emotion_embed = self.emotion_embedding(emotion_id)
        speaker_embed = self.speaker_embedding(speaker_id)

        # Event Encoder
        F_v = self.event_encoder(events, emotion_embed, speaker_embed)  # [B, T_v, 256]

        # Audio Encoder
        F_a = self.audio_encoder(mel_spec, pitch, energy, speaker_embed)  # [B, T_a, 256]

        # Cross-Modal Alignment
        F_align = self.alignment(F_v, F_a)  # [B, T_a, 256]

        # Text Encoder
        F_t = self.text_encoder(text)  # [B, T_text, 256]

        # Interpolate text features to match audio length
        if F_t.shape[1] != T_a:
            F_t = torch.nn.functional.interpolate(
                F_t.transpose(1, 2), size=T_a, mode='linear', align_corners=False,
            ).transpose(1, 2)

        # VITS: Prior and Posterior
        mu_prior, log_sigma_prior = self.prior_encoder(F_t, emotion_id, speaker_id)

        outputs = {
            'F_v': F_v,
            'F_a': F_a,
            'F_align': F_align,
            'mu_prior': mu_prior,
            'log_sigma_prior': log_sigma_prior,
        }

        if is_training:
            # Posterior flow (training only)
            mu_post, log_sigma_post = self.posterior_encoder(mel_spec)
            z_post = self.posterior_encoder.sample(mu_post, log_sigma_post)

            outputs['mu_post'] = mu_post
            outputs['log_sigma_post'] = log_sigma_post
            outputs['z_post'] = z_post

            # Flow matching target
            t = torch.rand(B, 1, device=mel_spec.device)
            noise = torch.randn_like(mel_spec)
            x_t = t.unsqueeze(-1) * mel_spec + (1 - t.unsqueeze(-1)) * noise
            v_target = mel_spec - noise

            outputs['x_t'] = x_t
            outputs['t'] = t
            outputs['v_target'] = v_target

            # CFM Decoder prediction
            hidden = self.knowledge_bridge(z_post)
            v_pred = self.cfm_decoder(x_t, t, hidden)
            outputs['v_pred'] = v_pred

        else:
            # Inference flow
            z_prior = mu_prior
            hidden = self.knowledge_bridge(z_prior)
            outputs['hidden'] = hidden

        return outputs


def train_one_epoch(
    model: EventSpeechModel,
    train_loader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    global_step: int,
    experiment_logger: ExperimentLogger,
    config: Dict,
) -> int:
    """Train for one epoch."""
    model.train()
    total_losses = {}

    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        events = batch['events'].to(device)
        mel_spec = batch['mel_spec'].to(device)
        pitch = batch['pitch'].to(device)
        energy = batch['energy'].to(device)
        text = batch['text'].to(device)
        emotion_id = batch['emotion_id'].to(device)
        speaker_id = batch['speaker_id'].to(device)

        optimizer.zero_grad()

        with autocast(enabled=config.get('training', {}).get('fp16', True)):
            # Forward pass
            outputs = model(
                events=events,
                mel_spec=mel_spec,
                pitch=pitch,
                energy=energy,
                text=text,
                emotion_id=emotion_id,
                speaker_id=speaker_id,
                is_training=True,
            )

            # Compute losses
            losses = criterion(
                pred_mel=None,
                target_mel=mel_spec,
                v_pred=outputs['v_pred'],
                v_target=outputs['v_target'],
                mu_post=outputs['mu_post'],
                log_sigma_post=outputs['log_sigma_post'],
                mu_prior=outputs['mu_prior'],
                log_sigma_prior=outputs['log_sigma_prior'],
                audio_features=outputs['F_a'],
                visual_features=outputs['F_v'],
                h_lip=torch.zeros_like(outputs['F_v']),
                h_au=torch.zeros_like(outputs['F_v']),
                h_rhythm=torch.zeros_like(outputs['F_v']),
                h_prosody=torch.zeros_like(outputs['F_v']),
                epoch=epoch,
            )

        # Backward pass with gradient scaling
        scaler.scale(losses['total']).backward()

        # Gradient clipping
        max_grad_norm = config.get('training', {}).get('max_grad_norm', 1.0)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step += 1

        # Accumulate losses
        for key, value in losses.items():
            if key not in total_losses:
                total_losses[key] = 0.0
            total_losses[key] += value.item()

        # Log metrics
        if global_step % config.get('logging', {}).get('log_interval', 50) == 0:
            experiment_logger.log_loss(losses, global_step, prefix='train')

        if batch_idx % 100 == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Loss: {losses['total'].item():.4f}"
            )

    # Average losses
    for key in total_losses:
        total_losses[key] /= len(train_loader)

    return global_step


def validate(
    model: EventSpeechModel,
    val_loader,
    criterion: MultiTaskLoss,
    evaluator: Evaluator,
    device: torch.device,
    epoch: int,
    config: Dict,
) -> Dict[str, float]:
    """Validate model on validation set."""
    model.eval()
    total_losses = {}
    all_metrics = []

    with torch.no_grad():
        for batch in val_loader:
            events = batch['events'].to(device)
            mel_spec = batch['mel_spec'].to(device)
            pitch = batch['pitch'].to(device)
            energy = batch['energy'].to(device)
            text = batch['text'].to(device)
            emotion_id = batch['emotion_id'].to(device)
            speaker_id = batch['speaker_id'].to(device)

            outputs = model(
                events=events,
                mel_spec=mel_spec,
                pitch=pitch,
                energy=energy,
                text=text,
                emotion_id=emotion_id,
                speaker_id=speaker_id,
                is_training=False,
            )

            # Generate mel spectrogram
            hidden = outputs['hidden']
            pred_mel = model.cfm_decoder.sample_euler(hidden, num_steps=20)

            # Compute metrics
            metrics = evaluator.evaluate_batch(
                pred_mel=pred_mel,
                target_mel=mel_spec,
            )
            all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)

    return avg_metrics


def main(rank: int = 0, world_size: int = 1, config: Dict = None):
    """Main training function."""
    if config is None:
        config = {}

    # Load config
    config_path = config.get('config_path', 'configs/eventspeech_a100.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)

    # Setup distributed training
    if world_size > 1:
        manager = DistributedManager(rank, world_size, config)
        device = manager.device
        is_main = manager.is_main
    else:
        device = get_device(rank)
        is_main = True

    # Initialize model
    model = EventSpeechModel(config)
    model = model.to(device)

    if world_size > 1:
        model = manager.wrap_model(model)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('training', {}).get('learning_rate', 2e-4),
        betas=(
            config.get('training', {}).get('beta1', 0.8),
            config.get('training', {}).get('beta2', 0.99),
        ),
        weight_decay=config.get('training', {}).get('weight_decay', 0.01),
    )

    # Setup scheduler
    total_steps = config.get('training', {}).get('num_epochs', 200) * 1000  # Approximate
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.get('training', {}).get('learning_rate', 2e-4),
        total_steps=total_steps,
        pct_start=config.get('training', {}).get('warmup_steps', 10000) / total_steps,
    )

    # Setup loss
    criterion = MultiTaskLoss(config)

    # Setup gradient scaler for mixed precision
    scaler = GradScaler(enabled=config.get('training', {}).get('fp16', True))

    # Setup data loaders
    data_dir = config.get('data_dir', 'data')
    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
        config, data_dir, world_size, rank,
    )

    # Setup logger
    if is_main:
        experiment_logger = ExperimentLogger(
            config=config,
            project_name=config.get('logging', {}).get('project', 'eventspeech'),
            checkpoint_dir=config.get('checkpoint', {}).get('dir', 'checkpoints'),
        )
    else:
        experiment_logger = None

    # Setup evaluator
    evaluator = Evaluator(config)

    # Training loop
    num_epochs = config.get('training', {}).get('num_epochs', 200)
    global_step = 0

    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch}")

        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            global_step=global_step,
            experiment_logger=experiment_logger,
            config=config,
        )

        # Validate
        if is_main and (epoch + 1) % config.get('training', {}).get('validation_interval', 5) == 0:
            val_metrics = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                evaluator=evaluator,
                device=device,
                epoch=epoch,
                config=config,
            )

            # Log validation metrics
            if experiment_logger:
                experiment_logger.log_metrics(val_metrics, global_step, prefix='val')

            # Check for best model (LSE-C metric)
            lse_c = val_metrics.get('lse_c', float('-inf'))
            if experiment_logger:
                is_best = experiment_logger.update_best_model(lse_c, epoch, 'lse_c')
                if is_best:
                    experiment_logger.save_checkpoint(
                        model=model.module if hasattr(model, 'module') else model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step=global_step,
                        metrics=val_metrics,
                        is_best=True,
                    )

        # Save periodic checkpoint
        if is_main and (epoch + 1) % config.get('training', {}).get('save_interval', 10) == 0:
            if experiment_logger:
                experiment_logger.save_checkpoint(
                    model=model.module if hasattr(model, 'module') else model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=global_step,
                    metrics={},
                    is_best=False,
                )

    # Cleanup
    if is_main and experiment_logger:
        experiment_logger.finish()

    if world_size > 1:
        manager.__exit__(None, None, None)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='EventSpeech Training')
    parser.add_argument('--config', type=str, default='configs/eventspeech_a100.yaml',
                        help='Path to config file')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of GPUs for distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    args = parser.parse_args()

    config = {'config_path': args.config}

    if args.world_size > 1:
        import torch.multiprocessing as mp
        mp.spawn(main, args=(args.world_size, config), nprocs=args.world_size, join=True)
    else:
        main(rank=0, world_size=1, config=config)
