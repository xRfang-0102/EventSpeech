import os
import sys
import yaml
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, List
import json

from models import EventEncoder, AudioEncoder, CrossModalAlignment
from models import PriorEncoder, PosteriorEncoder, KnowledgeBridge
from models import CFMDecoder
from utils.evaluator import Evaluator
from utils.logger import ExperimentLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventSpeechInference:
    """Inference module for EventSpeech.

    Supports:
    - Single sample inference
    - Batch inference
    - Evaluation mode with all metrics
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: Dict,
        device: torch.device = None,
    ):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # Setup evaluator
        self.evaluator = Evaluator(config)

        logger.info(f"Loaded model from {checkpoint_path}")

    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load model from checkpoint."""
        from train import EventSpeechModel

        model = EventSpeechModel(self.config)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")

        return model.to(self.device)

    @torch.no_grad()
    def generate_mel(
        self,
        events: torch.Tensor,
        text: torch.Tensor,
        emotion_id: torch.Tensor,
        speaker_id: torch.Tensor,
        num_steps: int = 20,
        solver: str = 'euler',
    ) -> torch.Tensor:
        """Generate mel spectrogram from events and text.

        Args:
            events: Event voxels [B, T_v, 3, H, W]
            text: Text sequence [B, T_text]
            emotion_id: Emotion class ID [B]
            speaker_id: Speaker class ID [B]
            num_steps: Number of ODE solver steps
            solver: ODE solver type ('euler' or 'rk4')

        Returns:
            Generated mel spectrogram [B, 80, T_a]
        """
        B = events.shape[0]
        T_a = events.shape[1] + 1  # T_v = T_a - 1

        # Get embeddings
        emotion_embed = self.model.emotion_embedding(emotion_id.to(self.device))
        speaker_embed = self.model.speaker_embedding(speaker_id.to(self.device))

        # Encode events
        F_v = self.model.event_encoder(
            events.to(self.device), emotion_embed, speaker_embed,
        )

        # Encode text
        F_t = self.model.text_encoder(text.to(self.device))

        # Interpolate text to match audio length
        if F_t.shape[1] != T_a:
            F_t = torch.nn.functional.interpolate(
                F_t.transpose(1, 2), size=T_a, mode='linear', align_corners=False,
            ).transpose(1, 2)

        # Prior encoder
        mu_prior, _ = self.model.prior_encoder(
            F_t, emotion_id.to(self.device), speaker_id.to(self.device),
        )

        # Knowledge bridge
        z = mu_prior  # Use prior mean for inference
        hidden = self.model.knowledge_bridge(z)

        # Generate mel spectrogram
        if solver == 'rk4':
            mel_gen = self.model.cfm_decoder.sample_rk4(hidden, num_steps=num_steps)
        else:
            mel_gen = self.model.cfm_decoder.sample_euler(hidden, num_steps=num_steps)

        return mel_gen

    @torch.no_grad()
    def inference_single(
        self,
        events: torch.Tensor,
        text: torch.Tensor,
        emotion_id: torch.Tensor,
        speaker_id: torch.Tensor,
        num_steps: int = 20,
        solver: str = 'euler',
    ) -> Dict[str, torch.Tensor]:
        """Run inference on a single sample.

        Returns:
            Dictionary with generated mel and features
        """
        mel_gen = self.generate_mel(
            events, text, emotion_id, speaker_id,
            num_steps=num_steps, solver=solver,
        )

        return {
            'mel_spec': mel_gen,
        }

    @torch.no_grad()
    def inference_batch(
        self,
        batch: Dict[str, torch.Tensor],
        num_steps: int = 20,
        solver: str = 'euler',
    ) -> Dict[str, torch.Tensor]:
        """Run inference on a batch.

        Args:
            batch: Dictionary with input tensors
            num_steps: Number of ODE solver steps
            solver: ODE solver type

        Returns:
            Dictionary with batch outputs
        """
        events = batch['events']
        text = batch['text']
        emotion_id = batch['emotion_id']
        speaker_id = batch['speaker_id']

        return self.inference_single(
            events, text, emotion_id, speaker_id,
            num_steps=num_steps, solver=solver,
        )

    def evaluate(
        self,
        test_loader,
        num_steps: int = 20,
        solver: str = 'euler',
        output_dir: str = 'results',
    ) -> Dict[str, float]:
        """Evaluate model on test set.

        Args:
            test_loader: Test data loader
            num_steps: Number of ODE solver steps
            solver: ODE solver type
            output_dir: Directory to save results

        Returns:
            Dictionary of evaluation metrics
        """
        os.makedirs(output_dir, exist_ok=True)

        all_predictions = []
        all_references = []

        for batch_idx, batch in enumerate(test_loader):
            # Generate predictions
            outputs = self.inference_batch(batch, num_steps=num_steps, solver=solver)

            # Store predictions and references
            for i in range(batch['mel_spec'].shape[0]):
                all_predictions.append({
                    'mel_spec': outputs['mel_spec'][i].cpu(),
                })
                all_references.append({
                    'mel_spec': batch['mel_spec'][i].cpu(),
                })

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches")

        # Evaluate
        metrics = self.evaluator.evaluate_dataset(all_predictions, all_references)

        # Save results
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        self.evaluator.save_results(metrics, results_path)

        # Print results
        logger.info("=" * 50)
        logger.info("Evaluation Results:")
        logger.info("=" * 50)
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.6f}")
        logger.info("=" * 50)

        return metrics

    def generate_and_save(
        self,
        test_loader,
        output_dir: str = 'generated',
        num_steps: int = 20,
        solver: str = 'euler',
    ):
        """Generate and save mel spectrograms.

        Args:
            test_loader: Test data loader
            output_dir: Directory to save generated mels
            num_steps: Number of ODE solver steps
            solver: ODE solver type
        """
        os.makedirs(output_dir, exist_ok=True)

        for batch_idx, batch in enumerate(test_loader):
            outputs = self.inference_batch(batch, num_steps=num_steps, solver=solver)

            # Save each sample
            for i in range(outputs['mel_spec'].shape[0]):
                sample_idx = batch_idx * test_loader.batch_size + i
                save_path = os.path.join(output_dir, f'sample_{sample_idx:06d}.pt')
                torch.save(outputs['mel_spec'][i].cpu(), save_path)

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Generated {batch_idx + 1}/{len(test_loader)} batches")


def main():
    parser = argparse.ArgumentParser(description='EventSpeech Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/eventspeech_a100.yaml',
                        help='Path to config file')
    parser.add_argument('--mode', type=str, default='eval',
                        choices=['eval', 'generate', 'single'],
                        help='Inference mode')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Path to output directory')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='Number of ODE solver steps')
    parser.add_argument('--solver', type=str, default='euler',
                        choices=['euler', 'rk4'],
                        help='ODE solver type')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize inference module
    inference = EventSpeechInference(
        checkpoint_path=args.checkpoint,
        config=config,
        device=device,
    )

    # Create data loader
    from datasets import EVTSPKDataset
    from torch.utils.data import DataLoader

    test_dataset = EVTSPKDataset(
        data_dir=args.data_dir,
        split='test',
        config=config,
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Run inference
    if args.mode == 'eval':
        metrics = inference.evaluate(
            test_loader=test_loader,
            num_steps=args.num_steps,
            solver=args.solver,
            output_dir=args.output_dir,
        )
        logger.info("Evaluation complete!")

    elif args.mode == 'generate':
        inference.generate_and_save(
            test_loader=test_loader,
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            solver=args.solver,
        )
        logger.info("Generation complete!")

    elif args.mode == 'single':
        # Example single inference
        sample = next(iter(test_loader))
        outputs = inference.inference_batch(
            sample, num_steps=args.num_steps, solver=args.solver,
        )
        logger.info(f"Generated mel shape: {outputs['mel_spec'].shape}")


if __name__ == '__main__':
    main()
