import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Dict, Optional, Tuple, List
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class EVTSPKDataset(Dataset):
    """Multi-modal dataset for EVT-SPK benchmark.

    Loads neuromorphic events, audio features, text sequences,
    emotion IDs, and speaker embeddings for training/validation/testing.
    Supports data augmentation for training and clean evaluation.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        config: Optional[Dict] = None,
        augment: bool = False,
        max_text_length: int = 200,
        max_audio_frames: int = 512,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.config = config or {}
        self.augment = augment and (split == 'train')
        self.max_text_length = max_text_length
        self.max_audio_frames = max_audio_frames

        # Load manifest
        manifest_path = self.data_dir / f'{split}_manifest.jsonl'
        self.samples = self._load_manifest(manifest_path)

        # Augmentation parameters
        if self.augment:
            self.temporal_jitter_ms = self.config.get('augmentation', {}).get('temporal_jitter_ms', 5)
            self.event_dropout_rate = self.config.get('augmentation', {}).get('event_dropout_rate', 0.1)
            self.freq_mask_param = self.config.get('augmentation', {}).get('freq_mask_param', 27)
            self.time_mask_param = self.config.get('augmentation', {}).get('time_mask_param', 40)
            self.pitch_shift_range = self.config.get('augmentation', {}).get('pitch_shift_semitones', 2)
            self.time_stretch_range = self.config.get('augmentation', {}).get('time_stretch_range', [0.9, 1.1])

        logger.info(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_manifest(self, manifest_path: Path) -> List[Dict]:
        """Load dataset manifest file."""
        samples = []
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    samples.append(sample)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Returns:
            Dictionary containing:
            - events: Event voxel grid [T_v, 3, H, W]
            - mel_spec: Mel spectrogram [1, 80, T_a]
            - pitch: Pitch contour [T_a]
            - energy: Energy contour [T_a]
            - text: Text sequence [T_text]
            - emotion_id: Emotion class ID
            - speaker_id: Speaker class ID
            - text_length: Length of text sequence
            - audio_length: Length of audio sequence
        """
        sample_info = self.samples[idx]
        sample_id = sample_info['id']

        # Load event voxels
        events = self._load_events(sample_id)
        events = torch.from_numpy(events).float()

        # Load audio features
        mel_spec, pitch, energy = self._load_audio_features(sample_id)

        # Load text sequence
        text = self._load_text(sample_id)
        text = torch.from_numpy(text).long()

        # Load emotion and speaker IDs
        emotion_id = torch.tensor(sample_info.get('emotion_id', 0)).long()
        speaker_id = torch.tensor(sample_info.get('speaker_id', 0)).long()

        # Apply augmentation if training
        if self.augment:
            events = self._augment_events(events)
            mel_spec = self._augment_mel(mel_spec)

        # Record lengths before padding
        text_length = torch.tensor(len(text)).long()
        audio_length = torch.tensor(mel_spec.shape[-1]).long()

        # Pad sequences
        events = self._pad_events(events)
        mel_spec = self._pad_mel(mel_spec)
        text = self._pad_text(text)
        pitch = self._pad_prosody(pitch)
        energy = self._pad_prosody(energy)

        return {
            'events': events,
            'mel_spec': mel_spec,
            'pitch': pitch,
            'energy': energy,
            'text': text,
            'emotion_id': emotion_id,
            'speaker_id': speaker_id,
            'text_length': text_length,
            'audio_length': audio_length,
        }

    def _load_events(self, sample_id: str) -> np.ndarray:
        """Load pre-computed event voxels."""
        event_path = self.data_dir / 'events' / f'{sample_id}.npy'
        if event_path.exists():
            return np.load(event_path)
        return np.zeros((1, 3, 260, 346), dtype=np.float32)

    def _load_audio_features(
        self,
        sample_id: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load mel spectrogram and prosody features."""
        audio_path = self.data_dir / 'audio' / f'{sample_id}.pt'
        if audio_path.exists():
            data = torch.load(audio_path, weights_only=True)
            mel_spec = data['mel_spec']
            pitch = data['prosody']['pitch']
            energy = data['prosody']['energy']
        else:
            mel_spec = torch.zeros(1, 80, 1)
            pitch = torch.zeros(1)
            energy = torch.zeros(1)
        return mel_spec, pitch, energy

    def _load_text(self, sample_id: str) -> np.ndarray:
        """Load phoneme/text sequence."""
        text_path = self.data_dir / 'text' / f'{sample_id}.npy'
        if text_path.exists():
            return np.load(text_path)
        return np.zeros(1, dtype=np.int64)

    def _pad_events(self, events: torch.Tensor) -> torch.Tensor:
        """Pad event voxels to max_audio_frames - 1."""
        target_len = self.max_audio_frames - 1
        current_len = events.shape[0]
        if current_len >= target_len:
            return events[:target_len]
        pad_size = target_len - current_len
        return torch.nn.functional.pad(events, (0, 0, 0, 0, 0, 0, 0, pad_size))

    def _pad_mel(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Pad mel spectrogram to max_audio_frames."""
        target_len = self.max_audio_frames
        current_len = mel_spec.shape[-1]
        if current_len >= target_len:
            return mel_spec[..., :target_len]
        pad_size = target_len - current_len
        return torch.nn.functional.pad(mel_spec, (0, pad_size))

    def _pad_text(self, text: torch.Tensor) -> torch.Tensor:
        """Pad text sequence to max_text_length."""
        target_len = self.max_text_length
        current_len = len(text)
        if current_len >= target_len:
            return text[:target_len]
        pad_size = target_len - current_len
        return torch.nn.functional.pad(text, (0, pad_size))

    def _pad_prosody(self, prosody: torch.Tensor) -> torch.Tensor:
        """Pad prosody features to max_audio_frames."""
        target_len = self.max_audio_frames
        current_len = len(prosody)
        if current_len >= target_len:
            return prosody[:target_len]
        pad_size = target_len - current_len
        return torch.nn.functional.pad(prosody, (0, pad_size))

    def _augment_events(self, events: torch.Tensor) -> torch.Tensor:
        """Apply event augmentation.

        Includes temporal jittering and event dropout for
        sim-to-real domain transfer robustness.
        """
        # Temporal jittering (±5ms)
        if self.temporal_jitter_ms > 0:
            jitter_frames = int(self.temporal_jitter_ms / 20.0)  # 20ms per frame
            if jitter_frames > 0:
                shift = np.random.randint(-jitter_frames, jitter_frames + 1)
                if shift != 0:
                    events = torch.roll(events, shifts=shift, dims=0)

        # Event dropout (10% random masking)
        if self.event_dropout_rate > 0:
            mask = torch.rand(events.shape[0]) > self.event_dropout_rate
            events = events * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()

        return events

    def _augment_mel(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to mel spectrogram."""
        mel = mel_spec.clone()
        _, n_mels, n_frames = mel.shape

        # Frequency masking
        if self.freq_mask_param > 0:
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, max(1, n_mels - f))
            mel[:, f0:f0 + f, :] = 0

        # Time masking
        if self.time_mask_param > 0:
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, max(1, n_frames - t))
            mel[:, :, t0:t0 + t] = 0

        return mel


def create_dataloaders(
    config: Dict,
    data_dir: str,
    world_size: int = 1,
    rank: int = 0,
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]]:
    """Create training and validation dataloaders.

    Args:
        config: Configuration dictionary
        data_dir: Path to data directory
        world_size: Number of distributed processes
        rank: Current process rank

    Returns:
        Tuple of (train_loader, val_loader, train_sampler, val_sampler)
    """
    batch_size = config.get('training', {}).get('batch_size', 32)
    num_workers = config.get('training', {}).get('num_workers', 4)

    # Training dataset with augmentation
    train_dataset = EVTSPKDataset(
        data_dir=data_dir,
        split='train',
        config=config,
        augment=True,
    )

    # Validation dataset without augmentation
    val_dataset = EVTSPKDataset(
        data_dir=data_dir,
        split='val',
        config=config,
        augment=False,
    )

    # Create samplers for DDP
    train_sampler = None
    val_sampler = None

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, train_sampler, val_sampler
