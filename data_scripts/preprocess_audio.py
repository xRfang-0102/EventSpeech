import numpy as np
import torch
import torchaudio
import librosa
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Audio preprocessing pipeline for EventSpeech.

    Implements the strict preprocessing pipeline:
    1. 80Hz high-pass filtering
    2. Spectral subtraction denoising
    3. Resampling to 22050 Hz
    4. Loudness normalization to -23 LUFS (EBU R128)
    """

    def __init__(
        self,
        target_sr: int = 22050,
        highpass_freq: float = 80.0,
        target_lufs: float = -23.0,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ):
        self.target_sr = target_sr
        self.highpass_freq = highpass_freq
        self.target_lufs = target_lufs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )

    def preprocess(
        self,
        waveform: torch.Tensor,
        sr: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Full preprocessing pipeline.

        Args:
            waveform: Raw audio waveform [1, T] or [T]
            sr: Original sample rate

        Returns:
            Tuple of (mel_spec, prosody_features)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Step 1: High-pass filtering (80Hz cutoff)
        waveform = self._highpass_filter(waveform, sr)

        # Step 2: Spectral subtraction denoising
        waveform = self._spectral_subtraction(waveform, sr)

        # Step 3: Resample to target sample rate
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
            sr = self.target_sr

        # Step 4: Loudness normalization to -23 LUFS
        waveform = self._normalize_loudness(waveform, sr)

        # Extract mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-7)

        # Extract prosody features
        prosody = self._extract_prosody(waveform, sr)

        return mel_spec, prosody

    def _highpass_filter(
        self,
        waveform: torch.Tensor,
        sr: int,
    ) -> torch.Tensor:
        """Apply high-pass filter to remove low-frequency noise."""
        # Design Butterworth high-pass filter
        nyquist = sr / 2.0
        cutoff = self.highpass_freq / nyquist

        # Simple high-pass using FFT
        fft = torch.fft.rfft(waveform)
        freqs = torch.linspace(0, nyquist, fft.shape[-1])
        mask = (freqs >= self.highpass_freq).float()
        fft = fft * mask
        waveform = torch.fft.irfft(fft, n=waveform.shape[-1])

        return waveform

    def _spectral_subtraction(
        self,
        waveform: torch.Tensor,
        sr: int,
        noise_frames: int = 10,
        alpha: float = 2.0,
        beta: float = 0.01,
    ) -> torch.Tensor:
        """Apply spectral subtraction for denoising."""
        # Estimate noise from first few frames
        n_fft = self.n_fft
        hop_length = self.hop_length

        # Compute STFT
        stft = torch.stft(
            waveform.squeeze(0),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=self.win_length,
            return_complex=True,
        )

        magnitude = torch.abs(stft)
        phase = torch.angle(stft)

        # Estimate noise spectrum from first frames
        noise_estimate = magnitude[:, :noise_frames].mean(dim=1, keepdim=True)

        # Spectral subtraction
        magnitude_sub = magnitude - alpha * noise_estimate
        magnitude_sub = torch.maximum(
            magnitude_sub,
            beta * magnitude,
        )

        # Reconstruct
        stft_clean = magnitude_sub * torch.exp(1j * phase)
        waveform_clean = torch.istft(
            stft_clean,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=self.win_length,
            length=waveform.shape[-1],
        )

        return waveform_clean.unsqueeze(0)

    def _normalize_loudness(
        self,
        waveform: torch.Tensor,
        sr: int,
    ) -> torch.Tensor:
        """Normalize loudness to target LUFS using EBU R128."""
        # Simple loudness normalization
        # For proper LUFS calculation, we approximate with RMS
        rms = torch.sqrt(torch.mean(waveform ** 2))

        # Target RMS for -23 LUFS (approximate)
        target_rms = 10 ** ((self.target_lufs + 0.691) / 20.0)

        if rms > 1e-7:
            gain = target_rms / rms
            waveform = waveform * gain

        # Clip to prevent overflow
        waveform = torch.clamp(waveform, -1.0, 1.0)

        return waveform

    def _extract_prosody(
        self,
        waveform: torch.Tensor,
        sr: int,
    ) -> Dict[str, torch.Tensor]:
        """Extract pitch, energy, and duration features."""
        wav_np = waveform.squeeze(0).numpy()

        # Extract F0 using librosa
        f0, voiced_flag, _ = librosa.pyin(
            wav_np,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            hop_length=self.hop_length,
        )

        # Replace NaN with 0
        f0 = np.nan_to_num(f0, nan=0.0)

        # Extract energy
        energy = librosa.feature.rms(
            y=wav_np,
            hop_length=self.hop_length,
        )[0]

        # Ensure same length
        min_len = min(len(f0), len(energy))
        f0 = f0[:min_len]
        energy = energy[:min_len]

        # Log transform for pitch and energy
        pitch = torch.from_numpy(np.log(f0 + 1e-7)).float()
        energy = torch.from_numpy(np.log(energy + 1e-7)).float()

        return {
            'pitch': pitch,
            'energy': energy,
        }


def preprocess_audio_file(
    audio_path: str,
    output_path: Optional[str] = None,
    config: Optional[Dict] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Preprocess a single audio file.

    Args:
        audio_path: Path to input audio file
        output_path: Path to save processed features (optional)
        config: Configuration dictionary (optional)

    Returns:
        Tuple of (mel_spec, prosody_features)
    """
    if config is None:
        config = {}

    preprocessor = AudioPreprocessor(
        target_sr=config.get('sample_rate', 22050),
        highpass_freq=config.get('highpass_freq', 80.0),
        target_lufs=config.get('target_lufs', -23.0),
        n_fft=config.get('n_fft', 1024),
        hop_length=config.get('hop_length', 256),
        win_length=config.get('win_length', 1024),
        n_mels=config.get('n_mels', 80),
        f_min=config.get('f_min', 0.0),
        f_max=config.get('f_max', 8000.0),
    )

    # Load audio
    waveform, sr = torchaudio.load(audio_path)

    # Preprocess
    mel_spec, prosody = preprocessor.preprocess(waveform, sr)

    # Save if output path provided
    if output_path is not None:
        torch.save({
            'mel_spec': mel_spec,
            'prosody': prosody,
        }, output_path)

    return mel_spec, prosody
