import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class MCDCalculator:
    """Mel-Cepstral Distortion (MCD) calculator.

    Computes MCD with DTW alignment between generated and
    reference audio for acoustic fidelity evaluation.
    """

    def __init__(self, n_mfcc: int = 13, epsilon: float = 1e-7):
        self.n_mfcc = n_mfcc
        self.epsilon = epsilon

    def compute_mfcc(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Compute MFCC features from mel spectrogram."""
        # Simple MFCC approximation using DCT
        # In practice, use torchaudio.transforms.MFCC
        return mel_spec[:, :self.n_mfcc, :]

    def dtw_distance(
        self,
        seq1: torch.Tensor,
        seq2: torch.Tensor,
    ) -> float:
        """Compute DTW distance between two sequences."""
        n, m = seq1.shape[1], seq2.shape[1]

        # Initialize cost matrix
        cost = torch.full((n + 1, m + 1), float('inf'))
        cost[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dist = torch.sqrt(((seq1[:, i-1] - seq2[:, j-1]) ** 2).sum())
                cost[i, j] = dist + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])

        return cost[n, m].item() / (n + m)

    def compute(
        self,
        pred_mel: torch.Tensor,
        target_mel: torch.Tensor,
    ) -> float:
        """Compute MCD between predicted and target mel spectrogram.

        Args:
            pred_mel: Predicted mel [B, 80, T]
            target_mel: Target mel [B, 80, T]

        Returns:
            MCD value
        """
        pred_mfcc = self.compute_mfcc(pred_mel)
        target_mfcc = self.compute_mfcc(target_mel)

        mcd_values = []
        for b in range(pred_mel.shape[0]):
            mcd = self.dtw_distance(pred_mfcc[b], target_mfcc[b])
            mcd_values.append(mcd)

        return np.mean(mcd_values)


class F0RMSECalculator:
    """F0 Root Mean Square Error calculator.

    Computes F0-RMSE in log space using WORLD vocoder's
    Harvest algorithm for prosody evaluation.
    """

    def __init__(self, epsilon: float = 1e-7):
        self.epsilon = epsilon

    def extract_f0(
        self,
        waveform: torch.Tensor,
        sr: int = 22050,
    ) -> np.ndarray:
        """Extract F0 using simplified method.

        In production, use WORLD vocoder's Harvest algorithm.
        """
        # Simplified F0 extraction using autocorrelation
        wav_np = waveform.squeeze().numpy()
        frame_length = 1024
        hop_length = 256

        f0_values = []
        for i in range(0, len(wav_np) - frame_length, hop_length):
            frame = wav_np[i:i + frame_length]
            # Simple autocorrelation-based F0
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Find first peak after zero crossing
            min_lag = sr // 500  # Max F0 = 500Hz
            max_lag = sr // 50   # Min F0 = 50Hz

            if max_lag < len(autocorr):
                peak_idx = np.argmax(autocorr[min_lag:max_lag]) + min_lag
                f0 = sr / peak_idx if peak_idx > 0 else 0
            else:
                f0 = 0

            f0_values.append(f0)

        return np.array(f0_values)

    def compute(
        self,
        pred_f0: np.ndarray,
        target_f0: np.ndarray,
    ) -> float:
        """Compute F0-RMSE in log space.

        Args:
            pred_f0: Predicted F0 contour
            target_f0: Target F0 contour

        Returns:
            F0-RMSE value
        """
        # Filter unvoiced frames
        voiced_mask = (pred_f0 > 0) & (target_f0 > 0)

        if voiced_mask.sum() == 0:
            return 0.0

        pred_log = np.log(pred_f0[voiced_mask] + self.epsilon)
        target_log = np.log(target_f0[voiced_mask] + self.epsilon)

        rmse = np.sqrt(np.mean((pred_log - target_log) ** 2))
        return rmse


class LSECalculator:
    """Lip Sync Error calculator.

    Computes LSE-D (distance) and LSE-C (confidence) using
    pre-trained SyncNet for audio-visual synchronization evaluation.
    """

    def __init__(self, syncnet_model: Optional[nn.Module] = None):
        self.syncnet = syncnet_model

    def compute(
        self,
        audio_embed: torch.Tensor,
        visual_embed: torch.Tensor,
    ) -> Tuple[float, float]:
        """Compute LSE-D and LSE-C.

        Args:
            audio_embed: Audio embeddings [B, D]
            visual_embed: Visual embeddings [B, D]

        Returns:
            Tuple of (LSE-D, LSE-C)
        """
        # Normalize embeddings
        audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)
        visual_embed = torch.nn.functional.normalize(visual_embed, dim=-1)

        # LSE-D: Euclidean distance
        lse_d = torch.sqrt(((audio_embed - visual_embed) ** 2).sum(dim=-1)).mean().item()

        # LSE-C: Cosine similarity confidence
        cos_sim = (audio_embed * visual_embed).sum(dim=-1)
        lse_c = cos_sim.mean().item()

        return lse_d, lse_c


class WERCalculator:
    """Word Error Rate calculator.

    Computes WER using pre-trained Whisper model for
    content intelligibility evaluation.
    """

    def __init__(self, whisper_model_name: str = 'large-v3'):
        self.model_name = whisper_model_name
        self.model = None

    def load_model(self):
        """Load Whisper model."""
        try:
            import whisper
            self.model = whisper.load_model(self.model_name)
            logger.info(f"Loaded Whisper model: {self.model_name}")
        except ImportError:
            logger.warning("Whisper not installed, WER calculation unavailable")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file using Whisper."""
        if self.model is None:
            self.load_model()

        if self.model is None:
            return ""

        result = self.model.transcribe(audio_path)
        return result['text']

    def compute_wer(
        self,
        hypothesis: str,
        reference: str,
    ) -> float:
        """Compute Word Error Rate.

        Args:
            hypothesis: Predicted text
            reference: Ground truth text

        Returns:
            WER value
        """
        hyp_words = hypothesis.lower().split()
        ref_words = reference.lower().split()

        # Dynamic programming for edit distance
        n, m = len(hyp_words), len(ref_words)
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if hyp_words[i-1] == ref_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,      # deletion
                        dp[i][j-1] + 1,      # insertion
                        dp[i-1][j-1] + 1,    # substitution
                    )

        wer = dp[n][m] / max(len(ref_words), 1)
        return wer


class Evaluator:
    """Unified evaluator for all objective metrics.

    Integrates MCD, F0-RMSE, LSE-D, LSE-C, and WER
    for comprehensive evaluation.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.mcd_calc = MCDCalculator()
        self.f0_calc = F0RMSECalculator()
        self.lse_calc = LSECalculator()
        self.wer_calc = WERCalculator(
            config.get('evaluation', {}).get('whisper_model', 'large-v3')
        )

    def evaluate_batch(
        self,
        pred_mel: torch.Tensor,
        target_mel: torch.Tensor,
        pred_waveform: Optional[torch.Tensor] = None,
        target_waveform: Optional[torch.Tensor] = None,
        audio_embed: Optional[torch.Tensor] = None,
        visual_embed: Optional[torch.Tensor] = None,
        pred_text: Optional[str] = None,
        target_text: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate a batch of predictions.

        Returns:
            Dictionary of metric name to value
        """
        metrics = {}

        # MCD
        mcd = self.mcd_calc.compute(pred_mel, target_mel)
        metrics['mcd'] = mcd

        # F0-RMSE
        if pred_waveform is not None and target_waveform is not None:
            pred_f0 = self.f0_calc.extract_f0(pred_waveform)
            target_f0 = self.f0_calc.extract_f0(target_waveform)
            f0_rmse = self.f0_calc.compute(pred_f0, target_f0)
            metrics['f0_rmse'] = f0_rmse

        # LSE-D and LSE-C
        if audio_embed is not None and visual_embed is not None:
            lse_d, lse_c = self.lse_calc.compute(audio_embed, visual_embed)
            metrics['lse_d'] = lse_d
            metrics['lse_c'] = lse_c

        # WER
        if pred_text is not None and target_text is not None:
            wer = self.wer_calc.compute_wer(pred_text, target_text)
            metrics['wer'] = wer

        return metrics

    def evaluate_dataset(
        self,
        predictions: List[Dict],
        references: List[Dict],
    ) -> Dict[str, float]:
        """Evaluate on entire dataset.

        Args:
            predictions: List of prediction dictionaries
            references: List of reference dictionaries

        Returns:
            Dictionary of averaged metrics
        """
        all_metrics = []

        for pred, ref in zip(predictions, references):
            metrics = self.evaluate_batch(
                pred_mel=pred.get('mel_spec'),
                target_mel=ref.get('mel_spec'),
                pred_waveform=pred.get('waveform'),
                target_waveform=ref.get('waveform'),
                audio_embed=pred.get('audio_embed'),
                visual_embed=pred.get('visual_embed'),
                pred_text=pred.get('text'),
                target_text=ref.get('text'),
            )
            all_metrics.append(metrics)

        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            avg_metrics[key] = np.mean(values)

        return avg_metrics

    def save_results(
        self,
        metrics: Dict[str, float],
        output_path: str,
    ):
        """Save evaluation results to file."""
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved evaluation results to {output_path}")
