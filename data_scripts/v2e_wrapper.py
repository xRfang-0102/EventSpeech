import numpy as np
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class V2EWrapper:
    """Wrapper for V2E simulator with controlled parameters.

    Converts video frames to neuromorphic events using the V2E simulator
    with fixed contrast threshold (0.15) and leak current (0).
    """

    def __init__(
        self,
        contrast_threshold: float = 0.15,
        leak_current: float = 0.0,
        shot_noise_rate: float = 0.0,
        refractory_period: float = 0.0,
    ):
        self.contrast_threshold = contrast_threshold
        self.leak_current = leak_current
        self.shot_noise_rate = shot_noise_rate
        self.refractory_period = refractory_period
        self._validate_params()

    def _validate_params(self):
        assert 0 < self.contrast_threshold < 1.0, \
            f"Contrast threshold must be in (0, 1), got {self.contrast_threshold}"
        assert self.leak_current >= 0, \
            f"Leak current must be non-negative, got {self.leak_current}"

    def convert_video_to_events(
        self,
        video_frames: np.ndarray,
        fps: float = 30.0,
        timestamp_offset: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Convert video frames to event stream.

        Args:
            video_frames: Video frames array [N, H, W, C] in uint8
            fps: Frames per second
            timestamp_offset: Offset for timestamps

        Returns:
            Dictionary with events (x, y, t, p) arrays
        """
        num_frames, height, width, channels = video_frames.shape

        # Convert to grayscale if needed
        if channels == 3:
            gray_frames = np.mean(video_frames, axis=-1).astype(np.float64)
        else:
            gray_frames = video_frames[..., 0].astype(np.float64)

        # Normalize to [0, 1]
        gray_frames = gray_frames / 255.0

        events_x = []
        events_y = []
        events_t = []
        events_p = []

        dt = 1.0 / fps

        for i in range(1, num_frames):
            curr_frame = gray_frames[i]
            prev_frame = gray_frames[i - 1]

            # Log intensity change
            log_curr = np.log(curr_frame + 1e-7)
            log_prev = np.log(prev_frame + 1e-7)
            delta = log_curr - log_prev

            # Positive events (brightness increase)
            pos_mask = delta > self.contrast_threshold
            # Negative events (brightness decrease)
            neg_mask = delta < -self.contrast_threshold

            pos_y, pos_x = np.where(pos_mask)
            neg_y, neg_x = np.where(neg_mask)

            timestamp = timestamp_offset + i * dt

            if len(pos_x) > 0:
                events_x.append(pos_x)
                events_y.append(pos_y)
                events_t.append(np.full(len(pos_x), timestamp))
                events_p.append(np.ones(len(pos_x)))

            if len(neg_x) > 0:
                events_x.append(neg_x)
                events_y.append(neg_y)
                events_t.append(np.full(len(neg_x), timestamp))
                events_p.append(-np.ones(len(neg_x)))

        if len(events_x) == 0:
            return {
                'x': np.array([], dtype=np.int32),
                'y': np.array([], dtype=np.int32),
                't': np.array([], dtype=np.float64),
                'p': np.array([], dtype=np.float32),
            }

        return {
            'x': np.concatenate(events_x).astype(np.int32),
            'y': np.concatenate(events_y).astype(np.int32),
            't': np.concatenate(events_t).astype(np.float64),
            'p': np.concatenate(events_p).astype(np.float32),
        }

    def events_to_voxel_grid(
        self,
        events: Dict[str, np.ndarray],
        height: int,
        width: int,
        num_bins: int,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> np.ndarray:
        """Convert events to voxel grid representation.

        Args:
            events: Dictionary with x, y, t, p arrays
            height: Grid height
            width: Grid width
            num_bins: Number of temporal bins
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)

        Returns:
            Voxel grid [num_bins, height, width]
        """
        if len(events['x']) == 0:
            return np.zeros((num_bins, height, width), dtype=np.float32)

        x = events['x']
        y = events['y']
        t = events['t']
        p = events['p']

        if start_time is None:
            start_time = t.min()
        if end_time is None:
            end_time = t.max()

        # Normalize timestamps to [0, num_bins-1]
        t_norm = (t - start_time) / (end_time - start_time + 1e-7) * (num_bins - 1)

        voxel_grid = np.zeros((num_bins, height, width), dtype=np.float32)

        # Bilinear interpolation across time bins
        t_floor = np.floor(t_norm).astype(np.int32)
        t_ceil = np.minimum(t_floor + 1, num_bins - 1)
        t_frac = t_norm - t_floor

        for i in range(len(x)):
            xi, yi = int(x[i]), int(y[i])
            if 0 <= xi < width and 0 <= yi < height:
                voxel_grid[t_floor[i], yi, xi] += p[i] * (1 - t_frac[i])
                voxel_grid[t_ceil[i], yi, xi] += p[i] * t_frac[i]

        return voxel_grid

    def get_params(self) -> Dict:
        return {
            'contrast_threshold': self.contrast_threshold,
            'leak_current': self.leak_current,
            'shot_noise_rate': self.shot_noise_rate,
            'refractory_period': self.refractory_period,
        }
