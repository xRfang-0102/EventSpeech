import numpy as np
import torch
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class EventVoxelizer:
    """Voxelizes event streams into fixed-dimensional grids.

    Implements 20ms temporal binning with bilinear interpolation
    and polarity-based accumulation for acoustic frame alignment.
    """

    def __init__(
        self,
        height: int = 260,
        width: int = 346,
        time_bin_ms: float = 20.0,
        num_polarities: int = 3,
        normalize: bool = True,
    ):
        self.height = height
        self.width = width
        self.time_bin_ms = time_bin_ms
        self.time_bin_sec = time_bin_ms / 1000.0
        self.num_polarities = num_polarities
        self.normalize = normalize

    def voxelize(
        self,
        events: Dict[str, np.ndarray],
        num_bins: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> np.ndarray:
        """Voxelize events into a 3D grid.

        Args:
            events: Dictionary with x, y, t, p arrays
            num_bins: Number of time bins (if None, auto-calculate)
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            Voxel grid [num_bins, num_polarities, height, width]
        """
        if len(events['x']) == 0:
            if num_bins is None:
                num_bins = 1
            return np.zeros(
                (num_bins, self.num_polarities, self.height, self.width),
                dtype=np.float32
            )

        x = events['x'].astype(np.float64)
        y = events['y'].astype(np.float64)
        t = events['t'].astype(np.float64)
        p = events['p'].astype(np.float32)

        if start_time is None:
            start_time = t.min()
        if end_time is None:
            end_time = t.max()

        if num_bins is None:
            duration = end_time - start_time
            num_bins = max(1, int(np.ceil(duration / self.time_bin_sec)))

        voxel_grid = np.zeros(
            (num_bins, self.num_polarities, self.height, self.width),
            dtype=np.float32
        )

        # Assign polarity indices: -1 -> 0, +1 -> 1, 0 -> 2
        p_idx = np.clip((p + 1).astype(np.int32), 0, self.num_polarities - 1)

        # Calculate time bin indices with bilinear interpolation
        t_norm = (t - start_time) / (end_time - start_time + 1e-7) * (num_bins - 1)
        t_floor = np.floor(t_norm).astype(np.int32)
        t_ceil = np.minimum(t_floor + 1, num_bins - 1)
        t_frac = t_norm - t_floor

        # Spatial bounds check
        valid_mask = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height) & \
                     (t_floor >= 0) & (t_floor < num_bins)

        x_valid = x[valid_mask].astype(np.int32)
        y_valid = y[valid_mask].astype(np.int32)
        p_valid = p_idx[valid_mask]
        t_floor_valid = t_floor[valid_mask]
        t_ceil_valid = t_ceil[valid_mask]
        t_frac_valid = t_frac[valid_mask]

        # Accumulate events with bilinear temporal interpolation
        for i in range(len(x_valid)):
            voxel_grid[t_floor_valid[i], p_valid[i], y_valid[i], x_valid[i]] += \
                (1 - t_frac_valid[i])
            voxel_grid[t_ceil_valid[i], p_valid[i], y_valid[i], x_valid[i]] += \
                t_frac_valid[i]

        if self.normalize:
            # Normalize each bin independently
            for b in range(num_bins):
                max_val = np.abs(voxel_grid[b]).max()
                if max_val > 0:
                    voxel_grid[b] /= max_val

        return voxel_grid

    def voxelize_fixed_bins(
        self,
        events: Dict[str, np.ndarray],
        num_bins: int,
        duration: float,
    ) -> np.ndarray:
        """Voxelize events with fixed number of bins.

        Args:
            events: Dictionary with x, y, t, p arrays
            num_bins: Fixed number of time bins
            duration: Total duration in seconds

        Returns:
            Voxel grid [num_bins, num_polarities, height, width]
        """
        start_time = 0.0
        end_time = duration
        return self.voxelize(events, num_bins, start_time, end_time)

    def voxelize_batch(
        self,
        events_batch: list,
        num_bins: Optional[int] = None,
    ) -> torch.Tensor:
        """Voxelize a batch of event streams.

        Args:
            events_batch: List of event dictionaries
            num_bins: Fixed number of bins (optional)

        Returns:
            Batch tensor [B, num_bins, num_polarities, height, width]
        """
        voxel_list = []
        for events in events_batch:
            voxel = self.voxelize(events, num_bins)
            voxel_list.append(voxel)

        # Pad to same number of bins if needed
        if num_bins is None:
            max_bins = max(v.shape[0] for v in voxel_list)
            padded_list = []
            for v in voxel_list:
                if v.shape[0] < max_bins:
                    pad_size = max_bins - v.shape[0]
                    padded = np.pad(v, ((0, pad_size), (0, 0), (0, 0), (0, 0)))
                    padded_list.append(padded)
                else:
                    padded_list.append(v)
            voxel_list = padded_list

        return torch.from_numpy(np.stack(voxel_list, axis=0)).float()


class AcousticFrameAligner:
    """Aligns event voxels to acoustic frames.

    Ensures T_v = T_a - 1 as specified in the architecture.
    """

    def __init__(self, audio_hop_ms: float = 20.0):
        self.audio_hop_ms = audio_hop_ms

    def align(
        self,
        voxel_grid: np.ndarray,
        num_acoustic_frames: int,
    ) -> np.ndarray:
        """Align voxel grid to acoustic frame count.

        Args:
            voxel_grid: Input voxel grid [T_v, C, H, W]
            num_acoustic_frames: Target number of acoustic frames T_a

        Returns:
            Aligned voxel grid [T_a - 1, C, H, W]
        """
        target_frames = num_acoustic_frames - 1
        current_frames = voxel_grid.shape[0]

        if current_frames == target_frames:
            return voxel_grid
        elif current_frames > target_frames:
            # Truncate
            return voxel_grid[:target_frames]
        else:
            # Pad with zeros
            pad_size = target_frames - current_frames
            return np.pad(
                voxel_grid,
                ((0, pad_size), (0, 0), (0, 0), (0, 0)),
                mode='constant',
                constant_values=0,
            )
