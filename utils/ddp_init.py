import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def setup_ddp(
    rank: int,
    world_size: int,
    backend: str = 'nccl',
):
    """Initialize distributed training environment.

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend (nccl or gloo)
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(rank)

    logger.info(f"Initialized DDP: rank={rank}, world_size={world_size}")


def cleanup_ddp():
    """Cleanup distributed training environment."""
    dist.destroy_process_group()
    logger.info("Cleaned up DDP")


def get_device(local_rank: int) -> torch.device:
    """Get device for current process.

    Args:
        local_rank: Local GPU rank

    Returns:
        torch.device for computation
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU")

    return device


def setup_for_distributed(is_master: bool):
    """Disable printing for non-master processes."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_main_process(rank: int) -> bool:
    """Check if current process is the main process."""
    return rank == 0


def reduce_tensor(
    tensor: torch.Tensor,
    world_size: int,
    op: str = 'mean',
) -> torch.Tensor:
    """Reduce tensor across all processes.

    Args:
        tensor: Input tensor
        world_size: Number of processes
        op: Reduction operation ('mean' or 'sum')

    Returns:
        Reduced tensor
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if op == 'mean':
        rt /= world_size
    return rt


def launch_training(
    train_fn,
    world_size: int,
    config: dict,
):
    """Launch distributed training.

    Args:
        train_fn: Training function to execute
        world_size: Number of GPUs
        config: Configuration dictionary
    """
    mp.spawn(
        train_fn,
        args=(world_size, config),
        nprocs=world_size,
        join=True,
    )


class DistributedManager:
    """Manager for distributed training lifecycle.

    Handles setup, cleanup, and common distributed operations.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        config: dict,
    ):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.device = get_device(rank)
        self.is_main = is_main_process(rank)

        # Setup DDP
        backend = config.get('ddp', {}).get('backend', 'nccl')
        setup_ddp(rank, world_size, backend)

        # Setup printing
        setup_for_distributed(self.is_main)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_ddp()
        return False

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model with DistributedDataParallel."""
        model = model.to(self.device)
        find_unused = self.config.get('ddp', {}).get('find_unused_parameters', False)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.rank] if self.device.type == 'cuda' else None,
            output_device=self.rank if self.device.type == 'cuda' else None,
            find_unused_parameters=find_unused,
        )
        return model

    def create_sampler(self, dataset, shuffle: bool = True):
        """Create DistributedSampler for dataset."""
        return torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
        )

    def barrier(self):
        """Synchronize all processes."""
        dist.barrier()

    def reduce_metrics(self, metrics: dict) -> dict:
        """Reduce metrics across all processes."""
        reduced = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                reduced[key] = reduce_tensor(value, self.world_size).item()
            else:
                reduced[key] = value
        return reduced
