import torch
import torch.distributed as dist


def get_device_module(device_type: str):
    """Return the torch device module for the given device type."""
    if device_type == "cuda":
        return torch.cuda
    elif device_type == "npu":
        return torch.npu
    else:
        raise ValueError(f"Unsupported device type: {device_type}")


def get_dist_backend(device_type: str) -> str:
    """Return the distributed backend for the given device type."""
    if device_type == "cuda":
        return "nccl"
    elif device_type == "npu":
        return "hccl"
    else:
        raise ValueError(f"Unsupported device type: {device_type}")


def get_default_device_str(device_type: str) -> str:
    """Return the torch device string for the given device type."""
    return device_type


def set_device(device_type: str, rank: int):
    """Set the current device."""
    mod = get_device_module(device_type)
    mod.set_device(rank)


def make_device(device_type: str, rank: int = 0) -> torch.device:
    """Create a torch.device object."""
    return torch.device(f"{device_type}:{rank}")


def synchronize(device_type: str):
    """Synchronize the device."""
    mod = get_device_module(device_type)
    mod.synchronize()


def empty_cache(device_type: str):
    """Empty the device memory cache."""
    mod = get_device_module(device_type)
    mod.empty_cache()


def mem_get_info(device_type: str) -> tuple[int, int]:
    """Return (free, total) device memory in bytes."""
    mod = get_device_module(device_type)
    return mod.mem_get_info()


def memory_stats(device_type: str) -> dict:
    """Return device memory statistics."""
    mod = get_device_module(device_type)
    return mod.memory_stats()


def reset_peak_memory_stats(device_type: str):
    """Reset peak memory statistics."""
    mod = get_device_module(device_type)
    mod.reset_peak_memory_stats()


def is_graph_available(device_type: str) -> bool:
    """Check whether graph capture (CUDA/NPU graph) is supported."""
    if device_type == "cuda":
        return True
    # NPU graph support depends on torch_npu version; disable by default
    return False


def create_graph(device_type: str):
    """Create a device graph object for capture/replay."""
    mod = get_device_module(device_type)
    if device_type == "cuda":
        return mod.CUDAGraph()
    else:
        raise NotImplementedError(f"Graph capture not supported for {device_type}")


def graph_context(graph, pool=None, device_type: str = "cuda"):
    """Context manager for graph capture. `graph` is the device graph object."""
    mod = get_device_module(device_type)
    return mod.graph(graph, pool)


def init_distributed(device_type: str, rank: int, world_size: int, master_addr: str = "tcp://localhost:2333"):
    """Initialize the distributed process group."""
    backend = get_dist_backend(device_type)
    if not dist.is_initialized():
        dist.init_process_group(backend, master_addr, world_size=world_size, rank=rank)
