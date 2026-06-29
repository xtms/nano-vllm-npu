import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    device_type: str = "cuda"
    device_id: int | list[int] = 0
    memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.device_type in ("cuda", "npu")
        if isinstance(self.device_id, list):
            assert all(d >= 0 for d in self.device_id)
            assert len(self.device_id) == self.tensor_parallel_size, \
                f"device_id list length ({len(self.device_id)}) must match tensor_parallel_size ({self.tensor_parallel_size})"
        else:
            assert self.device_id >= 0
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

    def get_device_id(self, rank: int) -> int:
        """Resolve the physical device ID for the given rank."""
        if isinstance(self.device_id, list):
            return self.device_id[rank]
        return self.device_id + rank
