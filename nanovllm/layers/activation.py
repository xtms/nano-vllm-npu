import torch
from torch import nn
import torch.nn.functional as F
from nanovllm.utils.compile import optional_compile


class SiluAndMul(nn.Module):

    @optional_compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
