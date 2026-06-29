"""Conditional torch.compile — disabled on NPU where Triton inductor is unstable."""

import torch


def _is_npu_env() -> bool:
    """Detect if we're running in an Ascend NPU environment."""
    try:
        import torch_npu  # noqa: F401
        return True
    except ImportError:
        return False


def optional_compile(fn=None):
    """torch.compile replacement that is a no-op on NPU.

    Can be used as a decorator with or without arguments:
        @optional_compile
        def forward(self, x): ...

        @optional_compile()
        def forward(self, x): ...
    """
    if _is_npu_env():
        if fn is not None:
            return fn
        return lambda f: f
    else:
        if fn is not None:
            return torch.compile(fn)
        return torch.compile
