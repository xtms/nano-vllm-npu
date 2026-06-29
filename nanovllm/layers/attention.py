import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.utils.context import get_context


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """Store key and value tensors into the KV cache at the given slot indices.

    Pure PyTorch implementation — works on both CUDA and NPU without Triton.

    k_cache/v_cache shape: [num_blocks, block_size, num_kv_heads, head_dim]
    slot_mapping contains flat indices into the first two dims: slot = block_idx * block_size + offset
    """
    mask = slot_mapping != -1
    if mask.any():
        valid_slots = slot_mapping[mask]
        num_kv_heads = k_cache.size(2)
        head_dim = k_cache.size(3)
        # Flatten [blocks, block_size, ...] → [blocks * block_size, ...] for flat slot indexing
        k_flat = k_cache.view(-1, num_kv_heads, head_dim)
        v_flat = v_cache.view(-1, num_kv_heads, head_dim)
        k_flat[valid_slots] = key[mask]
        v_flat[valid_slots] = value[mask]


def _gather_kv_from_cache(k_cache: torch.Tensor, v_cache: torch.Tensor, block_table: torch.Tensor, seq_lens: torch.Tensor):
    """Gather contiguous K, V tensors from a paged KV cache using block_table.

    Args:
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        block_table: [batch_size, max_blocks_per_seq] — block indices, -1 for padding
        seq_lens: [batch_size] — actual sequence lengths in tokens

    Returns:
        k: [batch_size, max_seqlen, num_kv_heads, head_dim] (zero-padded)
        v: [batch_size, max_seqlen, num_kv_heads, head_dim] (zero-padded)
    """
    batch_size = block_table.size(0)
    block_size = k_cache.size(1)
    num_kv_heads = k_cache.size(2)
    head_dim = k_cache.size(3)
    max_seqlen = block_table.size(1) * block_size

    k = k_cache.new_zeros(batch_size, max_seqlen, num_kv_heads, head_dim)
    v = v_cache.new_zeros(batch_size, max_seqlen, num_kv_heads, head_dim)

    for b in range(batch_size):
        seq_len = seq_lens[b].item()
        if seq_len == 0:
            continue
        gathered_k = []
        gathered_v = []
        for block_idx in block_table[b]:
            if block_idx < 0:
                break
            gathered_k.append(k_cache[block_idx])
            gathered_v.append(v_cache[block_idx])
        if gathered_k:
            kv_cat = torch.cat(gathered_k, dim=0)[:seq_len]  # [seq_len, num_kv_heads, head_dim]
            vv_cat = torch.cat(gathered_v, dim=0)[:seq_len]
            k[b, :seq_len] = kv_cat
            v[b, :seq_len] = vv_cat

    return k, v


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

        # Try to import flash-attn; fall back to SDPA if unavailable
        try:
            from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
            self._flash_attn_varlen_func = flash_attn_varlen_func
            self._flash_attn_with_kvcache = flash_attn_with_kvcache
            self._use_flash_attn = True
        except ImportError:
            self._flash_attn_varlen_func = None
            self._flash_attn_with_kvcache = None
            self._use_flash_attn = False

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if self._use_flash_attn:
                o = self._flash_prefill(q, k, v, k_cache, v_cache, context)
            else:
                o = self._sdpa_prefill(q, k, v, k_cache, v_cache, context)
        else:    # decode
            if self._use_flash_attn:
                o = self._flash_decode(q, k_cache, v_cache, context)
            else:
                o = self._sdpa_decode(q, k_cache, v_cache, context)
        return o

    def _flash_prefill(self, q, k, v, k_cache, v_cache, context):
        if context.block_tables is not None:    # prefix cache
            k, v = k_cache, v_cache
        o = self._flash_attn_varlen_func(q, k, v,
                                         max_seqlen_q=context.max_seqlen_q,
                                         cu_seqlens_q=context.cu_seqlens_q,
                                         max_seqlen_k=context.max_seqlen_k,
                                         cu_seqlens_k=context.cu_seqlens_k,
                                         softmax_scale=self.scale, causal=True,
                                         block_table=context.block_tables)
        return o

    def _flash_decode(self, q, k_cache, v_cache, context):
        o = self._flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                          cache_seqlens=context.context_lens,
                                          block_table=context.block_tables,
                                          softmax_scale=self.scale, causal=True)
        return o

    def _sdpa_prefill(self, q, k, v, k_cache, v_cache, context):
        """SDPA fallback for prefill — works on CUDA and NPU.

        SDPA expects 4D tensors in [batch, heads, seq, embed] order.
        Our tensors are in [batch, seq, heads, embed] order, so we transpose.
        """
        outputs = []
        if context.block_tables is not None:
            # Prefix cache: gather K, V from cache blocks (all tokens including cached).
            seq_lens_k = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
            k_gathered, v_gathered = _gather_kv_from_cache(k_cache, v_cache, context.block_tables, seq_lens_k)
            # k_gathered, v_gathered: [batch, max_seqlen, num_kv_heads, head_dim]
            # -> transpose to [batch, num_kv_heads, max_seqlen, head_dim] for SDPA
            k_gathered = k_gathered.transpose(1, 2)
            v_gathered = v_gathered.transpose(1, 2)
            for i in range(context.cu_seqlens_q.size(0) - 1):
                q_start = context.cu_seqlens_q[i].item()
                q_end = context.cu_seqlens_q[i + 1].item()
                seq_len_k = seq_lens_k[i].item()
                if q_end == q_start or seq_len_k == 0:
                    continue
                # q: [total_tokens, num_heads, head_dim] -> [1, num_heads, seqlen_q, head_dim]
                qi = q[q_start:q_end].unsqueeze(0).transpose(1, 2)
                # k, v: [batch, num_kv_heads, max_seqlen, head_dim] -> slice to [1, num_kv_heads, seq_len_k, head_dim]
                ki = k_gathered[i:i+1, :, :seq_len_k]
                vi = v_gathered[i:i+1, :, :seq_len_k]
                if self.num_heads > self.num_kv_heads:
                    ratio = self.num_heads // self.num_kv_heads
                    ki = ki.repeat_interleave(ratio, dim=1)
                    vi = vi.repeat_interleave(ratio, dim=1)
                oi = F.scaled_dot_product_attention(
                    qi, ki, vi,
                    scale=self.scale, is_causal=True,
                )
                # oi: [1, num_heads, seqlen_q, head_dim] -> transpose back -> [seqlen_q, num_heads, head_dim]
                outputs.append(oi.squeeze(0).transpose(0, 1))
        else:
            # No prefix cache: q, k, v are all flat with same cu_seqlens_q boundaries
            for i in range(context.cu_seqlens_q.size(0) - 1):
                q_start = context.cu_seqlens_q[i].item()
                q_end = context.cu_seqlens_q[i + 1].item()
                if q_end == q_start:
                    continue
                # [seqlen, num_heads, head_dim] -> [1, num_heads, seqlen, head_dim]
                qi = q[q_start:q_end].unsqueeze(0).transpose(1, 2)
                ki = k[q_start:q_end].unsqueeze(0).transpose(1, 2)
                vi = v[q_start:q_end].unsqueeze(0).transpose(1, 2)
                if self.num_heads > self.num_kv_heads:
                    ratio = self.num_heads // self.num_kv_heads
                    ki = ki.repeat_interleave(ratio, dim=1)
                    vi = vi.repeat_interleave(ratio, dim=1)
                oi = F.scaled_dot_product_attention(
                    qi, ki, vi,
                    scale=self.scale, is_causal=True,
                )
                # [1, num_heads, seqlen, head_dim] -> [seqlen, num_heads, head_dim]
                outputs.append(oi.squeeze(0).transpose(0, 1))

        return torch.cat(outputs, dim=0) if outputs else q.new_zeros(0, self.num_heads, self.head_dim)

    def _sdpa_decode(self, q, k_cache, v_cache, context):
        """SDPA fallback for decode — works on CUDA and NPU.

        SDPA expects 4D tensors in [batch, heads, seq, embed] order.
        """
        k, v = _gather_kv_from_cache(k_cache, v_cache, context.block_tables, context.context_lens)
        # k, v: [batch, max_seqlen, num_kv_heads, head_dim]
        # -> transpose to [batch, num_kv_heads, max_seqlen, head_dim] for SDPA
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat KV for GQA if needed (along heads dim, dim=1 after transpose)
        if self.num_heads > self.num_kv_heads:
            ratio = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(ratio, dim=1)
            v = v.repeat_interleave(ratio, dim=1)

        # q: [batch, num_heads, head_dim] -> [batch, num_heads, 1, head_dim]
        q = q.unsqueeze(2)

        o = torch.empty_like(q)  # [batch, num_heads, 1, head_dim]
        for b in range(q.size(0)):
            seq_len = context.context_lens[b].item()
            qb = q[b:b+1, :, :, :]                              # [1, num_heads, 1, head_dim]
            kb = k[b:b+1, :, :seq_len, :]                         # [1, num_heads, seq_len, head_dim]
            vb = v[b:b+1, :, :seq_len, :]
            o[b:b+1] = F.scaled_dot_product_attention(qb, kb, vb, scale=self.scale)

        return o.squeeze(2)  # [batch, num_heads, head_dim]
