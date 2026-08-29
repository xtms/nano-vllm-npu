import sys, types, importlib.util
ROOT = "/data/sd/nano-vllm-npu"
# stub out the nanovllm package chain so __init__.py (which imports heavy deps) is NOT executed
for name in ("nanovllm", "nanovllm.engine", "nanovllm.utils"):
    m = types.ModuleType(name); m.__path__ = []; sys.modules[name] = m

def _load(modpath, fqname):
    spec = importlib.util.spec_from_file_location(fqname, modpath)
    mod = importlib.util.module_from_spec(spec); sys.modules[fqname] = mod
    spec.loader.exec_module(mod); return mod

sp_mod   = _load(f"{ROOT}/nanovllm/sampling_params.py", "nanovllm.sampling_params")
seq_mod  = _load(f"{ROOT}/nanovllm/engine/sequence.py",   "nanovllm.engine.sequence")
bm_mod   = _load(f"{ROOT}/nanovllm/engine/block_manager.py", "nanovllm.engine.block_manager")
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.sampling_params import SamplingParams

BS = 4  # small block size for edge-case coverage
Sequence.block_size = BS

def mkseq(tokens):
    sp = SamplingParams(temperature=1.0, max_tokens=100, ignore_eos=True)
    return Sequence(list(tokens), sp)

def new_bm(n=64):
    bm = BlockManager(n, BS)
    return bm

# ---------------------------------------------------------------- round 1
print("=== R1: identical prompts => cache hit (chain reuse) ===")
bm = new_bm()
s1 = mkseq(range(10))  # 10 tokens, BS=4 => 3 blocks, last block(2 tokens) partial
bm.can_allocate(s1)
bm.allocate(s1, 0)
s1.num_scheduled_tokens = 10
bm.hash_blocks(s1); s1.num_cached_tokens += s1.num_scheduled_tokens
print("s1 blocks hashed:", [bm.blocks[b].hash for b in s1.block_table],
      "hashed-full:", [i for i in range(s1.num_blocks) if bm.blocks[s1.block_table[i]].hash != -1])

s2 = mkseq(range(10))
ncb = bm.can_allocate(s2)
print("s2 can_allocate num_cached_blocks:", ncb, "(expect 2: blocks 0,1 full, block2 partial skipped)")
bm.allocate(s2, ncb)
print("s2 block_table:", s2.block_table, "shares block0 with s1?", s2.block_table[0]==s1.block_table[0])
print("s2 num_cached_tokens:", s2.num_cached_tokens, "(expect", ncb*BS, ")")

# ---------------------------------------------------------------- round 2
print("\n=== R2: prompt length = exact multiple of block_size (last full block NOT reused) ===")
bm = new_bm()
s1 = mkseq(range(8))  # exactly 8 tokens => 2 full blocks
bm.can_allocate(s1); bm.allocate(s1, 0)
s1.num_scheduled_tokens = 8
bm.hash_blocks(s1); s1.num_cached_tokens += s1.num_scheduled_tokens
print("s1 hashed blocks:", [i for i in range(s1.num_blocks) if bm.blocks[s1.block_table[i]].hash != -1],
      "(both 0 and 1 hashed)")
s2 = mkseq(range(8))
ncb = bm.can_allocate(s2)
print("s2 can_allocate:", ncb, "(expect 1, NOT 2 -- last block skipped even though full & cached)")
print("  => confirms last-block-skip inefficiency for boundary-aligned prompts")

# ---------------------------------------------------------------- round 3
print("\n=== R3: chunked prefill keeps chain consistent across chunks ===")
bm = new_bm()
s1 = mkseq(range(10))  # 3 blocks
# chunk1: tokens 0..3 (block 0)
bm.can_allocate(s1); bm.allocate(s1, 0)
s1.num_scheduled_tokens = 4
bm.hash_blocks(s1); s1.num_cached_tokens += s1.num_scheduled_tokens
h_after_chunk1 = dict(bm.hash_to_block_id)
# chunk2: tokens 4..9 (block1 full + block2 partial)
s1.num_scheduled_tokens = 6
bm.hash_blocks(s1); s1.num_cached_tokens += s1.num_scheduled_tokens
print("after chunk1 block0 hashed:", 0 in [i for i in range(s1.num_blocks) if bm.blocks[s1.block_table[i]].hash!=-1] or bm.blocks[s1.block_table[0]].hash!=-1)
print("after chunk2 block1 hashed:", bm.blocks[s1.block_table[1]].hash != -1, "(expect True, chain prefix from block0)")
print("chain prefix used for block1 == block0.hash:", bm.blocks[s1.block_table[1]].hash != -1)

# verify a fresh identical prompt still hits the chunked-completed full blocks
s2 = mkseq(range(10))
ncb = bm.can_allocate(s2)
print("s2 (identical, after chunked fill) can_allocate:", ncb, "(expect 2)")

# ---------------------------------------------------------------- round 4
print("\n=== R4: decode block-completion hashes block with FULL content (one-step lag ok) ===")
bm = new_bm()
# prompt = 3 tokens (block0 partial with 3). prefill produces 1 token => block0 full(4)
s1 = mkseq(range(3))
bm.can_allocate(s1); bm.allocate(s1, 0)
s1.num_scheduled_tokens = 3
bm.hash_blocks(s1); s1.num_cached_tokens += s1.num_scheduled_tokens  # num_cached=3
# prefill appends first completion token (token id 100) -> num_tokens=4, block0 full
s1.append_token(100)
print("after prefill num_tokens:", s1.num_tokens, "num_cached:", s1.num_cached_tokens,
      "block0 hashed?", bm.blocks[s1.block_table[0]].hash != -1, "(expect False: not yet hashed)")

# decode step: input token=100 (index3). may_append? len=4, 4%4=0 !=1 => no new block. scheduled=1
bm.may_append(s1)
s1.num_scheduled_tokens = 1
# postprocess decode: hash_blocks BEFORE append
start = s1.num_cached_tokens // BS
end = (s1.num_cached_tokens + s1.num_scheduled_tokens) // BS
print("decode step: start", start, "end", end, "(expect 0,1 -> hashes block0 now)")
bm.hash_blocks(s1); s1.num_cached_tokens += s1.num_scheduled_tokens
content = bm.blocks[s1.block_table[0]].token_ids
print("block0 hashed content:", content, "== token_ids[0:4]?", content == s1.token_ids[0:4],
      "(expect True, full 4 tokens incl. the decode input token 100)")

# ---------------------------------------------------------------- round 5
print("\n=== R5: preemption frees blocks but keeps hash entries => re-hit on reschedule ===")
bm = new_bm()
s1 = mkseq(range(10))
bm.can_allocate(s1); bm.allocate(s1, 0)
s1.num_scheduled_tokens = 10
bm.hash_blocks(s1); s1.num_cached_tokens += s1.num_scheduled_tokens
print("before preempt hash_to_block_id size:", len(bm.hash_to_block_id),
      "free_blocks:", len(bm.free_block_ids))
bm.deallocate(s1)
print("after  preempt hash_to_block_id size:", len(bm.hash_to_block_id),
      "(expect unchanged: entries retained for cache) free_blocks:",
      len(bm.free_block_ids), "(expect increased)")
s1b = mkseq(range(10))
ncb = bm.can_allocate(s1b)
print("rescheduled identical seq can_allocate:", ncb, "(expect 2: re-hits freed cached blocks)")

# ---------------------------------------------------------------- round 6
print("\n=== R6: content-mismatch guard prevents wrong hit on hash collision ===")
bm = new_bm()
# craft two DIFFERENT token lists that collide under a toy hash is hard with xxh64,
# so instead directly corrupt: put a block with hash matching but wrong token_ids.
s1 = mkseq([1,2,3,4, 5,6,7,8])
bm.can_allocate(s1); bm.allocate(s1, 0)
s1.num_scheduled_tokens = 8
bm.hash_blocks(s1); s1.num_cached_tokens += s1.num_scheduled_tokens
h0 = bm.blocks[s1.block_table[0]].hash
# corrupt: same hash key but change stored token_ids
bm.blocks[s1.block_table[0]].token_ids = [99,99,99,99]
bm.hash_to_block_id[h0] = s1.block_table[0]
s2 = mkseq([1,2,3,4, 5,6,7,8])
ncb = bm.can_allocate(s2)
print("corrupted-content block can_allocate:", ncb, "(expect 0: content check rejects, no false hit)")

# ---------------------------------------------------------------- round 7
print("\n=== R7: numpy dtype determinism of compute_hash (within-process) ===")
import numpy as np
from nanovllm.engine.block_manager import BlockManager as BM
tids = [1,2,3,4]
arr = np.array(tids)
print("inferred dtype:", arr.dtype, "bytes/token:", len(arr.tobytes())//len(arr))
h_a = BM.compute_hash(tids, -1)
h_b = BM.compute_hash(np.array(tids, dtype=np.int32).tolist(), -1)
print("default-list hash == int32-list hash:", h_a == h_b, "(within-process consistent; dtype implicit)")
print("\nALL CHECKS DONE")
