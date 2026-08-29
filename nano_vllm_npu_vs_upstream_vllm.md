## nano-vllm-npu 与上游 vLLM 逐项对比

延续 [CODEXTMS-22](mention://issue/01a048bd-0e1b-7e8d-be35-9593afd320d2) 整体架构分析与 [CODEXTMS-23](mention://issue/01a048c1-5f03-715e-ab5c-5193cb298059) 分模块分析,本篇将 nano-vllm-npu 与上游 vLLM v1 **逐模块、逐特性**对比。

- **nano-vllm-npu**:`/data/sd/nano-vllm-npu`,分支 `main`,提交 `4f6d847`(支持 NPU,并支持指定占用卡),Python 代码 ~1915 行。
- **上游 vLLM**:`/data/sd/vllm`,提交 `0fc695fc6d`(main,2026-08),`vllm/v1/` 架构 + `vllm/model_executor`、`vllm/platforms`、`vllm/compilation`、`vllm/entrypoints` 等。

> 说明:nano-vllm 并非 vLLM 仓库的 fork,而是一次**独立的轻量复刻**——用 ~1900 行重新实现 vLLM 的核心思想(paged KV cache + 前缀缓存、张量并行、CUDA graph、chunked prefill、continuous batching),并在其上叠加 Ascend NPU 适配。因此本对比聚焦"同一概念在两侧如何实现、nano 简化/缺失了什么、NPU 适配改了什么"。所有 `file:line` 引用相对两个仓库根目录。
>
> 相关深度分析(聚焦单模块,与本篇互补):[`chunked_prefill_boundary_analysis.md`](chunked_prefill_boundary_analysis.md)(chunked prefill 六类边界)、[`nano_vllm_npu_hashchain_correctness_analysis.md`](nano_vllm_npu_hashchain_correctness_analysis.md)(前缀缓存哈希链正确性 + 校验脚本)。

---

## 〇、对比汇总表

| 维度 | nano-vllm-npu | 上游 vLLM v1 |
|---|---|---|
| 代码规模 | ~1915 行,23 个 .py | 数十万行,数百模块 |
| 控制平面 | 单进程同步(scheduler+rank0 同进程) | 多进程异步(EngineCore 进程 + ZMQ + worker 进程) |
| 并发模型 | `mp.spawn` + SharedMemory + Event,锁步执行 | ZMQ 消息队列 + NCCL `torch.distributed`,异步流水 |
| 调度策略 | 两阶段 prefill 优先,仅队首可 chunked | 统一单阶段,prefill/decode 自由混合,任意请求可 chunked |
| 抢占 | `running.pop()`(LIFO)重算 | 优先级感知 / FCFS,可回滚当步状态 |
| 请求队列 | 普通 deque | FCFS / Priority heap |
| 序列状态机 | 3 状态(WAITING/RUNNING/FINISHED) | 12 状态 + 多种 WAITING_FOR_* 阻塞态 |
| Block 大小 | 256(必须 %256==0) | 默认 16,可配置 |
| 前缀缓存哈希 | xxhash 链,`int→block_id`,**token 二次校验** | sha256(可配),`BlockHashWithGroupId`,无 token 校验 |
| LRU 驱逐 | 无(freed 块保留哈希直至重分配) | 有(`FreeKVCacheBlockQueue` + touch/evict) |
| 注意力后端 | 1 个(flash-attn 或 SDPA 回退) | ~20 个可插拔后端 + 平台选择器 |
| KV 写入 | 纯 PyTorch 索引(`k_flat[slots]=key`) | 自定义 CUDA/Triton op(`reshape_and_cache_flash`) |
| 采样 | Gumbel-max,仅 temperature,禁 greedy | top-k/top-p/min-p/penalty/logprobs/结构化/投机采样 |
| 张量并行 | TP 1-8,Megatron 式 column/row | TP×PP×DP×PCP×DCP + 异步 TP + 自定义 all-reduce |
| CUDA graph | decode 专用,固定 bs 列表,NPU 禁用 | FULL/PIECEWISE/FULL_DECODE_ONLY,可配置 |
| torch.compile | `optional_compile`,仅采样器,NPU 降级 no-op | 完整 Inductor + 自定义 pass + AOT 缓存 + 分段编译 |
| 模型 | 仅 Qwen3 dense | 数百模型 + MoE/VL/ASR 等变体 |
| 量化 | 无 | awq/gptq/fp8/bitsandbytes/CompressedTensors… |
| API 服务 | 4 路由,同步阻塞,无流式 | 全异步,SSE 流式,跨请求连续批,工具/推理解析 |
| 平台 | CUDA + NPU(Ascend) | CUDA/ROCm/TPU/XPU/CPU(+OOT 插件),无 NPU |

---

## 一、整体架构与代码规模

nano-vllm-npu 把 vLLM 的"前端 → EngineCore → Executor → Worker → ModelRunner"五层压缩成**单进程单对象**:`LLMEngine`(`engine/llm_engine.py:15`)在主进程内同时持有 tokenizer、`Scheduler`、rank-0 `ModelRunner`;TP>1 时仅把 rank 1..N-1 作为 `mp.Process` spawn 出去(`llm_engine.py:24-30`),通过一块 1 MiB `SharedMemory` + 每 worker 一个 `mp.Event` 锁步同步(`model_runner.py:64-106`)。`LLM`(`llm.py:4`)只是 `class LLM(LLMEngine): pass` 的别名。

上游 vLLM v1 是**多进程异步**架构:前端 `AsyncLLM`/`LLMEngine`(`v1/engine/async_llm.py:70`、`v1/engine/llm_engine.py:47`)与 `EngineCore`(`v1/engine/core.py:95`)分属不同进程,经 ZMQ(DEALER/PUSH/XSUB,`core.py:1051,1466,1564`)通信;`EngineCoreProc`(`core.py:860`)在后台进程跑 busy loop,内含 input/output IO 线程;GPU worker 又由 `Executor`(`v1/executor/multiproc_executor.py:103`)另行 spawn,每 rank 一个进程,经 `MessageQueue` 广播 `SchedulerOutput`、回传 `ModelRunnerOutput`。控制平面与数据平面彻底解耦。

| 层 | nano | vLLM v1 |
|---|---|---|
| 前端 | `LLMEngine`(`llm_engine.py:15`)同步 | `AsyncLLM`/`LLMEngine`(`async_llm.py:70`/`llm_engine.py:47`)+ `InputProcessor`/`OutputProcessor` |
| 核心 | (合并在前端) | `EngineCore`/`EngineCoreProc`(`core.py:95,860`)独立进程 |
| 执行器 | (无) | `Executor`(`v1/executor/abstract.py:37`)→ `MultiprocExecutor`/`RayExecutor` |
| Worker | rank0 同进程,rank>0 `mp.Process` | 每 rank 一个 `Worker` 进程(`v1/worker/gpu_worker.py:117`)+ `WorkerWrapperBase` |
| Runner | `ModelRunner`(`model_runner.py:29`,280 行) | `GPUModelRunner`(`gpu_model_runner.py:418`,~7561 行) |
| IPC | SharedMemory + Event + `tcp://localhost:2333` | ZMQ + `MessageQueue` + 动态端口 `torch.distributed` |

---

## 二、引擎层(Engine)

### 2.1 职责与结构

nano `LLMEngine` 一个类包揽全部:`add_request`(`llm_engine.py:43`)encode prompt→`Sequence` 入 `scheduler.waiting`;`step`(`llm_engine.py:49`)做 `schedule()→call("run")→postprocess()`;`generate`(`llm_engine.py:60`)是阻塞 `while not is_finished(): step()` 循环,末尾统一 decode 返回 `list[str]`。无 EngineCore 进程、无 async、无流式、无 detokenizer、无 Input/OutputProcessor、无 Renderer。

vLLM v1 拆分清晰:`LLMEngine`(`llm_engine.py:47`)是同步前端,只持有 `InputProcessor`(`:93`)、`OutputProcessor`(`:96`)、`EngineCoreClient`(`:104`),其 `step`(`:287`)做 `engine_core.get_output()→output_processor.process_outputs()`;`AsyncLLM`(`async_llm.py:70`)是 asyncio 前端,有 `output_handler` 任务(`:170`)持续排空 `EngineCoreOutputs`;`EngineCore`(`core.py:95`)才是真正的调度+执行核心,持有 `model_executor`、`scheduler`、`structured_output_manager`、`batch_queue`(PP 流水,`:192`)、`async_scheduling`(`:220`)。

### 2.2 关键差异

- **无异步/无独立核心进程/无 ZMQ**:nano 单进程同步;vLLM 多进程 + ZMQ + busy loop + IO 线程。
- **无流式**:nano `generate` 全部完成才返回字符串;vLLM 经 `OutputProcessor` + `stream_interval` 增量流式 `RequestOutput`。
- **无 InputProcessor/OutputProcessor/Renderer**:nano 在 `add_request` 内联 tokenize、`generate` 内联 decode;vLLM 有专门阶段 + 独立 detokenizer 进程。
- **无 DP/PP/batch_queue**:nano 仅 TP(`config.py:15`);vLLM 有 DP(`core.py:1701`)、PP 流水(`core.py:192-198`)、`DPMoEEngineCoreActor`。
- **无 spec decode / 结构化输出 / grammar**:nano `SamplingParams` 仅 `{temperature, max_tokens, ignore_eos}`;vLLM `EngineCore` 持有 `structured_output_manager`、`use_spec_decode`、`get_grammar_bitmask`。
- **无 encoder/decoder、多模态、LoRA、KV connector / P-D 分离、sleep/wake、profile、stats logging**:vLLM 全有,nano 全无。
- **block 默认 256**(`config.py:19`)vs **16**(`config/cache.py:46`);`Config` 13 字段扁平 dataclass vs `VllmConfig` ~20 嵌套子配置。

---

## 三、调度器(Scheduler)

### 3.1 调度策略

nano `Scheduler`(`scheduler.py:8`)是**严格两阶段、prefill 优先**:`schedule`(`scheduler.py:25`)先处理 `waiting`(prefill),仅当无 prefill 可调度才进 decode 分支。**一个 step 要么全 prefill 要么全 decode,不混合**。chunked prefill **仅限队首**长 prompt(`scheduler.py:42-43`:`if remaining < num_tokens and scheduled_seqs: break`);后续短序列整批同 batch。`num_scheduled_tokens = min(num_tokens, remaining)`(`scheduler.py:46`),未算完则留 `waiting[0]` 下轮续算。

vLLM v1 `Scheduler`(`v1/core/sched/scheduler.py:65`)是**统一单阶段**:`schedule`(`scheduler.py:340`)注释明确"无单独 prefill/decode 阶段",每个请求用 `num_computed_tokens` 追赶 `num_tokens_with_spec`。先扫 RUNNING(`:378-551`,decode 或半 prefill),再扫 WAITING(`:562-868`,新 prefill),**可在同一 batch 自由混合** running 与新 admitted 请求。**任意请求都可 chunked**,受 `long_prefill_token_threshold`(`:685`)与 `token_budget`(`:410`)约束;`enable_chunked_prefill=False` 时超 budget 直接 break(`:692-697`)。

### 3.2 抢占

nano(`scheduler.py:75-79`):仅 decode 步 `can_append` 不足时触发,`running.pop()`(LIFO)→ `deallocate`(全释放,`num_cached_tokens` 归 0)→ 回 `waiting` 重算。纯**重算式**抢占。

vLLM v1(`scheduler.py:460-509` + `_preempt_request:974`):`allocate_slots` 返回 None 时触发。`PRIORITY` 策略选 `(priority, arrival_time)` 最大者(`:474-478`);FCFS 选 `running.pop()`(`:499`)。可回滚**当步已调度**状态(`:480-497`),设 `status=PREEMPTED`、`num_computed_tokens=0`、`prepend` 回 waiting。

### 3.3 队列与状态

nano 用普通 `deque`(`scheduler.py:16-17`)。vLLM v1 有 `FCFSRequestQueue`/`PriorityRequestQueue`(`request_queue.py:75,131`,后者 `heapq` 按 `(priority, arrival_time, request_id)` 排序),`skipped_waiting`(`scheduler.py:167`),以及 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`/`WAITING_FOR_REMOTE_KVS`/`WAITING_FOR_STREAMING_REQ`(`request.py:322-325`)等阻塞态晋升逻辑。

### 3.4 返回与后处理

nano `schedule` 返回 `(list[Sequence], is_prefill: bool)`;`postprocess`(`scheduler.py:81`)内联做 hash+append token+判结束+释放块。vLLM v1 返回富 `SchedulerOutput`(new/cached reqs、`num_scheduled_tokens`、spec tokens、encoder inputs、common-prefix blocks、`new_block_ids_to_zero`、`kv_connector_metadata`);拆分为 `_update_after_schedule`(`scheduler.py:997`,推进计数器)+ `update_from_output`(`scheduler.py:1329`,应用 token、spec 接受、stop 判定、释放)。

### 3.5 关键差异小结

- 两阶段 prefill-then-decode(nano)vs 统一单阶段自由混合(vLLM)。
- 仅队首 chunk(nano)vs 任意请求可 chunk + 可关闭(vLLM)。
- 普通 deque(nano)vs FCFS/Priority heap + skipped_waiting(vLLM)。
- LIFO 重算抢占(nano)vs 优先级感知/FCFS + 当步回滚(vLLM)。
- nano 无 encoder/spec/grammar/connector/mamba/pause/streaming/async-scheduling;vLLM 全有。
- nano 无 `SchedulerInterface` ABC;vLLM 形式化契约(`interface.py:36`)。

---

## 四、序列/请求(Sequence / Request)

### 4.1 角色

nano `Sequence`(`sequence.py:14`)一身兼三职:既是 vLLM 的 `Request`,也是 `SequenceGroup`,也是单条序列——无 group/seq 拆分、无 `n>1` fan-out。携带 prompt token、output token、block_table、调度计数器与少量采样参数。`SequenceStatus` 仅 3 态(`sequence.py:8`)。

vLLM v1 `Request`(`v1/request.py:59`)字段约 40 个,12 种 `RequestStatus`(`request.py:319-335`),携带 multimodal features、LoRA、structured-output、KV-transfer、block_hashes、spec tokens、streaming、events、prefill_stats 等。

### 4.2 token 计数

nano 双游标:`num_cached_tokens` + `num_scheduled_tokens`(`sequence.py:25-26`);`postprocess` 推进 `num_cached_tokens += num_scheduled_tokens`(`scheduler.py:84`)。vLLM v1 单游标 `num_computed_tokens`(`request.py:149`),`_update_after_schedule` 推进(`scheduler.py:1010`),并 `is_prefill_chunk`(`request.py:164`)+ `num_tokens_with_spec` 统一 prefill/decode/spec;`num_output_placeholders`(`request.py:141`)支持异步调度投机推进。

### 4.3 块哈希归属

nano `Sequence` **不**算哈希,由 `BlockManager.hash_blocks(seq)`(`block_manager.py:110`)按 `num_cached_tokens`/`num_scheduled_tokens` 计算。vLLM v1 `Request` 自持 `block_hashes: list[BlockHash]`(`request.py:175`)与绑定 `_block_hasher`(`:179`,来自 `EngineCore.request_block_hasher` `core.py:213`),`update_block_hashes`(`request.py:233`)在 append token 时增量计算。

### 4.4 IPC

nano `Sequence` 跨进程 pickle 到 TP worker,`__getstate__`(`sequence.py:72-74`)定制:prefill 传完整 `token_ids`,decode 只传 `last_token`,压低 IPC 负载。vLLM v1 `Request` **从不** pickle 到 worker——worker 只收 `SchedulerOutput` + `NewRequestData`/`CachedRequestData`(`scheduler.py:897-920`)。

### 4.5 采样参数

nano `SamplingParams` = `{temperature, max_tokens, ignore_eos}`,`assert temperature > 1e-10` 禁 greedy(`sampling_params.py:11`)。vLLM `SamplingParams` 含 top_k/top_p/min_p/n/stop/penalty/structured_output/guided_decoding/logprobs 等数十项。

### 4.6 关键差异小结

- nano 3 状态/~12 字段;vLLM 12 状态/~40 字段。
- nano 无 priority/n>1/multimodal/LoRA/structured-output/encoder/spec/KV-transfer/streaming/async 字段。
- nano 在 BlockManager 算哈希;vLLM 在 Request 内算(append 时增量)。
- nano 定制 pickle 优化 IPC;vLLM 从不 pickle Request 到 worker。
- nano `block_size` 是可变类属性全局设置(`sequence.py:15` + `llm_engine.py:21`);vLLM 按 KV-cache group 配置。

---

## 五、块管理器 / KV 缓存 / 前缀缓存

### 5.1 架构

nano `BlockManager`(`block_manager.py:26`,~120 行)一个扁平类包揽全部:flat `list[Block]`、`hash_to_block_id: dict[int,int]`、`free_block_ids: deque`、`used_block_ids: set`。KV 张量实际在 `ModelRunner.kv_cache`(`model_runner.py:138`),BlockManager 只管块 ID 与元数据。

vLLM v1 拆为多层:`KVCacheManager`(`v1/core/kv_cache_manager.py:110`,调度器门面)→ `KVCacheCoordinator`(多 group 路由)→ `SingleTypeKVCacheManager`(每 group)→ `BlockPool`(`v1/core/block_pool.py:130`),哈希工具在 `kv_cache_utils.py`。

### 5.2 哈希链

nano `compute_hash(token_ids, prefix=-1)`(`block_manager.py:35`):`xxhash.xxh64()`,若 `prefix != -1` 先 `update(prefix.to_bytes(8,"little"))` 再 `update(token_bytes)`,返回 64 位 int。块 i 哈希 = `xxh64(prev_hash || block_i_tokens)`。首块 `prefix=-1`(无前缀字节)。`hash_to_block_id: int→block_id`。

vLLM v1 `hash_block_tokens(...)`(`kv_cache_utils.py:563`):`BlockHash(hash_fn((parent_block_hash, tuple(tokens), extra_keys)))`,`parent_block_hash` 默认 `NONE_HASH`(每进程 `os.urandom(32)`,`kv_cache_utils.py:111`,除非 `PYTHONHASHSEED` 固定)。默认 sha256(`core.py:208`),`extra_keys` 携带 multimodal/LoRA/prompt-embed/cache_salt(`kv_cache_utils.py:525,417,484,499`)。键为 `BlockHashWithGroupId` = hash + group_id(`:56`),同内容不同 group 不冲突。

### 5.3 前缀缓存命中与分配

nano `can_allocate(seq)`(`block_manager.py:58`):遍历完整块(`range(num_blocks-1)`,末块排除),链式算 `h`,查 `hash_to_block_id.get(h,-1)`。命中需 `block_id != -1` **且** `blocks[block_id].token_ids == token_ids`(**token 二次校验**,`:66`)。`allocate`(`:75`)命中块 `ref_count += 1` 共享,余块 `_allocate_block`。

vLLM v1 `get_computed_blocks(request)`(`kv_cache_manager.py:196`):`coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length=num_tokens-1)`(`:221`,强制末 token 重算以保证 logits)。**无 token 二次校验**——信任 sha256 + extra_keys。`allocate_slots`(`:238`)三段式:先 free/skipped + 容量检查(`:395`)→ 处理前缀(external-computed、`allocate_new_computed_blocks` 重引用,`:400`)→ `allocate_new_blocks` 新算(`:413`)→ `cache_blocks` 注册(`:430`)。

### 5.4 LRU 驱逐

nano **无 LRU**:块 `ref_count==0` 时入 `free_block_ids`,**哈希条目保留**直至该块被重分配(`_allocate_block:47` 删除)。freed-but-cached 块立即可被新请求命中复用,**无 touch/ref 计数 bump、无冷块驱逐**。FIFO deque,无序。

vLLM v1 **有 LRU**:`FreeKVCacheBlockQueue`(`kv_cache_utils.py:165`)双向链表,`ref_cnt==0` 的 cached 块作为驱逐候选;`popleft`/`popleft_n` 取最冷。`touch(blocks)`(`block_pool.py:402`)命中时把 `ref_cnt==0` 块移出队列并 `ref_cnt+=1` 复活;`_maybe_evict_cached_block`(`:365`)重用途时弹哈希 + `reset_hash()`;`evict_blocks`(`:443`)/`reset_prefix_cache`(`:462`)显式驱逐。

### 5.5 块大小与分配粒度

- nano `block_size=256`(`config.py:19`,断言 `%256==0`)——NPU 友好(块表项少、索引简单)。vLLM 默认 16(`config/cache.py:46`),可配置。
- nano **admission 时一次性分配整个 prompt 的块**(`block_manager.py:90-91` 循环 `num_cached_blocks..num_blocks`);`can_allocate` 若整 prompt `num_new_blocks > free` 返回 -1(`:71`)——**prompt 必须整体(扣除前缀缓存)装得下才准入**。vLLM v1 **按 chunk 分配**(`allocate_slots` 只为当前 chunk `num_new_tokens` 分配),**prompt 可超过空闲 KV 只要每 chunk 装得下**。

### 5.6 其余差异

| 项 | nano | vLLM v1 |
|---|---|---|
| null_block 占位 | 无 | 有(`block_pool.py:176`,稀疏/SWA/Mamba) |
| 混合 KV group | 单 group | `KVCacheCoordinator` + `SingleTypeKVCacheManager` 多 group 不同块大小 |
| 滑动窗口/skipped 块 | 无 | `remove_skipped_blocks`(`kv_cache_manager.py:448`) |
| common-prefix / cascade attention | 无 | `get_num_common_prefix_blocks`(`:485`) |
| external/connector KV | 无 | `num_external_computed_tokens`、`delay_cache_blocks`、KV 事件 |
| spec-decode lookahead 分配 | 无 | `num_lookahead_tokens`(`:244`) |
| 每步块清零 | 无(靠覆写 slot) | `take_new_block_ids` + `needs_kv_cache_zeroing`(`scheduler.py:926`) |
| 指标/事件 | 无 | `KVCacheMetricsCollector`、`KVCacheEvent`、`PrefixCacheStats` |

### 5.7 前缀缓存正确性对比

- nano:`xxh64` int + `dict[int,int]` + **token 二次校验** + 固定 seed(-1) + 无 extra_keys + 无 group_id + 无 LRU + ref-count 共享 + hash 存活至重分配。
- vLLM:**sha256(可配)** bytes + `BlockHashToBlockMap`(hash+groupid→block-or-dict) + **无 token 校验** + extra_keys(MM/LoRA/embeds/salt) + per-group 隔离 + 随机 per-process seed + **LRU + touch/evict** + common-prefix + 事件发布。
- nano 的 token 二次校验是廉价正确性兜底(防哈希碰撞);vLLM 用更强 sha256 + 更丰富键化换掉它。nano 无 LRU 对**离线批量**场景可接受(无长时闲竞争块);vLLM **在线服务**需 LRU 约束缓存内存。

---

## 六、执行器 / Worker(Model Runner)

### 6.1 架构

nano `ModelRunner`(`model_runner.py:29`,280 行)一个类兼任 executor+worker+runner。rank0 在主进程,rank>0 spawn 为 `mp.Process` 跑 `loop()`(`model_runner.py:84`):`while True: read_shm(); call(...)`。`LLMEngine.step` → `model_runner.call("run", seqs, is_prefill)`,rank0 `write_shm` 广播 pickle 后本地执行,worker 锁步同执行。

vLLM v1 五层:`EngineCore` 进程 → `Executor`(`v1/executor/abstract.py:37`)spawn 每 rank 一个 `Worker` 进程 → `Worker`(`gpu_worker.py:117`)→ `GPUModelRunner`(`gpu_model_runner.py:418`,~7561 行)。`Worker.execute_model`(`gpu_worker.py:806`)处理 PP 的 `AsyncIntermediateTensors`(`:851-893`)后委托 `model_runner.execute_model`(`gpu_model_runner.py:4005`)。

### 6.2 执行流

nano:`prepare_prefill|prepare_decode` 建 tensor + `set_context`(全局单例,`context.py:21`)→ `run_model`(`model_runner.py:219`):eager `model.compute_logits(model(...))` 或 graph replay → rank0 采样返回 token_ids,worker 返回 None。无异步调度、无调度/执行重叠、无输出队列。

vLLM v1:`_update_states`(应用 `SchedulerOutput` diff 到 `InputBatch`,`gpu_model_runner.py:1125`)→ `_prepare_inputs`(`:1872`)→ `_determine_batch_execution_and_padding`(`:3771`)→ `_build_attention_metadata`(`:2191`)→ `set_forward_context`(`forward_context.py:250`,线程局部)→ `_model_forward` → `compute_logits` → 返回 None(延迟)→ `sample_tokens`(`:4384`)。异步调度(`use_async_scheduling`,`:504`)用 `async_output_copy_stream`(`:696`)让 step N+1 调度与 step N 执行重叠。

### 6.3 IPC 对比

| | nano | vLLM v1 |
|---|---|---|
| 传输 | 1 块 `SharedMemory("nanovllm",2**20)` + 每 worker `mp.Event` | 每 rank worker 进程 + ZMQ `MessageQueue`(分块)+ NCCL `torch.distributed` |
| 数据 | pickle 整个 `Sequence` 列表(定制 `__getstate__` 瘦身) | 结构化 `SchedulerOutput` diff(块 id、`num_scheduled_tokens`),worker 用本地 `InputBatch` 重建 |
| 分布式初始化 | `dist.init_process_group(nccl/hccl, "tcp://localhost:2333", ...)`(`model_runner.py:45`)硬编码端口 | `init_distributed_environment(..., backend, timeout)`(`gpu_worker.py:1186`)+ 动态 `tcp://<loopback>:<open_port>`(`multiproc_executor.py:128`) |
| 模式 | 锁步 RPC(所有 rank 同一 `run` 同步执行) | 异步广播/应答 + EngineCore 解耦独立进程 |

### 6.4 KV 缓存布局

nano:`torch.empty(2, num_layers, num_blocks, block_size=256, kv_heads, head_dim)`(`model_runner.py:138`)——单一大张量,K 在 `[0]`、V 在 `[1]`、层在 dim1;每 attention 模块挂 `module.k_cache = kv_cache[0, layer_id]`(`:142`)。`slot_mapping = block_idx*block_size+offset`,`-1` 表无效。

vLLM v1:分配与形状解耦——`_allocate_kv_cache_tensors`(`gpu_model_runner.py:6999`)分配 raw `int8` buffer,`_reshape_kv_cache_tensors`(`:7040`)按后端 `get_kv_cache_shape` + `get_kv_cache_stride_order` 视图化(`:7093`),支持 NHD/HND 物理布局与 hybrid block。FA 布局 `(num_blocks, 2, block_size, kv_heads, head_size)`(`flash_attn.py:149`),无 `num_layers` 维(每层独立 `bind_kv_cache` `:7248`)。block_size 可配(默认 16/32),FA 要求 16 倍数(`flash_attn.py:94`)。`slot_mapping` 由 **Triton kernel** `_compute_slot_mapping_kernel`(`block_table.py:325`)计算。

### 6.5 关键差异小结

- nano 280 行单类融合 executor/worker/runner;vLLM 五层 7561 行 `GPUModelRunner`。
- nano 调度器在进程内同步;vLLM 异步 EngineCore↔worker 流水 + 异步调度。
- nano IPC:SharedMemory+Event 锁步;vLLM:ZMQ `MessageQueue` + NCCL + 独立 worker 进程。
- nano 仅 Qwen3 硬编码(`model_runner.py:54`);vLLM 模型无关(registry/qualname)。
- vLLM worker 有内存 profiling(`determine_available_memory`)、sleep/wake、weight-transfer、LoRA、KV connector、EPLB、profiler——nano 全无。
- vLLM 拆 `execute_model`(返回 None/IntermediateTensors)与 `sample_tokens`(延迟采样);nano 合并返回 token ids。

---

## 七、注意力(Attention)

### 7.1 后端体系

nano(`layers/attention.py`,222 行)单 `Attention(nn.Module)` 类 + 两模块级 helper。构造时 `try import flash_attn`(`:87-95`),不可用则回退 `F.scaled_dot_product_attention`(SDPA)。**无后端抽象、无 metadata builder、无 selector**——`forward` 内一个 `if self._use_flash_attn` 分支(`:103`)。注意力元数据从全局 `Context` 单例(`context.py:16`)读,不经参数传递。

vLLM v1(`v1/attention/`)可插拔后端:`AttentionBackend` ABC(`backend.py:55`)声明 `get_name`/`get_impl_cls`/`get_kv_cache_shape`/`supports_*`;`AttentionMetadataBuilder` ABC(`:523`);`AttentionImpl`(`:692`);`CommonAttentionMetadata`(`:360`);`AttentionCGSupport` 枚举(`:506`,ALWAYS/UNIFORM_BATCH/...);`get_attn_backend`(`selector.py:54`)经 `current_platform.get_attn_backend_cls`(`:121`)选择,~20 个后端实现(flash_attn/flashinfer/triton/rocm/flex/mamba/mla/cpu/...)。

### 7.2 KV 写入

nano `store_kvcache`(`attention.py:8`):纯 PyTorch 索引——`mask = slot_mapping != -1`,`k_flat[valid_slots] = key[mask]`。**无自定义 kernel**,NPU 兼容。

vLLM v1 `do_kv_cache_update`(`flash_attn.py:850`):调 `reshape_and_cache_flash(key, value, k_cache, v_cache, slot_mapping, kv_cache_dtype, _k_scale, _v_scale)`——**自定义 CUDA/Triton op**,支持 FP8 scale。

### 7.3 SDPA 回退路径

nano SDPA 回退含 **Python 双重循环**:`_sdpa_prefill`(`:134`)逐序列 `for` over `cu_seqlens_q`,`_sdpa_decode`(`:194`)`for b in range(batch_size)` gather 各块 + `repeat_interleave` 扩 GQA + `F.scaled_dot_product_attention`。大 batch 下是性能瓶颈——NPU 兼容性的代价。flash 路径用 `flash_attn_varlen_func`(`:115`)与 `flash_attn_with_kvcache`(`:127`)。

vLLM v1 始终用融合 `flash_attn_varlen_func`(`flash_attn.py:796`),无 Python 循环;支持 cascade attention(`:822`)、encoder/cross-attention(`:724`)。

### 7.4 关键差异小结

- 1 后端(nano)vs ~20 可插拔后端 + 平台选择器(vLLM)。
- 纯 PyTorch KV 写入(nano)vs 自定义 `reshape_and_cache_flash` CUDA/Triton op(vLLM)。
- nano SDPA 回退 Python 循环逐 batch;vLLM 始终融合 varlen kernel。
- nano KV 布局 `[2, layers, blocks, 256, kv_heads, head_dim]` block_size 256;vLLM 每层 `(blocks, 2, block_size, kv_heads, head_size)` 可配 block_size + 后端控 stride + hybrid block。
- nano 元数据 = 全局 `Context` 单例;vLLM 每层 `AttentionMetadata` 经 builder 构建、`set_forward_context` 传递。
- nano 无 cascade/sliding-window/FP8-KV/DCP/PCP/MLA/encoder/Mamba/linear attention;vLLM 全有。

---

## 八、CUDA 图 / 编译(CUDA Graph / Compilation)

### 8.1 图捕获

nano `capture_graph`(`model_runner.py:245`):`graph_bs = [1,2,4,8] + range(16, max_bs+1, 16)`(`:257`),逆序(大先)捕获共享 `graph_pool`(`:268`),仅 **decode** 用,预热一次后 `graph_context(graph, pool, device_type)` 捕获(`:265`)。**NPU 禁用**(`is_graph_available` 返回 False,`device.py:71`)。replay(`run_model:222`):选最小 `bs >= actual` 的图,拷输入进 `graph_vars`,`slot_mapping.fill_(-1)` 覆写,`graph.replay()`。prefill **始终 eager**。

vLLM v1:`CUDAGraphMode` 枚举 NONE/PIECEWISE/FULL/FULL_DECODE_ONLY/FULL_AND_PIECEWISE(`config/compilation.py:53`);`CudagraphDispatcher`(`v1/cudagraph_dispatcher.py:15`)预算 `(mode, BatchDescriptor)` 键(`:170`),`dispatch`(`:239`)运行时返回;`CUDAGraphWrapper`(`compilation/cuda_graph.py:145`)按 `forward_context` 捕获/重放;`capture_model`(`gpu_model_runner.py:6525`)经 `get_capture_descs`(`:6552`)。支持 encoder cudagraph、breakable cudagraph、ubatch cudagraph。

### 8.2 torch.compile

nano `optional_compile`(`compile.py:15`,32 行):检测 `torch_npu` 则**降级 no-op**(`:25-28`,因"Triton inductor 在 NPU 不稳定"),CUDA 则 `torch.compile(fn)`。仅装饰 `Sampler.forward`(`sampler.py:8`)与 `RotaryEmbedding.forward`(`rotary_embedding.py:38`)。无动态形状标记、无 AOT 缓存、无自定义 pass、无与图集成。

vLLM v1:`support_torch_compile` 装饰器(`decorators.py:86`)装饰模型片段,`VllmBackend`(`piecewise_backend.py`)在 attention 处分段 FX 图,非 attention 段用 Inductor + 自定义 pass(`passes/`)编译,attention 段保持 eager;AOT 编译落盘 `VLLM_CACHE_ROOT`(缓存,`decorators.py:545`);`maybe_use_cudagraph_partition_wrapper`(`:725`)把编译分区包进 `CUDAGraphWrapper(PIECEWISE)`;`optimization_level` O0-O3(`vllm.py:78`)预设整套编译+图+kernel 栈。

### 8.3 关键差异小结

- nano 固定 bs 列表整模型图 + `next(bs>=actual)` 派发;vLLM `CudagraphDispatcher` 多模式 + LoRA 特化键 + `BatchDescriptor` 键控。
- nano 图在 NPU 禁用(`is_graph_available`);vLLM 按 `CUDAGraphMode` + 后端支持门控(非按设备类型)。
- nano `torch.compile` 是 2 函数上的可选 no-op;vLLM 完整 Inductor + 自定义 pass + AOT 缓存 + 分段编译 + 图集成。
- nano 无分段编译/breakable/encoder/ubatch cudagraph;vLLM 全有。

---

## 九、张量并行与 IPC

### 9.1 并行维度

nano 仅 TP(`config.py:15`,断言 `1<=TP<=8`)。层级用 Megatron 式 column/row(`layers/linear.py`),进程级用 `mp.spawn` + SharedMemory/Event。无 PP、DP、CP。

vLLM v1:`parallel_config` 含 TP×PP×DP×PCP×DCP;`init_worker_distributed_environment`(`gpu_worker.py:1162`)→ `ensure_model_parallel_initialized(TP, PP, PCP, DCP)`(`:1195`);有 PP 的 `AsyncIntermediateTensors`(`irecv/isend_tensor_dict`,`:851-893`)、DP 协调器、自定义 all-reduce(`set_custom_all_reduce` `:1178`)、异步 TP(comm fusion,`config/compilation.py:133`)、KV transfer/connector group。

### 9.2 层级 TP(详见第十节)

两者概念相同(Megatron column 输出维切分无 all_reduce,row 输入维切分末尾 all_reduce,bias 仅 rank0),但 vLLM 量化感知、KV 头复制、fused-on-disk 处理等远超 nano。

### 9.3 关键差异小结

- nano TP = 层级并行 + `mp.spawn` + SharedMemory/Event + 硬编码 `tcp://localhost:2333`;vLLM = executor spawn worker 进程 + 动态 `torch.distributed` + ZMQ `MessageQueue` + 通用 TP×PP×DP×PCP×DCP。
- nano 无 PP(无 IntermediateTensors/irecv/isend);vLLM 全异步中间张量传输。
- nano 无 DP/CP/异步 TP/自定义 all-reduce/KV connector;vLLM 全有。
- nano rank0 是进程内调度器+驱动+采样器;vLLM rank0 worker 与 EngineCore 调度器分进程,仅经 ZMQ 通信,支持异步调度。
- nano 每步 pickle 整 `Sequence` 列表;vLLM 发结构化 `SchedulerOutput` diff,worker 本地重建输入。

---

## 十、线性层 / 张量并行(Linear / TP)

### 10.1 架构

nano `layers/linear.py`(156 行)扁平文件,`LinearBase` + `Replicated/ColumnParallel/MergedColumnParallel/QKVParallel/RowParallelLinear`。纯 `torch.distributed`,无量化、无"linear method"。权重加载算术式:`.narrow()`/`.chunk()` 按 `tp_rank`;`param.weight_loader` 手工绑定(`linear.py:26`)。

vLLM `model_executor/layers/linear.py`(1571 行)+ 参数类型体系(`BasevLLMParameter`/`ModelWeightParameter`/`PackedColumnParameter`/`RowvLLMParameter`...)。`LinearBase` 是 `PluggableLayer`(`:228`),持 `quant_method`(`:268`),GEMM 委托 `LinearMethodBase`(Unquantized/FP8/Marlin/AWQ/GPTQ/...)。两代 weight_loader:v1 手动 narrow/chunk vs v2 `param.load_*_weight`(类型参数自管),经 `WEIGHT_LOADER_V2_SUPPORTED`(`:45`)选择。

### 10.2 Megatron 方案(两侧相同)

- Column 并行:输出维切分,无 all_reduce。nano `output_size//tp_size`,`tp_dim=0`(`:62`);vLLM `output_partition_sizes`(`:451`)。
- Row 并行:输入维切分,末尾 all_reduce;bias 仅 rank0。nano `dist.all_reduce(y)`(`:154`);vLLM `tensor_model_parallel_all_reduce`(`:1552`)。**逻辑一致**。
- Merged gate/up:两等 output 拼接。nano `output_sizes: list[int]`(`:81`);vLLM 同(`:644`)。
- QKV 融合:q/k/v 按头切。

### 10.3 权重加载差异

nano `packed_modules_mapping` 映射 **HF 源名 → (融合参数名, shard_id)**(`qwen3.py:187`):`"q_proj": ("qkv_proj","q")`。loader 匹配后改名并调 `param.weight_loader(param, w, shard_id)`。`QKVParallelLinear.weight_loader`(`linear.py:114`)硬编码 q/k/v 偏移 + `chunk(tp_size)[tp_rank]`;**断言 `kv_heads % tp_size == 0`**,无 `v_head_size != head_size`,无 KV 头复制。

vLLM `packed_modules_mapping` 映射 **融合名 → [HF 源名]**(`qwen3.py:264`):`"qkv_proj": ["q_proj","k_proj","v_proj"]`,由 `AutoWeightsLoader`(`qwen3.py:335`)按位置推导 shard_id。`QKVParallelLinear`(`:972`)支持 `v_head_size`(`:1014`);`tp_size >= kv_heads` 时 `num_kv_head_replicas = tp_size // kv_heads` **复制 KV 头**(`:1026`);处理 fused-on-disk(Phi-3 风格)、block-quant scale、packed int4(AWQ)、Marlin tile、bitsandbytes 4-bit 偏移调整(`:70-108`)。

### 10.4 关键差异小结

- 156 行无量化(nano)vs 1571 行 + 参数体系 + ~15 量化方法 + CPU oneDNN/AMX/zentorch + ROCm aiter/tgemm GEMM 派发(vLLM)。
- nano 手工绑 `param.weight_loader`;vLLM `set_weight_attrs` + 类型化 `BasevLLMParameter`。
- nano 仅 v1 式手动 narrow/chunk;vLLM v1 + v2(`param.load_*_weight`)两代,按量化方法选。
- nano 断言 `kv_heads%tp==0`、无 `v_head_size`;vLLM 复制 KV 头 + 支持异 `v_head_size`。
- vLLM 检测 fused-on-disk 权重并拆分;nano 不能(checkpoint 必须含独立 q/k/v/gate/up)。
- vLLM `forward` 返回 `(output, bias)` 支持 `skip_bias_add` 融合;nano 返回裸 tensor。
- vLLM 暴露 `gather_output`/`reduce_results`/`input_is_parallel`/`disable_tp`;nano 全硬编码。

---

## 十一、采样器(Sampler)

### 11.1 算法

nano `Sampler`(`sampler.py:6`,13 行)单 `forward(logits, temperatures)`:
```
logits.div_(temperatures)      # 温度
probs = softmax(logits)
sample = probs.div_(empty_like.exponential_(1).clamp_min(1e-10)).argmax(-1)
```
即 **Gumbel-max**:`argmax(softmax(logits/T)/Exp(1)) == argmax(logits/T + Gumbel(0,1))`。仅 rank0 采样(`model_runner.py:241`)。`SamplingParams` 禁 greedy(`sampling_params.py:11`)。

vLLM v1 `Sampler`(`v1/sample/sampler.py:20`,439 行)9 步流水(raw logprobs 捕获 → float32 → allowed-token 白名单 → bad-words → 非argmax不变 processor(MinTokens/LogitBias)→ penalty → greedy-or-random 采样 → top-k logprobs gather → `SamplerOutput`)。random 路径核心与 nano 相同:`topk_topp_sampler.py:331` `q.exponential_(); probs.div(q).argmax(-1)`。**所以 nano 本质是 vLLM `forward_native` 减去 top-k/top-p/penalty/processor/logprobs/greedy,限 rank0。**

### 11.2 能力差异

| 能力 | nano | vLLM v1 |
|---|---|---|
| temperature | 是 | 是(混合 greedy+random 批,`:296`) |
| greedy | **禁**(`:11`) | 是(`argmax`,`:239`) |
| top-k/top-p/min-p | 无 | `TopKTopPSampler`(FlashInfer/aiter/Triton/native/cpu/xpu) |
| penalty(repetition/freq/presence) | 无 | `apply_all_penalties` |
| bad-words/allowed-token-id | 无 | 有 |
| logit bias/min-tokens | 无 | `MinTokens`/`LogitBias` processor |
| 结构化输出 | 无 | grammar 后端 |
| thinking-budget | 无 | `ThinkingBudgetStateHolder` |
| 投机采样拒绝 | 无 | `RejectionSampler`(`rejection_sampler.py:37`) |
| logprobs | 无 | raw/processed/top-k/per-token |
| 后端派发 | `@optional_compile` | `forward_cuda/hip/cpu/xpu` |
| 元数据 | 1-D `temperatures` | `SamplingMetadata`(~17 字段) |
| 输出 | 1-D token ids | `SamplerOutput(sampled_token_ids, logprobs_tensors)` |

---

## 十二、基础计算层(RMSNorm / Rotary / Activation / Embedding)

### 12.1 RMSNorm

nano `layernorm.py`(51 行):`RMSNorm` + 融合 `add_rms_forward`(残差融合,返回 `(out, residual)`),均 `@optional_compile`,纯 PyTorch upcast float32。

vLLM `layernorm.py`(328 行):`RMSNorm(CustomOp)` 派发 `ir.ops.rms_norm`/`fused_add_rms_norm`(融合 CUDA/HIP kernel)+ `GemmaRMSNorm`/`RMSNormGated`/`LayerNorm`/`poly_norm` + batch-invariant 模式 + `var_hidden_size` 覆盖 + `has_weight=False`。**残差融合 API 契约一致**(返回 `(out, residual)`),两侧 Qwen3 decoder 都依赖。

### 12.2 Rotary

nano `rotary_embedding.py`(60 行):单 NeoX `RotaryEmbedding` + `apply_rotary_emb` + `@lru_cache(1) get_rope`;**断言 `rotary_dim==head_size`**(`:29`);只读 `rope_scaling.get("rope_theta")`(`qwen3.py:54`),忽略 `rope_type`/`factor`——无法 YaRN/NTK/longrope 缩放。

vLLM `rotary_embedding/` 包(~17 变体:default/linear/NTK/dynamic-NTK/YaRN/deepseek-YaRN/longrope/llama3/mrope/fope/gemma4/xdrope/telechat3/dual_chunk/llama4_vision...)+ `base.py`(`RotaryEmbeddingBase`)+ `common.py`(`ApplyRotaryEmb` CustomOp)。支持部分 rotary、NeoX 与 GPT-J 两种、flashinfer/aiter 融合 kernel、跨层 KV 共享(`key=None`)、`enable_fp32_compute`。`get_rope` 按 `rope_parameters["rope_type"]` 派发。

### 12.3 Activation

nano `activation.py`(12 行):单 `SiluAndMul`,`forward`:`x,y=x.chunk(2,-1); F.silu(x)*y`,断言 `hidden_act=="silu"`(`qwen3.py:110`)。

vLLM `activation.py`(776 行):`SiluAndMul` `forward_native`(`:137`):`d=shape[-1]//2; F.silu(x[...,:d])*x[...,d:]`——**数学逐字节相同**。但加 `forward_cuda`(融合 `torch.ops._C.silu_and_mul`)+ ~12 变体(`SiluAndMulWithClamp`/`MulAndSilu`/`FatreluAndMul`/`GeluAndMul`/`SwigluOAIAndMul`/...)+ `get_act_and_mul_fn(name)` 注册表 + AWQ `ScaledActivation`。

### 12.4 Embedding / LMHead

nano `embed_head.py`(66 行):`VocabParallelEmbedding`(mask + `dist.all_reduce`)+ `ParallelLMHead`(**自己**做 logits matmul + gather 到 rank0 + prefill 末 token 选取)。断言 `vocab%tp_size==0`(`:19`),无 padding。prefill 取 `cu_seqlens_q[1:]-1` 末 token(`:59`);TP>1 `dist.gather(...,0)`(`:63`)。

vLLM 拆分:`VocabParallelEmbedding`(`vocab_parallel_embedding.py:192`,pad 到 64 + LoRA-added 分片 + 量化)+ `ParallelLMHead`(`:508`,**仅权重容器,`forward` 抛 RuntimeError** `:572`)+ `LogitsProcessor`(`logits_processor.py:19`,matmul + gather + scale + soft_cap)。Qwen3 用 `self.logits_processor(self.lm_head, hidden_states)`(`qwen3.py:332`)。支持 GGUF/packed-dim 加载、quant method、`get_top_tokens`(vocab 并行 argmax 免全 gather)。

### 12.5 关键差异小结

- RMSNorm:API 一致;nano 纯 PyTorch,vLLM 融合 kernel + 多变体。
- Rotary:nano 单 NeoX + 断言 `rotary_dim==head_size` + 无缩放;vLLM 17 变体 + `rope_type` 派发 + 部分 rotary + GPT-J + 融合 kernel + dual-chunk + YaRN/NTK/longrope。
- Activation:`SiluAndMul` 数学相同;nano 1 类 + 断言 silu;vLLM ~12 类 + 融合 kernel + 注册表。
- Embedding/LMHead:nano 合并权重+matmul+gather+prefill 选取,断言 `vocab%tp==0`;vLLM 拆 `ParallelLMHead`(权重,forward 抛错)+ `LogitsProcessor`(matmul+gather),pad 64,LoRA-added 分片,量化,GGUF,top-token 优化。

---

## 十三、模型定义(Qwen3)

### 13.1 结构

nano `models/qwen3.py`(216 行)单文件自包含:`Qwen3Attention`/`Qwen3MLP`/`Qwen3DecoderLayer`/`Qwen3Model`/`Qwen3ForCausalLM`。单架构、单 config 路径,无 MoE/quant/PP/LoRA/spec/multimodal。`forward(input_ids, positions)`,注意力元数据经全局 `Context` 传递。

vLLM `model_executor/models/qwen3.py`(340 行,复用 `Qwen2Model`/`Qwen2MLP`):`Qwen3ForCausalLM` 声明 `SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3`(`:261`)。另有 `qwen3_moe.py`/`qwen3_vl.py`/`qwen3_vl_moe.py`/`qwen3_5.py`/`qwen3_next.py`/`qwen3_asr*.py` 等覆盖整个 Qwen3 家族。`forward(input_ids, positions, intermediate_tensors, inputs_embeds)`,`attn_metadata` 显式传递。

### 13.2 结构相同处(忠实复刻)

- `Qwen3DecoderLayer.forward` 残差融合模式一致:`residual is None ? (residual=hs; hs=input_layernorm(hs)) : (hs,residual=input_layernorm(hs,residual)); hs=self_attn(...); hs,residual=post_attention_layernorm(hs,residual); hs=mlp(hs)`(nano `qwen3.py:146-159` vs vLLM `:216-236`)。
- `Qwen3Attention.forward` 形状一致:`qkv→split(q,k,v)→view heads→q_norm/k_norm→rotary→attn→o_proj`。
- `Qwen3MLP`:`gate_up_proj→SiluAndMul→down_proj`。
- `scaling = head_dim**-0.5`(nano `:39` vs vLLM `:95`);`max_position=4096*32`、`rope_theta` 默认 1000000。

### 13.3 nano 硬编码/缺失/风险

- **q/k norm 门控隐患**:nano 仅 `not qkv_bias` 时建 `q_norm`/`k_norm`(`qwen3.py:68-70`),而 `qkv_bias` 默认 `getattr(config,'attention_bias',True)`(`:133`)——若 config 无显式 `attention_bias=False`,nano **跳过 q/k norm**。真实 Qwen3 恒有 q/k norm;vLLM 无条件建(`:142`)。**正确性风险**。
- **KV 头 TP**:nano 断言 `kv_heads%tp_size==0`(`:34`);vLLM `kv_heads<tp_size` 时复制(`:82-91`)。
- **RoPE**:nano 仅 default NeoX + 断言 `rotary_dim==head_size` + 只读 `rope_theta`;vLLM 传全 `config.rope_parameters` 支持 YaRN/NTK/longrope/llama3(`:115-120`)。
- **量化/PP/LoRA/spec**:nano 无;vLLM 线程 `quant_config`/`prefix`、`PPMissingLayer`、`IntermediateTensors`、`AutoWeightsLoader`、`SupportsEagle*`。
- **multimodal/MoE/VL/ASR**:nano 仅 dense Qwen3;vLLM 数十变体。
- **logits 计算**:nano `compute_logits` 调 `self.lm_head(hidden)`(gather rank0,`:212`);vLLM 调 `self.logits_processor(self.lm_head, hidden)`(`:328`),权重与计算分离。
- **`packed_modules_mapping` 方向相反**:nano 源→(融合,shard);vLLM 融合→[源]。
- **compile**:nano 叶层 `@optional_compile`;vLLM 模型层 `@support_torch_compile(dynamic_arg_dims)`(`:244`)。

---

## 十四、配置(Config)

### 14.1 架构

nano `Config`(`config.py:6`,40 行,`@dataclass(slots=True)`)扁平 13 字段:`model, max_num_batched_tokens, max_num_seqs, max_model_len, device_type, device_id, memory_utilization, tensor_parallel_size, enforce_eager, hf_config, eos, kvcache_block_size, num_kvcache_blocks`。`__post_init__`(`:22`)载 HF `AutoConfig`、clamp `max_model_len`、跑 assert。`LLMEngine` 按 `Config` 字段名过滤 kwargs。nano 特色:`device_id: int|list[int]`(`:13`)同时支持偏移(`get_device_id` 返回 `device_id+rank`,`:40`)与显式列表(`:39`,断言长度匹配 TP),用于多卡 NPU/CUDA 指定占用。

vLLM `VllmConfig`(`config/vllm.py:295`,2264 行,pydantic `@config`)聚合 ~20 子配置:`model_config`/`cache_config`/`parallel_config`/`scheduler_config`/`device_config`/`load_config`/`compilation_config`/`lora_config`/`speculative_config`/`quant_config`/`structured_outputs_config`/`kv_transfer_config`/`observability_config`/`profiler_config`/`mamba_config`/`kernel_config`/`weight_transfer_config`...各住独立文件(`config/` 29 个)。`compute_hash()`(`:388`)哈希全部子配置供编译缓存;`optimization_level` O0-O3(`:78`)预设编译/图/kernel 栈;`set_current_vllm_config` contextvar 全局。

### 14.2 关键差异小结

- 1 dataclass 13 字段 40 行(nano)vs ~20 嵌套 pydantic 数千字段 29 文件(vLLM)。
- nano 裸 assert 校验;vLLM pydantic `model_validator`/`field_validator`。
- nano 无 hash;vLLM `compute_hash` 供编译缓存。
- nano `int|list` device-id 是 NPU 友好专用 helper;vLLM 经 Platform + env 解析物理 ID(`interface.py:234`)。
- nano 无量化/LoRA/spec/multimodal/structured-output/KV-transfer/offload/PP/DP/async/性能模式/优化级别。

---

## 十五、权重加载器(Weight Loader)

### 15.1 架构

nano `utils/loader.py`(28 行):`load_model(model, path)` glob 本地 `*.safetensors`,`safe_open(...,"pt","cpu")` 遍历。仅支持**本地 safetensors**。唯一变换是 `model.packed_modules_mapping`(`:13`)驱动的 packed 合并:匹配源名→改名融合目标→取参数上 `weight_loader` callable(`:22`)调 `weight_loader(param, w, shard_id)`;非 packed 走 `default_weight_loader`(`:8`,纯 `param.data.copy_`)。

vLLM `model_loader/`(15 模块):`BaseModelLoader` ABC(`base_loader.py:25`)+ `DefaultModelLoader`(`default_loader.py:43`,437 行)+ 15 种 `LoadFormat`(`__init__.py:34`:auto/hf/bitsandbytes/dummy/fastsafetensors/gguf/instanttensor/mistral/modelexpress/npcache/pt/runai_streamer/safetensors/sharded_state/tensorizer)+ `register_model_loader` 插件装饰器(`:72`)+ `Source` dataclass(`default_loader.py:49`,远程 HF/ModelScope 下载、多线程)+ 专用 loader(bitsandbytes/gguf/runai/tensorizer/sharded_state/ModelExpress)+ 在线量化 finalize(`base_loader.py:77`)+ `ep_weight_filter`(EP)。

### 15.2 共享约定

两者都把 `weight_loader` callable stashed 在 `nn.Parameter` 上,用 `packed_modules_mapping` 做 q/k/v 与 gate/up 融合——nano 是 vLLM 该机制的**最小复刻**。

### 15.3 关键差异小结

- 28 行(nano)vs 15 模块、`default_loader.py` 437 行(vLLM)。
- nano 仅本地 safetensors;vLLM 15 种 load format 含远程下载、GGUF、tensorizer、sharded state。
- nano 仅 FP/BF16;vLLM 有 bitsandbytes loader、`get_quant_config`、在线量化 finalize、量化 loader 子类。
- nano 无注册表;vLLM `register_model_loader` 插件。

---

## 十六、设备抽象 / 平台(Device / Platforms)

### 16.1 架构

nano `utils/device.py`(98 行)**自由函数模块**,无类、无插件、无枚举。每函数按 `device_type: str`("cuda"/"npu")派发:`get_device_module`(`:5`)→ `torch.cuda`/`torch.npu`;`get_dist_backend`(`:15`)→ nccl/hccl;`mem_get_info`(`:53`);`is_graph_available`(`:71`);`create_graph`/`graph_context`(`:79,88`);`set_device`(`:30`);`init_distributed`(`:94`)。

vLLM `platforms/`(8 模块)**类层次**:`Platform`(`interface.py:105`,1057 行)+ `PlatformEnum`(CUDA/ROCM/TPU/XPU/CPU/OOT/UNSPECIFIED)+ `CudaPlatform`/`RocmPlatform`/`TpuPlatform`/`XpuPlatform`/`CpuPlatform`/`ZenCpuPlatform` + 插件自动探测(`__init__.py:36`,`current_platform` 单例)。`Platform` 暴露 ~40 方法:`get_device_capability`、`get_attn_backend_cls`、`is_sleep_mode_available`、`get_compile_backend`、`get_pass_manager_cls`、`import_ir_kernels`、`device_id_to_physical_device_id`(`:234`)等。**此 checkout 无 NPU 平台**。

### 16.2 NPU 适配(nano 新增,共 8 处)

1. `device.py:9-12` `get_device_module` 返回 `torch.npu`——所有内存/图/同步经此路由。
2. `device.py:19-20` `get_dist_backend` 返回 `hccl`(华为集合通信库)。
3. `device.py:71-76` `is_graph_available` 对 NPU 返 **False**——`ModelRunner` 跳过图捕获,eager 运行。
4. `compile.py:6-12,25-28` `optional_compile` 检测 `torch_npu` 降级 **no-op**(Triton inductor 不稳定)。
5. `attention.py:8-26,86-95,134-222` `store_kvcache` 纯 PyTorch(去 Triton);`try import flash_attn`,不可用则 SDPA 回退 `_sdpa_prefill`/`_sdpa_decode` + Python 循环 `_gather_kv_from_cache`——NPU 上无 flash-attn/Triton paged kernel 时的保底。
6. `config.py:13,26-30,36-40` `device_type` 接受 `"npu"`;`device_id: int|list` 支持显式多卡映射。
7. `pyproject.toml:22-29` `npu` extra 钉 `torch-npu`;`cuda` extra 钉 `triton`+`flash-attn`。
8. `v1/run_api_server.py:184` 与 `example_npu.py:18` `--device-type` **默认 `npu`**(上游 nano-vllm 默认 cuda)。

vLLM 加 NPU 需新建 `platforms/npu.py` `Platform` 子类 + 注册 `PlatformEnum` + HCCL communicator + attention backend + IR kernel——远大于 nano 的 98 行。

### 16.3 关键差异小结

- 98 行自由函数(nano)vs 8 模块类层次 + 插件自动探测 + `current_platform` 单例(vLLM)。
- nano 支持 NPU 端到端;vLLM 无 NPU。
- nano 硬编码 flash-or-SDPA;vLLM `Platform.get_attn_backend_cls` + 后端注册表。
- nano 忽略 compute capability;vLLM 建 `DeviceCapability` 门控量化。
- vLLM 平台还管 IR-kernel 导入、compile backend、sleep mode、pass manager——nano 全无。

---

## 十七、编译(Compilation)

(与第八节 CUDA 图/编译内容一致,此处归纳编译视角。)

- nano `torch.compile` = `optional_compile`,仅 `Sampler` + `RotaryEmbedding`,NPU 降级 no-op;无 Inductor/pass/AOT 缓存/分段。
- vLLM `compilation/` 15 模块:`CompilationConfig`(`config/compilation.py`,1525 行)+ `CompilationMode`(NONE/STOCK_TORCH_COMPILE/DYNAMO_TRACE_ONCE/VLLM_COMPILE)+ `CUDAGraphMode`(5 种)+ `PassConfig`(fuse_norm_quant/fuse_act_quant/fuse_allreduce_rms/fuse_rope_kvcache/fuse_gemm_comms...)+ `passes/` 自定义 Inductor pass 包 + O0-O3 预设。
- nano 无优化级别;vLLM O0-O3 驱动整套 compile+graph+kernel 栈。

---

## 十八、API 服务 / 前端

### 18.1 架构

nano `v1/run_api_server.py`(211 行):`create_app`(`:41`)建 FastAPI,4 路由:`GET /health`(`:71`)、`GET /v1/models`(`:75`)、`POST /v1/completions`(`:89`)、`POST /v1/chat/completions`(`:125`)。启动时建一个 `LLM`(`:53`)+ tokenizer。每 handler `async def` 但**阻塞调** `llm.generate([prompt], sp)`(`:102,141`)——同步非流式,单请求占满引擎直至完成,无跨请求连续批。`uvicorn.run`(`:207`)。`ChatCompletionRequest` 仅 `messages`+`sampling_params`;`SamplingParamsRequest` 仅 `temperature/max_tokens/ignore_eos`;响应固定 shape,`finish_reason:"stop"` 硬编码。无 `stream`/`n`/`logprobs`/`tools`/`response_format`/`seed`/`stop`/`top_p`。

vLLM `entrypoints/openai/`(11+ 入口):`api_server.py`(705 行)`build_app`(`:156`)挂载多 router(chat/completion/batch/responses/generate/models/tokenization/disagg/elastic_ep/render/speech-to-text/pooling)+ CORS/auth/request-id/scaling 中间件 + uvloop + 异步 `EngineClient`(`build_async_engine_client`,`:76`)。`OpenAIServingChat`(`chat_completion/serving.py:83`,1469 行)/`OpenAIServingCompletion`(`completion/serving.py:54`,690 行)。`OpenAIServingModels`/`OpenAIServingRender`/`OpenAIServingTokenization`、`ToolParserManager`/`ReasoningParserManager`。离线 API `entrypoints/llm.py`(`LLM`,`:66`,913 行,mixin `BeamSearchOfflineMixin`/`PoolingOfflineMixin`/`OfflineInferenceMixin`)。

### 18.2 并发/批处理

nano:handler 阻塞 `llm.generate`,FastAPI 把同步代码丢线程池但引擎持单 GPU,**并发 HTTP 请求串行**——无跨请求连续批。引擎内批仅限单次 `generate()` 的多 prompt。

vLLM:`OpenAIServingChat` 提交每 `engine_input` 到异步引擎,返回 `AsyncGenerator[RequestOutput]`,非流式聚合成 `ChatCompletionResponse` 或流式 `chat_completion_stream_generator`(`serving.py:399`)。**异步引擎跨 HTTP 连接连续批**新请求进运行调度器,逐 token SSE 流式 + detokenization + tool/reasoning delta 解析 + usage 统计。

### 18.3 关键差异小结

- 单请求阻塞、无跨请求批(nano)vs 异步引擎跨请求连续批(vLLM)。
- 无流式(nano,恒 JSON)vs SSE 流式 + detokenizer 流式 + tool/reasoning delta(vLLM)。
- 4 路由(nano)vs chat/completion/batch/responses/generate/models/tokenization/disagg/pooling/speech-to-text/dev(vLLM)。
- 无 tools/reasoning/structured-output(nano)vs `ToolParserManager`/`ReasoningParserManager`/结构化后端(vLLM)。
- 无中间件(nano)vs CORS/auth/request-id/scaling/exception(vLLM)。
- 普通 uvicorn(nano)vs uvloop + `serve_http` launcher + forkserver 预导入(vLLM)。
- nano `LLM.generate` 30 行 tqdm;vLLM `LLM` 913 行 + beam search/pooling/chat/multimodal。

---

## 十九、特性支持矩阵

| 特性 | nano-vllm-npu | 上游 vLLM v1 |
|---|---|---|
| 前缀缓存 | **是** — xxhash 链 `hash_to_block_id`,token 二次校验(`block_manager.py:36,66`) | **是** — `enable_prefix_caching` 默认开,sha256(可配),extra_keys |
| Chunked prefill | **受限** — 仅队首 seq 可 chunk(`scheduler.py:42`) | **是** — 默认开,任意请求可 chunk,可配置阈值 |
| Continuous batching | **受限** — 单 `generate()` 内批;API 不跨请求批 | **是** — 异步引擎跨 HTTP 请求连续批 |
| CUDA graph | **仅 CUDA** — decode 专用,固定 bs,NPU 禁用 | **是** — FULL/PIECEWISE/...,prefill+decode,可配 |
| torch.compile | **极简** — `optional_compile` 仅采样器,NPU no-op | **是** — Inductor + pass + AOT + 分段,O0-O3 |
| 张量并行 TP | **是** — Megatron 式,TP 1-8,spawn+SharedMemory | **是** — TP + 全通信器 |
| 流水并行 PP | **否** | **是** |
| 数据并行 DP | **否** | **是** + DP 协调器 |
| 专家并行 EP / MoE | **否**(仅 dense Qwen3) | **是** + EPLB |
| 投机采样 | **否** | **是** — Eagle/n-gram/DFlash/MTP |
| 量化 | **否** — 仅 BF16/FP16 | **是** — awq/gptq/fp8/bitsandbytes/CompressedTensors |
| LoRA | **否** | **是** |
| 多模态 | **否** — 仅文本 | **是** — vision/audio/speech-to-text |
| 结构化输出 | **否** | **是** — grammar 后端 |
| 工具调用 | **否** | **是** — `ToolParserManager` |
| 推理解析 | **否** | **是** — `ReasoningParserManager` |
| Encoder-decoder | **否** | **是** |
| Sleep / wake | **否** | **是** — KV offload |
| KV transfer(P-D 分离) | **否** | **是** — `KVTransferConfig`/`ECTransferConfig` |
| 异步调度 | **否** — 同步 step 循环 | **是** — `async_scheduling` |
| 流式 SSE | **否** — 仅 JSON | **是** |
| Batch API | **否** | **是** |
| Responses API | **否** | **是** |
| 支持模型数 | **1** — Qwen3ForCausalLM | **数百** |
| 平台 | **2** — CUDA、NPU(Ascend) | **6+** — CUDA/ROCm/TPU/XPU/CPU/Zen(+OOT),无 NPU |

---

## 二十、NPU 适配差异总结

nano-vllm-npu 相对上游 vLLM 的**独有价值**在于 Ascend NPU 端到端支持,通过以下取舍实现:

| 适配点 | 做法 | 代价 |
|---|---|---|
| 设备抽象 | `device.py` 自由函数派发 cuda/npu | 无平台类/插件体系,扩展新后端需手改 |
| 通信后端 | HCCL(NPU)/NCCL(CUDA) | 仅 TP,无 PP/DP/CP |
| CUDA graph | NPU 禁用(`is_graph_available=False`) | NPU 无图加速,decode 走 eager |
| torch.compile | NPU 降级 no-op | NPU 无 Inductor 优化 |
| 注意力 | flash-attn 不可用时 SDPA 回退 + 纯 PyTorch KV 写入 | SDPA 回退含 Python 循环,大 batch 性能瓶颈 |
| 块大小 | 256(大块) | 块表项少、索引简单,但粒度粗、内存碎片高 |
| 前缀缓存 | xxhash + token 校验 + 无 LRU | 离线批量可接受,在线服务无驱逐控制 |

上游 vLLM 在此 checkout **无任何 NPU 代码路径**;加 NPU 需新建 `Platform` 子类 + HCCL communicator + attention backend + IR kernel,工程量远超 nano 的 98 行 `device.py`。

---

## 二十一、差距与改进建议

针对 nano-vllm-npu 若要向生产级演进,可参考上游 vLLM 逐步补齐(按性价比排序):

1. **q/k norm 门控修正**(`qwen3.py:68-70`):无条件建 `q_norm`/`k_norm`,消除 `attention_bias` 默认值导致的正确性风险——**最高优先,纯 bugfix**。
2. **任意请求 chunked prefill**:解除"仅队首"限制,允许 prefill/decode 同 batch 混合,提升吞吐。
3. **LRU 前缀缓存驱逐**:`FreeKVCacheBlockQueue` + touch,支撑在线服务长时运行。
4. **per-chunk 块分配**:`can_allocate` 改为按 chunk 分配,允许超长 prompt 准入。
5. **KV 头复制**:`tp_size > kv_heads` 时复制,支持小 KV 头模型多卡。
6. **流式 API + 跨请求连续批**:异步引擎 + SSE,转向在线服务。
7. **更多采样能力**:top-k/top-p/greedy/logprobs。
8. **NPU graph/compile**:随 torch_npu 版本成熟逐步启用图捕获与 inductor。
9. **量化 / LoRA / spec decode**:按业务需求引入。

---

## 附:模块关系速查

| 关注点 | nano 位置 | vLLM v1 对应 |
|---|---|---|
| 引擎主循环 | `engine/llm_engine.py:49,60` | `v1/engine/core.py:443` + `llm_engine.py:287` |
| 调度 | `engine/scheduler.py:25,75,81` | `v1/core/sched/scheduler.py:340` |
| 序列/请求 | `engine/sequence.py:14` | `v1/request.py:59` |
| 块/KV/前缀缓存 | `engine/block_manager.py:35,58,75,110` | `v1/core/kv_cache_manager.py:110` + `block_pool.py:130` |
| 执行器 | `engine/model_runner.py:29` | `v1/worker/gpu_worker.py:117` + `gpu_model_runner.py:418` |
| 注意力 | `layers/attention.py:8,97` | `v1/attention/backends/flash_attn.py:68` |
| 线性/TP | `layers/linear.py:12,96,131` | `model_executor/layers/linear.py:228` |
| 采样 | `layers/sampler.py:6` | `v1/sample/sampler.py:20` |
| 模型 | `models/qwen3.py:195` | `model_executor/models/qwen3.py:281` |
| 配置 | `config.py:7` | `config/vllm.py:295` |
| 权重加载 | `utils/loader.py:12` | `model_loader/default_loader.py:43` |
| 设备抽象 | `utils/device.py` | `platforms/interface.py:105` |
| 编译 | `utils/compile.py:15` + `model_runner.py:245` | `compilation/decorators.py:86` + `cuda_graph.py:145` |
| API 服务 | `v1/run_api_server.py:41` | `entrypoints/openai/api_server.py:156` |
