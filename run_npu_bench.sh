#!/bin/bash
# ============================================================
# nano-vllm NPU 性能测试脚本 (Ascend)
# ============================================================
set -e

# ---------- Ascend NPU 环境变量 ----------
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME}/cann-8.5.1
export LD_LIBRARY_PATH=${ASCEND_CANN_PACKAGE_PATH}/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${LD_LIBRARY_PATH}
export PATH=${ASCEND_CANN_PACKAGE_PATH}/bin:${PATH}
export PYTHONPATH=${ASCEND_CANN_PACKAGE_PATH}/python/site-packages:${PYTHONPATH}
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=1

# ---------- 配置 ----------
MODEL_PATH=${MODEL_PATH:-"/data2/models/Qwen3-0.6B"}
DEVICE_TYPE=${DEVICE_TYPE:-"npu"}
MEMORY_UTIL=${MEMORY_UTIL:-"0.9"}
TP_SIZE=${TP_SIZE:-"1"}
DEVICE_ID=${DEVICE_ID:-"0"}

# ---------- 性能测试 ----------
echo "========================================"
echo " nano-vllm NPU Benchmark"
echo "========================================"
echo " MODEL:           ${MODEL_PATH}"
echo " DEVICE_TYPE:     ${DEVICE_TYPE}"
echo " DEVICE_ID:       ${DEVICE_ID}"
echo " MEMORY_UTIL:     ${MEMORY_UTIL}"
echo " TP_SIZE:         ${TP_SIZE}"
echo "========================================"

python -u -c "
import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams

def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024

    path = '${MODEL_PATH}'
    print(f'Loading model from {path}...')

    llm = LLM(
        path,
        device_type='${DEVICE_TYPE}',
        device_id=${DEVICE_ID},
        memory_utilization=${MEMORY_UTIL},
        tensor_parallel_size=${TP_SIZE},
        max_model_len=4096,
    )
    print('Model loaded. Running warmup...')

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len))
        for _ in range(num_seqs)
    ]

    # warmup
    llm.generate(['Benchmark: '], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f'Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s')

if __name__ == '__main__':
    main()
"
