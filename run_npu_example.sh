#!/bin/bash
# ============================================================
# nano-vllm NPU API Server 启动脚本 (Ascend)
# ============================================================
set -x

# ---------- Ascend NPU 环境变量 ----------
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_TOOLKIT_HOME=${ASCEND_HOME}/ascend-toolkit/latest
export ASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME}/cann-8.5.1
export LD_LIBRARY_PATH=${ASCEND_CANN_PACKAGE_PATH}/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${LD_LIBRARY_PATH}
export PATH=${ASCEND_CANN_PACKAGE_PATH}/bin:${PATH}
export PYTHONPATH=${ASCEND_CANN_PACKAGE_PATH}/python/site-packages:${PYTHONPATH}

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=1

# ---------- 启动参数 ----------
MODEL=${MODEL:-"/data2/models/Qwen3-32B"}
PORT=${PORT:-"8006"}
DEVICE_TYPE=${DEVICE_TYPE:-"npu"}
MEMORY_UTIL=${MEMORY_UTIL:-"0.9"}
TP_SIZE=${TP_SIZE:-"2"}
DEVICE_ID=${DEVICE_ID:-"4,5"}
ENFORCE_EAGER=${ENFORCE_EAGER:-"false"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"4096"}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-"256"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"qwen-32b"}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python ${SCRIPT_DIR}/nanovllm/v1/run_api_server.py \
    --model ${MODEL} \
    --port ${PORT} \
    --device-type ${DEVICE_TYPE} \
    --device-id ${DEVICE_ID} \
    --memory-utilization ${MEMORY_UTIL} \
    --tensor-parallel-size ${TP_SIZE} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --served-model-name ${SERVED_MODEL_NAME} \
    $(if [ "${ENFORCE_EAGER}" = "true" ]; then echo "--enforce-eager"; fi)
