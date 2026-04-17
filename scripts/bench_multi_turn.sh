#!/usr/bin/env bash
# bench_multi_turn.sh — 多轮对话 agent 场景压测脚本（ShareGPT）
#
# 用法：
#   ./bench_multi_turn.sh [--clients N] [--conversations N] [--turns N] [--dry-run]
#
# 前置：vLLM 已启动，模型可通过 $VLLM_URL 访问
# 结果：/lssd/logs/multi_turn_<编号>.log + /lssd/logs/multi_turn_<编号>.xlsx

set -euo pipefail

# ────────────────────────────── 配置 ──────────────────────────────
VLLM_URL="http://localhost:8088"
MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507"   # vLLM --served-model-name

WORK_DIR="/lssd/multi_turn_bench"
LOG_DIR="/lssd/logs"
SCRIPT_DIR="${WORK_DIR}/scripts"
DATA_DIR="${WORK_DIR}/data"

# benchmark_serving_multi_turn.py 参数默认值
NUM_CLIENTS=10                  # 并发 client 数（模拟并发用户数）
MAX_ACTIVE_CONVERSATIONS=30     # 同时活跃的最大会话数
MAX_TURNS=20                    # 每个会话最大轮次数（截断 ShareGPT 长对话）
MAX_NUM_REQUESTS=200            # 总请求轮次数（达到后停止）
REQUEST_RATE=0                  # 0=尽快发送（saturate），>0=Poisson 到达率(req/s)
WARMUP=true                     # 是否先跑 warmup step

# ShareGPT 转换参数
SHAREGPT_MAX_ITEMS=200          # 从 ShareGPT 抽取的对话数
SHAREGPT_MIN_TURNS=4            # 最少轮次（过滤太短的对话）
SHAREGPT_MAX_TURNS=30           # 最多轮次
SEED=42

DRY_RUN=false

# ────────────────────────────── 参数解析 ──────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --clients)      NUM_CLIENTS=$2;             shift 2 ;;
        --conversations) MAX_ACTIVE_CONVERSATIONS=$2; shift 2 ;;
        --turns)        MAX_TURNS=$2;               shift 2 ;;
        --requests)     MAX_NUM_REQUESTS=$2;        shift 2 ;;
        --rate)         REQUEST_RATE=$2;            shift 2 ;;
        --dry-run)      DRY_RUN=true;               shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ────────────────────────────── 编号生成 ──────────────────────────────
mkdir -p "${LOG_DIR}"
BENCH_ID=$(ls "${LOG_DIR}"/multi_turn_*.log 2>/dev/null | grep -oP '\d+' | sort -n | tail -1 || echo 0)
BENCH_ID=$((BENCH_ID + 1))
LOG_FILE="${LOG_DIR}/multi_turn_${BENCH_ID}.log"
XLSX_FILE="${LOG_DIR}/multi_turn_${BENCH_ID}.xlsx"

echo "========================================"
echo " Multi-Turn Benchmark #${BENCH_ID}"
echo " clients=${NUM_CLIENTS}  active_conv=${MAX_ACTIVE_CONVERSATIONS}"
echo " max_turns=${MAX_TURNS}  max_requests=${MAX_NUM_REQUESTS}"
echo " request_rate=${REQUEST_RATE}  warmup=${WARMUP}"
echo " log → ${LOG_FILE}"
echo "========================================"

[[ "${DRY_RUN}" == "true" ]] && echo "[dry-run] 退出" && exit 0

# ────────────────────────────── 环境准备 ──────────────────────────────
mkdir -p "${SCRIPT_DIR}" "${DATA_DIR}"

# 1. 下载 benchmark 脚本（如果不存在）
BENCH_SCRIPT="${SCRIPT_DIR}/benchmark_serving_multi_turn.py"
CONVERT_SCRIPT="${SCRIPT_DIR}/convert_sharegpt_to_openai.py"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"

if [[ ! -f "${BENCH_SCRIPT}" ]]; then
    echo "[1/4] 下载 benchmark_serving_multi_turn.py ..."
    curl -sSL "https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/multi_turn/benchmark_serving_multi_turn.py" \
        -o "${BENCH_SCRIPT}"
fi

if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
    echo "[1/4] 下载 convert_sharegpt_to_openai.py ..."
    curl -sSL "https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/multi_turn/convert_sharegpt_to_openai.py" \
        -o "${CONVERT_SCRIPT}"
fi

if [[ ! -f "${REQUIREMENTS}" ]]; then
    echo "[1/4] 下载 requirements.txt ..."
    curl -sSL "https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/multi_turn/requirements.txt" \
        -o "${REQUIREMENTS}"
fi

for _f in bench_dataset.py bench_utils.py; do
    if [[ ! -f "${SCRIPT_DIR}/${_f}" ]]; then
        echo "[1/4] 下载 ${_f} ..."
        curl -sSL "https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/multi_turn/${_f}" \
            -o "${SCRIPT_DIR}/${_f}"
    fi
done

# 2. 安装依赖
echo "[2/4] 检查依赖 ..."
pip install -q -r "${REQUIREMENTS}" 2>/dev/null || true
pip install -q pandas tqdm openpyxl 2>/dev/null || true

# 3. 下载并转换 ShareGPT 数据集
SHAREGPT_RAW="${DATA_DIR}/sharegpt_raw.json"
SHAREGPT_CONVERTED="${DATA_DIR}/sharegpt_conv_${SHAREGPT_MAX_ITEMS}.json"

if [[ ! -f "${SHAREGPT_RAW}" ]]; then
    echo "[3/4] 下载 ShareGPT 数据集 ..."
    # 尝试从 HuggingFace 下载
    HF_URL="https://huggingface.co/datasets/philschmid/sharegpt-raw/resolve/main/sharegpt_20230401_clean_lang_split.json"
    if ! curl -sSL --max-time 120 "${HF_URL}" -o "${SHAREGPT_RAW}"; then
        echo "❌ HuggingFace 下载失败，请手动下载："
        echo "   ${HF_URL}"
        echo "   保存到 ${SHAREGPT_RAW}"
        exit 1
    fi
fi

if [[ ! -f "${SHAREGPT_CONVERTED}" ]]; then
    echo "[3/4] 转换 ShareGPT → OpenAI 格式 ..."
    # 注意：convert 脚本默认过滤掉纯 ASCII（英文）内容，保留含非 ASCII 字符的对话
    # 若数据集为英文对话，需在 convert_sharegpt_to_openai.py 中暂时注释掉
    # `return has_non_english_chars(content)` → `return True`
    python3 "${CONVERT_SCRIPT}" \
        "${SHAREGPT_RAW}" \
        "${SHAREGPT_CONVERTED}" \
        --seed="${SEED}" \
        --max-items="${SHAREGPT_MAX_ITEMS}" \
        --min-turns="${SHAREGPT_MIN_TURNS}" \
        --max-turns="${SHAREGPT_MAX_TURNS}"
fi

CONV_COUNT=$(python3 -c "import json; d=json.load(open('${SHAREGPT_CONVERTED}')); print(len(d))")
echo "  ShareGPT 对话数：${CONV_COUNT}"

# 4. 运行 benchmark
echo "[4/4] 启动 benchmark ..."
echo "  URL: ${VLLM_URL}"
echo "  Model: ${MODEL_NAME}"

WARMUP_ARG=""
[[ "${WARMUP}" == "true" ]] && WARMUP_ARG="--warmup-step"

RATE_ARG=""
[[ "${REQUEST_RATE}" != "0" ]] && RATE_ARG="--request-rate ${REQUEST_RATE}"

python3 "${BENCH_SCRIPT}" \
    --model "${MODEL_NAME}" \
    --served-model-name "${MODEL_NAME}" \
    --url "${VLLM_URL}" \
    --input-file "${SHAREGPT_CONVERTED}" \
    --num-clients "${NUM_CLIENTS}" \
    --max-active-conversations "${MAX_ACTIVE_CONVERSATIONS}" \
    --max-turns "${MAX_TURNS}" \
    --max-num-requests "${MAX_NUM_REQUESTS}" \
    --output-file "${XLSX_FILE}" \
    --request-timeout-sec 300 \
    ${WARMUP_ARG} \
    ${RATE_ARG} \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "✅ 完成！结果已保存："
echo "   日志：${LOG_FILE}"
echo "   Excel：${XLSX_FILE}"
