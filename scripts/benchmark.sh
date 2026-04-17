#!/bin/bash
# 用法:
#   ./benchmark.sh <served-model-name>                             # 纯吞吐压测（无共享前缀）
#   ./benchmark.sh <served-model-name> --prefix 256               # 生产模式（256 token system prompt）
#   ./benchmark.sh <served-model-name> --sync-to user@host:/path  # 压测完自动同步结果到本机
#
# 热身覆盖 parallel=1,2,4,8，确保各 batch size 的 CUDA graph/inductor 编译完成
# 结果保存至 benchmark_result/<timestamp>_<model>.md

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$1"
PREFIX_LENGTH=0
SYNC_TO=""

# 解析可选参数
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)  PREFIX_LENGTH="$2"; shift 2 ;;
        --sync-to) SYNC_TO="$2";       shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

COMMON_ARGS=(
  --model "$MODEL"
  --url http://127.0.0.1:8088/v1/chat/completions
  --api-key EMPTY
  --api openai
  --dataset random
  --max-tokens 200
  --min-tokens 150
  --prefix-length "$PREFIX_LENGTH"
  --min-prompt-length 4500
  --max-prompt-length 4500
  --tokenizer-path Qwen/Qwen2.5-0.5B-Instruct
  --extra-args '{"ignore_eos": true}'
)

# ── 采集 vLLM serve 信息（压测前） ───────────────────────────────────────────
VLLM_PID=$(ps aux | grep 'vllm serve' | grep -v grep | awk '{print $2}' | sort -n | tail -1)
if [[ -n "$VLLM_PID" ]]; then
    VLLM_CMD=$(ps -p "$VLLM_PID" -o args= 2>/dev/null | sed 's/ --/\n  --/g')
    VLLM_ENV=$(sudo cat /proc/"$VLLM_PID"/environ 2>/dev/null \
        | tr '\0' '\n' \
        | grep -E '^(VLLM_|NCCL_|NVSHMEM_|NVIDIA_|CUDA_)' \
        | sort)
else
    VLLM_CMD="（未找到 vllm serve 进程）"
    VLLM_ENV=""
fi

# ── 热身 ──────────────────────────────────────────────────────────────────────
echo "=== 热身阶段（parallel=1,2,4,8，覆盖主要 batch size 的编译）==="
evalscope perf \
  --parallel 1 2 4 8 \
  --number   4 8 12 16 \
  "${COMMON_ARGS[@]}"

# ── 正式压测（输出保存到临时文件） ────────────────────────────────────────────
BENCH_TMP=$(mktemp)
echo ""
echo "=== 正式压测（prefix-length=${PREFIX_LENGTH}）==="
evalscope perf \
  --parallel 1 2 4 8 20 40 60 \
  --number  10 20 40 80 200 400 600 \
  "${COMMON_ARGS[@]}" 2>&1 | tee "$BENCH_TMP"

# ── 写入结果文件 ──────────────────────────────────────────────────────────────
RESULT_DIR="${SCRIPT_DIR}/benchmark_result"
mkdir -p "$RESULT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SAFE=$(basename "$MODEL")
RESULT_FILE="${RESULT_DIR}/${TIMESTAMP}_${MODEL_SAFE}.md"

{
cat <<EOF
# 压测报告

- **时间**：$(date '+%Y-%m-%d %H:%M:%S')
- **模型**：${MODEL}
- **prefix-length**：${PREFIX_LENGTH}
- **输入长度**：4500 tokens（fixed）
- **输出长度**：150–200 tokens（ignore_eos=true）

## vLLM serve 命令

\`\`\`
${VLLM_CMD}
\`\`\`

## 环境变量

\`\`\`
${VLLM_ENV:-（无 VLLM_/NCCL_ 相关环境变量）}
\`\`\`

## 压测结果

\`\`\`
EOF
grep -A 40 'Performance Test Summary Report' "$BENCH_TMP" | tail -42
echo '```'
} > "$RESULT_FILE"

rm -f "$BENCH_TMP"
echo ""
echo "结果已保存至: ${RESULT_FILE}"

# ── 同步结果回本机（可选） ────────────────────────────────────────────────────
if [[ -n "$SYNC_TO" ]]; then
    echo ""
    echo "=== 同步 benchmark_result/ → ${SYNC_TO} ==="
    rsync -avz "${RESULT_DIR}/" "${SYNC_TO}/"
    if [[ $? -eq 0 ]]; then
        echo "同步完成"
    else
        echo "⚠ 同步失败，请手动执行：rsync -avz ${RESULT_DIR}/ ${SYNC_TO}/"
    fi
fi
