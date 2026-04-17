#!/bin/bash
# =============================================================================
# download_models.sh - 从 HuggingFace 批量下载模型到 /lssd/models/
#
# 用法:
#   ./download_models.sh               # 下载全部未完成的模型（顺序）
#   ./download_models.sh --list        # 列出所有模型及下载状态
#   ./download_models.sh --only qwen3  # 只下载名称含 qwen3 的模型（大小写不敏感）
#   ./download_models.sh --dry-run     # 预览命令，不实际下载
#
# 模型列表（注意：deepseek-ai/DeepSeek-V3.x 官方 repo 本身为 FP8 格式）：
#   Qwen3-235B-A22B-Instruct-2507:  BF16 / FP8 / NVFP4
#   Qwen3.5-397B-A17B:              BF16 / FP8 / NVFP4
#   DeepSeek-V3.1:                  FP8 / NVFP4
#   DeepSeek-V3.2:                  FP8 / NVFP4
#
# 依赖:
#   uv tool install huggingface_hub  （安装后 CLI 为 hf）
# =============================================================================
set -euo pipefail

MODELS_DIR="/lssd/models"
LOG_DIR="${MODELS_DIR}/logs"
DRY_RUN=false
ONLY_FILTER=""
LIST_MODE=false

# ── 参数解析 ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true;          shift ;;
        --list)       LIST_MODE=true;         shift ;;
        --only)       ONLY_FILTER="${2,,}";  shift 2 ;;
        -h|--help)    grep '^#' "$0" | head -20 | sed 's/^# \?//'; exit 0 ;;
        *)  echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 模型定义（repo_id  local_dir_name） ───────────────────────────────────────
# 格式: "HF_REPO_ID|LOCAL_DIR_NAME"
MODELS=(
    # Qwen3-235B-A22B-Instruct-2507
    "Qwen/Qwen3-235B-A22B-Instruct-2507|Qwen3-235B-A22B-Instruct-2507"
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8|Qwen3-235B-A22B-Instruct-2507-FP8"
    "nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4|Qwen3-235B-A22B-Instruct-2507-NVFP4"

    # Qwen3.5-397B-A17B
    "Qwen/Qwen3.5-397B-A17B|Qwen3.5-397B-A17B"
    "Qwen/Qwen3.5-397B-A17B-FP8|Qwen3.5-397B-A17B-FP8"
    "nvidia/Qwen3.5-397B-A17B-NVFP4|Qwen3.5-397B-A17B-NVFP4"

    # DeepSeek-V3.1（deepseek-ai 官方 repo 本身为 FP8）
    "deepseek-ai/DeepSeek-V3.1|DeepSeek-V3.1-FP8"
    "nvidia/DeepSeek-V3.1-NVFP4|DeepSeek-V3.1-NVFP4"

    # DeepSeek-V3.2（deepseek-ai 官方 repo 本身为 FP8）
    "deepseek-ai/DeepSeek-V3.2|DeepSeek-V3.2-FP8"
    "nvidia/DeepSeek-V3.2-NVFP4|DeepSeek-V3.2-NVFP4"
)

# ── 工具函数 ──────────────────────────────────────────────────────────────────
is_downloaded() {
    local dir="$1"
    # 目录存在、无 .incomplete 文件、且有 ≥5 个 .safetensors 文件视为已完成
    [[ -d "$dir" ]] || return 1
    find "$dir" -name "*.incomplete" -print -quit 2>/dev/null | grep -q . && return 1
    local count
    count=$(find "$dir" -maxdepth 2 \( -name "*.safetensors" -o -name "*.bin" \) 2>/dev/null | wc -l)
    [[ $count -ge 5 ]]
}

# ── 列表模式 ──────────────────────────────────────────────────────────────────
if $LIST_MODE; then
    printf "\n%-55s %-40s %s\n" "HF Repo" "本地目录" "状态"
    printf "%-55s %-40s %s\n" "$(printf '%.0s-' {1..55})" \
        "$(printf '%.0s-' {1..40})" "------"
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r repo local_name <<< "$entry"
        local_dir="${MODELS_DIR}/${local_name}"
        if is_downloaded "$local_dir"; then
            status="✅ 已下载"
        elif [[ -d "$local_dir" ]]; then
            status="⏳ 未完成"
        else
            status="❌ 未开始"
        fi
        printf "%-55s %-40s %s\n" "$repo" "$local_name" "$status"
    done
    echo ""
    exit 0
fi

# hf CLI 路径（uv tool install 默认装在 ~/.local/bin）
export PATH="$HOME/.local/bin:$PATH"

# ── 前置检查 ──────────────────────────────────────────────────────────────────
if ! $DRY_RUN; then
    if ! command -v hf &>/dev/null; then
        echo "错误: 未找到 hf CLI，请先安装："
        echo "  uv tool install huggingface_hub"
        exit 1
    fi
    mkdir -p "$LOG_DIR"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   download_models.sh - 批量下载模型到 ${MODELS_DIR}   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ── 下载循环 ──────────────────────────────────────────────────────────────────
total=${#MODELS[@]}
count=0
skipped=0

for entry in "${MODELS[@]}"; do
    IFS='|' read -r repo local_name <<< "$entry"

    # --only 过滤
    if [[ -n "$ONLY_FILTER" ]] && [[ "${local_name,,}" != *"$ONLY_FILTER"* ]]; then
        continue
    fi

    local_dir="${MODELS_DIR}/${local_name}"
    log_file="${LOG_DIR}/download_${local_name}.log"
    count=$(( count + 1 ))

    echo "─── [$count/$total] $local_name ───────────────────────────────"
    echo "    Repo : $repo"
    echo "    Dest : $local_dir"

    if is_downloaded "$local_dir"; then
        echo "    状态 : ✅ 已下载，跳过"
        skipped=$(( skipped + 1 ))
        echo ""
        continue
    fi

    echo "    日志 : $log_file"

    if $DRY_RUN; then
        echo "    [DRY-RUN] HF_HUB_ENABLE_HF_TRANSFER=1 \\"
        echo "      hf download $repo \\"
        echo "        --local-dir $local_dir"
        echo ""
        continue
    fi

    mkdir -p "$local_dir"
    echo "    开始下载..."

    # 流式输出到日志，同时在终端显示进度摘要
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_XET=1 \
    HF_HOME="${MODELS_DIR}/.hf_cache" \
        hf download "$repo" \
            --local-dir "$local_dir" \
        2>&1 | tee "$log_file" | grep -E 'Fetching|Downloading|complete|error|Error' || true

    if is_downloaded "$local_dir"; then
        echo "    ✅ 下载完成"
    else
        echo "    ⚠️  下载后未检测到模型文件，请检查日志：$log_file"
    fi
    echo ""
done

echo "═══════════════════════════════════════════════════════════════════"
echo "完成。已跳过（已存在）: $skipped，本次处理: $(( count - skipped ))"
echo ""
