#!/bin/bash
# =============================================================================
# mount_lssd.sh - 将 32 块 local SSD 合并为 RAID0 挂载到 /lssd
#
# 用法:
#   ./mount_lssd.sh              # 默认挂载到 /lssd
#   ./mount_lssd.sh --dry-run   # 预览命令，不实际执行
#   ./mount_lssd.sh --umount    # 卸载并销毁 /dev/md0
#
# 说明:
#   - 自动检测大小在 390–415 GB 之间的 NVMe 设备（匹配 375 GiB ≈ 402 GB local SSD）
#   - RAID0 chunk=512K，XFS 条带对齐（su=512k, sw=<检测到的盘数>）
#   - local SSD 重启后数据丢失，挂载配置不写入 /etc/fstab
# =============================================================================
set -euo pipefail

MOUNT_POINT="/lssd"
MD_DEV="/dev/md0"
CHUNK_SIZE="512"               # KB
DRY_RUN=false
UMOUNT=false

# SSD 大小过滤范围（字节），匹配 375 GiB（≈402 GB）local SSD（±12 GB 容差）
SSD_SIZE_MIN=390000000000
SSD_SIZE_MAX=415000000000

# ── 参数解析 ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true;  shift ;;
        --umount)    UMOUNT=true;   shift ;;
        -h|--help)   grep '^#' "$0" | head -15 | sed 's/^# \?//'; exit 0 ;;
        *)  echo "未知参数: $1"; exit 1 ;;
    esac
done

run() {
    echo "  + $*"
    $DRY_RUN || "$@"
}

# ── 检测 375 GB NVMe SSD ──────────────────────────────────────────────────────
mapfile -t DEVS < <(
    lsblk -dpno NAME,SIZE --bytes 2>/dev/null |
    awk -v min="$SSD_SIZE_MIN" -v max="$SSD_SIZE_MAX" \
        '$1 ~ /nvme/ && int($2) >= min && int($2) <= max { print $1 }' |
    sort
)
NUM_DEVS=${#DEVS[@]}

# ── 卸载模式 ─────────────────────────────────────────────────────────────────
if $UMOUNT; then
    echo "=== 卸载 ${MOUNT_POINT} 并销毁 ${MD_DEV} ==="
    if mountpoint -q "${MOUNT_POINT}" 2>/dev/null; then
        run sudo umount "${MOUNT_POINT}"
    else
        echo "  ${MOUNT_POINT} 未挂载，跳过"
    fi
    if [[ -b "${MD_DEV}" ]]; then
        run sudo /usr/sbin/mdadm --stop "${MD_DEV}"
        if [[ $NUM_DEVS -gt 0 ]]; then
            run sudo /usr/sbin/mdadm --zero-superblock "${DEVS[@]}"
        else
            echo "  警告: 未检测到 375 GB NVMe SSD，跳过 zero-superblock"
        fi
    else
        echo "  ${MD_DEV} 不存在，跳过"
    fi
    echo "完成。"
    exit 0
fi

# ── 前置检查 ─────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   mount_lssd.sh - 32× local SSD → RAID0 → ${MOUNT_POINT}          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# 检查设备数量
if [[ $NUM_DEVS -eq 0 ]]; then
    echo "错误: 未检测到大小在 390–415 GB 之间的 NVMe 设备（375 GiB）"
    echo "  lsblk 输出（供排查）:"
    lsblk -dpno NAME,SIZE --bytes | awk '$1 ~ /nvme/' | sed 's/^/    /'
    exit 1
fi
echo "  检测到 ${NUM_DEVS} 块 375 GiB NVMe SSD: ${DEVS[*]}"

# 检查是否已有文件系统（防止误操作）
HAS_FS=false
for d in "${DEVS[@]}"; do
    if sudo /usr/sbin/blkid "$d" &>/dev/null; then
        echo "  警告: $d 已有文件系统，可能被覆盖"
        HAS_FS=true
    fi
done
if $HAS_FS && ! $DRY_RUN; then
    read -r -p "  检测到已有文件系统，继续将销毁数据。确认？[y/N] " ans
    [[ "${ans,,}" == "y" ]] || { echo "已取消。"; exit 1; }
fi

# 检查 md0 是否已存在
if [[ -b "${MD_DEV}" ]]; then
    echo "错误: ${MD_DEV} 已存在，请先运行 --umount 或手动处理"
    exit 1
fi

# ── 步骤 1：创建 RAID0 ────────────────────────────────────────────────────────
echo ""
echo "=== [1/3] 创建 RAID0（${NUM_DEVS} 块 × 375 GiB = $(( NUM_DEVS * 375 )) GiB，chunk=${CHUNK_SIZE}K）==="
run sudo /usr/sbin/mdadm --create "${MD_DEV}" \
    --level=0 \
    --raid-devices="${NUM_DEVS}" \
    --chunk="${CHUNK_SIZE}" \
    "${DEVS[@]}" <<< "yes"

# ── 步骤 2：格式化 XFS ───────────────────────────────────────────────────────
echo ""
echo "=== [2/3] 格式化为 XFS（条带对齐 su=${CHUNK_SIZE}k, sw=${NUM_DEVS}）==="
run sudo /usr/sbin/mkfs.xfs -f \
    -d "su=${CHUNK_SIZE}k,sw=${NUM_DEVS}" \
    "${MD_DEV}"

# ── 步骤 3：挂载并创建子目录 ──────────────────────────────────────────────────
echo ""
echo "=== [3/3] 挂载到 ${MOUNT_POINT} 并创建子目录 ==="
run sudo mkdir -p "${MOUNT_POINT}"
run sudo mount "${MD_DEV}" "${MOUNT_POINT}"
run sudo chmod 1777 "${MOUNT_POINT}"

# 标准子目录布局（hf_home/torch_home/models/vllm_src/build/logs/install_logs）
run sudo mkdir -p \
    "${MOUNT_POINT}/hf_home" \
    "${MOUNT_POINT}/torch_home" \
    "${MOUNT_POINT}/models" \
    "${MOUNT_POINT}/vllm_src" \
    "${MOUNT_POINT}/build" \
    "${MOUNT_POINT}/logs" \
    "${MOUNT_POINT}/install_logs"
run sudo chmod -R 1777 \
    "${MOUNT_POINT}/hf_home" \
    "${MOUNT_POINT}/torch_home" \
    "${MOUNT_POINT}/models" \
    "${MOUNT_POINT}/vllm_src" \
    "${MOUNT_POINT}/build" \
    "${MOUNT_POINT}/logs" \
    "${MOUNT_POINT}/install_logs"

# ── 结果 ─────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   完成！                                                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
if ! $DRY_RUN; then
    df -h "${MOUNT_POINT}"
    echo ""
    echo "子目录:"
    ls -la "${MOUNT_POINT}/"
fi
echo ""
echo "⚠️  local SSD 重启后数据丢失，挂载不持久化到 /etc/fstab"
