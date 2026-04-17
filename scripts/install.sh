#!/bin/bash
# =============================================================================
# install.sh - vLLM 安装脚本（B200 专用）
#
# 用法:
#   ./install.sh                                           # 全量安装
#   ./install.sh --no-ep-kernels                          # 跳过 DeepEP
#   ./install.sh --list                                    # 查看步骤状态
#   ./install.sh --from 9                                  # 从第 9 步继续
#   ./install.sh --only 9                                  # 只跑第 9 步
#   ./install.sh --force 9                                 # 强制重跑第 9 步
#   ./install.sh --reset                                   # 清除所有 checkpoint
# =============================================================================
set -euo pipefail

# ── Root 检查 ─────────────────────────────────────────────────────────────────
# uv pip install --system 需要写入系统 site-packages，必须以 root 身份运行
if [[ $EUID -ne 0 ]]; then
  echo "错误: 请以 root 身份运行，例如："
  echo "  sudo -E $0 $*"
  echo "（-E 保留 PATH，确保 ~/.local/bin/uv 可被 sudo 找到）"
  exit 1
fi

# Ubuntu 24.04 externally-managed Python（PEP 668）：允许 uv 写入系统 site-packages
export UV_BREAK_SYSTEM_PACKAGES=true

# ── OS 自动检测 ───────────────────────────────────────────────────────────────
# 支持 Ubuntu/Debian（apt）和 Rocky Linux/RHEL（dnf）
PKG_MGR="apt"

detect_os() {
  local os_id="" os_like=""
  if [[ -f /etc/os-release ]]; then
    # shellcheck source=/dev/null
    source /etc/os-release
    os_id="${ID:-unknown}"
    os_like="${ID_LIKE:-}"
  fi
  if [[ "$os_id" == "ubuntu" ]] || [[ "$os_like" == *"debian"* ]]; then
    PKG_MGR="apt"
  elif [[ "$os_id" == "rocky" ]] || [[ "$os_id" == "rhel" ]] ||
    [[ "$os_like" == *"rhel"* ]] || [[ "$os_like" == *"fedora"* ]]; then
    PKG_MGR="dnf"
  else
    echo "警告: 未知 OS (${os_id:-unknown})，默认使用 apt-get"
    PKG_MGR="apt"
  fi
  echo "检测到 OS: ${os_id:-unknown} → 包管理器: ${PKG_MGR}"
}
detect_os

# ── B200 硬件配置（x86_64 / Blackwell SM100 / 180 GB HBM）───────────────────
GPU_ARCH_VLLM="10.0a"             # vLLM 主体编译目标
GPU_ARCH_EXT="9.0a 10.0a"         # DeepGEMM/DeepEP 扩展（向后兼容 H100）
NVSHMEM_PLATFORM="linux-x86_64"
GDRCOPY_DEB_ARCH="amd64"          # deb 文件名 arch
GDRCOPY_CUDA_VER="13.0"

# ── vLLM 分支 ────────────────────────────────────────────────────────────────
VLLM_BRANCH="nightly"

# ── DeepGEMM 固定版本（必装）───────────────────────────────────────────────
# v2.1.1.post3：引入 fp8_mqa_logits（DeepSeek-V3.2 必需），向下兼容 R1/V3
DEEPGEMM_VERSION="v2.1.1.post3"

# ── 安装开关 ─────────────────────────────────────────────────────────────────
INSTALL_EP_KERNELS=true

# ── 固定路径 ─────────────────────────────────────────────────────────────────
LSSD=/lssd
WORKSPACE=${LSSD}/vllm_src
EP_WORKSPACE=${LSSD}/build/ep_kernels
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0} # step_09 _ensure_nvcc 会强制设为 cuda-13.0
CUDA_VERSION_MAJOR=13
GDRCOPY_VER="2.5.1-1"
NVSHMEM_VER="3.3.24"
DEEPEP_COMMIT="73b6ea4a439ba03a695563f9fd242c8e4b02b37c"
CHECKPOINT_DIR=/var/tmp/vllm_install_checkpoints

# ── 全局编译 / UV 环境 ───────────────────────────────────────────────────────
export DEBIAN_FRONTEND=noninteractive
export UV_HTTP_TIMEOUT=500
export UV_INDEX_STRATEGY="unsafe-best-match"
export UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${LSSD}/hf_home"       # HuggingFace 模型缓存放 local SSD
export TORCH_HOME="${LSSD}/torch_home" # PyTorch 权重/扩展缓存

# ══════════════════════════════════════════════════════════════════════════════
# CLI 参数解析
# ══════════════════════════════════════════════════════════════════════════════
MODE="auto"
ARG_STEP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
  --no-ep-kernels)
    INSTALL_EP_KERNELS=false
    shift
    ;;
  --list)
    MODE="list"
    shift
    ;;
  --reset)
    MODE="reset"
    shift
    ;;
  --from)
    MODE="from"
    ARG_STEP="$2"
    shift 2
    ;;
  --only)
    MODE="only"
    ARG_STEP="$2"
    shift 2
    ;;
  --force)
    MODE="force"
    ARG_STEP="$2"
    shift 2
    ;;
  -h | --help)
    grep '^#' "$0" | head -15 | sed 's/^# \?//'
    exit 0
    ;;
  *)
    echo "未知参数: $1"
    exit 1
    ;;
  esac
done

# ── 编译参数 ─────────────────────────────────────────────────────────────────
export TORCH_CUDA_ARCH_LIST="${GPU_ARCH_EXT}"

mkdir -p "${CHECKPOINT_DIR}"

# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint 辅助函数
# ══════════════════════════════════════════════════════════════════════════════
_ckpt_file() { echo "${CHECKPOINT_DIR}/step_$(printf '%02d' "$1").done"; }
is_done() { [[ -f "$(_ckpt_file "$1")" ]]; }
mark_done() {
  echo "$(date '+%Y-%m-%d %H:%M:%S')  step $1 ${STEP_NAMES[$1-1]}" >"$(_ckpt_file "$1")"
  echo "✔  步骤 $1 完成"
}
reset_checkpoints() {
  rm -f "${CHECKPOINT_DIR}"/step_*.done
  echo "已清除所有 checkpoint。"
}

list_steps() {
  printf '%-5s  %-50s  %s\n' "步骤" "名称" "状态"
  printf '%-5s  %-50s  %s\n' "-----" "--------------------------------------------------" "------"
  for i in "${!STEP_NAMES[@]}"; do
    local num=$((i + 1))
    local st="[ 待运行 ]"
    is_done "$num" && st="[ ✔ 已完成 ]"
    printf '%-5s  %-50s  %s\n' "$num" "${STEP_NAMES[$i]}" "$st"
  done
}

run_step() {
  local num=$1
  local name="${STEP_NAMES[$((num - 1))]}"
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  [${num}/${TOTAL_STEPS}]  ${name}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  "step_$(printf '%02d' "$num")"
  mark_done "$num"
}

# ══════════════════════════════════════════════════════════════════════════════
# 各步骤函数
# ══════════════════════════════════════════════════════════════════════════════

step_01() {
  if [[ "$PKG_MGR" == "dnf" ]]; then
    # Rocky Linux / RHEL：gcc 版本由 OS 默认提供（Rocky 9 默认 gcc 11，Rocky 10 默认 gcc 14）
    # dnf-plugins-core 提供 dnf config-manager（step_04 添加 NVIDIA CUDA 仓库需要）
    dnf install -y dnf-plugins-core
    dnf install -y \
      git curl wget xz \
      python3-pip python3-devel \
      libibverbs libibverbs-devel numactl numactl-libs \
      gcc gcc-c++ \
      ninja-build \
      infiniband-diags rdma-core-devel \
      cmake make \
      findutils which
    # libsm/libxext/libgl 对应包（用于 opencv 等）
    dnf install -y libSM libXext mesa-libGL 2>/dev/null || true
  else
    # gcc 版本：Ubuntu 22.04 → gcc-10；Ubuntu 24.04+ → gcc-12
    # （gcc-10 已从 Ubuntu 24.04 官方仓库移除；CUDA 13.0 支持 gcc ≤ 13）
    local _gcc_ver="10"
    local _uver; _uver=$(. /etc/os-release 2>/dev/null; echo "${VERSION_ID:-22.04}" | tr -d '.')
    [[ "$_uver" -ge 2404 ]] && _gcc_ver="12"
    apt-get update -y
    apt-get install -y --no-install-recommends \
      software-properties-common \
      git curl sudo wget xz-utils \
      python3-pip \
      libibverbs-dev libnuma-dev \
      ffmpeg libsm6 libxext6 libgl1 \
      gcc-${_gcc_ver} g++-${_gcc_ver} \
      ninja-build numactl \
      ibverbs-utils infiniband-diags \
      build-essential cmake \
      htop sysstat iftop
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${_gcc_ver} 110 \
      --slave /usr/bin/g++ g++ /usr/bin/g++-${_gcc_ver}
    rm -rf /var/lib/apt/lists/*
    # 终端颜色支持（tmux / SSH 连接均正常渲染 256 色）
    grep -qF 'TERM=xterm-256color' /root/.bashrc 2>/dev/null || \
      echo 'export TERM=xterm-256color' >> /root/.bashrc
    grep -qF 'TERM=xterm-256color' /home/${SUDO_USER:-}/.bashrc 2>/dev/null || \
      echo 'export TERM=xterm-256color' >> /home/${SUDO_USER:-}/.bashrc 2>/dev/null || true
  fi
}

step_02() {
  if [[ "$PKG_MGR" == "dnf" ]]; then
    if ! python3.12 --version &>/dev/null; then
      # Rocky Linux 9.4+ 在标准 AppStream 中有 python3.12
      dnf install -y python3.12 python3.12-devel || {
        echo "错误: dnf 安装 python3.12 失败，请检查 RHEL/Rocky AppStream 仓库"
        exit 1
      }
    fi
    # RHEL 用 alternatives（与 update-alternatives 兼容）
    alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 2>/dev/null ||
      update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 || true
    alternatives --set python3 /usr/bin/python3.12 2>/dev/null ||
      update-alternatives --set python3 /usr/bin/python3.12 || true
    ln -sf /usr/bin/python3.12-config /usr/bin/python3-config 2>/dev/null || true
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
  else
    if ! python3.12 --version &>/dev/null; then
      for i in 1 2 3; do
        add-apt-repository -y ppa:deadsnakes/ppa && break || sleep 5
      done
      apt-get update -y
      apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv
      rm -rf /var/lib/apt/lists/*
    else
      # Python 3.12 已预装（Ubuntu 24.04），仍需确保 dev/venv 包存在（提供 Python.h 等头文件）
      apt-get install -y --no-install-recommends python3.12-dev python3.12-venv || true
    fi
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
    update-alternatives --set python3 /usr/bin/python3.12
    ln -sf /usr/bin/python3.12-config /usr/bin/python3-config 2>/dev/null || true
    # Ubuntu 24.04 是 externally-managed Python（PEP 668），不能直接装 pip；
    # 后续所有包管理均通过 uv（UV_BREAK_SYSTEM_PACKAGES=true），不需要系统 pip
  fi
}

step_03() {
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv --version
}

step_04() {
  if [[ "$PKG_MGR" == "dnf" ]]; then
    # Rocky 9：通过官方 NVIDIA CUDA 仓库安装 cuda-toolkit-13-0
    # step_09(_ensure_nvcc) 也会调用 dnf install cuda-toolkit-13-0，此处先配好仓库
    local _cuda_repo="https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo"
    echo "  [dnf/Rocky] 配置 NVIDIA CUDA 仓库: ${_cuda_repo}"
    dnf config-manager --add-repo "${_cuda_repo}" 2>/dev/null ||
      echo "  警告: 添加 CUDA 仓库失败，请检查网络或手动配置"
    echo "  CUDA_HOME=${CUDA_HOME}（nvcc 将在 step_09 中按需安装）"
    ldconfig "${CUDA_HOME}/compat/" 2>/dev/null || true
  else
    local CUDA_VER_DASH="13-0"
    # 若 CUDA 包尚未可见，安装 cuda-keyring 配置 NVIDIA apt 仓库（含 GPG key）
    if ! apt-cache show "cuda-nvcc-${CUDA_VER_DASH}" &>/dev/null; then
      local _uver; _uver=$(. /etc/os-release 2>/dev/null; echo "${VERSION_ID:-22.04}" | tr -d '.')
      local _keyring="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${_uver}/x86_64/cuda-keyring_1.1-1_all.deb"
      echo "  配置 NVIDIA CUDA apt 仓库: ubuntu${_uver}/x86_64"
      curl -fSL "${_keyring}" -o /tmp/cuda-keyring.deb && dpkg -i /tmp/cuda-keyring.deb \
        && rm -f /tmp/cuda-keyring.deb \
        || echo "  警告: cuda-keyring 配置失败，假设仓库已存在"
    fi
    apt-get update -y
    apt-get install -y --no-install-recommends \
      "cuda-nvcc-${CUDA_VER_DASH}" \
      "cuda-cuobjdump-${CUDA_VER_DASH}" \
      "cuda-cudart-${CUDA_VER_DASH}" \
      "cuda-cudart-dev-${CUDA_VER_DASH}" \
      "cuda-nvrtc-${CUDA_VER_DASH}" \
      "cuda-libraries-dev-${CUDA_VER_DASH}" \
      "libcublas-${CUDA_VER_DASH}" \
      libnccl-dev || echo "部分 CUDA 包可能已存在，继续..."
    ldconfig "${CUDA_HOME}/compat/" 2>/dev/null || true
    rm -rf /var/lib/apt/lists/*
  fi
}

step_05() {
  if [[ "$PKG_MGR" == "dnf" ]]; then
    # GDRCopy 无官方 RPM 包；如需 NVIDIA_GDRCOPY=1，请从源码编译：
    #   https://github.com/NVIDIA/gdrcopy
    echo "  [dnf/Rocky] GDRCopy 无官方 RPM，跳过（NVIDIA_GDRCOPY=1 将不可用）"
    return 0
  fi
  # GDRCopy（B200 x86_64）：deb 包从 NVIDIA developer.download
  local _ubuntu_ver
  _ubuntu_ver=$(. /etc/os-release 2>/dev/null; echo "${VERSION_ID:-22.04}" | tr '.' '_')
  local DEB="libgdrapi_${GDRCOPY_VER}_${GDRCOPY_DEB_ARCH}.Ubuntu${_ubuntu_ver}.deb"
  local URL="https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%20${GDRCOPY_CUDA_VER}/ubuntu${_ubuntu_ver}/x64/${DEB}"
  echo "下载 GDRCopy: ${URL}"
  curl -fSL "${URL}" -o "/tmp/${DEB}" || {
    echo "警告: GDRCopy 下载失败，跳过（NVIDIA_GDRCOPY=1 可能无效）"
    return 0
  }
  apt-get update -y
  apt-get install -y "/tmp/${DEB}"
  rm -f "/tmp/${DEB}"
  rm -rf /var/lib/apt/lists/*
}

step_06() {
  # PyTorch 2.10.0 发布于 2026-01-21；vLLM nightly 已升至 2.10（breaking change）
  # cu129 只有 PyTorch 2.9.x 的 wheel，2.10.0 官方提供 cu126/cu128/cu130
  # B200 driver 580 支持 CUDA 13.0，选 cu130（FlashMLA cu130+ 性能更优）
  uv pip install --system \
    torch==2.10.0 \
    torchaudio==2.10.0 \
    torchvision==0.25.0 \
    --extra-index-url https://download.pytorch.org/whl/cu130
}

step_07() {
  # flashinfer：跳过预装。VLLM_BRANCH=nightly 会自动拉取匹配版本，
  # step_13 安装 vLLM 后自动同步 cubin/jit-cache 版本
  echo "  跳过预装 flashinfer；step_13 完成后会自动同步 cubin/jit-cache 版本"
}

step_08() {
  # vLLM nightly 安装时会自动拉取绝大多数依赖（transformers、fastapi、ray 等）
  # 此处只安装 vLLM 未声明或需特定版本的包：
  #   numba       — profiling 工具，vLLM 不依赖；限定版本以兼容 numpy<2.2
  #   numpy<2.2   — numba 0.61.2 要求 numpy 1.x/2.x 但上限 2.2
  #   hf_transfer — HF 下载加速（vLLM 不自动安装）
  #   accelerate  — 大模型加载辅助（部分 vLLM 路径需要）
  #   bitsandbytes / timm — 可选量化/视觉扩展
  #   runai-model-streamer — GCS/S3 流式加载（vLLM 不自动安装）
  #   opencv-python-headless — 图像预处理（vLLM 不自动安装）
  #   ray[cgraph] — vLLM 只装 plain ray，cgraph extra 需手动指定
  #   build tools — cmake/ninja/wheel/setuptools 编译 DeepGEMM/DeepEP 需要
  uv pip install --system \
    "numpy>=1.24,<2.2" \
    "numba==0.61.2" \
    hf_transfer \
    accelerate \
    "bitsandbytes>=0.42.0" \
    "timm>=1.0.17" \
    "runai-model-streamer[s3,gcs]>=0.15.3" \
    "opencv-python-headless>=4.11.0" \
    "ray[cgraph]>=2.48.0" \
    "setuptools>=77.0.3,<81.0.0" \
    cmake packaging wheel setuptools-scm ninja \
    nvitop
}

step_09() {
  # DeepGEMM：必装。固定 tag ${DEEPGEMM_VERSION}（顶部定义）。
  # 需要 nvcc 编译 deep_gemm_cpp C++ 扩展；nvcc 必须与 PyTorch cu130 (CUDA 13.0) 对齐：
  #   - major 不一致 → torch.utils.cpp_extension 报 RuntimeError
  #   - minor 不一致 → UserWarning + CUDA_HOME 头文件与 runtime 不同源
  local DEEPGEMM_DIR="${LSSD}/build/deepgemm"

  _ensure_nvcc() {
    local cur_ver=""
    if command -v nvcc &>/dev/null; then
      cur_ver=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
    fi

    if [[ "$cur_ver" == "13.0" ]]; then
      echo "  nvcc 版本已是 13.0，无需变更"
    elif [[ "$PKG_MGR" == "apt" ]]; then
      echo "  ${cur_ver:+当前 nvcc ${cur_ver}，}安装 cuda-nvcc-13-0 (Ubuntu apt)..."
      apt-get update -y
      apt-get install -y --no-install-recommends cuda-nvcc-13-0 cuda-cuobjdump-13-0 || {
        echo "错误: cuda-nvcc-13-0/cuda-cuobjdump-13-0 安装失败"
        echo "请手动安装: https://developer.nvidia.com/cuda-downloads"
        exit 1
      }
    else
      # dnf（Rocky Linux / RHEL）
      if [[ -n "$cur_ver" ]]; then
        echo "  当前 nvcc 版本: ${cur_ver}（非 13.0），卸载后安装 13.0..."
        local cur_pkg
        cur_pkg=$(rpm -qf "$(command -v nvcc)" 2>/dev/null | head -1)
        if [[ -n "$cur_pkg" && "$cur_pkg" != *"not owned"* ]]; then
          echo "  卸载: ${cur_pkg}"
          dnf remove -y "${cur_pkg}" 2>/dev/null || true
        fi
      else
        echo "  nvcc 未找到，安装 CUDA 13.0 toolkit..."
      fi
      dnf install -y cuda-toolkit-13-0 2>/dev/null || {
        echo "错误: cuda-toolkit-13-0 安装失败"
        echo "请手动安装: https://developer.nvidia.com/cuda-downloads"
        exit 1
      }
    fi

    export CUDA_HOME="/usr/local/cuda-13.0"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    echo "  CUDA_HOME=${CUDA_HOME}, nvcc=$(nvcc --version | grep release)"
  }

  echo "安装 DeepGEMM ${DEEPGEMM_VERSION}（git tag）"
  _ensure_nvcc
  rm -rf "${DEEPGEMM_DIR}"
  git clone --depth=1 --branch "${DEEPGEMM_VERSION}" \
    https://github.com/deepseek-ai/DeepGEMM.git "${DEEPGEMM_DIR}"
  git -C "${DEEPGEMM_DIR}" submodule update --init --depth=1
  uv pip install --system --no-build-isolation "${DEEPGEMM_DIR}"
  rm -rf "${DEEPGEMM_DIR}"
}

step_10() {
  # NVSHMEM linux-x86_64 包
  local FILE="libnvshmem-${NVSHMEM_PLATFORM}-${NVSHMEM_VER}_cuda${CUDA_VERSION_MAJOR}-archive.tar.xz"
  local URL="https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/${NVSHMEM_PLATFORM}/${FILE}"
  mkdir -p "${EP_WORKSPACE}"
  cd "${EP_WORKSPACE}"
  echo "下载 NVSHMEM [${NVSHMEM_PLATFORM}]: ${URL}"
  curl -fSL "${URL}" -o "${FILE}"
  tar -xf "${FILE}"
  rm -rf nvshmem
  mv "${FILE%.tar.xz}" nvshmem
  rm -f "${FILE}"
  rm -rf nvshmem/lib/bin nvshmem/lib/share
  echo "NVSHMEM 解压至: ${EP_WORKSPACE}/nvshmem"
  cd /
}

step_11() {
  # pplx-kernels 已于 vLLM PR #33724（2026-02-26）从 nightly 移除，无需安装
  echo "  pplx-kernels 已由 vLLM nightly 移除，跳过"
}

step_12() {
  if [[ "${INSTALL_EP_KERNELS}" != "true" ]]; then
    echo "跳过 DeepEP（--no-ep-kernels）"
    return 0
  fi
  # DeepEP 需要 nvcc 编译 CUDA 扩展；确保 CUDA_HOME/bin 在 PATH 中
  export PATH="${CUDA_HOME}/bin:${PATH}"
  [[ -d "${EP_WORKSPACE}/nvshmem" ]] || {
    echo "错误: NVSHMEM 目录不存在（${EP_WORKSPACE}/nvshmem），请先运行步骤 10"
    exit 1
  }
  rm -rf "${EP_WORKSPACE}/DeepEP"
  git clone https://github.com/deepseek-ai/DeepEP "${EP_WORKSPACE}/DeepEP"
  git -C "${EP_WORKSPACE}/DeepEP" checkout "${DEEPEP_COMMIT}"
  NVSHMEM_DIR="${EP_WORKSPACE}/nvshmem" \
    uv pip install --system --no-build-isolation -vvv "${EP_WORKSPACE}/DeepEP"
}

step_13() {
  # vLLM nightly wheel
  echo "vLLM 分支: ${VLLM_BRANCH}  |  CUDA arch: ${GPU_ARCH_VLLM}"
  mkdir -p "${WORKSPACE}"

  uv pip install --system vllm --extra-index-url https://wheels.vllm.ai/nightly

  # vLLM nightly 可能升级 flashinfer-python；同步 cubin/jit-cache 版本
  local fi_ver
  fi_ver=$(python3 -c \
    "import importlib.metadata; print(importlib.metadata.version('flashinfer-python'))" \
    2>/dev/null ||
    python3 -c \
      "import importlib.metadata; print(importlib.metadata.version('flashinfer'))" \
      2>/dev/null || echo "")
  if [[ -n "$fi_ver" ]]; then
    echo "  同步 flashinfer cubin/jit-cache → ${fi_ver}"
    uv pip install --system "flashinfer-cubin==${fi_ver}" || true
    local pip_cmd
    pip_cmd=$(command -v pip3.12 || command -v pip3 || echo "")
    if [[ -n "$pip_cmd" ]]; then
      "${pip_cmd}" install --no-cache-dir "flashinfer-jit-cache==${fi_ver}" \
        --extra-index-url https://flashinfer.ai/whl/cu130 || true
    else
      uv pip install --system "flashinfer-jit-cache==${fi_ver}" \
        --extra-index-url https://flashinfer.ai/whl/cu130 || true
    fi
  fi

  python3 -c "import vllm; print('vLLM version:', vllm.__version__)"
}

step_14() {
  # LMCache：CPU DRAM KV cache offload（多轮长上下文压测必需）
  echo "安装 lmcache（KV cache offload）"
  uv pip install --system lmcache
}

step_15() {
  if ! command -v just &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh |
      bash -s -- --to /usr/local/bin
  fi
  just --version
}

step_16() {
  # evalscope：ModelScope 模型评测框架（能力评测 + 压测 + 可视化）
  # perf：压测模块（TTFT/TPOT，对标 vllm bench serve）
  # app：Gradio 可视化报告界面（evalscope app，端口 7861）
  uv pip install --system --no-cache-dir 'evalscope[perf,app]'
  /usr/local/bin/evalscope --version
}

step_17() {
  # nvidia-persistenced：Docker 容器运行时需要此 socket
  # 所属包：nvidia-compute-utils-xxx-server（随驱动一起安装）
  # unit 文件无 [Install] 段，驱动 postinstall 脚本在 multi-user.target.wants/ 创建软链。
  # 此步骤负责：① 清理之前误建的 autostart unit；② 兜底创建软链（驱动 postinstall 漏跑时）；
  #             ③ 确保当前会话 daemon 已运行。

  # ① 清理旧的冲突 unit（若存在）
  local custom_unit=/etc/systemd/system/nvidia-persistenced-autostart.service
  if [[ -f "$custom_unit" ]]; then
    systemctl disable nvidia-persistenced-autostart 2>/dev/null || true
    rm -f "$custom_unit"
    systemctl daemon-reload
    echo "  已清理旧的 nvidia-persistenced-autostart.service"
  fi

  # ② 兜底：确保开机自启软链存在（unit 无 [Install] 段，不能用 systemctl enable）
  local unit_src=/usr/lib/systemd/system/nvidia-persistenced.service
  local wants_dir=/etc/systemd/system/multi-user.target.wants
  if [[ -f "$unit_src" ]] && [[ ! -e "$wants_dir/nvidia-persistenced.service" ]]; then
    mkdir -p "$wants_dir"
    ln -sf "$unit_src" "$wants_dir/nvidia-persistenced.service"
    systemctl daemon-reload
    echo "  已创建开机自启软链（驱动 postinstall 漏跑兜底）"
  fi

  # ③ 当前会话：若 daemon 未运行则启动
  if [[ ! -f "$unit_src" ]]; then
    echo "  ⚠ nvidia-persistenced.service 不存在，NVIDIA 驱动可能未安装"
    return 0
  fi
  if ! systemctl is-active --quiet nvidia-persistenced.service; then
    systemctl start nvidia-persistenced.service || true
  fi
  echo "  nvidia-persistenced 状态: $(systemctl is-active nvidia-persistenced.service)"
  ls /run/nvidia-persistenced/socket 2>/dev/null && echo "  socket OK" || echo "  ⚠ socket 不存在"
}

step_18() {
  # 内核参数调优：vm + THP 持久化
  # 目标：减少 OS 对 LMCache CPU DRAM 的回收压力；THP 大页减少 CPU DRAM 分配碎片

  # ── sysctl 持久化 ────────────────────────────────────────────────────────────
  local sysctl_conf=/etc/sysctl.d/90-lmcache-tuning.conf
  cat >"${sysctl_conf}" <<'SYSCTL_EOF'
# LMCache / vLLM 内核调优参数
# 生成来源: install.sh step_18

# 降低 swap 倾向（默认 60）
# 1 = 几乎不 swap，避免 LMCache 500+ GB CPU DRAM 热数据被换出到磁盘
vm.swappiness = 1

# 降低 page cache 回收积极性（默认 100）
# 50 = OS 更倾向保留 page cache，减少 LMCache 命中数据被 dentry/inode 回收驱逐
vm.vfs_cache_pressure = 50

# 关闭 NUMA zone 本地强制回收（B200 3.8 TiB DRAM 跨 2 个 NUMA 节点）
# 0 = 不强制在单 zone 内回收页面，允许跨 NUMA 分配，LMCache 不因 zone 局部内存不足被驱逐
vm.zone_reclaim_mode = 0
SYSCTL_EOF

  sysctl -p "${sysctl_conf}"
  echo "  sysctl 参数已写入并应用: ${sysctl_conf}"

  # ── THP systemd 服务（持久化 /sys/kernel/mm/transparent_hugepage/* 写入）──
  # 注意：/sys 写入在 /etc/sysctl.d/ 中无效，必须用 systemd oneshot 服务
  # THP enabled=always  : 对所有匿名内存启用 THP，LMCache 大块 CPU DRAM 受益
  # THP defrag=defer    : 异步后台整理碎片（defer 是 defrag 的合法选项，非 enabled 的选项）
  local thp_service=/etc/systemd/system/thp-tuning.service
  cat >"${thp_service}" <<'SERVICE_EOF'
[Unit]
Description=Transparent Hugepage tuning for LMCache / vLLM
After=local-fs.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c 'echo always > /sys/kernel/mm/transparent_hugepage/enabled'
ExecStart=/bin/sh -c 'echo defer  > /sys/kernel/mm/transparent_hugepage/defrag'

[Install]
WantedBy=multi-user.target
SERVICE_EOF

  systemctl daemon-reload
  systemctl enable thp-tuning.service
  systemctl start  thp-tuning.service
  echo "  THP enabled : $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  echo "  THP defrag  : $(cat /sys/kernel/mm/transparent_hugepage/defrag)"
}

step_19() {
  echo "vLLM branch: ${VLLM_BRANCH}"
  python3 - <<'PYEOF'
import importlib.metadata as m, sys, torch

pkgs = [
    ("torch",             "PyTorch"),
    ("vllm",              "vLLM"),
    ("flashinfer",        "FlashInfer"),
    ("flashinfer_python", "FlashInfer-python"),
    ("deep_gemm",         "DeepGEMM"),
    ("deep_ep",           "DeepEP"),
    ("lmcache",           "lmcache"),
    ("evalscope",         "EvalScope"),
]
for mod, label in pkgs:
    try:
        print(f"  {label:<22}{m.version(mod)}")
    except m.PackageNotFoundError:
        print(f"  {label:<22}✗ 未安装", file=sys.stderr)

print(f"\n  {'CUDA runtime:':<22}{torch.version.cuda}  （torch 内嵌）")
try:
    print(f"  {'GPU:':<22}{torch.cuda.get_device_name(0)}")
except Exception:
    print(f"  {'GPU:':<22}（非 GPU 环境）")
PYEOF
}

# ══════════════════════════════════════════════════════════════════════════════
# 步骤名称表
# ══════════════════════════════════════════════════════════════════════════════
STEP_NAMES=(
  "系统基础包"
  "Python 3.12"
  "uv 包管理器"
  "CUDA 13.0 仓库配置（nvcc 在 step_09 按需安装）"
  "GDRCopy ${GDRCOPY_VER} (${GDRCOPY_DEB_ARCH} / CUDA ${GDRCOPY_CUDA_VER})"
  "PyTorch 2.10.0 (cu130)"
  "FlashInfer 0.5.3"
  "Python 通用依赖"
  "DeepGEMM ${DEEPGEMM_VERSION}"
  "NVSHMEM ${NVSHMEM_VER} (${NVSHMEM_PLATFORM})"
  "pplx-kernels（vLLM nightly 已移除，跳过）"
  "DeepEP ${DEEPEP_COMMIT} [EP]"
  "vLLM (branch=${VLLM_BRANCH}, arch=${GPU_ARCH_VLLM})"
  "lmcache（KV cache offload）"
  "just 配方运行器"
  "EvalScope (perf + app)"
  "nvidia-persistenced 开机自启（Docker 依赖）"
  "内核调优（vm.swappiness / vfs_cache_pressure / THP）"
  "环境验证"
)
TOTAL_STEPS=${#STEP_NAMES[@]}

# ══════════════════════════════════════════════════════════════════════════════
# 预检：打印推理相关包和驱动版本，等待确认后继续
# ══════════════════════════════════════════════════════════════════════════════
run_preflight() {
  local ckpt="${CHECKPOINT_DIR}/preflight.done"
  if [[ -f "$ckpt" ]]; then
    echo "  → 跳过预检（已确认，删除 ${ckpt} 可重新触发）"
    return 0
  fi

  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║   预检：当前系统推理相关包和驱动版本                             ║"
  echo "╚══════════════════════════════════════════════════════════════════╝"
  echo ""

  # NVIDIA 驱动 / CUDA driver version
  if command -v nvidia-smi &>/dev/null; then
    echo "── GPU / 驱动 ──────────────────────────────────────────────────────"
    nvidia-smi --query-gpu=name,driver_version \
      --format=csv,noheader 2>/dev/null |
      awk -F',' '{printf "  GPU:           %s\n  驱动版本:      %s\n", $1,$2}' || true
    # cuda_version 字段在 R580+ 已不再支持直接查询，改从驱动头部获取
    nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9.]+' | head -1 | \
      awk '{print "  CUDA（驱动）:  " $1}' || true
  else
    echo "  nvidia-smi 未找到（驱动未安装或环境不含 GPU）"
  fi

  echo ""
  echo "── nvcc（编译工具链）──────────────────────────────────────────────"
  if command -v nvcc &>/dev/null; then
    nvcc --version | grep -E "release|Build" | sed 's/^/  /'
  else
    echo "  nvcc: 未安装"
  fi

  echo ""
  echo "── Python 推理相关包 ───────────────────────────────────────────────"
  python3 - <<'PYEOF'
import importlib.metadata as m, sys
pkgs = [
    ("torch",         "PyTorch"),
    ("vllm",          "vLLM"),
    ("flashinfer",    "FlashInfer"),
    ("flashinfer_python", "FlashInfer-python"),
    ("deep_gemm",     "DeepGEMM"),
    ("deep_ep",       "DeepEP"),
    ("lmcache",       "lmcache"),
    ("evalscope",     "EvalScope"),
]
for mod, label in pkgs:
    try:
        print(f"  {label:<22}{m.version(mod)}")
    except m.PackageNotFoundError:
        print(f"  {label:<22}（未安装）")
PYEOF

  echo ""
  echo "── CUDA runtime（torch 内嵌）─────────────────────────────────────"
  python3 -c "import torch; print(f'  PyTorch CUDA runtime:  {torch.version.cuda}')" 2>/dev/null || true

  echo ""
  local ans=""
  read -r -p "  确认以上环境，继续安装？[y/N] " ans 2>/dev/null || true
  if [[ -n "$ans" ]] && [[ "${ans,,}" != "y" ]]; then
    echo "已取消。"
    exit 0
  elif [[ -z "$ans" ]]; then
    echo "  未获取输入（非交互模式），自动继续..."
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S')  preflight confirmed" >"$ckpt"
  echo ""
}

# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   vLLM 安装脚本（B200）                                           ║"
printf "║   DeepGEMM: %-12s  vLLM 分支: %-15s ║\n" "${DEEPGEMM_VERSION}" "${VLLM_BRANCH}"
printf "║   CUDA arch: %-12s  EP Kernels: %-15s ║\n" "${GPU_ARCH_VLLM}" "${INSTALL_EP_KERNELS}"
echo "╚══════════════════════════════════════════════════════════════════╝"

case "$MODE" in
list)
  list_steps
  exit 0
  ;;
reset)
  reset_checkpoints
  exit 0
  ;;
only)
  echo "模式: 仅运行步骤 ${ARG_STEP}"
  run_step "$ARG_STEP"
  ;;
force)
  echo "模式: 强制重跑步骤 ${ARG_STEP}"
  rm -f "$(_ckpt_file "$ARG_STEP")"
  run_step "$ARG_STEP"
  ;;
from)
  echo "模式: 从第 ${ARG_STEP} 步开始"
  run_preflight
  for i in $(seq "$ARG_STEP" "$TOTAL_STEPS"); do
    run_step "$i"
  done
  ;;
auto)
  echo "模式: 自动续跑"
  list_steps
  echo ""
  run_preflight
  for i in $(seq 1 "$TOTAL_STEPS"); do
    if is_done "$i"; then
      echo "  → 跳过步骤 ${i} [${STEP_NAMES[$((i - 1))]}]"
    else
      run_step "$i"
    fi
  done
  ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   安装完成！                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "── 数据目录（均在 /lssd）───────────────────────────────────────────────"
echo "  模型缓存:  ${HF_HOME}"
echo "  vLLM 源码: ${WORKSPACE}"
echo "  EP 组件:   ${EP_WORKSPACE}"
echo ""
echo "（版本汇总见 step_19 输出）"
