# B200 大模型推理测试 Step-by-Step 复现指南

> 按照本文档操作，即可在 GCP A4 B200 × 8 上完整复现所有测试。
> 硬件：B200 SXM × 8（180 GB HBM/卡）| 主机：`<b200-host>`

---

## Step 1：登录 B200 主机

```bash
ssh <user>@<b200-host>
cd ~/code/b200_vllm_opt
```

---

## Step 2：挂载本地 NVMe 存储

> B200 有 32 块 NVMe SSD，需要组 RAID0 挂载到 `/lssd`。**重启后数据丢失，需重新执行。**

```bash
./scripts/mount_lssd.sh
```

验证：

```bash
df -h /lssd          # 应显示 ~12 TB
ls /lssd/            # 子目录：hf_home/ torch_home/ models/ logs/
```

---

## Step 3：下载模型

```bash
./scripts/download_models.sh
```

模型存放在 `/lssd/models/`，常用模型：

| 模型路径 | 说明 |
|---------|------|
| `/lssd/models/Qwen3.5-397B-A17B-FP8` | Qwen3.5 FP8 |
| `/lssd/models/Qwen3.5-397B-A17B` | Qwen3.5 BF16 |
| `/lssd/models/Qwen3-235B-A22B-Instruct-2507-NVFP4` | Qwen3 NVFP4 |
| `/lssd/models/Qwen3-235B-A22B-Instruct-2507-FP8` | Qwen3 FP8 |
| `/lssd/models/DeepSeek-V3.2-FP8` | DeepSeek V3.2 FP8 |
| `/lssd/models/DeepSeek-V3.1-FP8` | DeepSeek V3.1 FP8 |

---

## Step 4：安装裸机环境（仅 DeepSeek V3.2 需要）

> Qwen 系列和 V3.1 都用 Docker，跳过此步。仅 V3.2 需要裸机（DeepGEMM JIT 编译）。

```bash
sudo -E ./scripts/install.sh
```

如果中断，可以断点续跑：

```bash
./scripts/install.sh --list          # 查看步骤状态
./scripts/install.sh --from 13       # 从第 13 步继续
./scripts/install.sh --force 9       # 强制重跑步骤 9
```

安装完成后关键版本：

| 组件 | 版本 |
|------|------|
| PyTorch | 2.10.0 (cu128) |
| nvcc | 12.8 |
| flashinfer | 0.6.6 |
| DeepGEMM | v2.1.1.post3 |


---

## Step 5：启动 vLLM 服务

> 所有命令都是直接可用的模板，替换 TP/DP/模型路径即可。
> 完整模板集合见 [03 命令行模板](03_command_reference.md)。

### 5a. Qwen3.5-397B FP8（Docker）

```bash
sudo docker run --rm --name "qwen35_test" \
  --runtime=nvidia --gpus all --ipc=host --network=host \
  -v /lssd:/lssd \
  -e HF_HOME=/lssd/hf_home \
  -e TORCH_HOME=/lssd/torch_home \
  -e HUGGINGFACE_HUB_CACHE=/lssd/hf_home/hub \
  "vllm/vllm-openai:cu130-nightly" \
  "/lssd/models/Qwen3.5-397B-A17B-FP8" \
  --served-model-name "Qwen/Qwen3.5-397B-A17B" \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --language-model-only \
  --port 8088 --host 0.0.0.0 \
  --tensor-parallel-size 8 \
  --data-parallel-size 1
```

### 5b. Qwen3-235B NVFP4（Docker）

```bash
sudo docker run --rm --name "qwen3_235b" \
  --runtime=nvidia --gpus all --ipc=host --network=host \
  -v /lssd:/lssd \
  -e HF_HOME=/lssd/hf_home \
  -e TORCH_HOME=/lssd/torch_home \
  -e HUGGINGFACE_HUB_CACHE=/lssd/hf_home/hub \
  "vllm/vllm-openai:cu130-nightly" \
  "/lssd/models/Qwen3-235B-A22B-Instruct-2507-NVFP4" \
  --served-model-name "nvidia/Qwen3-235B-A22B-NVFP4" \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --port 8088 --host 0.0.0.0 \
  --tensor-parallel-size 4 \
  --data-parallel-size 2 \
  --enable-expert-parallel
```

### 5c. DeepSeek V3.2 FP8（裸机）

```bash
CUDA_HOME=/usr/local/cuda-13.0 \
VLLM_ENGINE_READY_TIMEOUT_S=1800 \
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
vllm serve /lssd/models/DeepSeek-V3.2-FP8 \
  --served-model-name deepseek-ai/DeepSeek-V3.2 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.9 \
  --tokenizer-mode deepseek_v32 \
  --tool-call-parser deepseek_v32 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v3 \
  --port 8088 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --attention-backend FLASHINFER_MLA_SPARSE \
  --performance-mode throughput \
  --enable-expert-parallel
```

### 5d. DeepSeek V3.1 NVFP4（Docker，推荐）

> V3.1 实测 Docker vs 裸机性能差距 <8%（#66 Docker = 1048 toks/s vs #76 裸机 = 1115 toks/s）。
> 推荐用 Docker，省去裸机环境安装。

```bash
sudo docker run --rm --name "deepseek_v31" \
  --runtime=nvidia --gpus all --ipc=host --network=host \
  -v /lssd:/lssd \
  -e HF_HOME=/lssd/hf_home \
  -e TORCH_HOME=/lssd/torch_home \
  -e HUGGINGFACE_HUB_CACHE=/lssd/hf_home/hub \
  "vllm/vllm-openai:cu130-nightly" \
  "/lssd/models/DeepSeek-V3.1-NVFP4" \
  --served-model-name "deepseek-ai/DeepSeek-V3.1" \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.9 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v31 \
  --no-enable-prefix-caching \
  --port 8088 --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel
```

> 改用 FP8 时把模型路径改为 `/lssd/models/DeepSeek-V3.1-FP8`。

### 如何切换配置

替换以下参数即可测试不同配置：

| 要改什么 | 改哪里 | 常用值 |
|---------|--------|--------|
| 并行策略 | `--tensor-parallel-size` / `--data-parallel-size` | TP=1 DP=8（高吞吐）, TP=8 DP=1（低延迟） |
| 量化 | 模型路径 + served-model-name | FP8 / NVFP4 / BF16 |
| EP | 加/去 `--enable-expert-parallel` | DP>1 时加 |
| 注意力后端 | `--attention-backend` （仅 DeepSeek） | FLASHINFER_MLA_SPARSE / FLASHMLA_SPARSE |
| MTP 投机解码 | 加 `--speculative-config` | 见 [03 命令行模板](03_command_reference.md) §1.3 |

---

## Step 6：等待服务就绪

vLLM 启动需要 **15~20 分钟**（加载模型权重 + 编译 CUDA graph）。

确认就绪：

```bash
curl http://localhost:8088/v1/models
```

返回模型名即表示服务已启动。

---

## Step 7：运行压测

### 7a. 标准吞吐压测

```bash
./scripts/benchmark.sh "模型served-model-name"
```

例如：

```bash
./scripts/benchmark.sh "Qwen/Qwen3.5-397B-A17B"
./scripts/benchmark.sh "deepseek-ai/DeepSeek-V3.2"
./scripts/benchmark.sh "nvidia/Qwen3-235B-A22B-NVFP4"
```

> 模型名必须与 `--served-model-name` 一致。
> 压测会自动运行 warmup → 正式测试（并发 1, 2, 4, 8, 20, 40, 60），结果保存到 `benchmark_result/` 目录。

### 7b. 多轮对话压测

```bash
./scripts/bench_multi_turn.sh \
  --clients 40 \
  --conversations 240 \
  --turns 30 \
  --requests 5000
```

### 7c. evalscope 直接调用

```bash
evalscope perf \
  --url http://localhost:8088/v1/chat/completions \
  --model "Qwen/Qwen3.5-397B-A17B" \
  --api openai \
  --dataset openqa \
  --dataset-args '{"min_tokens": 4500, "max_tokens": 4500}' \
  --max-tokens 200 \
  --ignore-eos \
  --parallel 60 \
  --n 200 \
  --stream
```

---

## Step 8：查看和同步结果

结果自动保存到 `benchmark_result/` 目录。

```bash
# 查看最新结果
ls -lt benchmark_result/ | head

# 拉取到本地
rsync -avz <user>@<b200-host>:~/code/b200_vllm_opt/benchmark_result/ \
  ~/code/b200_vllm_opt/benchmark_result/
```

---

## Step 9：停止服务

```bash
# 停止 Docker
docker stop $(docker ps -q --filter ancestor=vllm/vllm-openai:cu130-nightly)

# 停止裸机 vLLM
kill $(pgrep -f "vllm serve")
```

> **压测前必须先停止已有 vLLM 进程。**

---

## 附录 A：推荐参数组合

### DeepSeek V3.2

| 场景 | TP | DP | EP | 注意力后端 | performance-mode |
|------|----|----|-----|-----------|-----------------|
| 高吞吐 | 1 | 8 | on | FLASHINFER_MLA_SPARSE | throughput |
| 均衡 | 2 | 4 | on | FLASHMLA_SPARSE | balanced |
| 低延迟 | 8 | 1 | off | FLASHINFER_MLA_SPARSE | 不加 |

### Qwen3.5-397B

| 场景 | TP | DP | EP |
|------|----|----|-----|
| 低延迟 | 8 | 1 | off |
| 高吞吐 | 2 | 4 | on |

### Qwen3-235B

| 场景 | TP | DP | EP |
|------|----|----|-----|
| NVFP4 高吞吐 | 4 | 2 | on |
| BF16 低延迟 | 8 | 1 | off |

---

## 附录 B：量化选择

| 模型 | 推荐 | 禁用 | 原因 |
|------|------|------|------|
| DeepSeek V3.2 | FP8 | NVFP4 | NVFP4 + FlashMLA-Sparse 兼容 bug (MO #763) |
| DeepSeek V3.1 | NVFP4 > FP8 | — | NVFP4 吞吐 +44%，精度正常 |
| Qwen3-235B | NVFP4 > FP8 > BF16 | — | NVFP4 官方 checkpoint 可用 |
| Qwen3.5-397B | FP8 | NVFP4 | NVFP4 精度严重下降 (vLLM #36094) |

---

## 附录 C：硬件信息

| 特性 | B200 | GB200 | H200 |
|------|------|-------|------|
| CPU 架构 | x86_64 | aarch64 (Grace) | x86_64 |
| GPU HBM | 180 GB | 180 GB | 141 GB |
| 架构 | SM100 (Blackwell) | SM100 (Blackwell) | SM90 (Hopper) |
| NVFP4 支持 | ✅ | ✅ | ❌ |
| FP8 算力 | 9,000 TFLOPS | 9,000 TFLOPS | 3,958 TFLOPS |

---

## 附录 D：禁止事项

| 规则 | 原因 |
|------|------|
| **禁止 `--enforce-eager`** | 必须通过正确 kernel 配置解决 CUDA graph 问题 |
| **禁止 V3.2 NVFP4 生产** | FlashMLA-Sparse 兼容 bug 未修复 |
| **禁止 Qwen3.5 NVFP4 生产** | 精度严重下降 (GSM8K=0.35) |
| **禁止 Qwen3.5 MTP ≥40 并发** | cudaErrorIllegalAddress 崩溃 |
| **FP8 必须 `VLLM_USE_DEEP_GEMM=0`** | 注意力层精度下降 bug (vLLM #37618) |
