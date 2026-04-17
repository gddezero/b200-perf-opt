# 测试命令行模板

> 不使用 serve.sh / docker-serve.sh，直接用命令行模板，替换参数即可。
> **能用 Docker 就用 Docker**（所有模型都可以 Docker 部署）。

---

## 1. Docker 模板（推荐）

### 1.1 Qwen3.5-397B — 基础模板

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
  --port 8088 \
  --host 0.0.0.0 \
  --tensor-parallel-size 8 \        # ← 替换：TP
  --data-parallel-size 1 \           # ← 替换：DP
  --performance-mode throughput      # ← 替换：throughput / balanced / interactivity
```

**可替换参数**：

| 要改什么 | 改哪里 | 可选值 |
|---------|--------|--------|
| TP/DP | `--tensor-parallel-size` / `--data-parallel-size` | TP=1 DP=8, TP=2 DP=4, TP=4 DP=2, TP=8 DP=1 |
| 量化 | 模型路径 + served-model-name | FP8: `Qwen3.5-397B-A17B-FP8`, BF16: `Qwen3.5-397B-A17B` |
| EP | 加 `--enable-expert-parallel` | DP>1 时加 |
| performance-mode | `--performance-mode` | throughput（高吞吐）/ balanced（均衡）/ interactivity（低延迟） |
| MTP | 加 `--speculative-config` | 见下方 MTP 模板 |

### 1.2 Qwen3.5-397B — 高吞吐（TP=2 DP=4 EP）

```bash
sudo docker run --rm --name "qwen35_throughput" \
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
  --port 8088 \
  --host 0.0.0.0 \
  --tensor-parallel-size 2 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --performance-mode throughput
```

### 1.3 Qwen3.5-397B — MTP 投机解码

在基础模板上加一行 `--speculative-config`：

```bash
sudo docker run --rm --name "qwen35_mtp" \
  --runtime=nvidia --gpus all --ipc=host --network=host \
  -v /lssd:/lssd \
  -e HF_HOME=/lssd/hf_home \
  -e TORCH_HOME=/lssd/torch_home \
  -e HUGGINGFACE_HUB_CACHE=/lssd/hf_home/hub \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  "vllm/vllm-openai:cu130-nightly" \
  "/lssd/models/Qwen3.5-397B-A17B-FP8" \
  --served-model-name "Qwen/Qwen3.5-397B-A17B" \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --language-model-only \
  --port 8088 \
  --host 0.0.0.0 \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --performance-mode interactivity \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

> `num_speculative_tokens` 改为 1/2/3。推荐 n=2。

### 1.4 DeepSeek V3.1（Docker，推荐）

> V3.1 实测 Docker vs 裸机性能差距 <8%（#66 vs #76），推荐 Docker 部署。

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
  --enable-expert-parallel \
  --performance-mode throughput
```

> 改 FP8：模型路径换 `/lssd/models/DeepSeek-V3.1-FP8`。

### 1.5 Qwen3-235B

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
  --port 8088 \
  --host 0.0.0.0 \
  --tensor-parallel-size 4 \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --performance-mode throughput
```

> 量化可选：NVFP4（最高吞吐）/ FP8 / BF16，改模型路径和 served-model-name。

---

## 2. 裸机模板（DeepSeek V3.2 专用）

> V3.2 必须裸机：依赖 DeepGEMM v2.1.1.post3 + FlashMLA-Sparse JIT 编译，Docker 镜像缺这些。
> V3.1 / Qwen 都用 Docker（见 §1）。

### 2.1 DeepSeek V3.2 FP8

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
  --tensor-parallel-size 1 \          # ← 替换
  --data-parallel-size 8 \            # ← 替换
  --attention-backend FLASHINFER_MLA_SPARSE \  # ← 或 FLASHMLA_SPARSE
  --performance-mode throughput \      # ← 或 balanced / interactivity
  --enable-expert-parallel
```

### 2.2 DeepSeek V3.2 — 低延迟（TP=8）

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
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --attention-backend FLASHINFER_MLA_SPARSE \
  --performance-mode interactivity
```

> 低延迟 TP=8 配置不加 `--enable-expert-parallel`，`--performance-mode` 用 `interactivity`。

### 2.3 DeepSeek V3.2 — MTP

在基础模板上加 `--speculative-config`：

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
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --attention-backend FLASHINFER_MLA_SPARSE \
  --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":2}'
```

> V3.1 也可以用裸机（仅当需要纯净环境复现旧测试时），命令同 §2.1 把模型路径换成 V3.1 即可。

---

## 3. 压测模板

### 3.1 标准吞吐压测

```bash
./scripts/benchmark.sh "模型served-model-name"
```

> 模型名必须与 `--served-model-name` 一致。

### 3.2 evalscope 直接调用

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

### 3.3 多轮对话压测

```bash
./scripts/bench_multi_turn.sh \
  --clients 40 \
  --conversations 240 \
  --turns 30 \
  --requests 5000
```

---

## 4. 参数替换速查

### 4.1 TP/DP 组合

| 场景 | TP | DP | EP | 说明 |
|------|----|----|-----|------|
| 高吞吐 | 1 | 8 | `--enable-expert-parallel` | 最大并发处理 |
| 均衡 | 2 | 4 | `--enable-expert-parallel` | 吞吐+延迟均衡 |
| 低延迟 | 8 | 1 | 不加 EP | 单请求最快 |

### 4.2 注意力后端（仅 DeepSeek）

| 参数 | 值 | 说明 |
|------|-----|------|
| `--attention-backend` | `FLASHINFER_MLA_SPARSE` | **推荐**，多数 TP 配置下更优 |
| `--attention-backend` | `FLASHMLA_SPARSE` | TP=2 时略优 |

> Qwen 不需要指定 attention-backend。

### 4.3 Performance Mode（适用所有模型）

| 值 | 场景 | 关键参数（来自 vLLM `arg_utils.py:2156`）|
|-----|------|---------------------------------------|
| `throughput` | 高并发批量处理（TP=1/2 + DP 高吞吐） | `max_num_batched_tokens=16384`、`max_num_seqs ×2`、稀疏 CUDA graph |
| `balanced` | 通用、LMCache 多轮对话场景 | `max_num_batched_tokens=8192`、标准 CUDA graph |
| `interactivity` | 低延迟 TP=8 单请求、MTP 场景 | `max_num_batched_tokens=8192`、CUDA graph 1–32 全覆盖 |

> **场景对照**（实测最优）：
> - TP=1/2/4 + DP + EP 高吞吐 → `throughput`
> - TP=8 单请求低延迟 → `interactivity`
> - LMCache 多轮长上下文 → `balanced`（`throughput` 会让 TTFT p99 翻倍，详见 [05 §7.1](05_kv_cache_and_lmcache.md)）
> - MTP 投机解码 → `interactivity`（CUDA graph 1–32 全覆盖避免 fallback eager）

### 4.4 模型路径与名称

| 模型 | 模型路径 | served-model-name |
|------|---------|------------------|
| Qwen3.5 FP8 | `/lssd/models/Qwen3.5-397B-A17B-FP8` | `Qwen/Qwen3.5-397B-A17B` |
| Qwen3.5 BF16 | `/lssd/models/Qwen3.5-397B-A17B` | `Qwen/Qwen3.5-397B-A17B` |
| Qwen3 NVFP4 | `/lssd/models/Qwen3-235B-A22B-Instruct-2507-NVFP4` | `nvidia/Qwen3-235B-A22B-NVFP4` |
| Qwen3 FP8 | `/lssd/models/Qwen3-235B-A22B-Instruct-2507-FP8` | `nvidia/Qwen3-235B-A22B-FP8` |
| V3.2 FP8 | `/lssd/models/DeepSeek-V3.2-FP8` | `deepseek-ai/DeepSeek-V3.2` |
| V3.1 FP8 | `/lssd/models/DeepSeek-V3.1-FP8` | `deepseek-ai/DeepSeek-V3.1` |

### 4.5 MTP speculative-config

| 模型 | method | 示例 |
|------|--------|------|
| Qwen3.5 | `qwen3_next_mtp` | `'{"method":"qwen3_next_mtp","num_speculative_tokens":2}'` |
| DeepSeek V3.2 | `deepseek_mtp` | `'{"method":"deepseek_mtp","num_speculative_tokens":2}'` |
| DeepSeek V3.1 | `deepseek_mtp` | `'{"method":"deepseek_mtp","num_speculative_tokens":2}'` |

> Qwen3-235B 不支持 MTP。

---

## 5. 环境变量

### 5.1 裸机必须设置

```bash
CUDA_HOME=/usr/local/cuda-13.0
VLLM_ENGINE_READY_TIMEOUT_S=1800
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
```

### 5.2 Docker 通过 -e 传入

```bash
-e HF_HOME=/lssd/hf_home
-e TORCH_HOME=/lssd/torch_home
-e HUGGINGFACE_HUB_CACHE=/lssd/hf_home/hub
-e VLLM_ENGINE_READY_TIMEOUT_S=1800      # MTP 时加
```

### 5.3 特殊场景

| 场景 | 环境变量 | 说明 |
|------|---------|------|
| LMCache | `-e PYTHONHASHSEED=0` | 修复跨进程 hash 不一致 |
| FP8 精度 | `VLLM_USE_DEEP_GEMM=0` | 禁用 DeepGEMM（精度 bug） |

---

## 6. 常用运维

```bash
# vLLM 升级后修复权限
sudo chmod -R a+w /usr/local/lib/python3.12/dist-packages/flashinfer \
  /usr/local/lib/python3.12/dist-packages/flashinfer_cubin \
  /usr/local/lib/python3.12/dist-packages/vllm

# 停止 Docker
docker stop $(docker ps -q --filter ancestor=vllm/vllm-openai:cu130-nightly)

# 停止裸机 vLLM
kill $(pgrep -f "vllm serve")

# 拉取结果
rsync -avz <user>@<b200-host>:~/code/b200_vllm_opt/benchmark_result/ \
  ~/code/b200_vllm_opt/benchmark_result/
```
