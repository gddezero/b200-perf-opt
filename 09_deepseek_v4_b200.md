# DeepSeek V4 (Pro & Flash) 在 B200 × 8 上的部署与压测

> **测试时间**：2026-04-24 ~ 2026-04-27
> **硬件**：GCP A4 B200 SXM × 8 (180 GB HBM3e/卡)
> **软件**：SGLang `lmsysorg/sglang:deepseek-v4-blackwell` + vLLM `vllm/vllm-openai:deepseekv4-cu130`
> **压测工具**：[evalscope perf](https://github.com/modelscope/evalscope) (`benchmark.sh` 包装)
> **标准负载**：4500 token 输入，150–200 token 输出（ignore_eos=true），无 prefix cache，random dataset

---

## 🎯 关键性能（实测 2026-04-25）

### V4-Pro（SGLang）+ V4-Pro（vLLM）+ V4-Flash（SGLang）峰值

| 模型 | 引擎 | 配置 | 编号 | 峰值并发 | 峰值 toks/s | @1 toks/s | @1 TTFT |
|---|---|---|:---:|:---:|---:|---:|---:|
| V4-Flash | SGLang | **高吞吐**（无 spec, max_req=1024）| #208 | 600 | **🏆 2,933** | 76 | 0.50 s |
| V4-Flash | SGLang | 平衡（EAGLE n=1, max_req=512）| #207 | 400 | 2,393 | 96 | 0.55 s |
| V4-Pro | SGLang | 高吞吐（无 spec, max_req=256）| #201 | 200 | 1,146 | 47 | 0.78 s |
| V4-Pro | vLLM | baseline（MTP n=2, 默认 max_seqs=1024）| #202 | 200 | 930 | 66 | 0.49 s |
| V4-Flash | SGLang | 低延迟（TP=8, MXFP4, EAGLE n=3）| #206 | 180 | 829 | 160 | 0.30 s |
| V4-Pro | SGLang | 平衡（EAGLE n=1, max_req=128）| #200 | 100 | 782 | 51 | 1.51 s |
| V4-Pro | SGLang | 低延迟（TP=8, MXFP4, EAGLE n=3）| #199 | 20 | 312 | 71 | 1.54 s |

**核心结论**：
1. **V4-Flash 高吞吐 = 单机峰值 2933 toks/s**，是 V4-Pro 高吞吐的 **2.56×**（2933/1146），源于 Flash 权重小 5×（21.83 vs 105.57 GB/卡）→ KV/slice **大 12.8×**（5.87M vs 0.46M tokens）→ max_req 上限 **4×**（256 → 1024）→ 实际并发承载 **3×**（200 → 600）
2. **SGLang 全面超越 vLLM baseline**：相同硬件 + 相同模型，SGLang 高吞吐 1146 vs vLLM 930 (#202 baseline)，+23%
3. **公式约束生效**：`max_running_requests × draft_tokens ≤ SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 是平衡/高吞吐配方的核心约束
4. **低延迟配置不能跑高并发**：V4-Pro @40、V4-Flash @200 起严重过载（EAGLE n=3 + chunked_prefill 4096 限制）

---

## 1. 硬件与模型概览

### 1.1 硬件

| 项 | 规格 |
|---|---|
| GPU | 8 × NVIDIA B200 SXM (180 GB HBM3e/卡) |
| 卡间 | NVLink 5.0 全互联 |
| OS | Ubuntu 24.04 |
| Driver | NVIDIA 580 |
| CUDA | 13.0 |
| Docker | 27.x |
| 模型存储 | `/lssd/` 本地 NVMe SSD（避免重复下载）|

### 1.2 模型权重

| 模型 | 架构 | 层数 | 权重/卡 (TP=8) | KV/slice (TP=8 DP=1) | 路径 |
|---|---|---:|---:|---:|---|
| DeepSeek-V4-Pro | DeepseekV4ForCausalLM | 61 | **105.57 GB** | 0.46 M tokens | `/lssd/models/DeepSeek-V4-Pro` |
| DeepSeek-V4-Flash | DeepseekV4ForCausalLM | 43 | **21.83 GB** | 5.87 M tokens | `/lssd/models/DeepSeek-V4-Flash` |

> Flash 权重为 Pro 的 **1/5**（21.83 / 105.57 = 21%），并非 1/8。但因为权重占用减小 → KV 空间 **~12.8 倍**（5.87M / 0.46M），这是 Flash 高吞吐性能反超的根本原因。

### 1.3 镜像

| 引擎 | 镜像 | 大小 |
|---|---|---|
| SGLang | `lmsysorg/sglang:deepseek-v4-blackwell` | 90 GB |
| vLLM | `vllm/vllm-openai:deepseekv4-cu130` | 31.5 GB |

---

## 2. 部署：SGLang 三套配置

### 2.1 通用前置

```bash
ssh <user>@<b200-host>

# 拉镜像（首次）
sudo docker pull lmsysorg/sglang:deepseek-v4-blackwell

# 预创建 cache 卷（多轮压测复用 JIT 编译产物）
sudo mkdir -p /lssd/cache && sudo chmod -R 777 /lssd/cache

# 切场景前必须停旧容器
sudo docker stop deepseek-v4-pro deepseek-v4-flash 2>/dev/null
sudo docker rm   deepseek-v4-pro deepseek-v4-flash 2>/dev/null
```

### 2.2 配置一：低延迟（TP=8 + MXFP4 + EAGLE n=3）

适用：单/小并发优先（≤ 40 for Pro，≤ 180 for Flash）。下面以 V4-Pro 为例，Flash 只需替换模型名 + max_req/DISPATCH（见小节末注）。

```bash
sudo docker run -d \
  --name deepseek-v4-pro \
  --gpus all --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v /lssd/models:/lssd/models:ro \
  -v /lssd/cache:/root/.cache \
  -e SGLANG_ENABLE_SPEC_V2=1 \
  -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  -e NCCL_IB_DISABLE=1 \
  --restart unless-stopped \
  lmsysorg/sglang:deepseek-v4-blackwell \
  python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path /lssd/models/DeepSeek-V4-Pro \
    --served-model-name DeepSeek-V4-Pro \
    --tp 8 \
    --moe-runner-backend flashinfer_mxfp4 \
    --speculative-algo EAGLE \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --chunked-prefill-size 4096 \
    --disable-flashinfer-autotune \
    --mem-fraction-static 0.82 \
    --enable-metrics \
    --enable-mfu-metrics \
    --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
    --host 0.0.0.0 --port 8088
```

> ⚠️ `--mem-fraction-static 0.82` 必须显式设；不设会用 SGLang 默认值（当前镜像~0.9 偏激进），EAGLE n=3 cudagraph capture 阶段必 OOM（实测 #209 v0 重现）。
> Flash 同配置：模型名换成 `DeepSeek-V4-Flash`，其余完全一致（KV 充裕，0.82 仍安全）。

### 2.3 配置二：平衡（TP=8 DP=8 + DeepEP + EAGLE n=1）

适用：中等并发 + 低延迟（80 ≤ p ≤ 400），平衡场景生产首选。下面以 V4-Pro 为例。

```bash
sudo docker run -d \
  --name deepseek-v4-pro \
  --gpus all --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v /lssd/models:/lssd/models:ro \
  -v /lssd/cache:/root/.cache \
  -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
  -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  -e NCCL_IB_DISABLE=1 \
  --restart unless-stopped \
  lmsysorg/sglang:deepseek-v4-blackwell \
  python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path /lssd/models/DeepSeek-V4-Pro \
    --served-model-name DeepSeek-V4-Pro \
    --tp 8 --dp 8 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
    --speculative-algo EAGLE \
    --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
    --max-running-requests 128 \
    --cuda-graph-max-bs 64 \
    --mem-fraction-static 0.82 \
    --enable-metrics \
    --enable-mfu-metrics \
    --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
    --host 0.0.0.0 --port 8088
```

> **公式约束**：`max_running_requests × draft_tokens ≤ DISPATCH`，即 V4-Pro `128 × 2 = 256 ≤ 256` ✓
> **Flash 同配置**：`--max-running-requests 512` + `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024`（Flash KV 12.8× 大，slice 上限可放大 4×）

### 2.4 配置三：高吞吐（TP=8 DP=8 + DeepEP + 无 spec）

适用：高并发、批吞吐优先（≥ 200），单机最高吞吐配方。下面以 V4-Pro 为例。

```bash
sudo docker run -d \
  --name deepseek-v4-pro \
  --gpus all --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v /lssd/models:/lssd/models:ro \
  -v /lssd/cache:/root/.cache \
  -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
  -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  -e NCCL_IB_DISABLE=1 \
  --restart unless-stopped \
  lmsysorg/sglang:deepseek-v4-blackwell \
  python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path /lssd/models/DeepSeek-V4-Pro \
    --served-model-name DeepSeek-V4-Pro \
    --tp 8 --dp 8 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
    --max-running-requests 256 \
    --cuda-graph-max-bs 64 \
    --mem-fraction-static 0.82 \
    --enable-metrics \
    --enable-mfu-metrics \
    --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
    --host 0.0.0.0 --port 8088
```

> **公式约束**：V4-Pro `256 × 1 = 256 ≤ 256` ✓（无 spec → draft_tokens=1）
> **Flash 同配置**：`--max-running-requests 1024` + `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024`

### 2.5 等待就绪 + 启动压测

```bash
# 捕获日志（fork 后台）
sudo docker logs -f deepseek-v4-flash >& /lssd/logs/sglang_<编号>.log &

# 等待就绪（cache 复用约 4-6 分钟，首次约 10-15 分钟）
until curl -sf http://127.0.0.1:8088/v1/models | grep -q DeepSeek-V4-Flash; do sleep 10; done

# 启动压测
cd ~/code/b200_vllm_opt
./benchmark.sh "DeepSeek-V4-Flash" \
  --parallel "1 2 4 8 20 40 60 80 100 120 140 160 180 200 300 400 500 600 700 800 900 1000" \
  --number   "10 20 40 80 200 400 600 800 1000 1200 1400 1600 1800 2000 3000 4000 5000 6000 7000 8000 9000 10000"
```

---

## 3. 部署：vLLM baseline

### 3.1 启动命令（V4-Pro）

```bash
sudo docker run -d \
  --name deepseek-v4-pro \
  --gpus all --privileged --ipc=host \
  --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v /lssd/models:/lssd/models:ro \
  -v /lssd/cache:/root/.cache \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  --restart unless-stopped \
  vllm/vllm-openai:deepseekv4-cu130 \
  /lssd/models/DeepSeek-V4-Pro \
  --served-model-name DeepSeek-V4-Pro \
  --host 0.0.0.0 --port 8088 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --data-parallel-size 8 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --attention_config.use_fp4_indexer_cache=True \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --speculative_config '{"method":"mtp","num_speculative_tokens":2}'
```

### 3.2 ⚠️ vLLM Docker 必须挂 cache 卷

vLLM 容器在 `docker rm` 后 **所有 JIT/Inductor/torch.compile 编译产物丢失**，下次启动需要 5–10 分钟全量重编译。

| 目录 | 内容 |
|---|---|
| `/root/.cache/vllm/` | torch.compile / Inductor 编译产物（含 piecewise CUDA graph 元数据）|
| `/root/.cache/torch/` | PyTorch Inductor 内核缓存 |
| `/root/.cache/flashinfer/` | flashinfer JIT cubin |
| `/root/.cache/deep_gemm/` | DeepGEMM JIT cubin |

**推荐**：单行整体挂载 `/root/.cache`：

```bash
-v /lssd/cache:/root/.cache
```

Docker 嵌套挂载行为：更具体的路径优先生效，HF 走主机，其它（vllm/torch/flashinfer/...）全部落到 `/lssd/cache`。

### 3.3 vLLM 三套对应方案（设计待跑）

按 V4-Pro `--performance-mode` 源码事实（`vllm/config/vllm.py:1368`、`vllm/engine/arg_utils.py:2156`），三套调整为：

| 场景 | --performance-mode | spec | --max-num-seqs | 拓扑 |
|---|---|---|---|---|
| 低延迟 | interactivity | MTP n=3 | 32 | TP=8（无 DP）|
| 平衡 | interactivity | MTP n=1 | 128 | DP=8 + EP |
| 高吞吐 | throughput | 无 | 256 或不设（→ 自动 ×2 = 2048）| DP=8 + EP |

> `interactivity` 模式 cudagraph 覆盖 `range(1, 33)` 整数，MTP 验证 batch 不会 fallback 到 eager。
> `throughput` 模式将 `max_num_batched_tokens` 和 `max_num_seqs` 各 ×2（仅当用户没显式传值）。
> 全代码库 grep `performance_mode`：**仅这两处真正生效**，文档声称的 "latency-oriented kernels" 实际不影响 backend 选择。

---

## 4. 性能数据

> 所有压测 Success Rate = 100%，下表省略；列说明：Avg Lat = 平均端到端延迟（s），TTFT = 首 token 延迟（avg / P99，单位 s），TPOT = 单 token 时间（avg / P99，单位 ms），toks/s = 输出生成吞吐。

### 4.1 V4-Pro SGLang 三套（#199 / #200 / #201）

#### #199 — 低延迟（TP=8 + MXFP4 + EAGLE n=3）

| 并发 | RPS | Avg Lat | TTFT avg | TTFT P99 | TPOT avg | TPOT P99 | toks/s | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.35 | 2.81 s | 1.544 s | 3.171 s | 6 ms | 7 ms | 71 | ⚠️ @1 冷启动 |
| 2 | 0.67 | 3.00 s | 0.828 s | 1.864 s | 11 ms | 17 ms | 133 | |
| 4 | 0.86 | 4.63 s | 1.057 s | 2.748 s | 18 ms | 30 ms | 172 | |
| 8 | 1.20 | 6.58 s | 0.895 s | 3.416 s | 29 ms | 49 ms | 241 | |
| **20** | **1.56** | 12.73 s | 1.362 s | 8.866 s | 57 ms | 85 ms | **312** | 🏆 sweet spot |
| 40 | 0.30 | 132.34 s | 2.469 s | 15.166 s | 653 ms | 1568 ms | 60 | ⚠️ 严重过载 |
| 60 | 0.16 | 370.18 s | 10.341 s | 24.513 s | 1808 ms | 4321 ms | 32 | ⚠️ 接近不可用 |

→ V4-Pro 低延迟 **不能跑 ≥ 40 并发**。

> **@1 TTFT 1.54s 偏高的原因**（更新版，详见 §4.6）：**warmup 不足**导致测量假象，**不是 EAGLE/Pro 真实性能问题**。#209 用同配方 + warmup n=100 重测，formal @1 TTFT avg 回落到 **0.49 s**（与 #201 无 spec、#202 vLLM MTP 同量级）。

#### #209 — 同 #199 配方 + warmup n=100（@1 验证）

| 阶段 | Avg TTFT | P50 | P90 | P95 | P99 | 备注 |
|---|---:|---:|---:|---:|---:|---|
| #199 formal (warmup n=4) | 1.544 s | — | — | — | 3.171 s | ⚠️ 失真 |
| **#209 formal (warmup n=100)** | **0.491 s** | **0.448 s** | **0.500 s** | **0.653 s** | **3.815 s** | 真实值 |

→ **真实 V4-Pro 低延迟 @1 TTFT ≈ 0.49 s**；#199 表中 1.54 s 是 evalscope 变长输入 × spec decoding × JIT 形状特异性的短 warmup 假象。详细推理过程、为什么 warmup n=4 不够 n=100 才行，见 **§4.6**。


#### #200 — 平衡（TP=8 DP=8 + DeepEP + EAGLE n=1, max_req=128）

| 并发 | RPS | Avg Lat | TTFT avg | TTFT P99 | TPOT avg | TPOT P99 | toks/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.26 | 3.90 s | 1.510 s | 4.082 s | 12 ms | 13 ms | 51 |
| 2 | 0.49 | 4.06 s | 0.932 s | 1.886 s | 16 ms | 22 ms | 98 |
| 4 | 0.71 | 5.57 s | 1.161 s | 4.597 s | 22 ms | 42 ms | 143 |
| 8 | 0.93 | 8.54 s | 1.575 s | 6.459 s | 35 ms | 60 ms | 186 |
| 20 | 1.60 | 12.38 s | 1.556 s | 6.694 s | 54 ms | 88 ms | 320 |
| 40 | 2.53 | 15.75 s | 1.470 s | 5.668 s | 72 ms | 101 ms | 506 |
| 60 | 3.06 | 19.47 s | 1.714 s | 8.184 s | 89 ms | 125 ms | 611 |
| 80 | 3.38 | 23.53 s | 2.046 s | 10.727 s | 108 ms | 148 ms | 676 |
| **100** | **3.91** | 25.46 s | 2.255 s | 12.723 s | 117 ms | 163 ms | **782** |

#### #201 — 高吞吐（TP=8 DP=8 + DeepEP + 无 spec, max_req=256）

| 并发 | RPS | Avg Lat | TTFT avg | TTFT P99 | TPOT avg | TPOT P99 | toks/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.23 | 4.26 s | 0.783 s | 1.054 s | 18 ms | 18 ms | 47 |
| 2 | 0.43 | 4.63 s | 0.990 s | 1.264 s | 18 ms | 19 ms | 86 |
| 4 | 0.78 | 5.14 s | 1.252 s | 2.753 s | 19 ms | 28 ms | 156 |
| 8 | 1.43 | 5.60 s | 1.423 s | 2.480 s | 21 ms | 27 ms | 285 |
| 20 | 2.09 | 9.56 s | 1.528 s | 4.487 s | 40 ms | 62 ms | 417 |
| 40 | 3.36 | 11.88 s | 2.428 s | 5.054 s | 48 ms | 69 ms | 671 |
| 60 | 3.83 | 15.62 s | 2.399 s | 7.631 s | 66 ms | 94 ms | 766 |
| 80 | 4.37 | 18.25 s | 2.669 s | 9.940 s | 78 ms | 111 ms | 874 |
| 100 | 4.74 | 21.03 s | 3.580 s | 12.551 s | 88 ms | 131 ms | 948 |
| 120 | 4.97 | 24.06 s | 3.788 s | 14.800 s | 102 ms | 150 ms | 995 |
| 140 | 4.98 | 28.03 s | 3.180 s | 17.282 s | 125 ms | 171 ms | 996 |
| 160 | 5.28 | 30.25 s | 3.674 s | 18.996 s | 134 ms | 188 ms | 1055 |
| 180 | 5.40 | 33.23 s | 4.965 s | 21.326 s | 142 ms | 196 ms | 1081 |
| **200** | **5.73** | 34.80 s | 4.882 s | 22.898 s | 150 ms | 212 ms | **1146** | 🏆 V4-Pro 单机峰值 |

### 4.2 V4-Flash SGLang 三套（#206 / #207 / #208）

#### #206 — 低延迟（TP=8 + MXFP4 + EAGLE n=3）

| 并发 | RPS | Avg Lat | TTFT avg | TTFT P99 | TPOT avg | TPOT P99 | toks/s | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.80 | 1.25 s | 0.298 s | 0.402 s | 5 ms | 5 ms | 160 | |
| 2 | 1.33 | 1.49 s | 0.324 s | 0.551 s | 6 ms | 7 ms | 267 | |
| 4 | 1.73 | 2.29 s | 0.468 s | 1.653 s | 9 ms | 20 ms | 347 | |
| 8 | 2.52 | 3.13 s | 0.466 s | 1.829 s | 13 ms | 20 ms | 504 | |
| 20 | 3.14 | 6.33 s | 0.759 s | 4.171 s | 28 ms | 47 ms | 628 | |
| 40 | 3.63 | 10.97 s | 1.020 s | 7.595 s | 50 ms | 77 ms | 725 | |
| 60 | 3.86 | 15.45 s | 1.366 s | 11.791 s | 71 ms | 111 ms | 773 | |
| 80 | 3.90 | 20.44 s | 1.794 s | 20.131 s | 94 ms | 151 ms | 780 | |
| 100 | 4.03 | 24.71 s | 2.258 s | 19.227 s | 113 ms | 178 ms | 807 | |
| 120 | 4.10 | 29.20 s | 2.504 s | 23.984 s | 134 ms | 225 ms | 819 | |
| 140 | 4.00 | 34.89 s | 3.351 s | 36.967 s | 159 ms | 282 ms | 800 | |
| 160 | 4.11 | 38.81 s | 3.114 s | 31.396 s | 179 ms | 302 ms | 822 | |
| **180** | **4.14** | 43.33 s | 3.589 s | 35.364 s | 200 ms | 325 ms | **829** | 🏆 sweet spot |
| 200 | 1.45 | 137.97 s | 3.838 s | 39.104 s | 674 ms | 1397 ms | 290 | ⚠️ 崩溃 |

→ Flash 低延迟上限 **180**（vs Pro 仅 20），上限提升 **9×**。

#### #207 — 平衡（TP=8 DP=8 + DeepEP + EAGLE n=1, max_req=512）

| 并发 | RPS | Avg Lat | TTFT avg | TTFT P99 | TPOT avg | TPOT P99 | toks/s | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.48 | 2.07 s | 0.553 s | 0.741 s | 8 ms | 8 ms | 96 | |
| 2 | 0.80 | 2.50 s | 0.570 s | 0.686 s | 10 ms | 11 ms | 159 | |
| 4 | 1.22 | 3.26 s | 0.621 s | 1.030 s | 13 ms | 19 ms | 244 | |
| 8 | 1.96 | 4.07 s | 0.663 s | 0.938 s | 17 ms | 24 ms | 392 | |
| 20 | 3.22 | 6.17 s | 0.774 s | 1.992 s | 27 ms | 39 ms | 644 | |
| 40 | 4.77 | 8.32 s | 0.906 s | 3.072 s | 37 ms | 55 ms | 955 | |
| 60 | 5.60 | 10.65 s | 1.080 s | 4.611 s | 48 ms | 64 ms | 1121 | |
| 80 | 6.94 | 11.45 s | 1.256 s | 5.503 s | 51 ms | 73 ms | 1388 | |
| 100 | 7.30 | 13.62 s | 1.371 s | 7.023 s | 62 ms | 84 ms | 1461 | |
| 120 | 8.53 | 14.00 s | 2.086 s | 8.304 s | 60 ms | 95 ms | 1706 | |
| 140 | 8.21 | 16.98 s | 1.612 s | 9.529 s | 77 ms | 106 ms | 1642 | |
| 160 | 8.77 | 18.15 s | 1.816 s | 10.487 s | 82 ms | 114 ms | 1755 | |
| 180 | 9.70 | 18.50 s | 1.947 s | 12.306 s | 83 ms | 117 ms | 1939 | |
| 200 | 9.73 | 20.48 s | 2.140 s | 13.307 s | 92 ms | 128 ms | 1947 | |
| 300 | 11.66 | 25.66 s | 3.193 s | 20.059 s | 113 ms | 169 ms | 2331 | |
| **400** | **11.96** | 33.34 s | 3.896 s | 25.835 s | 148 ms | 205 ms | **2393** | 🏆 |
| 500 | 8.66 | 57.62 s | 4.443 s | 32.572 s | 267 ms | 390 ms | 1733 | ⚠️ max_req 排队 |

→ 17 档 0 失败。500 回落原因：`max_req=512 / DP=8 = 64/slice`，500 conv = 62.5/slice 接近 64 上限触发排队（**非 KV 饱和**，KV 实际仅占 7%）。

#### #208 — 高吞吐（TP=8 DP=8 + DeepEP + 无 spec, max_req=1024）

> 数据从 evalscope 各档 `benchmark_summary.json` + `benchmark_percentile.json` 直读（log 层 SQLite 句柄耗尽影响 summary table，但每档原始 JSON 完整）。

| 并发 | RPS | Avg Lat | TTFT avg | TTFT P99 | TPOT avg | TPOT P99 | toks/s | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.38 | 2.62 s | 0.499 s | 0.581 s | 11 ms | 11 ms | 76 | |
| 2 | 0.71 | 2.81 s | 0.633 s | 0.889 s | 11 ms | 12 ms | 142 | |
| 4 | 1.39 | 2.88 s | 0.665 s | 0.740 s | 11 ms | 12 ms | 277 | |
| 8 | 2.62 | 3.05 s | 0.729 s | 0.914 s | 12 ms | 12 ms | 524 | |
| 20 | 4.10 | 4.85 s | 0.880 s | 1.667 s | 20 ms | 26 ms | 821 | |
| 40 | 6.67 | 5.99 s | 1.521 s | 3.145 s | 22 ms | 36 ms | 1333 | |
| 60 | 7.33 | 8.14 s | 1.230 s | 4.151 s | 35 ms | 47 ms | 1466 | |
| 80 | 9.02 | 8.86 s | 1.841 s | 5.445 s | 35 ms | 55 ms | 1804 | |
| 100 | 8.15 | 12.21 s | 1.537 s | 6.549 s | 54 ms | 73 ms | 1631 | |
| 120 | 8.87 | 13.46 s | 1.776 s | 7.748 s | 59 ms | 79 ms | 1775 | |
| 140 | 10.88 | 12.84 s | 3.186 s | 8.914 s | 49 ms | 92 ms | 2176 | |
| 160 | 10.41 | 15.33 s | 2.252 s | 10.311 s | 66 ms | 98 ms | 2082 | |
| 180 | 10.97 | 16.35 s | 2.924 s | 11.634 s | 68 ms | 106 ms | 2194 | |
| 200 | 11.51 | 17.34 s | 4.550 s | 12.643 s | 64 ms | 120 ms | 2301 | |
| 300 | 12.40 | 24.10 s | 3.843 s | 18.581 s | 102 ms | 151 ms | 2481 | |
| 400 | 14.17 | 28.16 s | 10.180 s | 24.877 s | 90 ms | 189 ms | 2835 | |
| 500 | 14.06 | 35.13 s | 16.314 s | 32.315 s | 95 ms | 165 ms | 2811 | |
| **600** | **14.67** | 40.84 s | 17.643 s | 36.283 s | 117 ms | 194 ms | **2933** | 🏆 单机峰值 |
| 700 | 14.47 | 47.73 s | 20.998 s | 43.696 s | 134 ms | 236 ms | 2895 | plateau |
| 800 | 14.48 | 53.94 s | 24.323 s | 48.596 s | 149 ms | 262 ms | 2895 | plateau |
| 900 | 14.58 | 61.02 s | 26.782 s | 55.796 s | 172 ms | 299 ms | 2915 | plateau |
| 1000 | — | — | — | — | — | — | — | parallel=1000 evalscope SQLite 句柄耗尽未产生数据 |

→ 21 档 0 失败（parallel=1000 因 evalscope SQLite 句柄耗尽未跑，与 SGLang 无关）。
→ 600-900 长 plateau，KV 利用率 13% 远未饱和；瓶颈在 DeepEP/scheduler。

### 4.3 V4-Pro vLLM baseline (#202)

DP=8 + EP + MTP n=2 + FP8 KV，**默认 max_num_seqs=1024**（未限制）。

| 并发 | RPS | Avg Lat | TTFT avg | TTFT P99 | TPOT avg | TPOT P99 | toks/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | 0.33 | 3.03 s | 0.490 s | 0.687 s | 13 ms | 15 ms | **66** | 🏆 单请求 (MTP 加速) |
| 2 | 0.47 | 4.27 s | 0.906 s | 7.911 s | 17 ms | 53 ms | 94 |
| 4 | 0.67 | 5.91 s | 0.987 s | 9.796 s | 25 ms | 100 ms | 135 |
| 8 | 0.96 | 8.14 s | 0.921 s | 8.703 s | 36 ms | 86 ms | 192 |
| 20 | 1.94 | 10.18 s | 0.930 s | 2.823 s | 46 ms | 73 ms | 388 |
| 40 | 2.31 | 17.14 s | 1.349 s | 4.815 s | 79 ms | 122 ms | 462 |
| 60 | 2.63 | 22.59 s | 1.703 s | 5.963 s | 105 ms | 166 ms | 527 |
| 80 | 3.06 | 25.93 s | 2.010 s | 7.899 s | 120 ms | 198 ms | 611 |
| 100 | 3.43 | 28.97 s | 2.248 s | 10.228 s | 134 ms | 226 ms | 686 |
| 120 | 3.59 | 33.21 s | 2.533 s | 11.557 s | 154 ms | 218 ms | 718 |
| 140 | 3.87 | 35.91 s | 2.530 s | 12.617 s | 168 ms | 270 ms | 774 |
| 160 | 4.40 | 36.11 s | 2.749 s | 15.360 s | 168 ms | 237 ms | 881 |
| 180 | 4.21 | 42.52 s | 3.029 s | 18.212 s | 198 ms | 275 ms | 841 |
| **200** | **4.65** | 42.73 s | 3.056 s | 17.351 s | 199 ms | 326 ms | **930** | 🏆 |

### 4.4 SGLang vs vLLM 对比（V4-Pro）

| 并发 | SGLang #201 toks/s | vLLM #202 toks/s | SGLang 优势 |
|---:|---:|---:|---:|
| 1 | 47 | **66** | -29%（vLLM MTP 占优）|
| 20 | **417** | 388 | +7% |
| 60 | **766** | 527 | **+45%** |
| 100 | **948** | 686 | **+38%** |
| 200 | **1146** | 930 | **+23%** |

**结论**：单请求 vLLM MTP n=2 占优；并发 ≥ 20 SGLang 高吞吐配置全面反超。

### 4.5 V4-Pro vs V4-Flash 对比

| 配置 | V4-Pro 峰值 toks/s | V4-Flash 峰值 toks/s | Flash 倍数 |
|---|---:|---:|---:|
| 低延迟 | 312 (#199 @20) | 829 (#206 @180) | 2.66× |
| 平衡 | 782 (#200 @100) | 2393 (#207 @400) | 3.06× |
| 高吞吐 | 1146 (#201 @200) | **2933 (#208 @600)** | **2.56×** |

### 4.6 @1 并发 TTFT 异常分析 + #209 对照实验

#### 4.6.1 现象

| 编号 | 模型 | 引擎 | spec | @1 Avg | @1 P99 | P99/Avg | 异常 |
|---|---|---|---|---:|---:|---:|:---:|
| **#199** | V4-Pro 低延迟 | SGLang | EAGLE n=3 | **1.544 s** | **3.171 s** | **2.05** | ⚠️ |
| **#200** | V4-Pro 平衡 | SGLang | EAGLE n=1 | **1.510 s** | **4.082 s** | **2.70** | ⚠️ |
| #201 | V4-Pro 高吞吐 | SGLang | 无 | 0.783 s | 1.054 s | 1.35 | ✓ |
| #202 | V4-Pro baseline | vLLM | MTP n=2 | 0.490 s | 0.687 s | 1.40 | ✓ |
| #206 | V4-Flash 低延迟 | SGLang | EAGLE n=3 | 0.298 s | 0.402 s | 1.35 | ✓ |
| #207 | V4-Flash 平衡 | SGLang | EAGLE n=1 | 0.553 s | 0.741 s | 1.34 | ✓ |
| #208 | V4-Flash 高吞吐 | SGLang | 无 | 0.499 s | 0.581 s | 1.16 | ✓ |

异常仅在 **V4-Pro + SGLang + EAGLE** 交集出现。

#### 4.6.2 #209 对照实验（warmup n=100, formal n=100, 同 #199 配方）

为定位根因，重跑 #199 配方但把 warmup 从 `n=4` 加到 `n=100`，formal 同样 `n=100`：

| Percentile | warmup phase TTFT | formal phase TTFT | 变化 |
|---:|---:|---:|---:|
| p50 | 0.46 s | 0.45 s | ≈ |
| p80 | 0.55 s | 0.48 s | -13% |
| **p90** | **2.10 s** | **0.50 s** | **-76%** |
| **p95** | **3.75 s** | **0.65 s** | **-83%** |
| **p98** | **7.03 s** | **1.05 s** | **-85%** |
| **p99** | **17.37 s** | **3.81 s** | **-78%** |
| **avg** | 0.95 s | **0.49 s** | **-48%** |

**结论性发现**：充分 warmup 后 V4-Pro + EAGLE n=3 在 @1 的真实 TTFT = **0.49s**（与 #201 无 spec 0.78s、#202 vLLM MTP 0.49s 量级一致）。原 #199 @1=1.54s 是 warmup 不足的测量假象。

#### 4.6.3 为什么 warmup n=4 不够，n=100 才行？

evalscope `--prefix-length 0 --min-prompt-length 4500 --max-prompt-length 4500` 看似固定长度，**但实际由 `random` 数据集生成的输入 token 长度跨度大**（实测 4664–9565 tokens，因 tokenizer 编码差异和随机 padding）：

| Warmup 样本 vs 输入长度覆盖 | 命中冷路径概率 |
|---|---|
| n=4：覆盖 ~4 种长度 | ~30% formal 请求命中冷路径 |
| n=100：覆盖 ~100 种长度 | ~1% formal 请求命中冷路径（仅 P99 残留 1 个 outlier） |

**SGLang + EAGLE + Pro 三件叠加**触发的冷路径：
1. **chunked-prefill 4096 边界**：长度 4500 vs 9000 走不同 chunk 数，对应不同 cudagraph batch
2. **EAGLE draft model 在不同 prefill 长度下的 cudagraph 形状**：draft model 也要做对应长度的前向
3. **DeepGEMM JIT autotune**：每次新输入长度首次出现时编译特定 GEMM kernel
4. **MoE flashinfer_mxfp4 在不同 token 数 batch 下的不同 dispatch 路径**

Flash 不显著的原因：① Flash 权重小 5×，draft model 也小 ~4×；② 即使有同样冷路径，Flash 上的绝对时间太短（0.3s 级），相对差异被吞掉。

#### 4.6.4 实践指引

- ✅ **生产真实 TTFT**：V4-Pro 低延迟 @1 真实 ≈ 0.5s，不是文档表中的 1.54s
- ⚠️ **压测方法学**：变长输入 + spec decoding 配置必须 `--warmup-number ≥ 100`，否则 @1 / @2 等小并发档位的 avg/P99 都失真
- ⚠️ **生产首请求不会更慢**：服务持续运行后所有路径已热，首个用户请求与第 1000 个无差异（除非 cudagraph 池被驱逐，未观察到）
- p99=3.81s 残留 outlier 仍可能与极端长输入（>9000 tokens）相关，但 99% 用例下 TTFT 在 0.4-0.65s 区间

---

## 5. 公式约束与调优规则

### 5.1 ⭐ 核心公式（DeepEP 模式）

```
max_running_requests × draft_tokens ≤ SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK
```

| 场景 | draft_tokens | 推荐配置（Flash）| 公式验证 |
|---|---|---|---|
| 低延迟 | 4（EAGLE n=3 + 1 verify）| (TP=8 无 DeepEP) | n/a |
| 平衡 | 2（EAGLE n=1 + 1）| max_req=512, DISPATCH=1024 | 512×2 = 1024 ✓ |
| 高吞吐 | 1（无 spec）| max_req=1024, DISPATCH=1024 | 1024×1 = 1024 ✓ |

### 5.2 max_req 在 DP 模式下的平分

`max-running-requests` 是 CLI 总值，DP=8 → 每 slice 实际生效 = 总值 / 8。

| 模型 | CLI max_req | DP | per slice |
|---|---:|---:|---:|
| Pro 平衡 #200 | 128 | 8 | 16 |
| Pro 高吞吐 #201 | 256 | 8 | 32 |
| Flash 平衡 #207 | 512 | 8 | 64 |
| Flash 高吞吐 #208 | 1024 | 8 | 128 |

⚠️ **诊断 RPS 回落的方法**：当 RPS 不再上涨，看 `当前并发数 / DP_size` 是否 ≈ `max_req / DP_size`，若是则属于调度排队，提高 `max_req`（同步提高 DISPATCH）即可解决。

### 5.3 SGLang `--max-running-requests` 默认值（源码事实）

| 路径 | 默认值 | 说明 |
|---|---|---|
| 字段定义 | `None` | `server_args.py:313` |
| DeepseekV4 路径 | **256** | `server_args.py` 自动覆盖 |
| Spec decoding 路径 | 48 | 仅当未被前路径设置时生效（被覆盖了）|

**⚠️ 行为**：DeepseekV4 + EAGLE → 实际生效 **256**（不是 48）。建议**显式设置** `--max-running-requests N` 避免不可预期。

### 5.4 vLLM `--performance-mode` 行为（源码事实）

源码版本：`0.17.1rc1.dev237+g0a0a1a198`。

| 模式 | cudagraph_capture_sizes | max_num_batched_tokens | max_num_seqs |
|---|---|---:|---:|
| balanced（默认）| `[1,2,4]` + `range(8,256,8)` + `range(256,max,16)` | 8192 | 1024 |
| interactivity | `range(1, 33)` 整数全覆盖 + step 8/16 段 | 8192 | 1024 |
| throughput | 同 balanced（更长尾段） | **16384**（×2）| **2048**（×2）|

> 全代码库 grep `performance_mode`：**仅 cudagraph_capture_sizes 和 max_num_batched_tokens/max_num_seqs 受影响**，无 kernel/backend 切换。
> 文档声称的 "latency-oriented kernels" 实际不会触发任何代码路径。

---

## 6. 环境变量参考

### 6.1 SGLang

| 变量 | 用途 | 推荐值 |
|---|---|---|
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | DeepEP per-rank dispatch token 上限 | 256（Pro 平衡）/ 1024（Flash） |
| `SGLANG_JIT_DEEPGEMM_PRECOMPILE` | DeepGEMM 启动时预编译 | **0**（禁用，加快启动）|
| `SGLANG_ENABLE_SPEC_V2` | EAGLE Spec v2 调度 | 1（仅低延迟用）|
| `NCCL_IB_DISABLE` | 禁用 IB（单机部署）| 1 |

### 6.2 vLLM

| 变量 | 用途 | 推荐值 |
|---|---|---|
| `VLLM_ENGINE_READY_TIMEOUT_S` | 启动超时（含 JIT + 权重加载）| 3600 |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | CUDA graph 显存预估，避免 profile_run OOM | 1 |
| `HF_HUB_OFFLINE` | 禁用 HF 远程访问（用本地 /lssd/models）| 1 |

### 6.3 通用

| 变量 | 用途 | 推荐值 |
|---|---|---|
| `CUDA_HOME` | 与 PyTorch cu130 匹配 | `/usr/local/cuda-13.0` |

---

## 7. 踩坑记录

### 7.1 V4-Pro 低延迟 `parallel ≥ 40` 严重过载

**症状**：#199 在 parallel=40 RPS 暴跌到 0.30（vs @20=1.56），TPOT 从 57ms 飙到 653ms。

**根因**：低延迟配置 `chunked_prefill=4096` + `EAGLE n=3, draft=4` 在 40 并发时：
- 每个 forward step 验证 batch ≈ 40 × 4 = 160 tokens
- 调度器 + cudagraph 设计针对 ≤ 32 batch 优化
- 超出 chunked_prefill 边界后 prefill/decode 互相阻塞

**对策**：
- V4-Pro 低延迟压测 cap `parallel ≤ 40`
- V4-Flash 同配置 cap 提升到 **180**（KV 大 9× → 调度容量翻倍）

### 7.2 V4-Flash 平衡 `TP=4 DP=2 + EAGLE` deadlock（已知 bug）

**症状**：旧版 #204（TP=4 DP=2，开 EAGLE）首次 chat 请求 hang，curl 永久无响应。

**根因**：SGLang `eagle_worker_v2.py:647` 在 DP attention 模式下抛
```
AssertionError: short-circuiting allreduce will lead to hangs
```
DP1 rank 走 `forward_idle` 路径短路了 EAGLE worker 期望的 4-way TP allreduce 同步。

**触发条件**：DP > 1 + TP > 1 + EAGLE 三件同时满足。

**Workaround**：
- 改用 TP=8 DP=8（每 DP slice TP=1，无 allreduce 短路问题）✅ 已验证 #207 跑成功
- 或 去掉 EAGLE（变成纯吞吐配方）
- 或 关 DP attention（失去 DP 优势）

### 7.3 V4-Flash 低延迟 `parallel=200` 崩溃

**症状**：#206 在 parallel=200 RPS 从 4.14 (@180) 暴跌到 1.45，TPOT 从 200 ms → 674 ms，toks/s 829 → 290。

**根因**：与 7.1 同类，超过 cudagraph 优化区域。

**对策**：V4-Flash 低延迟 cap `parallel ≤ 180`。

### 7.4 vLLM Docker 不挂 cache 卷 → 每次重编译 5-10 分钟

**症状**：`docker rm` 后下次启动 JIT 重新编译。

**根因**：vLLM 把 cache 写在 `/root/.cache/{vllm,torch,flashinfer,deep_gemm,...}`，这些目录在容器**可写层**里。

**对策**：见 §3.2，加挂 `-v /lssd/cache:/root/.cache`。

### 7.5 evalscope SQLite 句柄耗尽（连续 ≥ 22 档时）

**症状**：#208 跑到第 22 档 parallel=1000 直接报
```
ERROR: Exception in async function 'statistic_benchmark_metric': unable to open database file
```

**根因**：evalscope 每档生成一个 `outputs/.../parallel_X_number_Y/benchmark_data.db`，22+ 档同时打开超过 ulimit。

**对策**：
- 单次 sweep 控制在 ≤ 21 档
- 或拆成多个 ./benchmark.sh 调用（每次 ≤ 10 档）

### 7.6 evalscope `signal only works in main thread` 误报

**症状**：log 出现 `Traceback ... ValueError: signal only works in main thread of the main interpreter`，但实际压测继续运行。

**根因**：evalscope 内部 `BaseEventLoop.__del__` 在 GC 时清理 signal handler，被 `Exception ignored in __del__` 显式忽略。

**对策**：监控 grep 排除此模式（已加入 `grep -v "signal only works"`）。

---

## 8. 端到端时间线

| 任务 | 耗时 | 说明 |
|---|---|---|
| SGLang 镜像 pull | 5-15 min | 90 GB |
| vLLM 镜像 pull | 5-10 min | 31.5 GB |
| **首次启动**（V4-Pro，含 JIT）| 10-25 min | 权重加载 + cudagraph capture |
| **复用 cache 启动**（V4-Pro）| 4-6 min | 仅权重加载 + cudagraph |
| 首次启动（V4-Flash）| 7-12 min | Flash 权重小，快很多 |
| 复用 cache 启动（V4-Flash）| 3-5 min | |
| 单轮压测 V4-Pro 低延迟（#199, 7 档 1→60）| ~90 min | parallel=40/60 段过载，单档 ~60 min |
| 单轮压测 V4-Pro 平衡（#200, 9 档 1→100）| ~28 min | |
| 单轮压测 V4-Pro 高吞吐（#201, 14 档 1→200）| ~56 min | |
| 单轮压测 V4-Flash 低延迟（#206, 14 档 1→200）| ~80 min | parallel=200 段崩溃 |
| 单轮压测 V4-Flash 平衡（#207, 17 档 1→500）| ~70 min | |
| 单轮压测 V4-Flash 高吞吐（#208, 22 档 1→1000）| ~105 min | parallel=1000 evalscope SQLite 崩 |
| 单轮压测 vLLM baseline（#202, 14 档 1→200）| ~60 min | |

---

## 9. 操作手册速查

### 9.1 切换场景标准流程

```bash
# 1. 停旧容器
sudo docker stop deepseek-v4-flash 2>/dev/null
sudo docker rm   deepseek-v4-flash 2>/dev/null

# 2. 启新容器（见 §2 / §3）
sudo docker run -d ...

# 3. 捕获日志
sudo docker logs -f deepseek-v4-flash >& /lssd/logs/sglang_<编号>.log &

# 4. 等就绪
until curl -sf http://127.0.0.1:8088/v1/models | grep -q DeepSeek-V4-Flash; do sleep 10; done

# 5. 跑 benchmark
cd ~/code/b200_vllm_opt
./benchmark.sh "DeepSeek-V4-Flash" \
  --parallel "<档位>" --number "<档位>" \
  > /lssd/logs/bench_<编号>.log 2>&1 &

# 6. 监控（可选，cc-connect 推送到 Feishu）
tail -F /lssd/logs/bench_<编号>.log \
  | grep --line-buffered -E "结果已保存|Failed requests.*[1-9]|FATAL" \
  | notify-feishu --prefix "✅ #<编号>"
```

### 9.2 编号映射

| 编号 | 模型 | 引擎 | 配置 | 文件 |
|---:|---|---|---|---|
| #199 | V4-Pro | SGLang | 低延迟 | `20260424_134915_199_deepseek_v4_pro_sglang_lowlatency.md` |
| #200 | V4-Pro | SGLang | 平衡 | `20260424_142125_200_deepseek_v4_pro_sglang_balanced.md` |
| #201 | V4-Pro | SGLang | 高吞吐 | `20260424_152211_201_deepseek_v4_pro_sglang_throughput.md` |
| #202 | V4-Pro | vLLM | baseline | `20260424_171812_202_deepseek_v4_pro_vllm_baseline.md` |
| #206 | V4-Flash | SGLang | 低延迟 | `20260425_032400_206_deepseek_v4_flash_sglang_lowlatency.md` |
| #207 | V4-Flash | SGLang | 平衡 | `20260425_043116_207_deepseek_v4_flash_sglang_balanced.md` |
| #208 | V4-Flash | SGLang | 高吞吐 | `20260425_063127_208_deepseek_v4_flash_sglang_throughput.md` |
| #209 | V4-Pro | SGLang | 低延迟（warmup n=100 验证）| `20260427_034555_209_deepseek_v4_pro_sglang_lowlatency_warmup100.md` |

> #203 (V4-Flash 低延迟 旧版) 已被 #206 替代（旧版未跑到 200 档）；#204 (V4-Flash 平衡 旧版 TP=4 DP=2) 因 7.2 bug 失败未产生数据。
> #209 是 §4.6 @1 TTFT 异常的对照实验，证实 warmup 不足是 #199/#200 @1 数据失真的根因。

---

## 10. 下一步工作

- [ ] V4-Flash × vLLM 三套对照（按 §3.3 设计跑 #210-#212）
- [ ] V4-Pro / V4-Flash 长上下文 + LMCache MP 测试（参考 [05_kv_cache_and_lmcache.md](05_kv_cache_and_lmcache.md)）
- [ ] V4 多轮 agent 场景（ShareGPT or gen_lmcache_stress）
- [ ] V4-Pro 平衡用户在 #200 用 max_req=128 较保守，可测试 max_req=256 / 512 是否进一步提升
- [ ] SGLang vs vLLM 在低/平衡场景的细化对比

---

## 附录 A：参考资料

- 详细启动决策记录：[06_architecture_and_parallelism.md](06_architecture_and_parallelism.md)
- 全压测列表：[../benchmark_result/INDEX.md](../benchmark_result/INDEX.md)
- 标准化 CSV：[../benchmark_result/results.csv](../benchmark_result/results.csv)
- LMCache MP 命令：[05_kv_cache_and_lmcache.md](05_kv_cache_and_lmcache.md)
- vLLM 升级权限修复：见 [01_deployment_guide.md](01_deployment_guide.md)
