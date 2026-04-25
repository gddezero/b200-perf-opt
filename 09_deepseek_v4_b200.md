# DeepSeek V4 (Pro & Flash) 在 B200 × 8 上的部署与压测

> **测试时间**：2026-04-24 ~ 2026-04-25
> **硬件**：GCP A4 B200 SXM × 8 (192 GB HBM3e/卡)
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
1. **V4-Flash 高吞吐 = 单机峰值 2933 toks/s**，是 V4-Pro 高吞吐的 **2.56×**，源于 Flash 权重小 5×（21.83 vs 105.57 GB/卡）→ KV pool 大 9× → 调度容量翻倍
2. **SGLang 全面超越 vLLM baseline**：相同硬件 + 相同模型，SGLang 高吞吐 1146 vs vLLM 930 (#202 baseline)，+23%
3. **公式约束生效**：`max_running_requests × draft_tokens ≤ SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 是平衡/高吞吐配方的核心约束
4. **低延迟配置不能跑高并发**：V4-Pro @40、V4-Flash @200 起严重过载（EAGLE n=3 + chunked_prefill 4096 限制）

---

## 1. 硬件与模型概览

### 1.1 硬件

| 项 | 规格 |
|---|---|
| GPU | 8 × NVIDIA B200 SXM (192 GB HBM3e/卡) |
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

> Flash 权重为 Pro 的 **1/5**（21.83 / 105.57 = 21%），并非 1/8。但因为权重占用减小 → KV 空间大约 **9 倍以上**，这是 Flash 高吞吐性能反超的根本原因。

### 1.3 镜像

| 引擎 | 镜像 | 大小 |
|---|---|---|
| SGLang | `lmsysorg/sglang:deepseek-v4-blackwell` | 90 GB |
| vLLM | `vllm/vllm-openai:deepseekv4-cu130` | 31.5 GB |

---

## 2. 部署：SGLang 三套配置

### 2.1 通用前置

```bash
ssh maxwellx_google_com@forrest-b200-1

# 拉镜像（首次）
sudo docker pull lmsysorg/sglang:deepseek-v4-blackwell

# 预创建 cache 卷（多轮压测复用 JIT 编译产物）
sudo mkdir -p /lssd/cache && sudo chmod -R 777 /lssd/cache

# 切场景前必须停旧容器
sudo docker stop deepseek-v4-pro deepseek-v4-flash 2>/dev/null
sudo docker rm   deepseek-v4-pro deepseek-v4-flash 2>/dev/null
```

### 2.2 配置一：低延迟（TP=8 + MXFP4 + EAGLE n=3）

适用：单/小并发优先（< 40 for Pro，< 180 for Flash）。

```bash
sudo docker run -d \
  --name deepseek-v4-flash \
  --gpus all --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v /lssd/models:/lssd/models:ro \
  -v /lssd/cache:/root/.cache \
  -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  --restart unless-stopped \
  lmsysorg/sglang:deepseek-v4-blackwell \
  python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path /lssd/models/DeepSeek-V4-Flash \
    --served-model-name DeepSeek-V4-Flash \
    --tp 8 \
    --moe-runner-backend flashinfer_mxfp4 \
    --speculative-algo EAGLE \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --chunked-prefill-size 4096 \
    --disable-flashinfer-autotune \
    --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
    --host 0.0.0.0 --port 8088
```

> 切换到 V4-Pro 只需把 `DeepSeek-V4-Flash` 全部换成 `DeepSeek-V4-Pro`。

### 2.3 配置二：平衡（TP=8 DP=8 + DeepEP + EAGLE n=1）

适用：中等并发 + 低延迟（80 ≤ p ≤ 400），平衡场景生产首选。

```bash
sudo docker run -d \
  --name deepseek-v4-flash \
  --gpus all --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v /lssd/models:/lssd/models:ro \
  -v /lssd/cache:/root/.cache \
  -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
  -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  --restart unless-stopped \
  lmsysorg/sglang:deepseek-v4-blackwell \
  python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path /lssd/models/DeepSeek-V4-Flash \
    --served-model-name DeepSeek-V4-Flash \
    --tp 8 --dp 8 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
    --speculative-algo EAGLE \
    --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
    --max-running-requests 512 \
    --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
    --host 0.0.0.0 --port 8088
```

> **公式约束**：`max_running_requests × draft_tokens ≤ DISPATCH`，即 `512 × 2 = 1024 ≤ 1024` ✓
> V4-Pro 同配置取 `--max-running-requests 128`（V4-Pro KV 9× 小，每 slice 上限低）。

### 2.4 配置三：高吞吐（TP=8 DP=8 + DeepEP + 无 spec）

适用：高并发、批吞吐优先（≥ 200），单机最高吞吐配方。

```bash
sudo docker run -d \
  --name deepseek-v4-flash \
  --gpus all --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v /lssd/models:/lssd/models:ro \
  -v /lssd/cache:/root/.cache \
  -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
  -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  --restart unless-stopped \
  lmsysorg/sglang:deepseek-v4-blackwell \
  python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path /lssd/models/DeepSeek-V4-Flash \
    --served-model-name DeepSeek-V4-Flash \
    --tp 8 --dp 8 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
    --max-running-requests 1024 \
    --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
    --host 0.0.0.0 --port 8088
```

> **公式约束**：`1024 × 1 = 1024 ≤ 1024` ✓（无 spec → draft_tokens=1）
> V4-Pro 同配置取 `--max-running-requests 256`。

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

### 4.1 V4-Pro SGLang 三套（#199 / #200 / #201）

#### #199 — 低延迟（TP=8 + MXFP4 + EAGLE n=3）

| 并发 | RPS | TTFT | TPOT | toks/s | 备注 |
|---:|---:|---:|---:|---:|---|
| 1 | 0.35 | 1.54 s | 6 ms | 71 | |
| 8 | 1.20 | 0.90 s | 29 ms | 241 | |
| **20** | **1.56** | 1.36 s | 57 ms | **312** | 🏆 sweet spot |
| 40 | 0.30 | 2.47 s | 653 ms | 60 | ⚠️ 严重过载 |
| 60 | 0.16 | 10.34 s | 1808 ms | 32 | ⚠️ 接近不可用 |

→ V4-Pro 低延迟 **不能跑 ≥ 40 并发**。

#### #200 — 平衡（TP=8 DP=8 + DeepEP + EAGLE n=1, max_req=128）

| 并发 | RPS | TTFT | TPOT | toks/s |
|---:|---:|---:|---:|---:|
| 20 | 1.60 | 1.56 s | 54 ms | 320 |
| 40 | 2.53 | 1.47 s | 72 ms | 506 |
| 60 | 3.06 | 1.71 s | 89 ms | 611 |
| 80 | 3.38 | 2.05 s | 108 ms | 676 |
| **100** | **3.91** | 2.26 s | 117 ms | **782** | 🏆 |

#### #201 — 高吞吐（TP=8 DP=8 + DeepEP + 无 spec, max_req=256）

| 并发 | RPS | TTFT | TPOT | toks/s |
|---:|---:|---:|---:|---:|
| 100 | 4.74 | 3.58 s | 88 ms | 948 |
| 160 | 5.28 | 3.67 s | 134 ms | 1055 |
| **200** | **5.73** | 4.88 s | 150 ms | **1146** | 🏆 V4-Pro 单机峰值 |

### 4.2 V4-Flash SGLang 三套（#206 / #207 / #208）

#### #206 — 低延迟（TP=8 + MXFP4 + EAGLE n=3）

| 并发 | RPS | TTFT | TPOT | toks/s | 备注 |
|---:|---:|---:|---:|---:|---|
| 1 | 0.80 | 0.30 s | 5 ms | 160 | |
| 20 | 3.14 | 0.76 s | 28 ms | 628 | |
| 40 | 3.63 | 1.02 s | 50 ms | 725 | |
| 100 | 4.03 | 2.26 s | 113 ms | 807 | |
| **180** | **4.14** | 3.59 s | 200 ms | **829** | 🏆 sweet spot |
| 200 | 1.45 | 3.84 s | 674 ms | 290 | ⚠️ 崩溃 |

→ Flash 低延迟上限 **180**（vs Pro 仅 20），上限提升 **9×**。

#### #207 — 平衡（TP=8 DP=8 + DeepEP + EAGLE n=1, max_req=512）

| 并发 | RPS | TTFT | TPOT | toks/s | 备注 |
|---:|---:|---:|---:|---:|---|
| 100 | 7.30 | 1.37 s | 62 ms | 1461 | |
| 200 | 9.73 | 2.14 s | 92 ms | 1947 | |
| 300 | 11.66 | 3.19 s | 113 ms | 2331 | |
| **400** | **11.96** | 3.90 s | 148 ms | **2393** | 🏆 |
| 500 | 8.66 | 4.44 s | 267 ms | 1733 | ⚠️ max_req 排队 |

→ 17 档 0 失败。500 回落原因：`max_req=512 / DP=8 = 64/slice`，500 conv = 62.5/slice 接近 64 上限触发排队（**非 KV 饱和**，KV 实际仅占 7%）。

#### #208 — 高吞吐（TP=8 DP=8 + DeepEP + 无 spec, max_req=1024）

| 并发 | RPS | TTFT | TPOT | toks/s | 备注 |
|---:|---:|---:|---:|---:|---|
| 100 | 8.15 | 1.54 s | 54 ms | 1631 | |
| 200 | 11.51 | 4.55 s | 64 ms | 2301 | |
| 400 | 14.17 | 10.18 s | 90 ms | 2835 | |
| 500 | 14.06 | 16.31 s | 95 ms | 2811 | |
| **600** | **14.67** | 17.64 s | 117 ms | **2933** | 🏆 单机峰值 |
| 700 | 14.47 | 21.00 s | 134 ms | 2895 | plateau |
| 800 | 14.48 | 24.32 s | 149 ms | 2895 | plateau |
| 900 | 14.58 | 26.78 s | 172 ms | 2915 | plateau |

→ 21 档 0 失败（parallel=1000 因 evalscope SQLite 句柄耗尽未跑，与 SGLang 无关）。
→ 600-900 长 plateau，KV 利用率 13% 远未饱和；瓶颈在 DeepEP/scheduler。

### 4.3 V4-Pro vLLM baseline (#202)

DP=8 + EP + MTP n=2 + FP8 KV，**默认 max_num_seqs=1024**（未限制）。

| 并发 | RPS | TTFT | TPOT | toks/s |
|---:|---:|---:|---:|---:|
| **1** | 0.33 | 0.49 s | 13 ms | **66** | 🏆 单请求 (MTP 加速) |
| 20 | 1.94 | 0.93 s | 46 ms | 388 |
| 100 | 3.43 | 2.25 s | 134 ms | 686 |
| 160 | 4.40 | 2.75 s | 168 ms | 881 |
| **200** | **4.65** | 3.06 s | 199 ms | **930** | 🏆 |

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
| 单轮压测 V4-Pro 低延迟（#199, 14 档 1→200）| ~90 min | parallel=60 段单档 60 min |
| 单轮压测 V4-Pro 平衡（#200, 9 档 1→100）| ~28 min | |
| 单轮压测 V4-Pro 高吞吐（#201, 14 档 1→200）| ~56 min | |
| 单轮压测 V4-Flash 低延迟（#206, 14 档 1→200）| ~80 min | |
| 单轮压测 V4-Flash 平衡（#207, 17 档 1→500）| ~70 min | |
| 单轮压测 V4-Flash 高吞吐（#208, 22 档 1→1000）| ~105 min | parallel=1000 evalscope 崩 |
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

> #203 (V4-Flash 低延迟 旧版) 已被 #206 替代（旧版未跑到 200 档）；#204 (V4-Flash 平衡 旧版 TP=4 DP=2) 因 7.2 bug 失败未产生数据。

---

## 10. 下一步工作

- [ ] V4-Flash × vLLM 三套对照（按 §3.3 设计跑 #209-#211）
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
