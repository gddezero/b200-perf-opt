# LMCache CPU Offload 专题分析

> 本文档汇总 DeepSeek V3.2 FP8 上的 LMCache CPU DRAM Offload 测试与调优结论。
> 长上下文多轮对话场景中，LMCache 把 GPU 装不下的 KV 缓存到 CPU DRAM，再在需要时拉回。

---

## 1. 测试配置

| 项 | 值 |
|----|----|
| 模型 | DeepSeek V3.2 FP8（裸机） |
| 并行 | TP=1, DP=8, EP=on |
| 注意力后端 | FlashInfer-MLA-Sparse |
| KV cache dtype | fp8 |
| performance-mode | balanced |
| LMCache 模式 | MP（multiprocess） |
| LMCache L1 容量 | **1600 GB** (DRAM, blake3 hash) |
| 压测工具 | `benchmark_serving_multi_turn.py` |
| 压测负载 | 50 clients, max-active-conversations=500, 30 turns, **avg 84k token 输入** |
| 测试日期 | 2026-03-22 ~ 03-26 |

> **vLLM 中两个独立配置**：
> - **模型权重量化**（NVFP4 / FP8 / BF16）— 决定权重占多少显存
> - **`--kv-cache-dtype`**（auto / fp8 / bf16）— 决定每个 KV block 占多少字节
>
> vLLM **不支持** `--kv-cache-dtype nvfp4`。即使权重是 BF16，KV cache 也应用 `--kv-cache-dtype fp8`（容量翻倍，质量影响极小）。

---

## 2. LMCache on/off 对比（c=48）

| 指标 | OFF (#194) | ON (#195) | 提升 |
|------|---:|---:|:---:|
| RPS | 0.223 | 2.200 | **+9.87×** |
| TTFT avg | 13,906 ms | 1,177 ms | **11.8× 降低** |
| TTFT p99 | 58,012 ms | 4,037 ms | **14.4× 降低** |
| TPOT avg | 667 ms | 96 ms | **6.9× 降低** |

**结论**：长上下文多轮对话场景下 LMCache 价值巨大，RPS 提升约 10 倍，TTFT 从分钟级降到秒级。

---

## 3. 命中分层（LMCache ON）

```
┌─────────────────────────────────────────────┐
│                  GPU VRAM                     │
│  ┌─────────┐  ┌─────────┐                    │
│  │ KV Cache │──│ Prefix  │ ← 20.9% 命中     │
│  │ (2.36M)  │  │ Cache   │                    │
│  └────┬─────┘  └─────────┘                    │
│       │ LRU eviction                          │
├───────┼───────────────────────────────────────┤
│       ▼                                       │
│  ┌───────────────────┐                        │
│  │ LMCache L1 (DRAM) │ ← 81.0% 命中          │
│  │      1600 GB      │                        │
│  └───────────────────┘                        │
│       │                                       │
│       ▼ miss                                  │
│  [ 实际计算 ] ← 仅 4%                         │
└─────────────────────────────────────────────┘
```

| 层级 | 命中率 | 速率 | prompt 占比 |
|------|--------|------|-----------|
| GPU prefix cache（本地） | 20.9% | 4,487 tok/s | 23.8% |
| LMCache external（DRAM） | **81.0%**（max 97.1%） | 13,662 tok/s | 72.5% |
| 实际计算（local_compute） | — | 748 tok/s | **4.0%** |

**96% 的 prompt token 由缓存提供，仅 4% 需要真正计算。**

---

## 4. 硬件状态与瓶颈

### 4.1 GPU 硬件状态（LMCache ON）

| 指标 | avg | 说明 |
|------|-----|------|
| SM Active | 40.8% | GPU 40% 时间有 warp 运行 |
| Tensor Core Active | 7.0% | GEMM 计算极少 |
| Power | 518W | TDP 52%，轻载 |
| GPU 温度 | 53.6°C | 非常低 |

**GPU 处于 IO bound 状态**，大量时间等待 LMCache 通过 PCIe 搬运 KV 数据。

### 4.2 互连带宽

| 通道 | avg | max | 理论上限 | 利用率 |
|------|-----|-----|---------|--------|
| NVLink TX+RX (per GPU, 双向) | 17.8 GB/s | 34.8 GB/s | 1800 GB/s（B200 NVLink 5）| **~1%** |
| PCIe TX (GPU→LMCache store) | 30.0 MB/s | 37.2 MB/s | ~64 GB/s（PCIe 5.0 x16 单向）| <0.1% |
| PCIe RX (LMCache→GPU, 理论需求) | ~465 MB/s | — | ~64 GB/s | ~0.7% |

> PCIe RX 理论需求推算：13,662 tok/s × 34 KB/token ≈ 465 MB/s
> NVLink、PCIe 利用率都极低，**不是带宽瓶颈**——慢的是 PCIe 频繁小包传输的延迟，而非吞吐。

---

## 5. 高并发饱和分析

### 5.1 因果链

c=96 实测 TTFT avg = 19.7s。逐项拆解：

| 组成部分 | 时间 | 依据 |
|---------|------|------|
| LMCache prefetch | 1–3 ms | 实测：`Prefetch request completed: 41/41 prefix hits in 1.7ms`，per-key 0.025–0.085ms |
| Prefill 计算 | ~0.3 s | 推算：c=48 / DP=8 = 每 engine 6 并发；84k × 4% = 3,360 token/req 需算 → 6 × 3,360 ≈ 20k token；B200 单卡 FP8 prefill 实测约 60–80k tok/s → 20k / 60k ≈ 0.3 s |
| **vLLM 调度队列等待** | **~19.4 s** | **倒推**：19.7s（实测 TTFT）− 0.3s（prefill）− 1ms（prefetch） ≈ 19.4s |

```
高并发 (c≥96)
  → GPU VRAM KV cache 饱和 (利用率 91-96%)
  → vLLM 调度器等待现有 request 释放 KV slot
  → 新 request 在调度队列等待约 19.4s
  → TTFT = 19.4s (队列等待) + 0.3s (prefill) + 1ms (prefetch) ≈ 19.7s
```

> **关键证据**：c=48 时 TTFT=1.177s（GPU KV 未饱和、无排队），而 c=96 时 TTFT=19.7s。
> prefill 计算和 LMCache fetch 时间在两个并发下基本不变，差出来的 ~18s 全部来自调度队列等待。
> LMCache CPU 利用率仅 0.3%，19s 等待期间 LMCache 早已就绪，**瓶颈在 GPU KV slot，不在 LMCache**。

### 5.2 并发阶梯（LMCache ON，热 DRAM 稳态）

| 并发 | 测试编号 | RPS | TTFT avg | TTFT p99 | GPU KV 利用率 |
|------|--------|-----|----------|----------|-------------|
| **48** | **#195** | **2.200** | **1.177s** | **4.037s** | 未饱和 |
| 96 | #193 | 1.848 | 19.7s | 62.3s | ~91-96% |
| 128 | #191 | 1.754 | 38.7s | 81.3s | 饱和 |
| 160 | #192 | 1.843 | 53.5s | 104.4s | 饱和 |

**c=48 是最优工作点**（吞吐+延迟双优）。原"最优 c=96"结论已更正。

### 5.3 根因

- max-active-conversations=500, DP=8 → 每 engine 对话池 ~62 个
- 每对话 ~100k tokens → 总需求 6.2M tokens/engine
- GPU KV 容量仅 2.36M tokens/engine → 只能装 ~23 个对话的 KV
- 并发越高 → GPU 前缀缓存被更快占满 → 新请求排队等待 KV slot

### 5.4 两种瓶颈并存

| 维度 | 瓶颈 | 证据 |
|------|------|------|
| **生成吞吐** 不饱和 | LMCache PCIe KV transfer (IO bound) | SM Active 40%, Tensor Core 7% |
| **TTFT 高并发恶化** | GPU VRAM KV 饱和 → 调度队列等待 | TTFT avg 19.7s (@c=96), prefill 仅 0.3s |

> 原始结论"LMCache PCIe 是 TTFT 瓶颈"已更正。高并发 TTFT 恶化的根因是 GPU KV cache 饱和导致 vLLM 调度器排队，**不是** LMCache 慢。

---

## 6. LMCache 配置要点

### 6.1 推荐配置（MP 模式）

启动 LMCache MP server（必须先于 vLLM 启动）：

```bash
nohup python3 -m lmcache.v1.multiprocess.http_server \
    --host 0.0.0.0 \
    --port 6555 \
    --chunk-size 1024 \
    --max-workers 8 \
    --hash-algorithm blake3 \
    --l1-size-gb 1600 \
    --l1-use-lazy \
    --l1-init-size-gb 20 \
    --eviction-policy LRU \
    --eviction-trigger-watermark 0.8 \
    --eviction-ratio 0.2 \
    > /lssd/logs/lmcache_mp.log 2>&1 &
```

vLLM 侧加参数：

```bash
--kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":6555}}'
```

> MP 模式优势：L1 内存池全 TP 共享（1 个池 vs 8 个独立池），blake3 hash 跨进程确定性（无需 PYTHONHASHSEED=0）。

### 6.2 重启 SOP

变更 vLLM 参数后必须同步重启：

1. 停 vLLM
2. 停 LMCache MP server
3. 启新 LMCache MP server（约 6 分钟初始化）
4. 启新 vLLM（约 15 分钟）

---

## 7. 注意事项

### 7.1 throughput 模式不适合多轮 LMCache

#### 实测对比

| performance-mode | RPS | TTFT p99 | 适用 |
|------------------|-----|----------|------|
| balanced (#195) | 2.200 | 4.037 s | **推荐** |
| throughput (#197) | 2.171 | 8.220 s | 不推荐（LMCache ON） |
| throughput (#196) | 0.198 | 103.6 s | **禁用**（LMCache OFF，c=48 仍然崩盘） |

throughput 模式 + LMCache ON：RPS 几乎无损（-1.3%），但 TTFT p99 翻倍（4s → 8s）。
throughput 模式 + LMCache OFF：RPS −11%，TTFT p99 飙至 103s，完全不可接受。

#### 三种 performance-mode 参数差异（来自 vLLM 源码 `arg_utils.py:2156`）

| 维度 | `balanced`（默认）| `throughput` | `interactivity` |
|------|------------------|--------------|----------------|
| `max_num_batched_tokens` | 8192 | **16384（×2）** | 8192 |
| `max_num_seqs` | 默认 | **×2** | 默认 |
| CUDA graph 覆盖 size | 标准 | 1, 2, 4, 8, 16, 24, 32, 40, …（稀疏）| **1–32 全覆盖** |
| 调度倾向 | 中等 | 积极攒批，等更多请求一起 prefill | 低延迟优先，请求来了立即调度 |

#### throughput 为什么在 LMCache 场景下变差

1. **大 batch 导致排队**：`throughput` 把 `max_num_batched_tokens` 从 8192 翻倍到 16384，调度器倾向凑更多 prefill request 一起跑。多轮长上下文场景中 prompt 已经 84k token，少数请求就能填满；新来的小请求要等"凑批"才能开 prefill → TTFT p99 飙升。
2. **IO bound 场景大 batch 没用**：当前瓶颈是 PCIe KV transfer（SM Active 40%, Tensor Core 7%, TDP 52%），GPU 算力本就空闲。把 batch 放大只能让 prefill 更慢启动，而无法加速 prefill 本身。
3. **稀疏 CUDA graph 拖慢小批**：如果开 MTP，验证阶段 batch size 不固定（3, 5–7, 9–15 等），`throughput` 这些 size 没有 CUDA graph，要 fallback 到 eager → 多一笔 kernel launch overhead。

**结论**：LMCache 多轮场景用 **balanced**（默认）；如果开 MTP 改用 **interactivity**；**绝不用 throughput**。

### 7.2 MTP 在 LMCache 场景为负向

MTP n=2 在 LMCache 长上下文场景中性能下降（测试 #186），原因是 LMCache KV 还原开销与 MTP 预测开销叠加。**LMCache 场景不开 MTP**。

### 7.3 已知数据无效

| 测试编号 | 问题 |
|---------|------|
| #186 | MTP n=2 + throughput 模式专项实验，配置异常，不做基准引用 |
| #187–#189 | LMCache H2D 100% AssertionError，KV 从未还原到 GPU，**数据无效，禁止引用** |
