# 模型架构与并行策略分析

> 本文档整理 MoE 并行策略、TP/DP/EP 权衡、各模型架构特点及其对部署配置的影响。

---

## 1. 并行维度定义

| 维度 | 全称 | 作用 | 分割对象 |
|------|------|------|---------|
| TP | Tensor Parallel | 分割注意力头和 FFN | 模型参数 |
| DP | Data Parallel | 多副本独立处理请求 | 输入 batch |
| EP | Expert Parallel | 在 TP×DP 卡间分配专家 | MoE 专家层 |

**约束**：TP × DP ≤ GPU 总数（8），EP 在 TP×DP 卡间自动分配。

---

## 2. 并行配置权衡

### 2.1 性能特征

| 配置 | EP | 吞吐 | TTFT | TPOT | KV 容量 | 适用 |
|------|:---:|------|------|------|---------|------|
| TP=1 DP=8 | on | **最高** | 高 | 中 | 最大 (独立 KV) | 高并发吞吐 |
| TP=2 DP=4 | on | 高 | 中 | 中 | 中 | 均衡 |
| TP=4 DP=2 | on | 中 | 较低 | 较低 | 较小 | — |
| TP=8 DP=1 | off | 最低 | **最低** | **最低** | 最小 (共享 KV) | 低延迟单请求 |
| TP=8 DP=1 | on | 略高于 off | 接近 off | 接近 off | 最小 (共享 KV) | TP=8 + EP 备选 |

> **EP 规则**：DP>1 时必须开启 EP（否则每个 DP 副本都需放下全部 expert，显存爆炸）；TP=8 DP=1 时 EP 可选——off 让 8 卡共享所有 expert（低延迟最优），on 把 expert 切到 8 卡（吞吐略高，延迟略升）。

### 2.2 独立 KV vs 共享 KV

KV cache 的分布方式由 **TP × DP** 决定（与 EP 无关）：

#### 独立 KV（TP=1 DP=N）

```
GPU0: [完整模型] [KV 池 0] ← 服务 request A, B, C
GPU1: [完整模型] [KV 池 1] ← 服务 request D, E, F
...
GPU7: [完整模型] [KV 池 7] ← 服务 request V, W, X
```

- N 个 DP 副本各有一份完整模型 + 独立 KV 池
- 总 KV 字节数 = N × 单卡 KV（最大）
- 单 request 只能用所在副本的那一份，无法跨副本共享
- 适合：高并发、大量独立请求

#### 共享 KV（TP=N DP=1）

```
GPU0: [模型 1/N] [KV 1/N (head 0~k)]    ─┐
GPU1: [模型 1/N] [KV 1/N (head k+1~..)] ─┤
...                                          ├─ N 卡共同服务一个 request
GPU7: [模型 1/N] [KV 1/N (last head)]    ─┘
```

- 模型按 attention head 切到 N 卡，每卡只持 1/N KV
- 逻辑上 1 个统一 KV pool，所有 request 共用
- 单个长 prompt 横跨 N 卡，能用上聚合容量
- 适合：单请求超长上下文 / 极低延迟

#### 对比速查

| | 独立 KV (TP=1 DP=8) | 共享 KV (TP=8 DP=1) |
|---|---|---|
| KV 池数量 | 8 个 | 1 个 |
| 总 KV 字节数 | 最大（8 × 单卡）| 1 × 总和 |
| 同时活跃 request 数 | 高 | 低 |
| 单 token 算力 | 1 卡 | 8 卡 |
| TTFT / TPOT | 高 / 中 | 最低 / 最低 |

**简记**：独立 KV = 8 个小池，并发吞吐赢；共享 KV = 1 个大池，单请求延迟赢。

### 2.3 EP 与 KV 分布的关系

**EP 只切 MoE expert 层，attention/KV 不在 EP 切分范围内。**

EP 对 KV 的影响是通过显存挤压**间接发生**的：

| 配置 | Expert 显存 | 剩余 KV 空间 | KV 分布 |
|------|-----------|------------|---------|
| TP=1 DP=8 + EP=on | 8 卡分担 1 套 expert（每卡 1/8） | 大 | **8 个独立池** |
| TP=1 DP=8 + EP=off | 每个 DP 副本复制全部 expert（×8） | 极小或 OOM | 通常不可行 |
| TP=8 DP=1 + EP=on | 8 卡切 1 套 expert（每卡 1/8） | 大 | **1 个共享池**（最大） |
| TP=8 DP=1 + EP=off | 每卡复制全部 expert（×8） | 小 | **1 个共享池**（最小） |

**简记**：
- **独立 vs 共享** 由 TP/DP 决定（结构性）
- **池子大小** 由 EP 决定（显存预算）
- DP>1 必须开 EP；TP=8 DP=1 时 EP 可选

### 2.4 原理

- **TP 增大**：每请求使用更多卡 → prefill 更快 (TTFT↓) → 但 batch 能力下降 (吞吐↓)
- **DP 增大**：更多独立副本 → batch 吞吐↑ → 但每副本算力减少 (TTFT↑)
- **EP**：专家在 TP×DP 卡间均匀分布，TP>1 时通常开启

### 2.5 GPU 布局图

```
TP=1, DP=8, EP=8:
  GPU0: [Model] [E0 E1]     ← 独立处理请求
  GPU1: [Model] [E2 E3]     ← 独立处理请求
  ...
  GPU7: [Model] [E14 E15]   ← 独立处理请求

TP=8, DP=1, EP=off:
  GPU0~7: [Model 1/8] [All Experts]  ← 8卡协作处理单请求

TP=2, DP=4, EP=8:
  Group 0: GPU0+1 [Model 1/2] [E0..E3]  ← 2卡协作
  Group 1: GPU2+3 [Model 1/2] [E4..E7]  ← 2卡协作
  Group 2: GPU4+5 [Model 1/2] [E8..E11] ← 2卡协作
  Group 3: GPU6+7 [Model 1/2] [E12..E15]← 2卡协作
```

---

## 3. 各模型架构特点

### 3.1 DeepSeek V3.2

| 特性 | 值 |
|------|-----|
| 注意力 | MLA (Multi-head Latent Attention) + DSA (Sparse Attention) |
| 专家 | 256 experts, 8 active |
| Q heads | 128 |
| 推荐 TP | TP=1（128 Q head 不分割效率最高）|
| 注意力后端 | FlashMLA-Sparse / FlashInfer-MLA-Sparse |
| 约束 | NVFP4 与 FlashMLA-Sparse 不兼容 (MO #763)；NVFP4 必须搭 FlashInfer-MLA-Sparse 后端，生产推荐 FP8 |

**TP 规则**：
- TP=1：性能最优，128 Q head 不分割，KV cache 容量最大
- TP=2/4/8：均可启动（2026-03-14 测试 12/12 PASS），但吞吐递减

### 3.2 DeepSeek V3.1

| 特性 | 值 |
|------|-----|
| 注意力 | MLA |
| 专家 | 256 experts, 8 active |
| 推荐 TP | TP=1 DP=8 EP=on |
| 特殊依赖 | DeepGEMM（特定 commit） |

### 3.3 DeepSeek R1

| 特性 | 值 |
|------|-----|
| 注意力 | MLA |
| 专家 | 纯 EP，无 TP |
| 推荐 TP | TP=1 DP=8 |
| 说明 | 架构最简单，纯 MLA 无 DSA |

### 3.4 Qwen3.5-397B-A17B

| 特性 | 值 |
|------|-----|
| 注意力 | 标准 GQA |
| 专家 | 512 experts, 11 active (10 routed + 1 shared) |
| KV heads | 2 |
| 推荐 TP | TP=8（低延迟）或 TP=2 DP=4（高吞吐） |
| 约束 | TP 必须是 2 的倍数（KV head=2） |
| 禁用项 | DeepGEMM（Qwen 不使用）, MTP (高并发崩溃) |

**显存估算**：
- FP8 权重：~49.6 GB/卡 (TP=8)，~99.25 GB/卡 (TP=4)
- BF16 权重：~101 GB/卡 (TP=8) → 仅 TP=8 可行

### 3.5 Qwen3-235B-A22B

| 特性 | 值 |
|------|-----|
| 注意力 | 标准 GQA |
| 专家 | 64 experts, 4 active |
| 推荐配置 | NVFP4 TP=4 DP=2 EP=on（高吞吐） |
| 特殊 | 不支持 MTP |
| NVFP4 | 官方 checkpoint 可用，精度正常 |

---

## 4. 注意力后端对比

### 4.1 DeepSeek 系列可用后端

| 后端 | 全称 | 适用 |
|------|------|------|
| FlashMLA-Sparse | NVIDIA FlashMLA + Sparse Attention | DeepSeek V3.2 (MLA+DSA) |
| FlashInfer-MLA-Sparse | FlashInfer 实现的 MLA | DeepSeek V3.x |

### 4.2 TP 维度对比（V3.2 FP8，@60 toks/s）

| TP | FlashMLA | FlashInfer | 胜者 | 差距 |
|----|----------|-----------|------|------|
| 1 | #100 | **#105** | FlashInfer | +7% |
| 2 | **#101** | #106 | FlashMLA | +5% |
| 4 | #102 | **#107** | FlashInfer | +15% |
| 8 (EP=off) | #103 | **#108** | FlashInfer | +12% |
| 8 (EP=on) | #104 | **#109** | FlashInfer | +10% |

**结论**：FlashInfer 在 4/5 配置下胜出，推荐作为默认。仅 TP=2 时 FlashMLA 略优。

### 4.3 Qwen 系列可用后端

Qwen 使用标准 GQA 注意力，可用 FlashAttention-4 (FA4)、FlashInfer 等标准后端。
FA4 不支持 FP8 KV cache，实际推荐 FlashInfer。

---

## 5. 量化对并行策略的影响

### 5.1 权重大小 → 可用 TP 配置

| 模型 | BF16 | FP8 | NVFP4 |
|------|------|-----|-------|
| V3.2 (672B dense) | — | 每卡 ~83GB (TP=8) | 每卡 ~42GB (TP=8) |
| V3.1 (685B dense) | — | ~86GB (TP=8) | ~43GB (TP=8) |
| Qwen3.5 (397B) | ~101GB (TP=8) | ~49.6GB (TP=8) | ~25GB (TP=8) |
| Qwen3 (235B) | ~59GB (TP=4) | ~30GB (TP=4) | ~15GB (TP=4) |

### 5.2 权重越小 → KV Cache 越大（越大越好）

NVFP4 权重比 BF16 小一半 → 释放更多 HBM 给 KV Cache → KV Cache 池越大越好，原因：

1. **更多并发 request**：每个 request 的 KV 占固定字节，池越大，可同时活跃的 request 越多 → 吞吐越高
2. **更长上下文**：单个长 prompt 的 KV 能完整放下，避免 swap/recompute
3. **更高 prefix cache 命中率**：多轮对话场景中保留更多历史 KV，命中率提升 → TTFT 降低
4. **减少调度排队**：高并发下 GPU KV 不易饱和，避免新 request 等待 KV slot 释放（详见 [05 §5 高并发饱和分析](05_kv_cache_and_lmcache.md)）

**实测对照**（V3.2，B200×8 TP=1 DP=8）：

| 量化 | 权重总占用 | 单卡剩余给 KV | KV tokens/engine |
|------|-----------|------------|------------------|
| FP8 | ~664 GB | ~97 GB | ~2.36M |
| NVFP4 | ~335 GB | ~138 GB | ~3.36M（+42%） |
