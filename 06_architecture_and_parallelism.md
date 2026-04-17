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

### 2.2 原理

- **TP 增大**：每请求使用更多卡 → prefill 更快 (TTFT↓) → 但 batch 能力下降 (吞吐↓)
- **DP 增大**：更多独立副本 → batch 吞吐↑ → 但每副本算力减少 (TTFT↑)
- **EP**：专家在 TP×DP 卡间均匀分布，TP>1 时通常开启

### 2.3 GPU 布局图

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

### 5.2 权重越小 → KV 越大

NVFP4 权重比 BF16 小一半 → 释放更多 HBM 给 KV Cache → 多轮对话场景能容纳更多对话历史，提高 prefix cache 命中率。
