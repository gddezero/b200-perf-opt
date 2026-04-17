# 测试发现、调研与已知问题

> 本文档汇总测试过程中的技术发现、调研报告、已知 bug 和踩坑记录。

---

## 1. 已知 Bug（截至 2026-03-26）

| 问题 | 影响 | Issue | 状态 | 当前方案 |
|------|------|-------|------|---------|
| V3.2 NVFP4 + FlashMLA-Sparse | `sparse_prefill_fwd` 崩溃（要求 KV=BF16，NVFP4 不可配） | [MO #763](https://github.com/NVIDIA/Model-Optimizer/issues/763) | 未修复（stale auto-close） | 我们用 FlashInfer-MLA-Sparse 后端跑通（#115）；生产仍推荐 FP8 |
| Qwen3.5 NVFP4 精度严重下降 | GSM8K=0.35 | [vLLM #36094](https://github.com/vllm-project/vllm/issues/36094) | 未修复 | 用 FP8 |
| Qwen3.5 MTP 高并发 (≥40) 崩溃 | cudaErrorIllegalAddress | — | 未修复 | 命令行模板默认不开 MTP |
| Qwen3.5 FP8 + DeepGEMM 精度 | 注意力层 E8M0 误差 1.44–1.65× | [vLLM #37618](https://github.com/vllm-project/vllm/issues/37618) | 未修复 | `VLLM_USE_DEEP_GEMM=0` |

### 1.1 V3.2 NVFP4 + FlashMLA-Sparse 详细分析（MO #763）

**报错**：

```
RuntimeError: Expected kv.dtype() == torch::kBFloat16 to be true, but got false.
位置：_deps/flashmla-src/csrc/pybind.cpp:404 (sparse_prefill_fwd)
```

**根因**：

- `flash_mla.sparse_prefill_fwd` kernel 硬约束 KV tensor 必须是 BF16
- 但 `kv_cache_config.dtype` 只接受 `fp8 / nvfp4 / auto`，`auto` 不会选 BF16
- 结果：NVFP4 权重 + FlashMLA-Sparse 后端时无法在 API 层配出"BF16 KV"，权重加载成功但 prefill 阶段崩溃

**Issue 进展**：

- 报告人（evgeniiperepelkin，2026-01-12）实测 H200×8 + TRT-LLM 1.2.0rc7 复现
- 社区方案（`kv_cache_quant_algo: null` + `dtype: fp8`）报告人验证仍失败
- NVIDIA collaborator 建议升级 TRT-LLM 1.2.0rc8 + 用官方 [`nvidia/DeepSeek-V3.2-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4) checkpoint
- **2026-03-13 因 14 天无回复被 stale bot 自动关闭，并非真正修复**

**我们的实测验证**（B200×8，vLLM 0.17.1rc1 nightly）：

| 后端 | V3.2 NVFP4 | 测试编号 | 结论 |
|------|----------|---------|------|
| FLASHMLA_SPARSE | ❌ 启动失败/崩溃 | #110/#111/#112/#113/#114 | 与 #763 报错一致 |
| FLASHINFER_MLA_SPARSE | ✅ 全配置通过 | #115/#116/#117/#118/#119 | 事实 workaround，#115 达 1,177 toks/s |

> **关键发现**：#763 中没人尝试 FlashInfer 后端，但我们实测 FlashInfer-MLA-Sparse 完全规避了这个 BF16 KV 硬约束。这是社区未记录的 workaround。
>
> **生产建议**：仍推荐 V3.2 FP8（#105/#108），因为：
> 1. NVFP4 + FlashInfer 仅在 vLLM nightly 验证，未经长期稳定性测试
> 2. FP8 性能足够（@60 742 toks/s），且无任何后端兼容性风险
> 3. NVFP4 收益（+50%）不足以承担一个未修复 NVIDIA bug 的尾部风险

---

## 2. LMCache 与 KV Cache 关键发现

### 2.1 LMCache CPU Offload 效果

V3.2 FP8，84k token 多轮对话，LMCache L1=**1600 GB** (DRAM)：

| 层级 | 命中率 | 速率 | 占比 |
|------|--------|------|------|
| GPU prefix cache（本地） | 20.9% | 4,487 tok/s | 23.8% |
| LMCache external（DRAM） | **81.0%** | 13,662 tok/s | 72.5% |
| 实际计算 | — | 748 tok/s | 4.0% |

**96% 的 prompt token 由缓存提供，仅 4% 需要真正计算。**

### 2.2 LMCache 瓶颈分析

存在两种并存的瓶颈：

| 维度 | 瓶颈 | 证据 |
|------|------|------|
| 生成吞吐 | PCIe KV transfer (IO bound) | SM Active 40%, Tensor Core 7%, Power 52% TDP |
| TTFT 高并发恶化 | GPU VRAM KV 饱和 → 调度队列等待 | TTFT avg 19.7s (@c=96), prefill 计算仅 0.3s |

> 原始结论"LMCache PCIe 是 TTFT 瓶颈"已更正。高并发 TTFT 恶化的根因是 GPU KV cache 饱和导致 vLLM 调度器排队。

### 2.3 LMCache 最优并发

| 并发 | RPS | TTFT avg |
|------|-----|----------|
| **c=48 (#195)** | **2.200** | **1.177s** |
| c=96 (#193) | 1.848 | 19.7s |
| c=128 (#191) | 1.754 | 38.7s |

c=48 是吞吐与延迟的双优点。原"最优 c=96"结论已更正。

---

## 3. FlashAttention-4 调研

### 3.1 FA4 性能

- B200 实测：1605-1613 TFLOPs/s（71% 利用率）
- 主要突破：异步 softmax pipeline，绕过 MUFU/SFU 瓶颈

### 3.2 FA4 限制

| 限制 | 影响 |
|------|------|
| **不支持 FP8 KV cache** | 只能用 BF16 KV，显存翻倍 |
| 不支持 MLA | DeepSeek V3.x 使用 MLA，需要 FlashMLA |
| 仅 SM100+ | H200 不可用 |

### 3.3 结论

FA4 适合 Qwen 系列（标准 MHA/GQA），不适合 DeepSeek（MLA 架构）。但由于 FP8 KV cache 对显存至关重要，实际使用 FlashInfer 更普遍。

---

## 4. MoE 并行策略分析

### 4.1 三维并行

- **TP (Tensor Parallel)**：分割注意力头，降低单卡 TTFT
- **DP (Data Parallel)**：多副本处理请求，提高吞吐
- **EP (Expert Parallel)**：分割 MoE 专家，TP×DP 卡共享专家

### 4.2 GPU 布局示例

```
TP=1, DP=8, EP=8:  每卡独立处理请求 + 专家分布在 8 卡
  → 吞吐最高，但 KV cache 不共享

TP=8, DP=1, EP=off: 8 卡协作处理单请求
  → TTFT 最低，但吞吐最低

TP=2, DP=4, EP=8:  2 卡一组处理请求，4 组并行
  → 均衡配置
```

### 4.3 DeepSeek V3.2 TP 规则

- TP=1 性能最优（128 Q head 不分割，KV cache 最大）
- TP≥4 曾被认为"禁止"，但 2026-03-14 启动测试 12/12 全 PASS
- 显存约束才是主要瓶颈：NVFP4 权重 335GB，180GB/卡至少 4 卡

---

## 5. LMCache 踩坑记录

### 5.1 PYTHONHASHSEED 跨进程不一致

**现象**：LMCache stored 26.7M tokens，但 `external_prefix_cache_hits_total` 始终为 0。

**根因**：Python `hash()` 函数受 `PYTHONHASHSEED` 影响，vLLM 的 Scheduler 进程和 Worker 进程使用不同的随机 seed，导致 store 和 lookup 使用不同的 hash 值。

**修复**：
```bash
export PYTHONHASHSEED=0
```

### 5.2 Docker 下 numa_mode: auto 导致 LMCache 失败

**现象**：`mbind failed: Operation not permitted`，所有 TP worker 均失败。

**根因**：`mbind()` 系统调用需要 `CAP_SYS_NICE` capability，Docker 默认不包含。

**修复**：在 `scripts/lmcache_cpu.yaml` 中：
```yaml
numa_mode: disabled
```

### 5.3 gpu_util=0.90 导致 OOM

**现象**：高并发时 CUDA OOM。

**修复**：降为 `gpu-memory-utilization=0.85`。

### 5.4 expandable_segments 启动时间翻倍

**现象**：开启 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后启动时间从 ~15 分钟翻倍到 ~30 分钟。

**结论**：不使用此选项。

---

## 6. 环境配置决策记录

### 6.1 OS 选择

- **Ubuntu 24.04**（当前选择）：NVIDIA 驱动支持最好
- Rocky 9：可用但 dnf 包管理器检测需要额外适配

### 6.2 Driver 选择

- **R580**（当前选择）：稳定版
- R590：新版，但 DeepGEMM JIT 在 cu130 上有 bug

### 6.3 CUDA 版本策略

| 场景 | CUDA 版本 | 原因 |
|------|-----------|------|
| DeepSeek 裸机 | 12.8 | PyTorch cu128 wheel 可用 |
| Docker 镜像 | 13.0 | cu130-nightly 包含最新优化 |

> DeepGEMM JIT 编译在 CUDA 13.0 下有 bug，DeepSeek 必须用 12.8。

### 6.4 关键环境变量

| 变量 | 值 | 作用 |
|------|-----|------|
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | 1 | 命令行模板中已设置 |
| `VLLM_USE_DEEP_GEMM` | 0 | FP8 必须禁用（精度 bug） |
| `VLLM_USE_FLASHINFER_MOE_FP8` | 0 | 已弃用（当前 nightly 反而降速） |

---

## 7. 测试方法论演进

### 7.1 早期测试问题 (#33–#84)

- **过多手动变量**：每次测试手动设置 5~10 个环境变量
- **参数不一致**：`max_model_len`, `no-prefix-caching` 等参数在不同测试间不统一
- **无标准模板**：命令行格式随意

### 7.2 "严格模板"改革 (#45–#64)

- 发现旧参数集中 `VLLM_USE_FLASHINFER_MOE_FP8=0` + `max_model_len=125000` 限制了性能
- 移除这些限制后吞吐提升 **+27%**（#32 513 → #46 694 toks/s）

### 7.3 矩阵测试 (#100–#185)

- 系统化测试所有 TP/DP/EP/量化/后端组合
- 结果编号连续，`superseded_by` 追踪替代关系
- MTP n=1/2/3 专项测试
- 命令行参数收敛为统一模板（见 [03 命令行模板](03_command_reference.md)）

---

## 8. 被推翻的结论

| 原结论 | 实测结果 | 更正时间 |
|--------|---------|---------|
| "V3.2 TP≥4 禁止" | 12/12 启动测试全 PASS | 2026-03-14 |
| "LMCache PCIe 是 TTFT 瓶颈" | GPU KV 饱和 + 调度排队才是根因 | 2026-03-22 |
| "LMCache 最优并发 c=96" | c=48 (#195) RPS=2.200 > c=96 (#193) 1.848 | 2026-03-26 |
| "VLLM_USE_FLASHINFER_MOE_FP8=0 提速" | 当前 nightly 反而降速 | 2026-03 |
| "旧参数集 #32 是 FP8 最优" | 严格模板 #46 694 toks/s (+27%) | 2026-03-15 |
