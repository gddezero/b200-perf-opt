# B200 大模型推理性能测试

> **硬件**：GCP A4 B200 SXM × 8 (180 GB HBM/卡) | **框架**：vLLM 0.17.1rc1 nightly (cu130)
> **测试周期**：2026 年 3 月 | **有效压测**：~103 次（#85–#185 在 results.csv，#190–#197 LMCache MP 单独追踪）

---

## 1. 测试范围

### 1.1 测试模型

| 模型 | 参数量 | 架构 | 测试量化 | 测试编号 |
|------|-------|------|---------|---------|
| DeepSeek V3.2 | 672B | MoE + MLA + DSA | FP8, NVFP4 | #100–#119, #145–#169 |
| DeepSeek V3.1 | 685B | MoE + MLA | FP8, NVFP4 | #120–#129, #149–#177 |
| Qwen3.5-397B-A17B | 397B | MoE + GQA | BF16, FP8, NVFP4 | #130–#144, #158–#185 |
| Qwen3-235B-A22B | 235B | MoE + GQA | BF16, FP8, NVFP4 | #85–#99 |

### 1.2 测试维度

| 维度 | 覆盖范围 |
|------|---------|
| **并行策略** | TP=1/2/4/8, DP=1/2/4/8, EP on/off — 全组合矩阵 |
| **量化** | BF16（全精度）, FP8（8bit）, NVFP4（4bit，SM100 独有） |
| **注意力后端** | FlashMLA-Sparse vs FlashInfer-MLA-Sparse（仅 DeepSeek） |
| **MTP 投机解码** | n=0/1/2/3，测试低并发加速效果 |
| **LMCache KV Offload** | 长上下文多轮对话场景（84k token, 30 轮） |
| **B200 vs H200** | 同条件 FP8 对比 + NVFP4/MTP 独有加速 |

### 1.3 压测负载

| 参数 | 值 |
|------|-----|
| 输入长度 | 4,500 token（固定） |
| 输出长度 | ~175 token（ignore_eos=true） |
| 并发梯度 | 1 → 2 → 4 → 8 → 20 → 40 → 60 |
| 成功率 | 100%（所有汇报配置均无失败） |

---

## 2. 核心结论

### 2.1 B200 vs H200（同条件 FP8 对比）

| 模型 | 吞吐提升 @40 (RPS) | toks/s 提升 @40 | B200 测试编号 |
|------|:---:|:---:|:---:|
| Qwen3-235B | **1.7×** | **1.7×** | #98 |
| Qwen3.5-397B | **2.1×** | **2.1×** | #136 |
| DeepSeek V3.1 | **2.5×** | **2.5×** | #120 |
| DeepSeek V3.2 | **2.3×** | **2.3×** | #105 |

### 2.2 各模型最优配置与峰值性能

| 模型 | 量化 | 配置 | @60 toks/s | 测试编号 |
|------|------|------|-----------|---------|
| Qwen3.5-397B | FP8 | TP=2 DP=4 EP | **1,723** | #136 |
| Qwen3-235B | NVFP4 | TP=1 DP=8 EP（Docker） | **2,024** | #87 |
| DeepSeek V3.1 | NVFP4 | TP=1 DP=8 EP（Docker/裸机均可） | **1,236** | #125 |
| DeepSeek V3.2 | NVFP4 | TP=1 DP=8 EP（裸机） | **1,177** | #115 |
| DeepSeek V3.2 | FP8 | TP=1 DP=8 EP（裸机） | **742** | #105 |

### 2.3 关键发现

- **NVFP4 量化**（SM100 独有）在 FP8 基础上再提升 **31~65%** 吞吐
- **MTP n=2** 低并发 (c=1) 提升 **52~69%**，推荐低延迟场景使用
- **LMCache KV Offload** 多轮长上下文场景提升 **9.87× RPS**，96% prompt 免计算
- **TP=1 DP=8** 吞吐最高，**TP=8 DP=1** 延迟最低 — 按场景选择
- **FlashInfer** 在 4/5 TP 配置下优于 FlashMLA，推荐作为 DeepSeek 默认后端

### 2.4 已知限制

| 问题 | 影响 | 状态 |
|------|------|------|
| V3.2 NVFP4 + FlashMLA-Sparse 不兼容 | NVFP4 须用 FlashInfer 后端；生产推荐 FP8 | 未修复 (MO #763, stale auto-close) |
| Qwen3.5 NVFP4 精度下降 | 禁止生产使用 | 未修复 (vLLM #36094) |
| Qwen3.5 MTP 高并发崩溃 | c≥40 时 CUDA 错误 | 未修复 |
| FP8 + DeepGEMM 精度问题 | 必须 `VLLM_USE_DEEP_GEMM=0` | 未修复 (vLLM #37618) |

---

## 3. 如何复现测试

### 3.1 快速开始（5 步）

```
Step 1: ssh 登录 B200 主机
Step 2: ./scripts/mount_lssd.sh                          # 挂载存储
Step 3: 复制命令行模板，启动 vLLM                          # 等待 15~20 分钟
Step 4: ./scripts/benchmark.sh "模型served-model-name"    # 运行压测
Step 5: 查看 benchmark_result/ 下的结果
```

### 3.2 详细步骤

**完整操作流程**（从零开始到出结果）见：

→ [01 - Step-by-Step 复现指南](01_deployment_guide.md)

每一步都有可直接执行的命令，包括：
- 存储挂载、环境安装
- 所有模型的启动命令模板（Docker / 裸机）
- 参数替换说明（如何切换 TP/DP/量化/MTP）
- 压测执行和结果查看

### 3.3 命令行模板

所有模型的完整启动命令，直接复制替换参数即可：

→ [03 - 命令行模板](03_command_reference.md)

---

## 4. 文档目录

| 文档 | 一句话说明 |
|------|---------|
| [01 - Step-by-Step 复现指南](01_deployment_guide.md) | 从零到压测结果的完整操作流程 |
| [02 - 压测分析报告](02_benchmark_report.md) | B200 vs H200、最优配置、TP/DP/EP 规律、MTP 分析 |
| [03 - 命令行模板](03_command_reference.md) | 所有模型的启动/压测命令，直接复制替换参数 |
| [04 - 发现与问题](04_findings_and_issues.md) | 已知 Bug、踩坑记录、被推翻的结论 |
| [05 - LMCache 专题](05_kv_cache_and_lmcache.md) | DeepSeek V3.2 长上下文多轮对话 LMCache on/off 对比、调优 |
| [06 - 架构与并行策略](06_architecture_and_parallelism.md) | MoE 并行、TP/DP/EP 权衡、注意力后端对比 |
| [07 - 客户报告](07_customer_report.md) | 面向客户的性能对比数据 |
| [08 - 压测数据指南](08_benchmark_data_guide.md) | 197 次测试有效性、results.csv 使用方法 |
| [09 - DeepSeek V4 B200 部署](09_deepseek_v4_b200.md) | V4-Pro / V4-Flash SGLang/vLLM 三套配方、压测数据、@1 TTFT 异常分析 |
| [10 - V4-Pro 多轮长上下文压测](10_deepseek_v4_b200_multi_turn.md) | V4-Pro 低延迟 TP=8 多轮 26 档实测（60K/100K/200K base）、KV pool ≥90% 物理崩盘点、parallel 上限规则 |
| [SGLang KV cache 调研](sglang_kvcache_research.md) | DP 模式 KV pool 物理分布、SMG 单机不可用源码证据、客户端 `data_parallel_rank` 路由实测 cache hit 23%→77% |
| [scripts/](scripts/) | 文档涉及的全部脚本：mount_lssd.sh / download_models.sh / install.sh / benchmark.sh / bench_multi_turn.sh / lmcache_cpu.yaml |

---

## 5. 关键数字速查

| 指标 | 值 |
|------|-----|
| B200 vs H200 吞吐倍率 | **1.7~2.5×** |
| NVFP4 vs FP8 额外提升 | **+31~65%** |
| MTP n=2 低并发提升 | **+52~69%** |
| LMCache 多轮 RPS 提升 | **+9.87×** |
| Qwen3.5 FP8 峰值 | **8.61 RPS / 1,723 toks/s (#136)** |
| Qwen3 NVFP4 峰值 | **10.12 RPS / 2,024 toks/s (#87)** |
| V3.1 NVFP4 峰值 | **6.18 RPS / 1,236 toks/s (#125)** |
| V3.2 NVFP4 峰值 | **5.89 RPS / 1,177 toks/s (#115)** |
| results.csv 中有效测试 | 95 次（#85–#185 不含 superseded） |
| 含 LMCache MP 总计 | ~103 次 |

---

## 6. 原始文件对照表

| compiled_doc 文档 | 原始文件来源 |
|-------------------|-------------|
| 01_deployment_guide.md | GB200_vLLM_DeepSeek_部署指南.md, B200_Qwen3_部署实录.md, B200_Qwen35_*.md, B200_环境配置决策记录.md |
| 02_benchmark_report.md | benchmark_result/benchmark_analysis.md, b200_vs_h200_report_v2.md, INDEX.md, results.csv |
| 03_command_reference.md | 后期压测实际命令 (#105, #130, #136, #160 等), benchmark.sh, bench_multi_turn.sh |
| 04_findings_and_issues.md | kv_cache_test.md, FA4_调研报告.md, B200_环境配置决策记录.md |
| 05_kv_cache_and_lmcache.md | kv_cache_test.md (V3.2 LMCache 部分), kv_cache_analysis.md |
| 06_architecture_and_parallelism.md | MoE并行策略与模型参数分析.md, configs/hw/*.conf, configs/models/*.conf |
| 07_customer_report.md | benchmark_result/benchmark_analysis_customer.md, b200_vs_h200_report_v2.md, b200_aws_gcp.md |
| 08_benchmark_data_guide.md | benchmark_result/results.csv, INDEX.md |
| 09_deepseek_v4_b200.md | 2026-04 V4-Pro / V4-Flash 压测 #199–#210 (SGLang) + #202+ (vLLM) |
| 10_deepseek_v4_b200_multi_turn.md | 2026-04-28/29 V4-Pro 多轮 #211–#236 (60K/100K/200K base × parallel sweep), batch_multi_turn.sh + results_multiturn.csv |
| sglang_kvcache_research.md | 2026-04-27 SGLang 源码调研 (data_parallel_controller / radix_cache / hicache) + V4-Pro 实测 |
