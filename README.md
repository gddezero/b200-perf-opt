# B200 大模型推理性能测试

> **硬件**：GCP A4 B200 SXM × 8 (180 GB HBM/卡) | **框架**：vLLM 0.17.1rc1 nightly (cu130)
> **测试周期**：2026 年 3 月 | **有效压测**：~103 次

## 快速入口

→ **[完整文档目录与核心结论 (index.md)](index.md)**

## 文档结构

| 文档 | 说明 |
|------|------|
| [index.md](index.md) | 主索引：测试范围、核心结论、复现步骤、关键数字速查 |
| [01_deployment_guide.md](01_deployment_guide.md) | Step-by-Step 复现指南 |
| [02_benchmark_report.md](02_benchmark_report.md) | 压测分析：B200 vs H200、最优配置、TP/DP/EP/MTP 规律 |
| [03_command_reference.md](03_command_reference.md) | 所有模型的启动/压测命令模板 |
| [04_findings_and_issues.md](04_findings_and_issues.md) | 已知 Bug、踩坑记录、被推翻的结论 |
| [05_kv_cache_and_lmcache.md](05_kv_cache_and_lmcache.md) | DeepSeek V3.2 长上下文 LMCache 专题 |
| [06_architecture_and_parallelism.md](06_architecture_and_parallelism.md) | MoE 并行、注意力后端对比 |
| [07_customer_report.md](07_customer_report.md) | 面向客户的性能对比 |
| [08_benchmark_data_guide.md](08_benchmark_data_guide.md) | results.csv 使用方法、测试有效性 |
| [scripts/](scripts/) | 涉及的全部脚本（mount/install/benchmark 等） |

## 核心结论速览

- **B200 vs H200**：FP8 同条件下吞吐 1.7~2.5×
- **NVFP4 量化**（SM100 独有）：在 FP8 基础上再提升 31~65%
- **MTP n=2**：低并发提升 52~69%
- **LMCache**：长上下文多轮对话 RPS 提升 9.87×
