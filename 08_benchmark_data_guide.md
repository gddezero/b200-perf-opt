# 压测数据有效性指南

> 本文档说明 197 次压测中哪些仍然有效、哪些已废弃，以及如何正确使用 results.csv。

---

## 1. 测试阶段总览

| 阶段 | 编号范围 | 日期 | 模型 | 测试数 | 状态 |
|------|---------|------|------|-------|------|
| 探索期 | archive/ | 03-11 ~ 03-14 | 混合 | 58 | **废弃** |
| 第一轮 V3.2 | #33–#44 | 03-15 | V3.2 FP8/NVFP4 | 12 | **被 #100–#119 替代** |
| 严格模板 V3.2 | #45–#64 | 03-15 ~ 03-16 | V3.2 FP8/NVFP4 | 20 | **被 #100–#119 替代** |
| V3.1 Docker | #65–#74 | 03-16 | V3.1 Docker | 10 | 有效（Docker 基准） |
| V3.1 裸机 | #75–#84 | 03-17 | V3.1 裸机 | 10 | **被 #120–#129 替代** |
| **Qwen3-235B** | **#85–#99** | **03-17** | Qwen3-235B | **14** | **有效** |
| **V3.2 重测** | **#100–#119** | **03-17 ~ 03-18** | V3.2 FP8/NVFP4 | **20** | **有效（当前基准）** |
| **V3.1 重测** | **#120–#129** | **03-18** | V3.1 FP8/NVFP4 | **10** | **有效（当前基准）** |
| **Qwen3.5-397B** | **#130–#144** | **03-18** | Qwen3.5 BF16/FP8/NVFP4 | **15** | **有效（当前基准）** |
| **MTP 测试** | **#145–#185** | **03-18** | 多模型 MTP n=1/2/3 | **36** | **有效** |
| LMCache 实验 | #186 | 03-22 | V3.2 LMCache | 1 | 特殊实验（不做基准；不在 results.csv） |
| **LMCache 无效** | **#187–#189** | 03-22 | V3.2 LMCache | **3** | **数据无效（H2D AssertionError；不在 results.csv）** |
| **LMCache MP** | **#190–#197** | **03-22 ~ 03-26** | V3.2 LMCache MP | **8** | **有效（单独追踪于 kv_cache_test.md）** |

### 有效测试汇总

- **results.csv 核心有效**：#85–#99 (14) + #100–#185 (81) = **95 次**
- **加 LMCache MP**（在 `kv_cache_test.md`）：+ 8 = **103 次**
- **条件有效**：#65–#74（V3.1 Docker 基准，10 次，未被替代）
- **仅供参考**：#39–#44（早期 NVFP4，quant 标记为 unknown）, #186（LMCache 特殊实验）
- **废弃**：archive/, #33–#38, #45–#64, #75–#84（共 **74 次**，被后续重测替代）
- **无效**：#187–#189（LMCache H2D 断言错误，禁止引用）

---

## 2. superseded_by 替代关系

results.csv 中 `superseded_by` 列记录了测试替代关系（旧 → 新）：

### V3.2 FP8

| 旧编号 | 新编号 | 配置 |
|--------|--------|------|
| #33 | **#101** | TP=2 DP=4 EP FlashMLA |
| #34 | **#106** | TP=2 DP=4 EP FlashInfer |
| #35 | **#102** | TP=4 DP=2 EP FlashMLA |
| #36 | **#107** | TP=4 DP=2 EP FlashInfer |
| #37 | **#103** | TP=8 EP=off FlashMLA |
| #38 | **#108** | TP=8 EP=off FlashInfer |
| #45 | **#100** | TP=1 DP=8 EP FlashMLA |
| #46 | **#105** | TP=1 DP=8 EP FlashInfer |
| #47 | **#101** | TP=2 DP=4 EP FlashMLA |
| #48 | **#106** | TP=2 DP=4 EP FlashInfer |
| #49 | **#102** | TP=4 DP=2 EP FlashMLA |
| #50 | **#107** | TP=4 DP=2 EP FlashInfer |
| #51 | **#103** | TP=8 EP=off FlashMLA |
| #52 | **#108** | TP=8 EP=off FlashInfer |
| #53 | **#104** | TP=8 EP=on FlashMLA |
| #54 | **#109** | TP=8 EP=on FlashInfer |

### V3.2 NVFP4

| 旧编号 | 新编号 | 配置 |
|--------|--------|------|
| #55 | **#110** | TP=1 DP=8 EP FlashMLA |
| #56 | **#115** | TP=1 DP=8 EP FlashInfer |
| #57 | **#111** | TP=2 DP=4 EP FlashMLA |
| #58 | **#116** | TP=2 DP=4 EP FlashInfer |
| #59 | **#112** | TP=4 DP=2 EP FlashMLA |
| #60 | **#117** | TP=4 DP=2 EP FlashInfer |
| #61 | **#113** | TP=8 EP=off FlashMLA |
| #62 | **#118** | TP=8 EP=off FlashInfer |
| #63 | **#114** | TP=8 EP=on FlashMLA |
| #64 | **#119** | TP=8 EP=on FlashInfer |

### V3.1

| 旧编号 | 新编号 | 配置 |
|--------|--------|------|
| #75 | **#120** | TP=1 DP=8 EP FP8 |
| #76 | **#125** | TP=1 DP=8 EP NVFP4 |
| #77 | **#121** | TP=2 DP=4 EP FP8 |
| #78 | **#126** | TP=2 DP=4 EP NVFP4 |
| #79 | **#122** | TP=4 DP=2 EP FP8 |
| #80 | **#127** | TP=4 DP=2 EP NVFP4 |
| #81 | **#123** | TP=8 EP=off FP8 |
| #82 | **#128** | TP=8 EP=off NVFP4 |
| #83 | **#124** | TP=8 EP=on FP8 |
| #84 | **#129** | TP=8 EP=on NVFP4 |

---

## 3. results.csv 使用指南

### 3.1 列说明

| 列名 | 类型 | 说明 |
|------|------|------|
| num | int | 测试编号 |
| timestamp | string | 时间戳 (YYYYMMDD_HHMMSS) |
| date | date | 日期 |
| model | string | 模型名 (deepseek_v32, qwen35_397b, ...) |
| quant | string | 量化方式 (fp8, nvfp4, bf16, unknown) |
| attn | string | 注意力后端 (flashmla, flashinfer, default) |
| tp | int | Tensor Parallel |
| dp | int | Data Parallel |
| ep | string | Expert Parallel (on/off) |
| mtp_n | int | MTP 投机 token 数 (0=关闭) |
| deploy | string | 部署方式 (bare/docker) |
| gpu_util | float | GPU 显存利用率 |
| acceptance_len | float | MTP 平均接受长度 |
| acceptance_rate_pct | float | MTP 接受率 (%) |
| concurrency | int | 并发数 |
| rps | float | 每秒请求数 |
| avg_lat_s | float | 平均延迟 (秒) |
| ttft_s | float | 首 token 延迟 (秒) |
| tpot_ms | float | 每 token 延迟 (毫秒) |
| toks_s | float | 每秒 token 数 |
| success_pct | float | 成功率 (%) |
| superseded_by | int | 替代此测试的编号（非空=已废弃） |
| filename | string | 对应的 markdown 文件名 |

### 3.2 分析时必须过滤

```python
import pandas as pd

df = pd.read_csv('benchmark_result/results.csv')

# 必须：过滤掉被替代的测试
df_valid = df[df['superseded_by'].isna()]

# 必须：排除无效数据
df_valid = df_valid[~df_valid['num'].isin([187, 188, 189])]

# 推荐：排除 archive 外的早期测试
df_current = df_valid[df_valid['num'] >= 85]
```

### 3.3 注意事项

- `quant=unknown` 的行 (#39–#44) 实际是 NVFP4（从文件名可知），但 CSV 标记不准确
- 每个测试编号对应 7 行数据（concurrency = 1, 2, 4, 8, 20, 40, 60）
- MTP 测试 (#145–#185) 的 concurrency 范围缩小为 1, 2, 4, 8
- `gpu_util` 仅少数测试有值（#120, #126 = 0.85）

---

## 4. 测试配置矩阵

### 4.1 V3.2（#100–#119, 20 测试）

| 编号 | 量化 | 注意力 | TP | DP | EP |
|------|------|--------|----|----|-----|
| #100 | FP8 | FlashMLA | 1 | 8 | on |
| #101 | FP8 | FlashMLA | 2 | 4 | on |
| #102 | FP8 | FlashMLA | 4 | 2 | on |
| #103 | FP8 | FlashMLA | 8 | 1 | off |
| #104 | FP8 | FlashMLA | 8 | 1 | on |
| #105 | FP8 | FlashInfer | 1 | 8 | on |
| #106 | FP8 | FlashInfer | 2 | 4 | on |
| #107 | FP8 | FlashInfer | 4 | 2 | on |
| #108 | FP8 | FlashInfer | 8 | 1 | off |
| #109 | FP8 | FlashInfer | 8 | 1 | on |
| #110 | NVFP4 | FlashMLA | 1 | 8 | on |
| #111 | NVFP4 | FlashMLA | 2 | 4 | on |
| #112 | NVFP4 | FlashMLA | 4 | 2 | on |
| #113 | NVFP4 | FlashMLA | 8 | 1 | off |
| #114 | NVFP4 | FlashMLA | 8 | 1 | on |
| #115 | NVFP4 | FlashInfer | 1 | 8 | on |
| #116 | NVFP4 | FlashInfer | 2 | 4 | on |
| #117 | NVFP4 | FlashInfer | 4 | 2 | on |
| #118 | NVFP4 | FlashInfer | 8 | 1 | off |
| #119 | NVFP4 | FlashInfer | 8 | 1 | on |

### 4.2 MTP 测试编号速查

| 模型 | MTP=1 | MTP=2 | MTP=3 |
|------|-------|-------|-------|
| V3.2 FP8 EP=off | #145 | #162 | #166 |
| V3.2 FP8 EP=on | #146 | #163 | #167 |
| V3.2 NVFP4 EP=off | #147 | #164 | #168 |
| V3.2 NVFP4 EP=on | #148 | #165 | #169 |
| V3.1 FP8 EP=off | #149 | #170 | #174 |
| V3.1 FP8 EP=on | #150 | #171 | #175 |
| V3.1 NVFP4 EP=off | #151 | #172 | #176 |
| V3.1 NVFP4 EP=on | #152 | #173 | #177 |
| Qwen3.5 BF16 EP=off | #158 | #178 | #182 |
| Qwen3.5 BF16 EP=on | #159 | #179 | #183 |
| Qwen3.5 FP8 EP=off | #160 | #180 | #184 |
| Qwen3.5 FP8 EP=on | #161 | #181 | #185 |

> 缺失 #153–#157：原计划 Qwen3.5 mtp=0 基线，跳过（已有 #130–#144 无 MTP 基线）

---

## 5. 文件位置参考

| 内容 | 路径 |
|------|------|
| 结构化数据 | `benchmark_result/results.csv` |
| 单次测试详情 | `benchmark_result/YYYYMMDD_HHMMSS_##_*.md` |
| 测试索引 | `benchmark_result/INDEX.md` |
| 分析报告 | `benchmark_result/benchmark_analysis.md` |
| 客户报告 | `benchmark_result/benchmark_analysis_customer.md` |
| B200 vs H200 | `benchmark_result/b200_vs_h200_report_v2.md` |
| MTP 接受率 | `benchmark_result/acceptance_rates.csv` |
| 废弃数据 | `benchmark_result/archive/`（禁止读取） |
