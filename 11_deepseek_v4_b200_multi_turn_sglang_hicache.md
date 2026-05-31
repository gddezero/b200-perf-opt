# DeepSeek V4-Pro 多轮长上下文压测（B200 × 8）— SGLang HiCache 版

> **测试时间**：2026-05-16 ~ 2026-05-17
> **硬件**：GCP A4 B200 SXM × 8 (180 GB HBM3e/卡)
> **软件**：SGLang `lmsysorg/sglang:nightly-dev-cu13-20260516-d523ae12` (PR #24691 HiCache for V4 已 merged)
> **配方**：`docker_sglang_fp8_hicache_megamoe.sh` (latency, TP=8, EP=8, DeepEP MegaMoE + EAGLE n=3 + HiCache ratio=12)
> **数据集**：`multi_turn_{60,100,200,400}k_pg_sg.jsonl` (PG + ShareGPT 真材料, V3 interleaved 格式, turn-1 对齐 60K/100K/200K/400K)
> **压测工具**：`evalscope perf custom_multi_turn` (docker), NUM_PER_P=10
> **本次新增工具链**：`prom_snapshot.py` (counter + gauge + histogram 三模式) / `price_analyzer.py` (cost basis 反推) / `batch_multi_turn.sh v3.1` (集成 AUTO_STOP_ON_CRASH)
> **关联文档**：[10_deepseek_v4_b200_multi_turn.md](10_deepseek_v4_b200_multi_turn.md) (无 HiCache 基线，对照组)

---

# 第一部分：测试结果

## 1. 总览（22 档 4 dataset, ID #243-#270）

| dataset | parallel 跑了 | 跑了 / 想跑 | Sweet spot (RPS×latency) | Cost basis (uncached/cached/output $/1M) | AUTO_STOP 触发档 |
|---|---|---|---|---|---|
| 60K  | 1,2,4,8,16,32,64 | 7/7 ✅ | #244 p=2 (RPS 0.65, lat 5.9s) | **$1.53 / $0.15 / $7.64** | p=64 (host_fill=100%) |
| 100K | 1,2,4,8,16,32,64 | 7/7 ✅ | #251 p=2 (RPS 0.65, lat 6.3s) | **$1.21 / $0.12 / $6.06** | p=64 (host_fill=100%) |
| 200K | 1,2,4,8,16     | 5/7  | #258 p=2 (RPS 0.45, lat 8.4s) | **$0.96 / $0.10 / $4.78** | p=16 后跳 p=32/64 (host_fill=100% + device_hit<5%) |
| 400K | 1,2,4          | 3/7  | #264 p=1 (RPS 0.165, lat 11.6s) | **$1.21 / $0.12 / $6.06** | p=4 后跳 p=8/16/32/64 (host_fill=99.99% + device_hit=2.52%) |

**核心结论（6 条，与 doc 10 无 HiCache 基线最大区别）**：

1. ⭐ **HiCache 把"cache hit 塌方"崩盘救活**：doc 10 里 60K p=16 cache hit 1.23%（全 cache miss 重算），本次 60K p=16 cache hit **74.54%**（device 8.96% + host 65.58%，HiCache D↔H 接力命中）。崩盘形态从"hit 塌方"变成"latency 线性恶化 + queue 堆积"。
2. **HiCache 让 200K+ 上限提高 4×**：doc 10 200K p=4 是上限，本次 200K p=16 仍能跑完且 host_hit=70%。但代价是 P95 TTFT 从 17s 涨到 188s（queue 154s）。
3. **Sweet spot 全部稳定在 p=2**（除 400K p=1）：低 p 由 GPU pool 装下决定，高 p 不再崩但 latency 不可接受。
4. **400K 在 V4-Pro + B200×8 上接近物理上限**：单 conv 400K 占 53% GPU pool → 2 conv 800K > pool 762K → 即使 p=2 也要 HiCache 接力（host_hit 32%）；p=4 起 host_hit 81% + host_fill 几乎 100%。
5. **真崩盘信号从 "cache_hit < 50%" 变成 "host_fill=100% + device_hit<5%"**（HiCache 启用后）。AUTO_STOP_ON_CRASH 规则按此调整。
6. **Cost basis 与 doc 10 推论一致但更准**：200K cost basis $0.96 < 60K $1.53（context 越长单 token cost 越低，cache 复用率主导）。400K 反弹 ($1.21) 因 HiCache spillover overhead 占比上升。

## 2. 完整数据（按 dataset 分组）

### 2.1 60K 7 档（#243-#249）

| ID | p | dur | RPS | TTFT P95 | TPOT P95 | E2E P95 | **hit total** | **hit device** | **hit host** | evict/p | retract | full_pool peak | swa peak | hicache_fill peak | num_run peak | num_queue peak | spec_accept | server gen tput |
|:--:|:-:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|--:|--:|--:|:-:|:-:|--:|--:|
| #243 | 1 | 244s | 0.42 | 3.66s | 18.9 ms | 6.45s | **78.07%** | 78.07% | 0.00% | 1.63 | 0 | 0.103 | 0.791 | 8% | 1 | 0 | 2.50 | 89.0 t/s |
| #244 | 2 | 307s | 0.65 | 3.88s | 21.7 ms | 9.88s | **78.04%** | 78.04% | 0.00% | 2.61 | 0 | 0.196 | 0.835 | 16% | 2 | 1 | 2.69 | 145.0 t/s |
| **#245** | **4** | 394s | 1.02 | 3.93s | 24.2 ms | 17.41s | **80.19%** | 78.02% | **2.17%** | 2.87 | 0 | 0.387 | 0.933 | 27% | 4 | 3 | 2.78 | 224.6 t/s |
| **#246** | **8** | 588s | 1.36 | 3.91s | 54.4 ms | 26.17s | **80.78%** | 78.02% | **2.76%** | 2.93 | 0 | 0.759 | 0.970 | 42% | 8 | 6 | 2.74 | 297.2 t/s |
| #247 | 16 | 1240s | 1.29 | 19.09s | 83.6 ms | 52.89s | **74.54%** | 8.96% | **65.58%** | 9.06 | 0 | **0.998** | 0.916 | 55% | 16 | 15 | 2.70 | 284.3 t/s |
| #248 | 32 | 2431s | 1.32 | 57.80s | 88.4 ms | 91.40s | **74.97%** | 6.03% | **68.94%** | 8.98 | 0 | **0.999** | 0.956 | 77% | 19 | 29 | 2.63 | 291.0 t/s |
| #249 | 64 | 4938s | 1.30 | **184.63s** | 89.9 ms | **189.83s** | **74.40%** | 5.18% | **69.22%** | 9.01 | **6** | **0.999** | 0.983 | **100%** ⚠️ | 17 | 61 | 2.59 | 290.5 t/s |

**关键拐点**：
- **p=8→16**：device hit 从 78% 塌到 9%（GPU pool 不够，evict 加倍），host hit 从 3% 暴涨到 66%（HiCache 真接管），总 hit 仅微降到 74.5%
- **p=64**：retract 出现（6 个 active KV 被强制驱逐），P95 TTFT/E2E 接近 190s
- **gen throughput 在 p=8 饱和 297 t/s**，往后 p 增加只增加 queue 不增加吞吐

### 2.2 100K 7 档（#250-#256）

| ID | p | dur | RPS | TTFT P95 | TPOT P95 | E2E P95 | **hit total** | **hit device** | **hit host** | evict/p | retract | full_pool peak | hicache_fill peak | num_run peak | num_queue peak | spec_accept | server gen tput |
|:--:|:-:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|--:|--:|:-:|:-:|--:|--:|
| #250 | 1 | 267s | 0.30 | 7.43s | 20.2 ms | 11.50s | **80.66%** | 78.80% | 1.86% | 1.91 | 0 | 0.155 | 100% ⚠️ | 1 | 0 | 2.72 | 83.7 t/s |
| #251 | 2 | 328s | 0.49 | 7.46s | 21.4 ms | 17.34s | **82.26%** | 78.76% | 3.50% | 2.40 | 0 | 0.304 | 21% | 2 | 1 | 2.73 | 131.3 t/s |
| **#252** | **4** | 459s | 0.70 | 7.46s | 24.1 ms | 19.50s | **83.70%** | 79.21% | 4.49% | 2.53 | 0 | 0.604 | 36% | 4 | 3 | 2.79 | 190.9 t/s |
| #253 | 8 | 785s | 1.02 | 11.64s | 49.4 ms | 37.57s | **82.64%** | 31.01% | **51.64%** | 5.22 | 0 | **0.996** | 50% | 8 | 7 | 2.74 | 223.2 t/s |
| #254 | 16 | 1878s | 0.85 | 46.10s | 53.5 ms | 73.96s | **77.04%** | 6.06% | **70.99%** | 8.80 | 0 | **0.998** | 66% | 9 | 14 | 2.70 | 186.3 t/s |
| #255 | 32 | 3867s | 0.83 | 175.06s | 52.1 ms | 183.18s | **75.72%** | 4.81% | **70.91%** | 9.15 | 0 | **0.999** | 86% | 11 | 30 | 2.61 | 181.4 t/s |
| #256 | 64 | 8109s | 0.79 | **366.34s** | 55.8 ms | **370.61s** | **73.33%** | 4.86% | **68.47%** | 9.11 | **1** | **0.999** | **100%** ⚠️ | 11 | 61 | 2.58 | 172.9 t/s |

**关键拐点**：
- **p=4→8**：device hit 从 79% → 31%（GPU pool 撑不住 8 conv），host hit 从 4% → 52%（HiCache 接力）
- **p=8→16**：device hit 进一步塌到 6%，host hit 71%，evict/p 5→9
- **gen throughput 在 p=8 饱和 223 t/s** 之后下降（HiCache D↔H 来回搬反而降效）
- 100K p=1 hicache_fill_peak=100% 是因为单 conv 100K 直接把 5 turn 累计塞满（120K × ~73 entries 用满 HiCache 池）

### 2.3 200K 5 档（#257-#261，AUTO_STOP @ p=32）

| ID | p | dur | RPS | TTFT P95 | TPOT P95 | E2E P95 | **hit total** | **hit device** | **hit host** | evict/p | retract | full_pool peak | hicache_fill peak | num_run peak | num_queue peak | spec_accept | server gen tput |
|:--:|:-:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|--:|--:|:-:|:-:|--:|--:|
| #257 | 1 | 323s | 0.25 | 16.81s | 19.0 ms | 16.81s | **83.22%** | 81.29% | 1.93% | 2.07 | 0 | 0.287 | 100% ⚠️ | 1 | 0 | 2.92 | 64.6 t/s |
| **#258** | **2** | 439s | 0.36 | 17.65s | 21.2 ms | 19.50s | **84.86%** | 81.30% | 3.57% | 2.26 | 0 | 0.565 | 34% | 2 | 1 | 2.84 | 94.2 t/s |
| #259 | 4 | 874s | 0.36 | 19.27s | 45.1 ms | 39.31s | **85.58%** | 34.97% | **50.61%** | 3.69 | 0 | 0.868 | 56% | 4 | 3 | 2.83 | 94.9 t/s |
| #260 | 8 | 2450s | 0.26 | 59.64s | 49.6 ms | 86.12s | **78.86%** | 6.27% | **72.59%** | 8.75 | 0 | **0.998** | 83% | 5 | 7 | 2.75 | 69.2 t/s |
| #261 | 16 | 5424s | 0.24 | **187.82s** | 50.4 ms | 264.24s | **74.36%** | 4.13% | **70.22%** | 9.18 | 0 | **0.999** | **100%** ⚠️ | 6 | 15 | 2.70 | 62.6 t/s |
| ⛔ p=32/64 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | AUTO_STOP (host_fill 100% + device_hit < 5%) |

**关键拐点**：
- p=4→8 device hit 从 35% → 6%（200K p=4 就开始 HiCache 接管，比 60K 早得多）
- p=8 起 gen throughput 反降（HiCache 跨网带宽 / KV 调度 overhead）
- 200K p>4 不再有"RPS 增长"，只有 latency 恶化

### 2.4 400K 3 档（#264-#266，AUTO_STOP @ p=8）

| ID | p | dur | RPS | TTFT P95 | TPOT P95 | E2E P95 | **hit total** | **hit device** | **hit host** | evict/p | retract | full_pool peak | hicache_fill peak | num_run peak | num_queue peak | spec_accept | server gen tput |
|:--:|:-:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|--:|--:|:-:|:-:|--:|--:|
| **#264** | **1** | 606s | 0.165 | 36.12s | 20.1 ms | 47.25s | **81.37%** | 79.69% | 1.68% | 2.74 | 0 | 0.546 | **100%** ⚠️ | 1 | 0 | **3.01** | 35.9 t/s |
| #265 | 2 | 2739s | 0.073 | 83.17s | 80.2 ms | 86.50s | **83.48%** | 51.17% | **32.31%** | 3.48 | 0 | 0.547 | 61% | **1** ⚠️ | 1 | 2.82 | 15.0 t/s |
| #266 | 4 | 5927s | 0.067 | **182.97s** | 84.7 ms | 193.68s | **84.03%** | **2.52%** | **81.51%** | 8.63 | 0 | **0.997** | **99.99%** ⚠️ | 2 | 3 | 2.75 | 14.4 t/s |
| ⛔ p=8/16/32/64 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | AUTO_STOP (host_fill 99.99% + device_hit=2.52%) |

**400K 是 GPU pool 的物理瓶颈点**：
- p=1: 1 conv 占 GPU pool 54.6% (~407K tokens) — sweet spot
- p=2: 仍只能 1 个 conv 真 active（800K > pool 762K），1 在 queue → 实际串行 + HiCache 接力
- p=4: 全部 HiCache 接力（device hit 2.5%），host hit 81%
- spec_accept_length 在 400K 上反而最高 (3.01) — 长 prompt 让 EAGLE draft 更准
- gen throughput 在 p=2 暴跌 (145→15 t/s) 因 1 active + 1 queue 串行

### 2.5 4 dataset 横向对比

| 指标 | 60K | 100K | 200K | 400K |
|---|---|---|---|---|
| baseline TTFT P95 (p=1) | 3.66s | 7.43s | 16.81s | 36.12s |
| baseline TPOT P95 (p=1) | 18.9 ms | 20.2 ms | 19.0 ms | 20.1 ms |
| baseline real hit (p=1) | 78.07% | 80.66% | 83.22% | 81.37% |
| baseline spec accept | 2.50 | 2.72 | 2.92 | **3.01** |
| Sweet spot parallel | p=2 | p=2 | p=2 | **p=1** |
| Sweet spot RPS | 0.65 | 0.49 | 0.36 | 0.165 |
| Sweet spot cost basis (uncached) | $1.53 | $1.21 | **$0.96** | $1.21 |
| HiCache 接管起点 (device hit < 50%) | p=16 | p=8 | p=8 | p=4 |
| AUTO_STOP 触发 | p=64 (末档) | p=64 (末档) | p=32 (中断) | p=8 (中断) |
| 最大 gen tput | 297 t/s @p=8 | 223 t/s @p=8 | 95 t/s @p=4 | 36 t/s @p=1 |

**横向规律**：
1. baseline TTFT 与 context 长度近线性（4-7-16-36s vs 60K-100K-200K-400K = 1:2:4:7）— prefill 主导
2. baseline TPOT 与 context 长度无关（19-20ms 平台）— decode 单 token 时间稳定
3. **baseline cache hit 随 context 增长**（78→81→83→81%）— 多轮长 prefix 复用更高
4. **HiCache 接管起点随 context 越大越早**：60K 要 p=16 才接管 / 400K p=4 已全接管
5. **最大 gen throughput 与 context 反比** — 长 prompt 单 forward 占用 GPU 时间更长
6. **Sweet spot 都在 p=1/2** — 4 个 dataset 共通

---

# 第二部分：可复现的测试方法与步骤

## 3. 硬件与模型

### 3.1 硬件

GCP A4 B200 × 8 (180 GB HBM3e/卡, NVLink 5.0 全互联, CUDA 13.0, driver 590)。详见 [09_deepseek_v4_b200.md §1.1](09_deepseek_v4_b200.md)。

### 3.2 模型 + KV pool（ratio=12 vs doc 10 ratio=6 对比）

| 项 | doc 10 (无 HiCache) | doc 11 (HiCache ratio=12) |
|---|---|---|
| 模型 | DeepSeek-V4-Pro FP8 | 同 |
| weights / GPU (TP=8) | 113 GB | 113 GB |
| EAGLE draft weights / GPU | 0 | 2.89 GB（启用 spec decode）|
| `max_total_num_tokens` (GPU pool) | 1,069,568 | **762,112** (mem_fraction=0.80 留 cudagraph + EAGLE) |
| HiCache host pool capacity | — | **9,145,344 logical tokens** (= 762K × 12, 约 1.6 TB host RAM) |
| 容器 host RAM 占用 | ~860 GB (ratio=6) | **~1.63 TB** (ratio=12) |
| bytes_per_token (GPU per-card) | 38 KB (含 lightning indexer + compressor state) | ~32 KB |

⚠️ **不要混用 doc 10 的 KV 数据做规划**：doc 10 是 ratio=6 + 无 EAGLE 的配置；doc 11 ratio=12 + EAGLE，GPU pool 缩到 762K（doc 10 的 71%），换来 host pool 9.15M（doc 10 的 8.55×）。HiCache 把短缺挪到 host。

详细 bytes_per_token 推导见 doc 10 §10。

## 4. 部署：HiCache ratio=12 + MegaMoE + EAGLE（本次唯一配方）

### 4.1 配方文件

`gpu/b200/inference/recipes/deepseek_v4_pro/latency/docker_sglang_fp8_hicache_megamoe.sh`

关键参数（vs doc 10 的差异）：

| 参数 | doc 10 | doc 11 | 备注 |
|---|---|---|---|
| `--mem-fraction-static` | 0.82 | **0.80** | 留 cudagraph + EAGLE + NCCL buffer |
| `--enable-hierarchical-cache` | 不开 | **开** | HiCache D↔H 写穿 |
| `--hicache-ratio` | — | **12** | host pool = GPU pool × 12 |
| `--hicache-write-policy` | — | **write_through** | GPU 写同步落 host, evict 后立即可读 |
| `--hicache-io-backend` | — | **direct** | D↔H 用 direct memcpy |
| `--hicache-mem-layout` | — | **page_first_direct** | V4 必需（走 UnifiedRadixTree 新 path） |
| `--moe-runner-backend` | flashinfer_mxfp4 | **同** | |
| `--moe-a2a-backend` | — | **deepep** | MegaMoE expert all-to-all (ep_size 自动=tp_size=8) |
| `--deepep-mode` | — | **auto** | normal_dispatch+combine 各 96 sm |
| `--speculative-algorithm` | EAGLE | **同 (n=3, draft=4, topk=1)** | |
| 环境变量 | — | **`SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` + `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0` + DeepEP 5 个 env** | 见配方文件头部注释 |

### 4.2 镜像

`lmsysorg/sglang:nightly-dev-cu13-20260516-d523ae12` (image digest `sha256:06c4d4a2...`)

⚠️ 必须用 2026-05-16 以后的 nightly：PR #24691 (HiCache for V4) 才合并进去。详见 memory `reference_v4_hicache_pr24691_landed_2026_05_16.md`。

### 4.3 启动 + 健康检查

```bash
bash gpu/b200/inference/recipes/deepseek_v4_pro/latency/docker_sglang_fp8_hicache_megamoe.sh
# 等 ~7 min model loading + EAGLE draft load + HiCache init
docker logs sglang-v4-pro-hicache-megamoe 2>&1 | grep -c "HiCache D"  # 应 = 8 (8 worker init)
curl http://localhost:8088/v1/models | jq .
curl -s http://localhost:8088/metrics | grep hicache_host_total_tokens
# 应见: hicache_host_total_tokens = 9145344
```

## 5. 数据集（4 个 dataset, V3 interleaved 格式）

`gen_multi_turn_dataset.py` V3 写 user/assistant interleaved JSONL array（evalscope custom_multi_turn loader 要求）。

```bash
# 通过 evalscope docker 跑（host 无 transformers 包）
docker run --rm \
  -v $HOME/shared:$HOME/shared \
  -v /tmp/v4-tokenizer-with-chat-template:/tmp/v4-tokenizer-with-chat-template:ro \
  -w $HOME/shared/multi_turn_bench \
  evalscope:1.7.0 \
  python3 scripts/gen_multi_turn_dataset.py \
    --data-dir data \
    --tokenizer /tmp/v4-tokenizer-with-chat-template \
    --turn-lens 60000,5000,5000,5000,5000 \
    --num-conv 100 \
    --out custom_dataset/multi_turn_60k_pg_sg.jsonl --seed 42
```

| 数据集 | turn-1 长度 | 5 turn 累计 | 文件 |
|---|---:|---:|---|
| `multi_turn_60k_pg_sg.jsonl`  | 60K  | ~70K  | 29 MB |
| `multi_turn_100k_pg_sg.jsonl` | 100K | ~110K | 45 MB |
| `multi_turn_200k_pg_sg.jsonl` | 200K | ~210K | 85 MB |
| `multi_turn_400k_pg_sg.jsonl` | 400K | ~410K | 172 MB |

⚠️ **400K 第一次跑挂的踩坑**：B200 上 `gen_multi_turn_dataset.py` 是旧版本（V1 格式 `{"messages": [...]}`），evalscope 不认 → 全 line skip → "Dataset produced no conversations" exit=1。Fix：rsync 本地 V3 版到 B200 后重跑。详见 §17.1。

## 6. batch_multi_turn.sh v3.1（本次新增 prom 全量采集 + AUTO_STOP）

每档自动：
1. POST `/flush_cache` 清 RadixTree
2. `prom_snapshot.py snapshot` → before.json（counter + gauge + histogram baseline）
3. 启 `prom_snapshot.py sample` 后台周期采（默认 5s）
4. `evalscope perf --dataset custom_multi_turn --multi-turn --max-turns 5 --max-tokens 500`
5. kill sampler + `prom_snapshot.py snapshot` → after.json
6. `prom_snapshot.py finalize` → `metrics/prom_<NNN>_<TAG>_p<P>.json` + `_summary.txt`
7. （新）AUTO_STOP_ON_CRASH 检查：`num_retracted_peak > 0` / `evalscope exit != 0` / `host_fill > 99% + device_hit < 5%` → 立即 break PLIST
8. （新）batch 结束自动跑 `price_analyzer.py` → `metrics/pricing_<TAG>_<TS>.md`

```bash
ID_START=243 PLIST="1 2 4 8 16 32 64" \
DATASET=$HOME/shared/multi_turn_bench/custom_dataset/multi_turn_60k_pg_sg.jsonl \
MAX_PROMPT=80000 \
TAG=ratio12_60k \
NUM_PER_P=10 \
AUTO_STOP_ON_CRASH=1 \
COST_PER_HOUR=40 \
EVALSCOPE_CMD="bash $HOME/shared/multi_turn_bench/scripts/evalscope_in_docker.sh" \
bash $HOME/shared/multi_turn_bench/scripts/batch_multi_turn.sh
```

## 7. 新工具链：prom_snapshot.py / price_analyzer.py

### 7.1 prom_snapshot.py（SGLang `/metrics` 全量指标三模式工具）

stdlib-only Python，4 类指标全采：

| 类 | 例 | 用途 |
|---|---|---|
| Counter delta | `prompt_tokens_total`, `cached_tokens_total{cache_source=device/host/storage}`, `evicted_tokens_total`, `realtime_tokens_total{mode=...}` | 真实 cache hit 三层 + 驱逐 |
| Histogram percentiles | `time_to_first_token_seconds`, `inter_token_latency_seconds`, `e2e_request_latency_seconds`, `queue_time_seconds`, `per_stage_req_latency_seconds`, `eviction_duration_seconds` | TTFT/TPOT/e2e P50/P95/P99 服务端真值 |
| Gauge peak/avg (5s 周期) | `full_token_usage`, `swa_token_usage`, `mamba_usage`, `hicache_host_used_tokens`, `num_running_reqs`, `num_queue_reqs`, `num_retracted_reqs`, `spec_accept_length`, `gen_throughput` | KV 池峰值、调度、HiCache、Spec decode |
| Computed summary | `real_cache_hit_pct`, `device_cache_hit_pct`, `host_cache_hit_pct`, `evict_per_prompt`, `full_token_usage_peak`, `hicache_host_fill_peak_pct`, `num_running_peak`, `spec_accept_length_avg`, `server_gen_throughput_tok_s` | 综合诊断 |

完整字段表见 `shared/multi_turn_bench/scripts/README.md` § 2.5。

### 7.2 price_analyzer.py（token 定价反推）

读 prom_*.json 解方程 `cost/h = p_in × (T_uncached + r_cached·T_cached + r_out·T_out)` 反推 cost basis，多 markup 给推荐零售价。

输出 7 段 markdown：三档真实吞吐 / cost basis / sweet spot / 多 markup 报价 / 行业对照 / 敏感性 / caveat。

详见 `shared/multi_turn_bench/scripts/README.md` § 2.6 + 方法论 `shared/docs/token_pricing_methodology.md`。

### 7.3 Grafana dashboard 升级

`monitoring/grafana_dashboards/sglang-b200.json` 新增 4 个 section：⑨ HiCache / ⑩ KV Pool Detail / ⑪ Speculative Decode / ⑫ Crash Signals。修复了 2 个 broken panel（`sglang:cache_hit_rate` gauge 是 SGLang bug 永远 0）。详见 memory `feedback_bind_mount_inode_trap.md`（部署 + bind mount inode 坑）。

## 8. 端到端时间线

| 任务 | 耗时 | 说明 |
|---|---|---|
| 数据集生成（4 个 dataset, 各 100 conv）| ~30 min | 通过 evalscope docker 跑 |
| SGLang 重启 ratio=12 | ~8 min | 含 EAGLE draft load + HiCache D↔H init (8 worker) |
| **60K 7 档 (#243-#249)** | **~2h 49min** | p=64 单档 1h 23min |
| **100K 7 档 (#250-#256)** | **~4h 23min** | p=64 单档 2h 15min |
| **200K 5 档 (#257-#261)** | **~3h 47min** | p=16 单档 1h 30min (AUTO_STOP @p=32) |
| **400K 3 档 (#264-#266)** | **~2h 31min** | p=4 单档 1h 39min (AUTO_STOP @p=8) |
| **22 档总耗时** | **~13h 30min** | 含档间 flush + snapshot + finalize + auto pricing |

## 9. 编号映射

| 编号区间 | dataset | parallel 档 | tag |
|:---:|:---:|---|---|
| #243-#249 | 60K  | 1,2,4,8,16,32,64 | `ratio12_60k`  |
| #250-#256 | 100K | 1,2,4,8,16,32,64 | `ratio12_100k` |
| #257-#261 | 200K | 1,2,4,8,16       | `ratio12_200k` (AUTO_STOP @ 32) |
| #264-#266 | 400K | 1,2,4            | `ratio12_400k` (AUTO_STOP @ 8) |

⚠️ #262-#263 跳号：原本 200K p=32/64 槽位预留，AUTO_STOP 触发后未跑（编号一次性，跳号合规）。

⛔ **作废数据**（不要引用）：#237-#242 (ratio=6 + 旧 batch 无 prom 采集)、#242 (verify_prom)、#264 重跑前的初始失败 (V1 格式 dataset)。详见 `benchmark_result/metrics/SUPERSEDED.md`。

---

# 第三部分：分析、总结、踩坑

## 10. V4-Pro KV 占用两口径

完全引用 [doc 10 §10](10_deepseek_v4_b200_multi_turn.md#10-v4-pro-每-token-kv-占用两个口径必须区分)（口径 A 理论摊销 ~4.85 KB FP8 / 口径 B 引擎物理预算 38 KB/token）。doc 11 的 GPU pool 缩到 762K 是因为 mem_fraction 0.82→0.80 + EAGLE draft 2.89 GB 占用，不是 bytes_per_token 变了。

## 11. ⭐ HiCache 改变了 KV pool 的"装得下多少"语义

### 11.1 GPU pool vs HiCache host pool

| 池 | 容量 | 物理介质 | 单 token 写入成本 |
|---|---|---|---|
| GPU full pool | 762,112 tokens | HBM (TP=8 共享) | 0（write-through 不阻塞 forward）|
| HiCache host pool | 9,145,344 tokens (= 762K × 12) | host RAM (1.63 TB) | D→H memcpy（PCIe 带宽）|

write_through 模式：GPU 每写一个 KV block 异步复制到 host pool；GPU 内 radix evict 时 host 副本仍在 → 后续命中从 host 读回（counter `cached_tokens_total{cache_source=host}`）。

### 11.2 doc 10 vs doc 11 崩盘形态对比

| 行为 | doc 10 (无 HiCache) | doc 11 (HiCache ratio=12) |
|---|---|---|
| 60K p=16 cache hit | 1.23%（全 cache miss 重算）| 74.54%（device 9% + host 65.58%）|
| 100K p=16 行为 | 跨过测试范围 | 仍可跑 (cache hit 77%, queue 9s) |
| 200K p>4 行为 | 调度自限 num_running=2 | p=16 num_running=6 + queue=15 |
| 崩盘信号 | `cache hit < 50%` + `evict/p > 5` + `RPS 倒退` | `host_fill = 100%` + `device_hit < 5%` + `latency 线性恶化` |
| 救援方向 | 降并发 | 加 HiCache ratio / 加 host RAM |

⭐ **HiCache 把"崩盘"从"灾难"变成"trade off"**：崩盘的 KV 复用率被保住，但 latency 用 TTFT / queue_time 抵了出去。生产用 SLA 而非 cache_hit 来定上限。

## 12. evalscope multi-turn 累计机制

完全引用 [doc 10 §12](10_deepseek_v4_b200_multi_turn.md#12-️-evalscope-multi-turn-累计机制为什么必须用-custom_multi_turn)。本次仍用 custom_multi_turn 真材料数据集，turn-1 对齐 60K/100K/200K/400K + 后续 turn ~2.7K incr（PG ShareGPT 池的限制，无法精确 5K）。

## 13. percentile 数据点统计有效性

本次 `NUM_PER_P=10`（doc 10 是 50）→ 每档 turn 数缩小到 `p × 10 × avg_turns(≈4.5)`：

| parallel | turn 数 | P95 数据点 (5%) | P99 数据点 (1%) |
|---:|---:|---:|---:|
| 1 | 45 | 2 ❌ | 0 ❌ |
| 4 | 180 | 9 ✓ | 1 ❌ |
| 8 | 360 | 18 ✓ | 3 ❌ |
| 16 | 720 | 36 ✓ | 7 ✓ |
| 32 | 1440 | 72 ✓ | 14 ✓ |
| 64 | 2880 | 144 ✓ | 28 ✓ |

⚠️ **本次 p=1/2 的 P95 都基于 < 5 数据点，置信度低**（doc 11 表格里 p=1 的 TTFT P95 实际可能有 ±20% 波动）。生产用 doc 10 的 NUM_PER_P=50 重测低 p 档才靠谱。

## 14. cache hit 三口径（本次新增 host 分层）

| 口径 | 来源 | 60K p=2 typical | 60K p=16 |
|---|---|---:|---:|
| eval_cache_hit_pct | evalscope `Average approx KV cache hit rate (%)` | ~91% | ~88% |
| **real_cache_hit_pct** | SGLang counter `cached_tokens_total / prompt_tokens_total` | 78.04% | 74.54% |
| **├─ device_cache_hit_pct** | `{cache_source=device}` | 78.04% | 8.96% |
| **├─ host_cache_hit_pct** | `{cache_source=host}` | 0.00% | 65.58% |
| **└─ storage_cache_hit_pct** | `{cache_source=storage}` | 0.00% | 0.00% |

⭐ **HiCache 启用后总 hit% 仍是核心指标**，但 device/host 分层揭示 cost 结构：
- device hit：免费（HBM 内 radix lookup）
- host hit：D↔H memcpy 成本（PCIe 带宽 + readback 延迟）
- 60K p=16 总 hit 74.54% 看着不错，但 device 仅 9% → 大部分命中走 host → TTFT 仍涨到 19s

`price_analyzer.py` 默认 `r_cached = 0.10`（覆盖 device + host 平均成本）。如果要更精细，可拆三档 ratio。

## 15. 物理崩盘点定位（HiCache 改写规则）

### 15.1 doc 10 崩盘条件 vs doc 11

| 条件 | doc 10（无 HiCache）| doc 11（HiCache ratio=12）|
|---|---|---|
| 触发 | GPU pool 占用 ≥ 90% | **GPU pool 满 + HiCache host pool 满 + device hit < 5%** |
| 表现 | cache hit 从 74% 一步塌方到 0-4% | latency 线性恶化（TTFT 4→185s），cache hit 维持 74% |
| RPS | 倒退 -68% | 平台稳定（受 gen tput 上限锁死）|
| evict/p | 跳变到 13-14 | 维持 9 平台（HiCache 接管驱逐成本）|
| retract | 极少 | p=64 才出现（60K=6 / 100K=1） |

### 15.2 AUTO_STOP_ON_CRASH 触发规则

`batch_multi_turn.sh v3.1` 内置三档检测：
1. evalscope `Failed requests > 0` 或 exit != 0
2. prom `num_retracted_peak > 0`
3. prom `host_fill_peak > 99% + device_hit < 5%`

任一触发 → 立即 break PLIST 跳到下一 dataset。本次触发：
- 200K p=32 之前 (#261 p=16 已 host_fill=100% + device_hit=4.13%，p=32 必然触发)
- 400K p=4 (#266 host_fill=99.99% + device_hit=2.52%)

## 16. parallel 上限规则（本次更新）

### 16.1 三类上限取 min

```
1. 物理上限（HiCache ratio=12 + AUTO_STOP）:
   60K:  p ≤ 64 全可跑（末档 host_fill=100% 触发 AUTO_STOP 但已完成）
   100K: p ≤ 64 全可跑（同上）
   200K: p ≤ 16（p=32 trigger AUTO_STOP）
   400K: p ≤ 4 （p=8 trigger AUTO_STOP）

2. SLA 上限（按 TPOT P95）:
   TPOT < 50 ms: 60K p≤8, 100K p≤8, 200K p≤8, 400K p≤2
   TPOT < 100 ms: 60K p≤32, 100K p≤64, 200K p≤16, 400K p≤4

3. 经济上限（边际 RPS ≥ 5%）:
   60K:  p=2→4 +57%, p=4→8 +33%, p=8→16 -5% → 经济拐点 p=8
   100K: p=2→4 +43%, p=4→8 +46%, p=8→16 -16% → 经济拐点 p=8
   200K: p=2→4 +0%, p=4→8 -28% → 经济拐点 p=4
   400K: p=1→2 -56%（HiCache 接力开销）→ 经济拐点 p=1
```

### 16.2 生产推荐表（按 SLA 严格度）

| dataset | 严格 SLA (TPOT<50ms) | 平衡 SLA (TPOT<100ms) | 最大吞吐 | ⛔ 禁用 |
|---|:---:|:---:|:---:|:---:|
| 60K  | **p=2-4** | p=8 | p=16-32（latency 已不可接受）| p=64 (retract>0) |
| 100K | **p=2-4** | p=8 | p=8-16 | p=64 (retract>0) |
| 200K | **p=2** | p=8 | p=8 | p>=16 (TTFT>180s) |
| 400K | **p=1** | p=1 | p=2 (50% throughput 损失) | p>=4 (HiCache 几乎打爆) |

## 17. 踩坑记录

### 17.1 400K dataset V1 格式 → evalscope 全 line skip

**症状**：第一次跑 #264 (400K p=1) evalscope exit=1，log 里满是 `Skipping line: expected a non-empty JSON array`，`Dataset "custom_multi_turn" produced no conversations!`。

**根因**：B200 上 `~/shared/multi_turn_bench/scripts/gen_multi_turn_dataset.py` 是旧版本（V1）写 `{"messages": [...]}`，evalscope custom_multi_turn loader 要求 raw array `[{...}, {...}]`（V3 interleaved 格式）。60K/100K/200K dataset 是本地 V3 生成的没问题，400K 因为新生成走了 B200 上的 V1 脚本。

**Fix**：rsync 本地 V3 版到 B200 后重跑：
```bash
rsync -avz ~/code/ai_infra/shared/multi_turn_bench/scripts/gen_multi_turn_dataset.py \
  maxwellx_google_com@forrest-b200-01:~/shared/multi_turn_bench/scripts/
# 然后通过 evalscope docker 重新生成 400K
```

**预防**：每次启动 batch 前 `diff` 本地和 B200 的 gen 脚本，或干脆把 dataset 生成集成进 batch 脚本前置步骤。

### 17.2 SGLang strict_mem_check_during_idle 误杀

**症状**：HiCache 启用后 server 偶尔启动几分钟后挂掉，log 里 `ValueError: pool memory leak detected!`。

**根因**：SGLang 的 idle-phase leak checker 误把 HiCache 的 unflushed write 当作 leak。

**Fix**：recipe 加 `-e SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0`（已合并到 `docker_sglang_fp8_hicache_megamoe.sh`）。

### 17.3 prom counter sum 漏 host label

**症状**：`batch_multi_turn.sh v3` 早期 awk `cached_tokens_total` 没按 cache_source label sum，REAL_CACHE_HIT 显示偏低甚至负值。

**根因**：HiCache 启用后 `cached_tokens_total` 有 `cache_source=device|host|storage` 三个 label，每个独立 series。`{c=$NF}` 只取最后一行覆盖前面的。

**Fix**：`awk '... {c+=$NF}'`（累加），或更稳的 prom_snapshot.py `aggregate_counter` 按 label group sum。

### 17.4 Monitor false positive on stale CRASH text

**症状**：监控 batch 进度时 grep CRASH detected 立即触发"sweep 已崩"，实际是历史 batch log 残留。

**根因**：monitor 用 `grep -l "CRASH"` 在 logs/ 找文件，没限定本次 batch 时间窗口，老 sweep 的 CRASH 标记会一直 match。

**Fix**：用 `find ... -newer <current_batch_log>` 限定 mtime，或 monitor 直接 pgrep batch process。

### 17.5 Grafana dashboard 空数据（bind mount inode trap）

**症状**：SGLang dashboard 各 panel 全空，但 `/metrics` 端点有数据。

**根因**：litellm-hk 上 `prometheus.yml` 之前用 vim/sed -i 改过（unlink+create 换 inode），docker bind mount 仍指着旧 inode 的 prometheus.yml（targets 写错主机名），所以 scrape 一直 down。

**Fix**：(1) 用 `python3 write_text` / `cat > file` 保留 inode 改文件 (2) 已破坏 → `sudo docker restart monitor-prometheus`。详见 memory `feedback_bind_mount_inode_trap.md`。

### 17.6 Dashboard host 模板变量 stale label

**症状**：`$host` 下拉框显示 `b200, b200-01, b200-02, g4-01`（多出 stale `b200`）。

**根因**：旧 prom 配置用 `host: 'b200'` label，新约定 `host: 'b200-01'`。TSDB 保留过去 sample 的 label index 100 年（`--storage.tsdb.retention.time=100y`），即使 `delete_series` 也不能立即清 head block。

**Fix**：dashboard variable query 改宽 + 加 regex 过滤：`label_values(up, host)` + `regex: /^([a-z0-9]+-\d+)$/`。

## 18. 各 dataset 关键观察补充

### 18.1 60K 关键观察

- p=2 是 sweet spot（cost basis $1.53 最低 + latency 9.88s 仍可接受）
- p=8 是 SLA 极限（TPOT P95 54.4 ms，再上去会破 100ms）
- p=16 是 HiCache 接管起点（device hit 78%→9%）
- p=64 出现 retract（6 个），TTFT P95 184s 已不可接受
- spec_accept_length 在 p=4-8 最高 (2.78)，p=64 微降到 2.59（draft 准确率受调度抢占影响）

### 18.2 100K 关键观察

- p=4 是 sweet spot（cost basis 跟 60K p=2 同量级 $1.21 + latency 19.5s）
- p=8 device hit 已塌到 31%（比 60K 提早 8 档触发 HiCache 接管）
- p=16 起 gen throughput 实际下降（HiCache D↔H 跨网带宽限制 gen 路径）
- p=64 TTFT P95 366s，跟 60K p=64 类似（gen 端瓶颈而不是 prefill 端）

### 18.3 200K 关键观察

- p=2 sweet spot 但 latency 已经 19.5s（业务能接受才有意义）
- p=4 device hit 35%（HiCache 接管比 100K 更早）
- p=8 起 gen throughput 暴跌（94→69 t/s），HiCache spillover 严重
- p=16 host_fill=100% 触发 AUTO_STOP，p=32 跳过

### 18.4 400K 关键观察

- p=1 是唯一 sweet spot：单 conv 占 GPU pool 54.6%，其他全 host 接力
- p=2 num_run_peak=1（不是 2）：物理上 2 个 400K conv 装不下 GPU pool，必排队
- p=4 device hit 仅 2.52%，host hit 81.5%（完全 HiCache 接管）
- spec_accept_length 在 400K 反而最高 (3.01)：长 prompt context 让 EAGLE draft 更准
- 单条请求成本：p=1 dur 606s / 10 turn = 60.6s/turn（10 conv × ~50K avg prefill incr × 5 turn）

## 19. 下一步工作

### 19.1 短期

- [ ] 重复 60K NUM_PER_P=50 跑 p=1-8 细化（doc 10 的统计精度）→ 跟 doc 10 直接对比 ratio=12 vs ratio=6 的 SLA 影响
- [ ] 200K / 400K 用 SGLang `--max-running-requests` 显式限并发，看是否能避开 host_fill 100% AUTO_STOP
- [ ] HiCache ratio=8 / 16 / 24 对比，找 60K p=64 / 100K p=64 不出 retract 的最小 ratio

### 19.2 中期

- [ ] vLLM + LMCache 0.4.5 + Mooncake 1P1D 跨节点方案对比（G.1 smoke 部分通过，详见 `gpu/b200/inference/docs/lmcache_pr3171_v4_smoke_g1_result.md`）
- [ ] DPA + DeepEP + SMG router cache_aware 多轮压测（doc 10 §4.2 提到的另一条路）
- [ ] 同 prompt 的 SGLang HiCache vs vLLM NIXL PD 物理对比（cost basis + latency）

### 19.3 长期

- [ ] HiCache L3 SSD backend 启用（解决 host RAM 上限）
- [ ] 上 GB200 / B300（更大 HBM）后 GPU pool 翻倍，对 400K+ 场景的影响

---

## 附录 A：参考资料

- doc 10（无 HiCache 对照组）：[10_deepseek_v4_b200_multi_turn.md](10_deepseek_v4_b200_multi_turn.md)
- HiCache PR #24691 实测：memory `reference_v4_hicache_pr24691_landed_2026_05_16.md`
- HiCache + MegaMoE + EAGLE 配方调试：memory `reference_v4_hicache_megamoe_latency_bench_2026_05_16.md`
- bind mount 编辑器 unlink 陷阱：memory `feedback_bind_mount_inode_trap.md`
- token 定价方法论：`shared/docs/token_pricing_methodology.md`
- LMCache PR #3171 V4 支持调研：`gpu/b200/inference/docs/lmcache_pr3171_v4_kv_research.md`
- LMCache 0.4.5 G.1 smoke 实测：`gpu/b200/inference/docs/lmcache_pr3171_v4_smoke_g1_result.md`
- prom 工具：`shared/multi_turn_bench/scripts/README.md` § 2.5 / 2.6
- 22 档 prom JSON：`gpu/b200/inference/benchmark_result/metrics/prom_2[4567]?_ratio12_*.json`
- 4 个 pricing 报告：`gpu/b200/inference/benchmark_result/metrics/pricing_ratio12_*.md`

## 附录 B：本次新增产物清单

| 类型 | 路径 |
|---|---|
| Recipe | `gpu/b200/inference/recipes/deepseek_v4_pro/latency/docker_sglang_fp8_hicache_megamoe.sh` |
| 数据集 (4 个) | `shared/multi_turn_bench/custom_dataset/multi_turn_{60,100,200,400}k_pg_sg.jsonl` |
| 工具 (3 个) | `shared/multi_turn_bench/scripts/{prom_snapshot,price_analyzer}.py` + `batch_multi_turn.sh v3.1` |
| 22 prom JSON | `gpu/b200/inference/benchmark_result/metrics/prom_2[4567]?_ratio12_*.json` |
| 4 pricing 报告 | `gpu/b200/inference/benchmark_result/metrics/pricing_ratio12_{60k,100k,200k,400k}_*.md` |
| Grafana dashboard | `gpu/b200/inference/monitoring/grafana_dashboards/sglang-b200.json` v9 (+ HiCache/KV/Spec/Crash section) |
| 方法论文档 | `shared/docs/token_pricing_methodology.md` |
