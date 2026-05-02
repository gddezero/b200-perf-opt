# DeepSeek V4-Pro 多轮长上下文压测（B200 × 8）

> **测试时间**：2026-04-28 ~ 2026-04-29
> **硬件**：GCP A4 B200 SXM × 8 (180 GB HBM3e/卡)
> **软件**：SGLang `lmsysorg/sglang:deepseek-v4-blackwell`（低延迟 TP=8 单一配方）
> **压测工具**：`evalscope perf --dataset custom_multi_turn`，多轮 5-turn ShareGPT/PG 真材料
> **数据集**：`/lssd/datasets/multi_turn_{60k,100k,real_pg}_sg.jsonl`（100 conv × 5 turn，turn-1 长度对齐 60/100/200K）
> **跑法**：`batch_multi_turn.sh v3`，每档 number = parallel × 50（200K 同），档间 POST `/flush_cache` 清 RadixTree

---

## 🎯 关键性能（26 档实测）

### 三 base × parallel 全景（按 base 分组的 sweet spot）

| base | parallel | RPS | TTFT P95 | TPOT P95 | Lat P95 | real_cache_hit | evict/p | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 60K | 1 | 0.248 | 4.18 s | 8 ms | 7.25 s | 74.69% | 1.86 | baseline |
| 60K | **4** | 0.439 | 4.95 s | 35 ms | 18.06 s | 74.51% | 2.06 | **🏆 经济拐点** |
| 60K | 5 | 0.450 | 4.97 s | 38 ms | 19.82 s | 74.24% | 2.06 | TPOT<50ms 上限 |
| 60K | 8 | 0.502 | 5.62 s | 66 ms | 32.09 s | 74.68% | 2.05 | TPOT<100ms 上限 |
| 60K | 10 | 0.505 | 6.67 s | 84 ms | 41.68 s | 74.29% | 2.08 | 余量 |
| 60K | 12 | 0.524 | 8.92 s | 94 ms | 44.43 s | 72.75% | **3.09** | ⚠️ 拥塞前期 |
| 60K | **16** | **0.167** | 39.78 s | **327 ms** | **150.31 s** | **1.23%** | **14.09** | **❌ 完全崩盘** |
| 100K | 1 | 0.195 | 7.25 s | 8 ms | 10.72 s | 76.73% | 1.87 | baseline |
| 100K | **4** | 0.321 | 7.94 s | 51 ms | 33.10 s | 76.30% | 1.91 | **🏆 经济拐点** |
| 100K | 7 | 0.355 | 8.77 s | 86 ms | 42.19 s | 74.83% | 2.04 | TPOT<100ms 上限 |
| 100K | 8 | 0.355 | 9.58 s | 103 ms | 48.08 s | 75.06% | **2.80** | ⚠️ 拥塞前期 |
| 100K | **10** | **0.111** | 37.00 s | **298 ms** | **145.85 s** | **4.41%** | **13.37** | **❌ 完全崩盘** |
| 100K | 12 | 0.105 | 63.50 s | 322 ms | 172.70 s | **0.00%** | 12.83 | ❌ |
| 200K | 1 | 0.156 | 16.08 s | 9 ms | 19.19 s | 78.20% | 1.73 | baseline |
| 200K | 2 | 0.184 | 16.77 s | 54 ms | 32.81 s | 78.21% | 1.72 | TPOT<60ms |
| 200K | 3 | 0.194 | 16.75 s | 79 ms | 37.53 s | 77.61% | 1.80 | TPOT<100ms 上限 |
| 200K | **4** | 0.195 | 17.42 s | 110 ms | 53.83 s | 76.31% | 1.90 | **🏆 经济拐点（KV 自限 nr=2）** |

**核心结论**：
1. **崩盘点 KV pool 占用 ≥ 90%**（不是 100%）：60K p=16 / 100K p=10 同时触发 evict_per_prompt 跳到 13-14 + cache hit 从 74-78% 塌方到 0-4% + RPS 倒退 -68% ~ -69%
2. **崩盘是跳变不是渐变**：evict_per_prompt 从 2-3 平台直接跳 13，没有 5-12 中间区
3. **最早崩盘信号是 real_cache_hit < 50%**（不是 Failed > 0），server 通过 num_running 限流硬撑导致 Failed=0 但每个请求实际全 cache miss
4. **200K base 因 KV 调度自限 num_running=2 仍稳**：p=4 名义占 75% 但实际占 37%；p>4 必须先压测验证
5. **生产 sweet spot**：60K p=3-5 / 100K p=3-4 / 200K p=2-3（严格 SLA TPOT<50ms），宽松 SLA<100ms 可上 60K p=8 / 100K p=7 / 200K p=3

---

## 1. 硬件与模型

### 1.1 硬件

同 [09_deepseek_v4_b200.md §1.1](09_deepseek_v4_b200.md)（B200 × 8 SXM, 180 GB HBM3e/卡, NVLink 5.0 全互联, CUDA 13.0, Driver 580）。

### 1.2 模型与 KV pool

| 项 | 值 |
|---|---|
| 模型 | `DeepSeek-V4-Pro` (`/lssd/models/DeepSeek-V4-Pro`) |
| 架构 | DeepseekV4ForCausalLM, 61 层, 5 池 (MLA + CSA stride 4 + HCA stride 128 + 2 compressor states) |
| 权重/卡 (TP=8) | 105.57 GB |
| **KV pool 容量** (TP=8 dp=1, mem-fraction-static=0.82) | **1,069,568 tokens** (`bytes_per_full_token=38,367 B` 含 lightning indexer + compressor state) |
| 多轮 KV 占用估算 | `parallel × avg_input_token`（不是 turn-5 peak） |

⚠️ `bytes_per_full_token=38KB` 不等于真实 KV/token，含固定开销。跨引擎对比无意义，只用于本配方内的容量预算。

### 1.3 V4-Pro 每 token KV 占用（两个口径必须区分）

V4-Pro KV 数据有**两个不同口径**，混用必出错（本项目曾两次幻觉，详见 [project_v4pro_kv_capacity.md](../.claude/projects/-home-admin-maxwellx-altostrat-com-code-b200-vllm-opt/memory/project_v4pro_kv_capacity.md)）。

#### 口径 A：理论稀疏摊销（论文 / HF blog 用）

每 token 真正新增到 KV 池的字节，长上下文渐近值。

V4-Pro 公式（架构：30 层 CSA stride=4 + 31 层 HCA stride=128 + SWA window 摊到 0）：

```
30 × 640B / 4 (CSA) + 31 × 640B / 128 (HCA)
≈ 4,800 + 155 = ~4,950 B (BF16) → ~9.6 KB（含其他细项）
÷ 2 (FP8) = ~4.85 KB/token
```

| dtype | KV/token | 适用 |
|---|---:|---|
| BF16 | ~9.6 KB | V4 论文/blog 公开数据 |
| **FP8** (本压测 `--kv-cache-dtype fp8_e4m3`) | **~4.85 KB** | V4-Pro 实际部署 |

对比 V3.2-Exp dense MLA（BF16 = 68.6 KB/token）：**V4 ≈ V3.2 的 7%**（HF blog "1/10 of V3.2" 说法成立，前提：1M 长上下文渐近 + 同 dtype 比 + 摊销稀疏池）。

#### 口径 B：引擎物理预算（启动日志 `bytes_per_full_token`）

引擎为每个 logical token 在池里**预留的总空间**，含真 KV + lightning indexer key + compressor inflight slot 等固定开销。

```
SGLang V4-Pro TP=8 dp=1 启动日志:
  bytes_per_full_token = 38,367 B ≈ 37.5 KB/token
  available_bytes      = 38.22 GB / TP shard
  full_token           = 1,069,568 tokens（KV pool 容量）
```

**38 KB / 卡 / token 拆解**（5 池配额）：

| 池 | 含义 | 摊销值 / 卡 |
|---|---|---:|
| `full` | lightning indexer query-side key（每 token 1 entry，全 head 不切）| ~640 B |
| `swa` | SWA window (token × 0.0997)，head 切 8 | ~640 B |
| `c4` | CSA latent (token / 4)，head 切 8 | ~600 B |
| `c128` | HCA latent (token / 128)，head 切 8 | ~20 B |
| `c4_state` + `c128_state` | compressor **inflight buffer**（O(running_reqs)，固定 GB 预留）| ~36 KB 摊销 |

→ 38 KB / 卡 / token 中，~1.3 KB 是真 KV 切片，~36 KB 是固定开销 + inflight 摊销 + indexer 副本。

#### ⚠️ 不要混用

| 场景 | 用哪个 |
|---|---|
| 算"V4 比 V3.2 KV 省多少" | **口径 A**（~7-10%）|
| 算"能装多少 token / 多少并发"、判断是否爆池 | **口径 B**（full_token = 1,069,568 / TP shard）|
| vLLM 76 KB ÷ SGLang 38 KB | ❌ 没意义（不同引擎不同口径）|
| 把 38 KB 当"V4 真实 KV/token" | ❌ 漏掉 inflight 不该算到单 token |

#### 验证算式（60K p=16 崩盘）

按 per shard 口径 B 算 KV pool 占用：

```
16 conv × 60K logical token = 960K logical tokens
单张卡占用 = 960K × 38,367 B = 36.83 GB
单张卡 pool = 38.22 GB
占用率 = 36.83 / 38.22 = 96.4%  ← 实测 num_used_tokens 95.7% 吻合
```

→ KV pool 占用 ≥ 90% 触发 RadixTree evict 跟不上 → 全塌方。这就是 §5.1 物理崩盘点公式的来源。

### 1.4 TP=8 dp=1 的 8 卡 KV 分布（最容易出错的地方）

**关键事实**：8 张卡每张都有完整的 38.22 GB KV pool（共 305.76 GB 物理显存），但**全集群只能装 1.07M 个 logical token，不是 ×8**。1 个 logical token 必须跨 8 卡同时持有不同 head 的 KV 切片。

#### 物理分布

| 项 | 每张卡 | 8 卡合计 |
|---|---|---|
| `available_bytes`（KV pool 实占）| **38.22 GB** | 305.76 GB（物理显存）|
| `full_token`（可装 logical tokens）| **1,069,568** | **1,069,568**（不是 ×8！）|
| `bytes_per_full_token` | 38,367 B / 卡 / logical token | — |

→ 1 个 logical token 在 8 卡上同时存在（每卡存自己负责的 head 的 KV 切片），所以"装得下多少 token"按 1 张卡的 full_token 算就是全集群容量。

#### 1 个 logical token 在 8 卡里实际存了什么

按 SGLang 启动日志的 5 池拆解（每张卡都有这 5 池）：

- **真 KV 数据（SWA + CSA + HCA latent）**：head 维度切 8，每张卡只持有自己的 1/8。8 卡相加 ≈ 5 KB / token（口径 A 真实 KV）
- **lightning indexer key**：query-side 全 head 复用，**8 卡每卡都存一份完整副本**
- **compressor inflight buffer (`c4_state`, `c128_state`)**：跟在飞请求数走，不跟 token 数走，每张卡都预留几 GB 的 inflight 槽位

#### TP=8 dp=1 vs DPA dp=8 对比

| 项 | TP=8 dp=1（本压测）| DPA dp=8 ep=8 |
|---|---|---|
| 每张卡 KV pool | 38.22 GB | 16.35 GB |
| 每张卡 full_token | 1,069,568 | 457,472 |
| **全集群 logical tokens** | **1,069,568**（单池）| **3,659,776**（8 × 457k 独立池）|
| 1 个 logical token 跨卡分布 | 8 卡同时持有 head 切片 | 1 个 engine 独占 |
| RadixTree 池数 | **1**（全卡共享同一棵 tree）| **8**（slice 间互不复用）|
| 多轮 prefix 复用 | ✅ 100%（同一 tree）| ❌ 跨 slice 复用率 ~50% |

→ TP=8 dp=1 物理空间利用率低（每卡 38GB vs DPA 16GB），但**单 RadixTree 全卡共享**，多轮场景 cache hit 真实 74-78%；DPA 物理空间利用率高（总容量 3.66M），但 8 个独立 RadixTree 严重碎化。

→ 这就是 §2.2 表里"低延迟 TP=8 是多轮场景首选"的物理依据。

---

## 2. 部署：低延迟 TP=8（多轮压测唯一配方）

### 2.1 启动命令

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
    --kv-cache-dtype fp8_e4m3 \
    --disable-flashinfer-autotune \
    --mem-fraction-static 0.82 \
    --enable-metrics \
    --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
    --host 0.0.0.0 --port 8088
```

### 2.2 为什么用低延迟而非 DPA + DeepEP

| 指标 | 低延迟 TP=8 | DPA + DeepEP dp=8 |
|---|---|---|
| KV pool | 1.07M tokens（单池）| 8 × 457K = 3.66M tokens（8 slice 分散）|
| 多轮 cross-slice prefix 复用 | ✅ 单池 RadixTree 100% 同源 | ❌ slice 间不共享 RadixTree（cache hit ~50%）|
| 单请求延迟 | TTFT 4-7s, TPOT 8ms | TTFT 16s+, TPOT 12ms+ |
| 多轮场景适用度 | ✅ 首选 | ⚠️ 需 SMG router cache_aware 才能挽回 |

详见 [reference_v4pro_lowlat_60k_multi_turn_sla.md](../.claude/projects/-home-admin-maxwellx-altostrat-com-code-b200-vllm-opt/memory/reference_v4pro_lowlat_60k_multi_turn_sla.md)。

---

## 3. 多轮压测方法

### 3.1 数据集生成（custom_multi_turn 真材料）

ShareGPT_en + PG 19 (Project Gutenberg) 拼接生成 turn-1 对齐到 60K/100K/200K 的 5-turn 多轮：

```bash
ssh maxwellx_google_com@forrest-b200-1
python3 ~/code/b200_vllm_opt/gen_multi_turn_dataset.py \
  --turn-lens 60000,5000,5000,5000,5000 \
  --num-conv 100 \
  --out /lssd/datasets/multi_turn_60k_pg_sg.jsonl
```

| 数据集 | turn-1 长度 | 用途 |
|---|---:|---|
| `multi_turn_60k_pg_sg.jsonl` | 60,000 | 60K base 压测 |
| `multi_turn_100k_pg_sg.jsonl` | 100,000 | 100K base 压测 |
| `multi_turn_real_pg_sg.jsonl` | 200,000 | 200K base 压测 |

### 3.2 ⚠️ evalscope multi-turn 累计机制（为什么必须用 custom_multi_turn）

evalscope `random_multi_turn` 的累计行为是**最大的反直觉点**，必须先理解才能设计有效的多轮压测。

**源码事实**（`evalscope/perf/multi_turn_benchmark.py:130-215`）：

```python
context: List[Dict] = []
for user_turn_idx, user_msg in enumerate(user_msgs):
    context.append(user_msg.copy())                     # 加新 user
    request = api_plugin.build_request(list(context))   # 全量 context 当请求
    benchmark_data = await client.post(request)
    context.append({'role': 'assistant',
                    'content': benchmark_data.generated_text})  # 加 assistant 真实回复
```

**`random_multi_turn` 数据集**（`plugin/datasets/multi_turn.py:106-129`）每轮新 user msg 长度独立从 `[min_prompt_length, max_prompt_length]` 均匀采样。

**累计公式**：`turn_N_input_tokens ≈ N × U`（U 是 `[min, max]` 内的采样值，max_tokens 输出 ≪ U 可忽略）

**举例 `--min 100000 --max 200000 --max-turns 5`**：

| turn | 单条 user msg | 累计请求 input |
|---:|---:|---:|
| 1 | 100-200K | 100-200K |
| 2 | 100-200K | 200-400K |
| 3 | 100-200K | 300-600K |
| 4 | 100-200K | 400-800K |
| 5 | 100-200K | **500K-1M（最坏 1M）** |

⚠️ 参数说 "max 200K" 实际请求最大 input 是 **5×200K=1M**。命名严重反直觉。

**evalscope 没有任何控制累计的参数**：

| 参数 | 真实语义 |
|---|---|
| `--min-prompt-length` / `--max-prompt-length` | **单条 user message** 长度（不是累计、不是请求总量）|
| `--min-turns` / `--max-turns` | conv 的 user turn 数量 |
| `--max-tokens` | 单 turn assistant 输出上限 |

**无 `--max-history-tokens` / `--max-context` / `--rolling-window` / `--keep-only-last-k-turns`**。runner 硬编码 append 全量上下文。

**实测验证**（5K-10K 单 user msg × 5 turn × 5 conv，2026-04-28）：

| conv | turn-1 | turn-2 | turn-3 | turn-4 | turn-5 | 比例 |
|:---:|---:|---:|---:|---:|---:|:---:|
| 0 | 8,286 | 16,338 | 22,118 | 28,804 | 34,759 | ×4.2 |
| 1 | 6,214 | 13,028 | 21,988 | 28,865 | 38,125 | ×6.1 |
| 2 | 5,423 | 11,894 | 17,304 | 25,163 | 34,301 | ×6.3 |

严格单调递增，turn-5 ≈ 5 × turn-1，完全匹配公式。

**custom_multi_turn vs random_multi_turn 对比**：

| 项 | random_multi_turn | custom_multi_turn（本压测用）|
|---|---|---|
| 数据来源 | 运行时随机生成 | 预先生成的 JSONL（`messages` 数组）|
| turn-1 长度 | `[min, max]` 均匀采样 → 方差大 | **预先固定**（60K/100K/200K 整齐对齐）|
| 后续 turn 长度 | 每轮独立 `[min, max]` 采样 | **预先固定**（每轮 +5K incr）|
| 累计上限 | 5 turn × max = N×max（不可控）| 5 turn 累计 ≈ turn-1 + 4×5K（**完全可控**）|
| cache hit 实验性 | 跨 conv 复用率低 | 跨 conv 同源 prefix → 复用率真实 74-78% |
| 复现性 | 每次随机不同 | seed 固定 100% 复现 |

**本压测累计长度**（custom_multi_turn 真材料）：

| base | turn-1 | turn-5 累计 | 用 random 等效 max |
|:---:|---:|---:|---:|
| 60K | 60K | ~80K | 16K (5×16=80) |
| 100K | 100K | ~120K | 24K |
| 200K | 200K | ~220K | 44K |

⚠️ **不要用 `random_multi_turn --max-prompt-length 200000`** 测 200K 多轮——5 turn 累计到 1M 会直接打爆 KV 池（1.07M 总容量），与本压测的"200K base"语义完全不同。

**生产环境真实多轮**：业务侧必须做滚动 truncation（保留最近 K 条 message），SGLang/router 都救不了 evalscope 这个 append-only 行为。详见 [reference_evalscope_multiturn_accumulation.md](../.claude/projects/-home-admin-maxwellx-altostrat-com-code-b200-vllm-opt/memory/reference_evalscope_multiturn_accumulation.md)。

### 3.3 batch_multi_turn.sh v3（含 evict 监控）

每档自动：
1. POST `/flush_cache` 清 RadixTree（保证档间公平）
2. 抓 `sglang:prompt_tokens_total / cached_tokens_total / evicted_tokens_total{cache_type="SWARadixCache"}` 三 counter 起始值
3. 跑 `evalscope perf --dataset custom_multi_turn --multi-turn --max-turns 5 --max-tokens 500`
4. 抓三 counter 终值 → 算 `dprompt`, `REAL_CACHE_HIT = dcached/dprompt`, `EVICT_PER_PROMPT = devict/dprompt`
5. 写 `/lssd/logs/batch_lowlat_<TAG>_<TS>.log`

```bash
ID_START=215 PLIST="1 2 3 4" \
DATASET=/lssd/datasets/multi_turn_60k_pg_sg.jsonl \
MAX_PROMPT=80000 \
TAG=lowlat_60k \
NUM_PER_P=50 \
./batch_multi_turn.sh
```

### 3.4 results_multiturn.csv（single source of truth）

```bash
# B200 跑完后本机拉数据
rsync -az maxwellx_google_com@forrest-b200-1:~/code/b200_vllm_opt/outputs/ outputs/
python3 build_results_multiturn_csv.py
```

35 列：标识 (id/date/tag/base_K/parallel/number) + summary (wall_s/rps/avg_lat_s/avg_in_tok/...) + 3 类 percentile (ttft/tpot/lat ×3) + spec metrics + cache (eval_cache_hit_pct, **real_cache_hit_pct**) + **evict (evict_tokens_delta, evict_per_prompt)** + failed + 数据点诊断 (p95/p99 sample)。

数据来源融合：v3 batch log（新档自动有 EVICT）+ `evict_backfill.json`（历史档 PromQL `increase(sglang:evicted_tokens_total[Δt] @end_ts)` 反查）。

### 3.5 percentile 数据点统计有效性

evalscope multi-turn 模式 percentile **按 turn 算不按 conv 算**，每档 turn 数 = `parallel × 50 × avg_turns(≈4.5)`：

| parallel | turn 数 | P95 数据点（5%）| P99 数据点（1%）|
|---:|---:|---:|---:|
| 1 | 225 | 11 ✓ | 2 ❌ |
| 4 | 900 | 45 ✓ | 9 ✓ |
| 8 | 1800 | 90 ✓ | 18 ✓ |
| 16 | 3600 | 180 ✓ | 36 ✓ |

**报告全部用 P95**（≥ 5 数据点统计有效线一律满足，P99 部分小档只有 2-9 个数据点，不可靠）。

### 3.6 cache hit 两个口径

| 口径 | 来源 | 60K typical | 100K typical | 200K typical |
|---|---|---:|---:|---:|
| **eval_cache_hit_pct** | evalscope `Average approx KV cache hit rate (%)` | **88%** | **92%** | **95%** |
| **real_cache_hit_pct** | SGLang counter delta `dcached / dprompt × 100` | **74%** | **76%** | **78%** |

evalscope 用 prefix overlap 估算，**乐观偏 14pp**。SGLang counter 是 RadixTree 真实命中率，是物理事实。**报告用 real_cache_hit**。详见 [feedback_sglang_cache_hit_metric_bug.md](../.claude/projects/-home-admin-maxwellx-altostrat-com-code-b200-vllm-opt/memory/feedback_sglang_cache_hit_metric_bug.md)。

---

## 4. 性能数据（26 档实测）

### 4.1 60K base 11 档（#211-218 + #227-229）

| 编号 | parallel | number | wall (s) | RPS | TTFT P95 | TPOT P95 | Lat P95 | eval_hit | **real_hit** | **evict/p** | failed | 备注 |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| #215 | 1 | 50 | 201.7 | 0.248 | 4.18 s | 8 ms | 7.25 s | 88.14% | 74.69% | 1.86 | 0 | baseline |
| #216 | 2 | 100 | 297.6 | 0.336 | 4.44 s | 28 ms | 11.39 s | 88.12% | 74.78% | 2.02 | 0 | |
| #217 | 3 | 150 | 390.0 | 0.385 | 8.06 s | 33 ms | 15.53 s | 88.14% | 74.63% | 2.00 | 0 | TPOT<50ms |
| **#218** | **4** | 200 | 455.4 | **0.439** | 4.90 s | 35 ms | 18.06 s | 88.12% | 74.51% | 2.06 | 0 | 🏆 **经济拐点** |
| #211 | 5 | 250 | 555.2 | 0.450 | 4.97 s | 38 ms | 19.82 s | 88.12% | 74.24% | 2.06 | 0 | TPOT<50ms 上限 |
| #212 | 6 | 300 | 632.1 | 0.475 | 5.31 s | 51 ms | 24.74 s | 88.10% | 74.57% | 2.04 | 0 | |
| #213 | 7 | 350 | 710.7 | 0.493 | 5.44 s | 62 ms | 28.84 s | 88.10% | 74.25% | 2.07 | 0 | |
| #214 | 8 | 400 | 796.7 | 0.502 | 5.62 s | 66 ms | 32.09 s | 88.11% | 74.68% | 2.05 | 0 | TPOT<100ms 上限 |
| #227 | 10 | 500 | 989.9 | 0.505 | 6.67 s | 84 ms | 41.68 s | 88.11% | 74.29% | 2.08 | 0 | 仍健康（KV 56%）|
| #228 | 12 | 600 | 1144.2 | 0.524 | 8.92 s | 94 ms | 44.43 s | 88.10% | 72.75% | **3.09** | 0 | ⚠️ **拥塞前期** |
| **#229** | **16** | 800 | **4787.9** | **0.167** | **39.78 s** | **327 ms** | **150.31 s** | 88.09% | **1.23%** | **14.09** | 0 | ❌ **完全崩盘** |

**60K 关键观察**：
- p=1→8 标准 8 档：RPS +103% / TPOT P95 +730% / TTFT P95 +34% / cache hit 平台稳 / evict 平台稳
- p=10→12：evict 首跳出健康区（2.08 → 3.09），cache hit 首次破平台（74→72.75%）
- p=16：KV pool 占用从 96% 名义 → **实测打满** → RadixTree evict 跟不上写入 → cache hit **从 74% 一步塌方到 1.23%** → 每请求几乎全 cache miss → TPOT P95 从 84ms 跳到 327ms (×3.9) → RPS 从 0.505 倒退到 0.167 (-67%) → wall_s 从 990s → 4788s (×4.8)
- 拐点序列（按敏感度）：cache hit 塌方 → evict 飙升 → TPOT 雪崩 → wall 暴涨 → RPS 倒退（Failed 仍 = 0，server 兜底）

### 4.2 100K base 11 档（#219-226 + #230-232）

| 编号 | parallel | number | wall (s) | RPS | TTFT P95 | TPOT P95 | Lat P95 | eval_hit | **real_hit** | **evict/p** | failed | 备注 |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| #219 | 1 | 50 | 256.5 | 0.195 | 7.25 s | 8 ms | 10.72 s | 92.00% | 76.73% | 1.87 | 0 | baseline |
| #220 | 2 | 100 | 393.6 | 0.254 | 7.95 s | 28 ms | 18.00 s | 91.99% | 75.91% | 1.89 | 0 | |
| #221 | 3 | 150 | 480.3 | 0.312 | 7.90 s | 36 ms | 19.71 s | 91.97% | 76.13% | 1.92 | 0 | TPOT<50ms |
| **#222** | **4** | 200 | 622.6 | **0.321** | 7.94 s | 51 ms | 33.10 s | 91.98% | 76.30% | 1.91 | 0 | 🏆 **经济拐点** |
| #223 | 5 | 250 | 722.0 | 0.346 | 8.00 s | 64 ms | 33.45 s | 91.97% | 75.39% | 2.00 | 0 | |
| #224 | 6 | 300 | 885.7 | 0.339 | 11.65 s | 82 ms | 37.31 s | 91.97% | 75.30% | 2.02 | 0 | RPS 已平台 |
| #225 | 7 | 350 | 985.2 | 0.355 | 8.77 s | 86 ms | 42.19 s | 91.98% | 74.83% | 2.04 | 0 | TPOT<100ms 上限 |
| #226 | 8 | 400 | 1126.1 | 0.355 | 9.58 s | 103 ms | 48.08 s | 91.98% | 75.06% | **2.80** | 0 | ⚠️ **拥塞前期**（首次 evict>2.5）|
| **#230** | **10** | 500 | **4519.4** | **0.111** | **37.00 s** | **298 ms** | **145.85 s** | 91.98% | **4.41%** | **13.37** | 0 | ❌ **完全崩盘** |
| #231 | 12 | 600 | 5721.1 | 0.105 | 63.50 s | 322 ms | 172.70 s | 91.98% | **0.00%** | 12.83 | 0 | ❌ 0% hit |
| #232 | 16 | 64 | 567.3 | 0.113 | 96.24 s | 240 ms | 185.92 s | 91.72% | **0.00%** | 13.10 | 0 | ❌ quick 验证（NUM_PER_P=4）|

**100K 关键观察**：
- p=1→8 标准 8 档：RPS +82% / TPOT P95 +1188% / cache hit 75% 平台微降 / evict 1.87→2.80（p=8 首次接近 3.0）
- 100K p=8 已到拥塞前期边缘（evict=2.80 vs 60K p=12 的 3.09 同量级，但 100K 还能跑 RPS 0.355 = 60K 0.524 的 68%）
- p=10：KV pool 93% 名义 → 与 60K p=16 同款全塌方，cache hit 76%→4.41%，TPOT 103→298ms (×2.9)
- p=12 / 16：完全 0% cache hit（每个请求 100% 重算），server 完全靠 num_running 限流兜底
- p=16 用 NUM_PER_P=4 (number=64) 快速验证，9 分钟跑完，证实 100K p=10/12/16 全在同稳态崩盘

### 4.3 200K base 4 档（#233-236）

| 编号 | parallel | number | wall (s) | RPS | TTFT P95 | TPOT P95 | Lat P95 | eval_hit | **real_hit** | **evict/p** | failed | 备注 |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| #233 | 1 | 50 | 321.3 | 0.156 | 16.08 s | 9 ms | 19.19 s | 95.56% | 78.20% | 1.73 | 0 | baseline |
| #234 | 2 | 100 | 544.9 | 0.184 | 16.77 s | 54 ms | 32.81 s | 95.56% | 78.21% | 1.72 | 0 | TPOT<60ms |
| #235 | 3 | 150 | 773.2 | 0.194 | 16.75 s | 79 ms | 37.53 s | 95.56% | 77.61% | 1.80 | 0 | TPOT<100ms 上限 |
| **#236** | **4** | 200 | 1026.5 | **0.195** | 17.42 s | 110 ms | 53.83 s | 95.56% | 76.31% | 1.90 | 0 | 🏆 **经济拐点（KV 自限 nr=2）** |

**200K 关键观察**：
- TTFT P95 几乎平台（16-17s, +8%）— 200K prefill 主导，与 parallel 弱相关
- TPOT P95 线性恶化（9→110ms, ×12）— 与 60K/100K 同模式
- cache hit **78% 平台**（高于 60K 的 74% 和 100K 的 76%，多轮长 prefix 复用率最高）
- **关键现象**：p=4 名义 KV 占 75%（4×200K÷1070K），**实测 num_running=2**（被 SGLang 调度自限），实际占用 ≈ 37%
- evict_per_prompt 1.73→1.90 全平台，无任何崩盘征兆
- p=5 推算 KV 占 94% 已过 90% 崩盘线，**未实测，必须先压测验证**（参考 60K/100K 实测路径）

### 4.4 三 base 横向对比

| 指标 | 60K | 100K | 200K |
|---|---|---|---|
| baseline TTFT P95 (p=1) | 4.18 s | 7.25 s | 16.08 s |
| baseline TPOT P95 (p=1) | 8 ms | 8 ms | 9 ms |
| baseline real_hit (p=1) | 74.69% | 76.73% | 78.20% |
| 经济拐点 parallel | **4** | **4** | **4** |
| 经济拐点 RPS | 0.439 | 0.321 | 0.195 |
| TPOT<50ms 最大 parallel | 5 | 3 | 2 |
| TPOT<100ms 最大 parallel | 8 | 7 | 3 |
| 拥塞前期 (evict ≥ 2.8) | p=12 | p=8 | 未到 |
| 完全崩盘 (cache hit < 50%) | **p=16** | **p=10** | 未触发 |
| 物理上限名义 (0.85×池/in) | 15 | 9 | 4-5 |

**横向规律**：
1. baseline TTFT 与 base 长度成正比（4-7-16s ≈ 1:1.7:3.8 与 60K:100K:200K = 1:1.67:3.33 吻合，prefill 主导）
2. baseline TPOT 与 base 长度无关（8-8-9ms 平台，decode 一次只处理 1 token）
3. real_cache_hit 随 base 增长（74→76→78%），多轮长 prefix 复用率更高
4. **经济拐点统一在 p=4**（不管 base 多大）— 经济拐点由调度开销决定不由 KV 决定
5. SLA 上限随 base 减小（60K p=8 / 100K p=7 / 200K p=3 满足 TPOT<100ms）— SLA 上限由 KV 占用决定
6. 崩盘点 KV pool 名义占用 ≥ 90%（60K p=16 / 100K p=10）

---

## 5. 物理崩盘点定位（核心结论）

### 5.1 崩盘条件 = KV pool 占用 ≥ 90%（不是 100%）

```
60K base, KV pool 1.07M:
- p=10: 占 56% → 健康
- p=12: 占 67% → 拥塞前期 (evict 3.09)
- p=14: 占 79% → 推算仍健康（未实测）
- p=15: 占 84% → 推算边缘
- p=16: 占 90% → ❌ 全塌方实锤

100K base:
- p=8:  占 75% → 拥塞前期 (evict 2.80)
- p=9:  占 84% → 推算边缘（未实测）
- p=10: 占 93% → ❌ 全塌方实锤

200K base:
- p=4:  名义占 75%，实测因调度自限 num_running=2 → 实际 37% → 健康
- p=5:  名义占 94% → 推算崩盘但调度可能延后（未实测）
```

**RadixTree evict 速率追不上 prompt 写入速率**是崩盘机制：当 KV pool 占用 > 90%，新写入的 prefix 立即被踢出，下一个请求复用率归零。

### 5.2 崩盘信号触发顺序（按敏感度排序）

| 优先级 | 信号 | 健康值 | 崩盘值 | 实测样本 |
|:---:|---|---|---|---|
| 🔴 1 | `real_cache_hit < 50%` | 74-78% | **0-4%** | 60K p=16: 1.23% / 100K p=10: 4.41% |
| 🟠 2 | `evict_per_prompt > 5` | 1.7-2.8 | **13-14** | 60K p=16: 14.09 / 100K p=10: 13.37 |
| 🟡 3 | `TPOT P95 > 200ms` | 8-100 ms | **240-330 ms** | 60K p=16: 327 / 100K p=10: 298 |
| 🟢 4 | `RPS 比上一档 < -50%` | 对数饱和 | **倒退 -68%** | 60K p=12→16: 0.524→0.167 |
| ⚪ 5 | `Failed > 0` | 0 | **仍 0** | 所有 26 档 Failed=0 |

⚠️ **不要等 Failed > 0 才报警** — Failed 是最后兜底信号，崩盘已发生。**优先看 real_cache_hit**（从 74% 一步塌方到 < 5%，最早最敏感）。

### 5.3 崩盘是跳变不是渐变

实测无中间状态：

```
evict_per_prompt 实测分布:
1.7-2.0: 健康平台   ████████████████████ (60K/100K/200K p=1-7)
2.5-3.5: 拥塞前期   ██                   (60K p=12, 100K p=8)
5-12   : (无样本)   (空白)
13-14  : 完全崩盘   ████                 (60K p=16, 100K p=10/12/16)
```

物理意义：RadixTree 处于稳定模式（evict 跟得上写入）→ 一旦 KV pool 占用越过 90%，evict 进入"竭力清空"模式（每请求踢出 13-14 倍 prefix）。没有"温和退化"过渡。

### 5.4 200K 调度自限现象

200K p=4 名义 KV 占 75%，但 `sglang:num_running_reqs=2`（不是 4）：SGLang 调度器看到 KV pool 占用接近阈值时主动限流，不让新请求进 running 队列（进 num_queue）。所以：

- 200K p=4 看起来"健康" = 实际只跑 2 路并发
- 不能套用 60K/100K 的"占用 ≥ 90% 必崩盘"外推到 200K
- 200K p>4 调度可能继续自限到 num_running=2-3，但风险是 KV 占用一旦真升到 95% 后立即触发同款塌方

---

## 6. parallel 上限规则与生产推荐

### 6.1 三类上限取 min（实测确认）

```
1. 物理上限（避免崩盘）:
   p_phys ≈ 0.85 × KV_pool / avg_input_token  (留 5% 安全余量, 不是 0.9)
   60K:  0.85 × 1070 / 60  ≈ 15  → 实测 p=12 拥塞前期 / p=16 崩盘
   100K: 0.85 × 1070 / 100 ≈ 9   → 实测 p=8 拥塞前期 / p=10 崩盘
   200K: 0.85 × 1070 / 200 ≈ 4-5 → 实测 p=4 健康（自限保护）

2. SLA 上限（按业务 TPOT/TTFT 反推）:
   TPOT P95 < 50ms  → 60K p≤5, 100K p≤3, 200K p≤2
   TPOT P95 < 100ms → 60K p≤8, 100K p≤7, 200K p≤3

3. 经济上限（边际 RPS ≥ 5%）:
   60K:  p=4→5 +2.5%, p=5→6 +5.6%, p=6→7 +3.8% → 经济拐点 p=4
   100K: p=4→5 +7.8%, p=5→6 -2.0%（已平台）   → 经济拐点 p=4
   200K: p=3→4 +0.5%（已无收益）              → 经济拐点 p=3-4
```

### 6.2 生产推荐表（按 SLA 严格度）

| base | 严格 SLA (TPOT<50ms) | 平衡 SLA (TPOT<100ms) | 最大吞吐（仍稳）| ⛔ 禁用 |
|---|:---:|:---:|:---:|:---:|
| 60K | **p=3-5** | p=8-10 | p=12（拥塞警报）| **p≥16** |
| 100K | **p=3-4** | p=7-8 | p=8（拥塞警报）| **p≥10** |
| 200K | **p=2-3** | p=3 | p=4（KV 自限）| **p>4 未验证** |

### 6.3 客户问"还能加吗"决策树

1. real_cache_hit > 60% 且 evict/p < 2.5 → **可加**
2. real_cache_hit 60-50% 或 evict/p 2.5-5 → **拥塞前期，不建议加**
3. real_cache_hit < 50% 或 evict/p > 5 → **已崩盘，立即降档**
4. Failed > 0 → **已严重过载，紧急降档 + 排查**

---

## 7. 踩坑记录

### 7.1 ⚠️ 崩盘后 Failed 仍 = 0，监控必须看 cache hit

**症状**：60K p=16 / 100K p=10/12/16 全部 Failed=0，但 RPS 倒退 -68%、TPOT ×4、cache hit 0%。

**根因**：SGLang 调度器在 KV pool 满时通过 `num_running_reqs` 限流（新请求进 `num_queue_reqs` 等），但 evalscope `--stream` 不超时不报错，所以请求最终都成功（只是排队几十秒），Failed=0。

**对策**：监控**优先 real_cache_hit < 50%**，其次 evict_per_prompt > 5。Failed 是最后兜底信号。

### 7.2 evalscope cache hit 估算偏乐观 14pp

**症状**：evalscope summary `Average approx KV cache hit rate (%)` 60K 给 88%，100K 给 92%，200K 给 95%；但 SGLang counter 实算 74-78%。

**根因**：evalscope 用 prefix overlap **估算**，不看实际 RadixTree 命中。

**对策**：从 SGLang `prometheus` `cached_tokens_total / prompt_tokens_total` counter delta 算真实命中率（`batch_multi_turn.sh v3` 已自动）。详见 [feedback_sglang_cache_hit_metric_bug.md](../.claude/projects/-home-admin-maxwellx-altostrat-com-code-b200-vllm-opt/memory/feedback_sglang_cache_hit_metric_bug.md)。

### 7.3 percentile 必须用 P95 不用 P99

**症状**：小 parallel (1-3) 档 P99 数据点只有 2-9 个，统计不可靠。

**根因**：evalscope multi-turn percentile 按 turn 算，turn 数 = `parallel × 50 × avg_turns(≈4.5)`；P99 取 1% → 小档样本不足。

**对策**：报告全用 P95（≥ 5 数据点的 ≥ 11 全档满足）。CSV 里 `p95_data_points` / `p99_data_points` 列直接写出，方便核验。

### 7.4 max-prompt-length 是 INCLUSION 不是 truncation

**症状**：用 `--max-prompt-length 35000` 跑 60K 数据集 → 100/100 conv 全被过滤掉，evalscope 无任何请求发出。

**根因**：evalscope 把 `max-prompt-length` 当 INCLUSION 阈值（turn-1 长度 > 阈值 → 整 conv 跳过），不做 truncation。

**对策**：`MAX_PROMPT` 必须 ≥ 数据集 turn-1 长度 + 5K 余量。本配方实战值：60K → 80000, 100K → 105000, 200K → 210000。

### 7.5 ShareGPT 短样本误导 cache hit 高估

**症状**：早期用 `share_gpt_en_multi_turn` 跑 60K 档，cache hit 显示 85%。

**根因**：share_gpt_en 中位 input 仅 625 tok / P99 仅 2194 tok，远低于 60K target。turn-1 全是相同短 prompt，cache hit 85% 是 prefix 复用假象，**不是真实长上下文行为**。

**对策**：用 PG (Project Gutenberg) + ShareGPT 拼接到 turn-1 整齐对齐 60K/100K/200K（custom_multi_turn 数据集已生成）。详见 [reference_evalscope_custom_multiturn_real_data.md](../.claude/projects/-home-admin-maxwellx-altostrat-com-code-b200-vllm-opt/memory/reference_evalscope_custom_multiturn_real_data.md)。

### 7.6 档间不 flush_cache → 后档拿前档 cache hit

**症状**：早期 batch 不 flush_cache，p=2 跑完后 p=4 起步 cache hit 直接 90%（不是真实 74%）。

**根因**：RadixTree 跨档持久化，前档积累的 prefix 给后档"白送"hit。

**对策**：`batch_multi_turn.sh` 每档前 POST `/flush_cache`（已自动），保证档间公平对比。

### 7.7 evict_per_prompt 跳变没有中间过渡

**症状**：实测 5-12 区间无样本，1.7-2.0 平台 / 2.5-3.5 拥塞前期 / 13-14 完全崩盘。

**根因**：RadixTree 在 KV pool 占用 < 90% 时 evict 跟得上 → 稳定平台；占用 ≥ 90% 时 evict 进入竭力清空模式 → 跳变。

**对策**：监控阈值用 evict/p 2.5（拥塞前期警报）和 evict/p 5（崩盘警报），无需中间档位。

---

## 8. 端到端时间线

| 任务 | 耗时 | 说明 |
|---|---|---|
| 数据集生成（100 conv × 5 turn）| 3-5 min | PG + ShareGPT 拼接 + tokenize 验证 |
| SGLang 启动（cache 复用）| 4-6 min | 低延迟 TP=8 配方 |
| **60K 8 档标准 #211-218 (p=1-8)** | ~85 min | wall_s 总和 4040s + flush 时间 |
| **60K 高档 #227-229 (p=10/12/16)** | **~117 min** | p=16 单档 80 min（崩盘后 wall ×4.8）|
| **100K 8 档标准 #219-226 (p=1-8)** | **~95 min** | wall_s 总和 5470s |
| **100K 高档 #230-232 (p=10/12/16-quick)** | **~180 min** | p=10 单档 75 min, p=12 单档 95 min |
| **200K 全档 #233-236 (p=1/2/3/4)** | ~45 min | wall_s 总和 2666s |
| **26 档总耗时** | **~9 小时** | 含档间 flush + counter snapshot |

档间 flush_cache + counter snapshot 开销稳定 ~10s/档。

---

## 9. 编号映射

| 编号 | base | parallel | number | TAG | 数据集 |
|:---:|:---:|:---:|:---:|---|---|
| #211 | 60K | 5 | 250 | lowlat_60k | multi_turn_60k_pg_sg.jsonl |
| #212 | 60K | 6 | 300 | lowlat_60k | 同 |
| #213 | 60K | 7 | 350 | lowlat_60k | 同 |
| #214 | 60K | 8 | 400 | lowlat_60k | 同 |
| #215 | 60K | 1 | 50 | lowlat_60k | 同 |
| #216 | 60K | 2 | 100 | lowlat_60k | 同 |
| #217 | 60K | 3 | 150 | lowlat_60k | 同 |
| #218 | 60K | 4 | 200 | lowlat_60k | 同 |
| #219 | 100K | 1 | 50 | lowlat_100k | multi_turn_100k_pg_sg.jsonl |
| #220 | 100K | 2 | 100 | lowlat_100k | 同 |
| #221 | 100K | 3 | 150 | lowlat_100k | 同 |
| #222 | 100K | 4 | 200 | lowlat_100k | 同 |
| #223 | 100K | 5 | 250 | lowlat_100k | 同 |
| #224 | 100K | 6 | 300 | lowlat_100k | 同 |
| #225 | 100K | 7 | 350 | lowlat_100k | 同 |
| #226 | 100K | 8 | 400 | lowlat_100k | 同 |
| **#227** | 60K | 10 | 500 | lowlat_60k_high | multi_turn_60k_pg_sg.jsonl |
| **#228** | 60K | 12 | 600 | lowlat_60k_high | 同 |
| **#229** | 60K | 16 | 800 | lowlat_60k_high | 同 |
| **#230** | 100K | 10 | 500 | lowlat_100k_high | multi_turn_100k_pg_sg.jsonl |
| **#231** | 100K | 12 | 600 | lowlat_100k_high | 同 |
| **#232** | 100K | 16 | **64** | lowlat_100k_p16 | 同（NUM_PER_P=4 quick）|
| **#233** | 200K | 1 | 50 | lowlat_200k | multi_turn_real_pg_sg.jsonl |
| **#234** | 200K | 2 | 100 | lowlat_200k | 同 |
| **#235** | 200K | 3 | 150 | lowlat_200k | 同 |
| **#236** | 200K | 4 | 200 | lowlat_200k | 同 |

---

## 10. 下一步工作

- [ ] 200K p=5 / p=6 实测验证（当前推算崩盘但 KV 调度自限可能延后）
- [ ] 60K p=13 / 14 / 15 细化（当前 p=12→16 跨度太大，找精确崩盘起点）
- [ ] 100K p=9 细化（当前 p=8→10 跨度，找精确崩盘起点）
- [ ] 平衡 / 高吞吐 配方下的多轮压测（DPA + DeepEP + SMG router cache_aware）
- [ ] 不同 max-tokens 对崩盘点的影响（当前都用 500）
- [ ] vLLM baseline 多轮压测对比（当前只有低延迟 SGLang）

---

## 附录 A：参考资料

- 单轮压测对照：[09_deepseek_v4_b200.md](09_deepseek_v4_b200.md)
- 完整压测流程文档：[../workflow_multi_turn.md](../workflow_multi_turn.md)
- 60K 8 档完整 SLA 数据 memory：`reference_v4pro_lowlat_60k_multi_turn_sla.md`
- 26 档物理崩盘点定位 memory：`reference_evict_parallel_model.md`
- KV 池容量两口径辨析 memory：`project_v4pro_kv_capacity.md`
- evalscope cache hit 偏差 memory：`feedback_sglang_cache_hit_metric_bug.md`
- custom_multi_turn 数据集生成 memory：`reference_evalscope_custom_multiturn_real_data.md`
- 标准化 CSV：[../benchmark_result/results_multiturn.csv](../benchmark_result/results_multiturn.csv)
- batch 脚本：[../batch_multi_turn.sh](../batch_multi_turn.sh)
- CSV 重建脚本：[../build_results_multiturn_csv.py](../build_results_multiturn_csv.py)
- PromQL 回填脚本：[../backfill_evict.py](../backfill_evict.py)
