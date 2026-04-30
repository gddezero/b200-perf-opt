# SGLang DP 模式 KV cache 调研与跨 slice 复用方案

**调研日期**：2026-04-27 起，最新更新 2026-04-28
**目标硬件**：单机 8×B200
**目标模型**：DeepSeek-V4-Pro（MoE）
**对照镜像**：`lmsysorg/sglang:deepseek-v4-blackwell`（server）+ `lmsysorg/sgl-model-gateway:latest`（router/SMG）
**源码路径**：`/sgl-workspace/sglang/python/sglang/srt/`

---

## TLDR

### 最佳生产配方 ⭐⭐（vE-v3，2026-04-28 实测）

| 项 | 值 |
|---|---|
| **真实 cache hit (counter delta)** | **78.01%** |
| Time taken (500 req multi-turn) | 33.9 min |
| RPS | 0.246 |
| Total throughput | 54,267 tok/s |
| Avg TTFT | 9.73s |
| **P99 TTFT** | **67.97s** |
| Avg TPOT | 0.155s (155 ms) |
| P99 TPOT | 0.392s |
| 成功率 | 500/500 (100%) |

**Server 启动**（V4-Pro，8×B200，1 个 sglang server 暴露 8 个 DP）：

```bash
python3 -m sglang.launch_server \
  --trust-remote-code --model-path /lssd/models/DeepSeek-V4-Pro \
  --served-model-name DeepSeek-V4-Pro \
  --tp 8 --dp 8 --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
  --max-running-requests 256 --cuda-graph-max-bs 64 --mem-fraction-static 0.82 \
  --enable-metrics --enable-metrics-for-all-schedulers \
  --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 --port 8088
```

**Router 启动**（SMG image，1.5s 启动，无 GPU 占用）：

```bash
sudo docker run -d --name sglang-router --restart no --network host \
  -v /lssd/models:/lssd/models:ro \
  lmsysorg/sgl-model-gateway:latest \
  --worker-urls http://127.0.0.1:8088 \
  --model-path /lssd/models/DeepSeek-V4-Pro \
  --tokenizer-path /lssd/models/DeepSeek-V4-Pro \
  --reasoning-parser deepseek-v4 --tool-call-parser deepseek \
  --dp-aware --policy cache_aware \
  --cache-threshold 0.5 \
  --balance-abs-threshold 8 \
  --balance-rel-threshold 1.2 \
  --prometheus-port 29000 \
  --host 0.0.0.0 --port 8000
```

**客户端**：直接打 `:8000`，零改动。Client `parallel=16` 与 `balance_abs=8` 配套（`abs ≤ parallel × 0.5` 才能触发 fallback；`parallel` 是 `dp_size=8` 的整数倍）。

### 完整测试对比矩阵（按时间倒序）

#### 真实数据 200K + 10K incr × 5 turn × 100 conv（2026-04-28，custom_multi_turn JSONL）

| 测试 | 配方 | parallel | abs | rel | thresh | 数据集 | 完成 | Time | RPS | Total tok/s | Avg lat | P99 lat | Avg TTFT | **P99 TTFT** | Avg TPOT | P99 TPOT | **真实 cache hit** | 备注 |
|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ⭐⭐ **vE-v3** | router cache_aware (SMG) | **16** | **8** | 1.2 | 0.5 | PG+ShareGPT | 500/500 | **33.9 min** | **0.246** | **54,267** | 64.83s | 151.19s | 9.73s | **67.97s** | 0.155s | 0.392s | **78.01%** | 生产首选 |
| vE-v5 | client `data_parallel_rank` wrapper（直连 :8088）| 16 | — | — | — | PG+ShareGPT | 500/500 | 79.1 min | 0.105 | 23,210 | 150.32s | 421.17s | 59.16s | 266.62s | 0.269s | 1.044s | 50.66% | hash 落 6 bin（DP3/5 各 24 conv），慢 2.34× |
| vE-v2 | router abs=4 + parallel=8 | 8 | 4 | 1.2 | 0.5 | PG+ShareGPT | 手动停 | — | — | — | — | — | — | — | — | — | — | parallel<dp_size，DP 无法填满 |
| vE-v1 | router abs=16 + parallel=8 | 8 | 16 | 1.2 | 0.5 | PG+ShareGPT | 手动停 | — | — | — | — | — | — | — | — | — | — | abs > parallel 凝固，3/8 DP 工作 |

#### random_multi_turn 30K-200K × 5 turn × 100 conv（2026-04-28，random 数据）

| 测试 | 配方 | parallel | seed | 完成 | Time | RPS | Avg lat | P99 TTFT | TPOT P99 | **真实 cache hit** | Avg input | 备注 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| vC | router v3 + 单轮 35K | 8 | 42 | 100/0 | **228.8s** | **0.437** | 18.1s | **8.93s** | 0.56s | **64.19%** | 60K | ⭐⭐⭐ TTFT P99 −97.5%，单轮长度限制是真根因 |
| vA | router v3 + 单轮 200K | 8 | 42 | 99/1 | 1096s | 0.090 | 82.5s | 352s | **0.53s** | 59.09% | 317K | DPA 集群慢化 + 1 abort（accumulate 700K）|
| vB | router v3 + 单轮 200K | 10 | 42 | 100/0 | 1329s | 0.075 | 122.4s | 348s | 2.10s | 56.14% | 347K | 10:8 不整除，慢 vA 18% + TPOT P99 4× |

#### 100 conv × 5 turn × 30K-200K, parallel=10（2026-04-27，#201 配方调参，random_multi_turn）

| 测试 | image | thresh | 其他 | 完成 | Time | RPS | Avg lat | P99 TTFT | **真实 cache hit** | inter_token P99 | 备注 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| ⭐ **v3 (baseline)** | **SMG** | 0.5 | default | 100/0 | **995s** | **0.10** | **98.2s** | **286s** | **61.63%** | 1616ms | SMG image 关键 |
| **v4-B** | SMG | 0.5 | tree=256MB | 100/0 | 1187s | 0.08 | 111.0s | 366s | 58.90% | **996ms** | inter_token P99 −38% (流式首选) |
| v6 | SMG | **0.6** | default | 100/0 | 1008s | 0.10 | 99.8s | 350s | 56.37% ⬇️ | — | thresh 调高反降 5pp |
| v4-A | SMG | 0.5 | tree=256MB+ev=600s | 100/0 | 1298s | 0.08 | 122.4s | 367s | 56.42% ⬇️ | — | 双改最差 −5pp/+30% |
| v2 | sglang | 0.5 | default | 100/0 | 1361s | 0.07 | 122.0s | 465s | 53.07% | 1567ms | sglang image (Python wrapper)|
| v1 | sglang | 0.3 | default | 100/0 | 1590s | 0.06 | 136.6s | 500s | 51.27% | 1274ms | thresh=0.3 baseline |

#### 短数据合成 4K × 5 turn × 20 sess（2026-04-28 vF / 2026-04-27 multi_turn_dp_aware.py）

| 测试 | 模式 | 完成 | Time | Avg turn lat | ΔPrompt | ΔCached | **真实 cache hit** | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---|
| ⭐ **vF-B** | client `data_parallel_rank` 直连 :8088 | 100 turns | **36.38s** | **5.82s** | 302,364 | **232,960** | **77.05%** | 验证字段在 V4-Pro 完全有效 |
| vF-A | round_robin 直连 :8088 (baseline) | 100 turns | 46.85s | 7.40s | 302,644 | 66,816 | 22.08% | baseline |

#### 早期对照（2026-04-27 #201, 20 conv × 3 turn × prompt 2K）

| 测试 | 模式 | RPS | Avg Lat | TTFT | **真实 cache hit** |
|---|---|---:|---:|---:|---:|
| C | router `--dp-aware` cache_aware | 1.59 | 2.83s | 0.54s | **46.85%** |
| B | client `data_parallel_rank` 直连 | 1.69 | 2.81s | 0.53s | 43.93% |
| A | round_robin 直连 baseline | 1.46 | 3.31s | 0.61s | 2.86% |

### 6 个核心结论（按重要性）

1. ⭐⭐ **生产首选 router cache_aware (vE-v3 配方)**：78% cache hit，对数据集多样性鲁棒，客户端零改动。
2. ⭐⭐ **`balance_abs ≤ parallel × 0.5` + `parallel` 必须是 `dp_size` 整数倍**：vE-v1 (abs=16, p=8) 凝固 3/8 DP；vB (p=10) 慢 vA (p=8) 18%；vE-v3 (p=16, abs=8) 完美 8/8 DP。
3. ⭐ **客户端 `data_parallel_rank` 字段在 V4-Pro DPA+DeepEP 下完全有效**（vF 验证 22%→77%），但**必须保证 messages[0] hash 散到全 dp_size bin**（vE-v5 反例：100 conv 共享 17 base → 6 bin → 慢 2.34×）。
4. ⭐ **multi-turn cache hit 真根因不是路由凝固，是 evalscope 累计 context 拖爆 DPA barrier**：vA 单轮 200K × 5 turn 累计到 P99 700K → DP 集群慢化；vC 单轮缩到 35K → P99 TTFT 从 352s 暴跌到 8.93s（−97.5%）。
5. ⭐ **SMG image (Rust native) vs sglang image (Python wrapper) 仅换镜像耗时 −27%、cache hit +9pp**（v3 vs v2）。生产强制用 `lmsysorg/sgl-model-gateway:latest`。
6. ⚠️ **router 调参易踩坑**：`cache_threshold=0.6` 反降 5pp；`max-tree-size 256MB + eviction 600s` 双改慢 30%、降 5pp。`max-tree-size 256MB` 单调让 inter_token P99 −38% 是唯一正向（流式输出场景才用）。

### 验证陷阱速记（必读）

- ❌ **不能用 curl 单 shot + sleep + curl /metrics** 判断 dp_rank 是否生效：DPA barrier 让瞬时 num_running 不准（vE-v4 误判源头）
- ✅ 必须用**批量 ≥20 conv × 多 turn + counter delta**（`cached_tokens_total / prompt_tokens_total` 增量）
- ❌ **evalscope `Average approx KV cache hit rate (%)` 是 prefix 估算**，与 GPU 实际复用差 17.5pp（vE-v3 95.55% vs counter 78.01%）
- ❌ **evalscope multi-turn `Avg/P99 TPOT` 是统计假象**：用 SGLang 后端 `inter_token_latency_seconds` 才真实

### 工具速查

- 真实数据 multi-turn 数据集生成：`reference_evalscope_custom_multiturn_real_data.md`
- 客户端 dp_rank 注入 wrapper：`/home/maxwellx_google_com/code/b200_vllm_opt/run_evalscope_dp.py`
- 不同 parallel 档的 router 参数：`reference_router_concurrency_param_combos.md`

### Open Questions

- EAGLE/MTP 在 vE-v3 完全没生效（`Avg decoded tokens per iter = 0.99`，`Spec accept rate = 0.0`），需查 V4-Pro 启动是否需 `--speculative-algorithm EAGLE3` 类参数

---

## 0. 背景与目标

### 问题陈述

V4-Pro 在 8×B200 上以 `--tp 8 --dp 8 --enable-dp-attention --moe-a2a-backend deepep` 拓扑运行，multi-turn 长上下文场景实测 **cache hit 仅 1.53%**（`cached_tokens_total / prompt_tokens_total`），远低于 evalscope 估算的 64%。

### 核心问题

1. KV cache 在 8 个 DP slice 之间的**物理分布**？是否存在共享？
2. SGLang 收到对话请求时，KV cache 如何**拆解 / 匹配**？
3. 在不放弃 DP attention 的前提下，能否**跨 slice 复用 prefix**？

### 调研结论速览

| 结论 | 状态 | 证据 |
|---|---|---|
| 8 个 DP slice = 8 个独立 OS 进程，KV pool/RadixTree per-process 独占 | ✅ 源码确认 | `data_parallel_controller.py:393-489` 等 |
| 路由层（4 种 LoadBalanceMethod）**完全不查 prefix** | ✅ 源码确认 | `data_parallel_controller.py:507-515` |
| **SMG co-launch 在单机 DPA 不可用，但 separate + `--dp-aware` 完美可用** | ✅ 源码 + 实测 | §4 / §6.4 |
| **SGLang 原生支持客户端 `data_parallel_rank` 字段，可零服务端改动钉同 slice** ⭐ | ✅ 源码 + 实测 | `data_parallel_controller.py:500-505`，OpenAI 协议透传 |
| **实测：dp_rank 路由让 cache hit 23% → 77.4%（3.36×），wall time −29%** | ✅ 实测 | 本文 §6 / vF |
| ⚠️ **客户端 dp_rank 仅当数据集 messages[0] hash 散到全部 dp_size bin 时有效** | ✅ vE-v5 反例 | §6.4.10：100 conv 共享 17 种 messages[0] → cache hit 50.66% / 慢 vE-v3 router cache_aware 2.34× |
| ⭐⭐ **生产首选 router cache_aware (abs=8 + parallel=16 + dp-aware)**：cache hit 78%, P99 TTFT 68s, 33.9 min | ✅ vE-v3 实测 | §6.4.9 |

---

## 1. KV pool 在 DP 模式下的物理分布

### 1.1 进程模型：8 slice = 8 个独立 OS 进程

`managers/data_parallel_controller.py:393-410` 的 `launch_dp_attention_schedulers` 调 `launch_tensor_parallel_group`；后者 `:443-489` 在 `for tp_rank in tp_rank_range` 内为每 rank `mp.Process(target=run_scheduler_process_func, ...)` 起新进程。

`layers/dp_attention.py:227-235` 的 `compute_dp_attention_world_info` 在 `tp=8 dp=8` 下计算：
```
attn_tp_size = tp_size / dp_size = 1
attn_dp_rank = tp_rank // 1 = tp_rank
```
8 个 tp_rank 各自对应 1 个独立 attn_dp_rank，每进程的 attention TP group 只有自己 1 个 rank。

### 1.2 KV pool：每 slice 持有完全独立的 5 池

`scheduler.py:602-620` `init_cache_with_memory_pool`：
```python
req_to_token_pool, token_to_kv_pool_allocator = self.tp_worker.get_memory_pool()
```
该调用在每个 scheduler 进程内执行，由该进程自己的 `tp_worker` 创建。NSA/MLA pool 的 5 个 buffer（`memory_pool.py:1813-1888` 的 `NSATokenToKVPool`，含 `index_k_with_scale_buffer` 等）都是 `torch.zeros(... device=device)` 在该进程绑定的那张 GPU 上分配，**没有任何 IPC handle 共享**。

实测 0.46M tokens/slice × 8 = 3.66M tokens 总容量，与"独立池"完全一致。

### 1.3 RadixTree：per-process 独占的 Python 对象

`scheduler.py:693`：
```python
self.tree_cache = RadixCache(params)
```
`scheduler.py:629-633` 传入 `tp_cache_group=self.attn_tp_cpu_group`，但 `attn_tp_size=1` 意味着该组只有自己 1 个成员。`radix_cache.py:252-296` 构造里**没有任何跨进程同步**——树就是进程内的 Python 对象图。

---

## 2. 请求到达后的匹配链路（端到端）

### 2.1 路由层完全不查 prefix

```
HTTP :8088
  → tokenizer_manager.py:299-306 (zmq PUSH)
  → data_parallel_controller.py:170-172 (PULL)
  → event_loop  :564-572
  → TypeBasedDispatcher  :233-242
  → round_robin_scheduler  :507-515
```

`round_robin_scheduler` 实现：
```python
self.workers[self.round_robin_counter].send_pyobj(req)
self.round_robin_counter = (self.round_robin_counter + 1) % len(self.workers)
```

只递增 counter，**不看 token 内容、不看长度、不看 session**。

其它 3 种 LoadBalanceMethod (`shortest_queue` / `decode_round_robin` / `minimum_tokens`) 同样只看队列长度（`DPBudget.dispatch :140-144`），都不查 prefix。

### 2.2 客户端绕过路由的唯一原生路径 ⭐

`data_parallel_controller.py:500-505`：
```python
def maybe_external_dp_rank_routing(self, req: Req):
    if req.data_parallel_rank is not None:
        logger.debug(f"Direct routing to DP rank {req.data_parallel_rank}")
        self.workers[req.data_parallel_rank].send_pyobj(req)
        return True
    return False
```

`round_robin_scheduler` 第一行就调它，**所有 LoadBalanceMethod 都被这个字段优先短路**：
```python
def round_robin_scheduler(self, req: Req):
    if self.maybe_external_dp_rank_routing(req):
        return
    ...
```

OpenAI 协议层完整透传：
- `entrypoints/openai/protocol.py:261` (CompletionRequest)
- `entrypoints/openai/protocol.py:538` (ChatCompletionRequest)
- `entrypoints/openai/serving_chat.py:214` 显式 `data_parallel_rank=request.data_parallel_rank`
- `entrypoints/openai/serving_completions.py:119` 同上

校验：`engine.py:243` `data_parallel_rank < 0 → ValueError`，传 `0..dp_size-1` 整数即可。

### 2.3 落到 slice 后的本地 prefix 匹配

```
scheduler.py:1600  req.init_next_round_input(self.tree_cache)
  → schedule_batch.py:860  tree_cache.match_prefix(RadixKey(token_ids=..., extra_key=self.extra_key))
  → radix_cache.py:340-410  本地 trie
  → 命中后返回 device_indices（KV 槽位整数索引）
```

命中部分通过 `cache_protected_len = len(prefix_indices)` (`schedule_batch.py:881`) 锁住；`cache_finished_req` (`radix_cache.py:429-473`) 在请求结束时把新 KV 挂回树并 `dec_lock_ref` 转入 evictable。

### 2.4 跨 slice KV 复用：源码层无原生路径

- `HiRadixCache` (`hiradix_cache.py:35-117`) 的 `MHATokenToKVPoolHost / MLATokenToKVPoolHost` 是**进程内 host RAM**，无 SHM/IPC
- 唯一跨 slice 共享路径是 `hicache_storage_backend`（HF3FS/NIXL/mooncake_store/lmcache/aibrix/EIC，`mem_cache/storage/` 各子目录）：每 slice 把自家 host pool 写到外部 KV store，其它 slice 通过 `prefetch_from_storage` (`scheduler.py:1613`) 拉回——**本质是 host→storage→host 复制，不是 GPU KV 直接共享**，且仅当 `hicache_storage_pass_prefix_keys` 启用时才走

---

## 3. 高吞吐 #201 vs 平衡 #200 在 KV cache 行为上的差异

### 3.1 关键差异：`max_running_requests`

`scheduler.py:546-559` 把它从 `tp_worker.get_worker_info()` 取出，影响：
- `update_running_batch` 的 `batch_is_full` 判定（`:1882-1893`）
- `req_to_metadata_buffer_idx_allocator` 大小（`:924` `buffer_size = self.max_running_requests * 2`）

256 vs 128 意味着同一 slice 同时驻留请求多 1 倍，每请求平均能占的 KV 从 ~5.9K token 降到 ~2.9K token。

### 3.2 Evict 触发：分配不下时才驱逐

`mem_cache/common.py:248-253`：
```python
if allocator.available_size() < num_tokens:
    tree_cache.evict(num_tokens)
```

`radix_cache.py:544-569` 用 `EvictionStrategy`（默认 LRU，`scheduler.py:634` + server_args 默认 `"lru"`）从叶子开始 heap pop。

### 3.3 multi-turn 命中实质

Turn 1 完成后 KV 不立即丢——`cache_finished_req` 走 `dec_lock_ref` 转 evictable，**只要 turn 间隔内没被新请求压力驱逐就还能命中**。但 256 并发 + 64K 上下文场景里，turn 间几秒就会被新批次推走。

---

## 4. SMG (sglang_router) 调研：co-launch 不可用，**separate + `--dp-aware` 完美可用**

**⚠️ 修正（2026-04-27 二次调研）**：之前的"SMG 单机完全不可用"结论**只对 co-launch 模式成立**。后续发现 router separate 模式有 `--dp-aware` 参数，**完美适用单机 DPA + cache_aware 路由**，已实测验证。完整对比见 §6。


### 4.1 SMG `--dp-size N` 的真实语义

`sgl-model-gateway/bindings/python/src/sglang_router/launch_server.py:86-95`：
```python
def launch_server_process(server_args, worker_port, dp_id):
    server_args = copy.deepcopy(server_args)
    server_args.port = worker_port
    server_args.base_gpu_id = dp_id * server_args.tp_size   # 关键 1
    server_args.dp_size = 1                                 # 关键 2
    proc = mp.Process(target=run_server, ...)
```

含义：
- 拉 N 个独立 sglang server，每个独占 `tp_size` 张 GPU
- 每个 server 内部 `dp_size` 被强制改 1

`--tp 8 --dp 8` 时 worker0 占 GPU 0–7，worker1 想从 GPU 8 起 → `invalid device ordinal`（2026-04-27 实测崩溃，DP0–DP7 全报）。

### 4.2 SMG 对 `--enable-dp-attention` 的支持：无

- 在 `sgl-model-gateway/bindings/python/src/sglang_router/`（`launch_server.py`、`launch_router.py`、`router_args.py`）中 grep `dp_attention` / `enable_dp` **0 命中**（无任何转发或专门处理）
- 即便参数能透传，因 line 93 强制 `dp_size=1`，而 `dp_dpa_smg_guide.md:114` 明确 "`--dp-size` must be greater than 1 for DPA to work" → 每个 worker 内 DPA 自动被关掉
- 文档无任何 "DPA + SMG co-launch" 的 MoE 示例

### 4.3 cache_aware 工作层级：仅 worker URL 间

`dp_dpa_smg_guide.md:303` "Maintains an approximate radix tree for **each worker** based on request history"；`router_args.py` 的 `worker_urls` 是字符串 URL 列表。**无任何源码进入单个 sglang 进程的内部 DP rank**。

### 4.4 单机场景 SMG 的可行性

| 模式 | 结果 |
|---|---|
| Co-launch (`--tp 8 --dp 8`) | ❌ 崩溃（64 GPU 需求，实只 8 GPU） |
| Co-launch (`--tp 1 --dp 8`) | ⚠️ 拓扑变更，丢失 DPA + EP 性能；MoE EP 在 TP=1 上能否跑通存疑 |
| **Separate + `--dp-aware`** | **✅ 完美可用，详见 §6.5** |

`dp_dpa_smg_guide.md:226` "Separate Launch (**Multi-Node**)" 把 separate 模式定位到多节点，但**单机 + `--dp-aware`** 同样可行（文档未明示，实测验证）。

### 4.5 `--dp-aware` 工作机制（源码 ground truth）

router 启动后的 4 步工作流（`sgl-model-gateway/src/`）：

1. **`discover_dp.rs:46-78`** 调后端 `/server_info` 读 `dp_size`：
   ```rust
   if !config.dp_aware { return Ok(StepResult::Success); }
   let dp_info = get_dp_info(&config.url, ...)?;
   context.set("dp_info", dp_info);
   ```

2. **`create_worker.rs:286-334`** 把 1 个 URL 展开成 N 个虚拟 worker：
   ```rust
   for rank in 0..dp_info.dp_size {
       let mut builder = DPAwareWorkerBuilder::new(normalized_url, rank, dp_info.dp_size)
       workers.push(Arc::new(builder.build()));
   }
   ```
   内部约定 URL 格式 `http://host:port@rank`。

3. **`policies/registry.rs:267-277`** cache_aware 在虚拟 worker 间建 radix tree：
   ```rust
   init_cache_aware_policy(workers: &[Arc<dyn Worker>], ...)
   ```
   即 prefix tree 是 `(URL, rank)` 粒度。

4. **`router.rs:487-524`** 转发请求时剥 `@rank` 注入 `data_parallel_rank`：
   ```rust
   const DP_RANK_KEY: &str = "data_parallel_rank";
   let (worker_url_prefix, dp_rank) = Self::extract_dp_rank(worker_url)?;
   map.insert(DP_RANK_KEY.to_string(), serde_json::json!(dp_rank));
   ```

后端 SGLang 的 `maybe_external_dp_rank_routing` (`data_parallel_controller.py:500-505`) 接收并直接路由到对应 DP slice — 与客户端手动传字段走的是同一条路径。


---

## 5. SGLang 其他相关源码事实

### 5.1 `ep_size` 在 deepep 模式下被强制覆盖

`server_args.py:1860-1865` `_handle_a2a_moe`（post-init 阶段无条件执行）：
```python
def _handle_a2a_moe(self):
    if self.moe_a2a_backend == "deepep":
        ...
        self.ep_size = self.tp_size
        logger.warning(
            f"DeepEP MoE is enabled. The expert parallel size is adjusted "
            f"to be the same as the tensor parallel size[{self.tp_size}]."
        )
```

`mooncake` / `ascend_fuseep` 同样覆盖。

含义：
- DeepSeek V4 / V3 官方配方**不写 `--ep 8`** 是因为写了也会被覆盖
- ep_size 永远 = tp_size
- 真正控制 EP 规模的只有 `--tp`

### 5.2 `sglang:cache_hit_rate` gauge bug

`scheduler_metrics_mixin.py:378` decode 路径硬编码 `cache_hit_rate = 0.0`，只 prefill 路径正确算。`self.stats.cache_hit_rate = cache_hit_rate` 是瞬时 gauge，取最后 step 值，decode step 数远多于 prefill → gauge 几乎永远是 0。

**正确算法**：
```promql
sglang:cached_tokens_total / sglang:prompt_tokens_total
```

---

## 6. 关键发现：客户端 `data_parallel_rank` 路由 ⭐

### 6.1 用法

OpenAI Chat Completions 请求体加非标字段：
```json
{
  "model": "DeepSeek-V4-Pro",
  "messages": [...],
  "data_parallel_rank": 3
}
```

Python OpenAI SDK：
```python
import hashlib
session_id = "user-123-conv-456"
dp_rank = int(hashlib.md5(session_id.encode()).hexdigest(), 16) % 8

resp = client.chat.completions.create(
    model="DeepSeek-V4-Pro",
    messages=[...],
    extra_body={"data_parallel_rank": dp_rank},
)
```

### 6.2 实测对比（2026-04-27 V4-Pro #201 高吞吐配方）

测试参数：`multi_turn_dp_aware.py`，20 sessions × 5 turns × prompt_len=4000 × max_tokens=200

| 指标 | round_robin (baseline) | dp_rank (hash session) | 提升 |
|---|---|---|---|
| **cache hit (cached/prompt)** | **23.0%** (68,864/299,287) | **77.4%** (234,496/303,061) | **3.36×** |
| wall time | 53.5 s | 38.1 s | −29% |
| avg turn latency | 8.10 s | 6.07 s | −25% |
| session 分布 | counter 递增（无亲和） | 7 slice 命中（hash 不均，slice 1 落空） | — |

数学一致性：
- round_robin 不是 0% → counter 递增 + 异步并发交叠下，同 session 5 turn 仍有 ~23% 概率落同 slice
- dp_rank 强制钉同 slice → hit 接近 prefix overlap 上限

### 6.3 与其他方案对比

| 方案 | 改动量 | 跨 turn cache | 跨 session prefix | 适用场景 |
|---|---|---|---|---|
| **router separate + `--dp-aware`** ⭐⭐ | 加 1 router 容器 | ✅ 真 token-level radix | **✅ 命中** | 生产首选 |
| **客户端 hash dp_rank** | 客户端按 session hash | ✅ session 粒度 | ❌ 无 | 不想加组件 |
| SMG co-launch | 拓扑重构 | ⚠️ 单机 DPA 不适用 | — | 不可用 |
| hicache + storage backend | 加参数 + SSD | ⚠️ V4 走 HiSparseCoordinator host pool 硬编码 | ✅ | dense 模型可行 |
| 默认 round_robin | 零 | ❌ 2.86% | ❌ | 单 turn / 无亲和需求 |

### 6.4 router `--dp-aware` 实测

#### 6.4.1 小规模快速对比（2026-04-27 V4-Pro #201, 20 conv × 3 turn × prompt 2K）

| 模式 | RPS | Avg Lat | TTFT | 真实 cache hit (counter delta) |
|---|---:|---:|---:|---:|
| A: 直连 :8088 round_robin | 1.46 | 3.31 s | 0.61 s | **2.86%** |
| B: 客户端 hash dp_rank | 1.69 | 2.81 s | 0.53 s | **43.93%** |
| C: router `--dp-aware` cache_aware | 1.59 | 2.83 s | 0.54 s | **46.85%** |

观察：
- B/C 真实 cache hit 几乎相等（差异在 noise 范围）
- B 略快（无 router 一跳延迟），C cache hit 略高（token-level prefix tree）
- random 数据集无跨 session prefix 共享，C 的真正优势未体现；在共享 system prompt 场景 C 应显著领先
- C 客户端零改动，是生产推荐

#### 6.4.2 真负载完整压测（V4-Pro #201, 100 conv × 5 turn × 30K-200K input × max-tokens 500）

> **2026-04-27 三轮实测进展**：v1 default (sglang img) → v2 tuned (sglang img) → **v3 tuned (SMG img)**。
> 其中 v3 换用 `lmsysorg/sgl-model-gateway:latest` image，**仅换 image，参数与 v2 完全相同**，性能大幅跃升。

```bash
./multi_turn_benchmark.sh DeepSeek-V4-Pro /lssd/models/DeepSeek-V4-Pro \
  --number 100 --parallel 10 --port <8000 router | 8088 直连>
```

##### 整体性能（六轮对比）

| 维度 | v1 default<br>(sglang, t=0.3) | v2 tuned<br>(sglang, t=0.5) | **v3 tuned<br>(SMG, t=0.5)** | v4-A<br>(+tree256MB +ev600s) | v4-B<br>(+tree256MB) | v6<br>(t=0.6) |
|---|---:|---:|---:|---:|---:|---:|
| **耗时** | 1590 s | 1361 s | **995 s** ⭐ | 1298 s ⚠️ | 1187 s | 1008 s |
| RPS | 0.06 | 0.07 | **0.10** | 0.08 | 0.08 | 0.10 |
| Avg Lat | 136.6 s | 122.0 s | **98.2 s** | 122.4 s | 111.0 s | 99.8 s |
| Avg TTFT | — | 55.6 s | **37.7 s** | 46.3 s | 41.8 s | 42.0 s |
| **TTFT P99** | 499.99 s | 465.08 s | **286.18 s** | 367.3 s | 366.2 s | 349.5 s |
| **真实 cache hit** | 51.27% | 53.07% | **61.63%** ⭐ | 56.42% ⬇️ | 58.90% | 56.37% ⬇️ |
| Avg toks/s | 13.41 | 13.76 | 17.19 | 16.48 | **20.15** | 16.04 |
| Router Mem | 614 MiB | 614 MiB | 400 MiB | 400 MiB | 400 MiB | 400 MiB |
| 成功率 | 100% | 100% | 100% | 100% | 100% | 100% |

**v6 实测推翻 subagent 预测**：cache_threshold=0.5 → 0.6 不是 +1~2pp 而是 **−5.3pp**（61.63%→56.37%）。原因：原本 50-60% 重叠请求被强制走 fallback shortest-queue → 路由到无 prefix 历史 DP → 全 cold prefill → DP0/DP7 命中从 60%+ 跌到 40%。

**结论**：cache_threshold=0.5 是 multi-turn 30K-200K 场景的真 sweet spot，单方向调整都是负向。

##### inter-token latency（必须区分两种口径！）

| 口径 | v1 | v2 | v3 | v4-A | v4-B | 说明 |
|---|---:|---:|---:|---:|---:|---|
| **evalscope avg TPOT** ⚠️ | 0.556s | 0.907s | 0.346s | 0.261s | 0.236s | ❌ 含 turn 间客户端处理 + 网络 + 下个 turn 等待 |
| **evalscope P99 TPOT** ⚠️ | 15.93s | 57.34s | 7.96s | 1.78s | 0.575s | ❌ 同上，multi-turn 失真 |
| **SGLang inter_token P50** ✅ | 377ms | 430ms | 368ms | 407ms | **302ms** | ✅ 后端真实测量，含 chunked prefill 抢占 |
| **SGLang inter_token P95** ✅ | 588ms | 594ms | 594ms | 590ms | **570ms** | ✅ |
| **SGLang inter_token P99** ✅ | 1274ms | 1567ms | 1616ms | 1502ms | **996ms** | ✅ v4-B 最稳 |

##### Router 参数独立效应（控制变量分析）

| 参数 | 比较 | 影响 |
|---|---|---|
| `eviction-interval 60→600` | v4-B (60s) vs v4-A (600s) | ❌ 600s 让 cache hit 降 2.5pp、耗时 +9% |
| `max-tree-size 64MB→256MB` | v3 (64MB) vs v4-B (256MB) | ❌ 256MB 让 cache hit 降 2.7pp、耗时 +19% |
| `max-tree-size 64MB→256MB` (only) | v3 vs v4-B inter_token | ✅ **inter_token P99 −38%（1616→996ms）— 唯一正向** |

> **重要纠正**：之前表格里的 `evalscope P99 TPOT` 列（v2 突增 +260% 等）是 **multi-turn 统计假象**，因为 evalscope 的 TPOT 算法 `(E2E - TTFT) / output_tokens` 在多轮场景把 turn 间的等待也算进分子。**真实 inter-token latency 看 SGLang 后端 `inter_token_latency_seconds`**：
> - 各轮 P50 都在 370-430ms 量级（chunked prefill 与 decode 持续交错的固有开销）
> - P99 都在 1.3-1.6s 量级（被长 prompt 抢占 decode 的极端情况）
> - 不同 router 配置对真实 inter-token 影响很小（差异在 ±200ms 内），证明 router 调参不会显著影响后端 decode 体验

> **v2 → v3 唯一变化是 docker image**：sglang 镜像里的 sglang_router 是 Python wrapper，SMG 镜像 (`lmsysorg/sgl-model-gateway:latest`) 是 native Rust 实现，**总耗时和 cache hit 差距巨大**（不是 inter-token latency）。
>
> **v3 → v4-A 调 router 内部参数（`--max-tree-size 256MB --eviction-interval-secs 600`）实测负向**：cache hit 反而下降 5pp，总耗时 +30%。结论：v3 的 router 默认 `max-tree-size=64MB / eviction-interval=60s` 已经是 sweet spot，不要瞎调。

> **evalscope 表中 "Approx Cache Hit 865184% / 773606%" 是已知 bug**（详见 `feedback_sglang_cache_hit_metric_bug.md`），只信 counter delta。

##### per-DP 路由分布对比

| DP | v1 | v2 | **v3** | 备注 |
|---:|---:|---:|---:|---|
| 0 | 7 | 13 | 12 | 均衡 |
| 1 | 11 | 12 | 12 | 均衡 |
| 2 | 15 | 11 | 14 | 均衡 |
| 3 | 17 | 17 | 5 | v3 最少 |
| **4** | **33** | 10 | 13 | v3 完全平衡 |
| **5** | **0** | 10 | 13 | 持续工作 |
| 6 | 22 | 5 | 13 | v3 回升 |
| **7** | 17 | **23** | 19 | 仍 spillover 但减轻 |
| **极差比** | **33:0 = ∞** | 23:5 = 4.6× | **19:5 = 3.8×** | 持续收窄 |

##### per-DP 终态 cache hit (realtime_tokens prefill_cache / (prefill_cache + prefill_compute))

| DP | v1 hit | v2 hit | **v3 hit** | 解读 |
|---:|---:|---:|---:|---|
| 0 | 61.2% | 62.1% | 51.7% | 均衡分流后略降 |
| 1 | 65.0% | 59.6% | 59.2% | ~ |
| 2 | 67.1% | 65.5% | 66.8% | 持平 |
| 3 | 59.0% | 60.7% | 66.3% | ↑ |
| 4 | 62.9% | 65.7% | 63.4% | ~ |
| 5 | 35.9% | 16.9% | **62.0%** | v3 真正承担稳定负载 |
| 6 | 63.0% | 64.0% | 54.2% | 略降 |
| **7** | **0.98%** | 19.3% | **64.7%** | ✅ **cold-spillover 完全消失** |

**v3 全 8 个 DP cache hit 都 ≥ 51%**，DP 间最大差距仅 15pp（v1/v2 时差距 60pp+）。这是质的飞跃。

##### Router 自身性能（不是瓶颈）

| 维度 | 值 | 评价 |
|---|---|---|
| 容器 CPU | **0.01%** | 跑 100 conv 全程零开销 |
| 容器 Mem | 614 MiB / 3.8 TiB | 可忽略 |
| router_request_duration mean | 877 ms | 122 个请求 96% 在 ≤ 2.5s |
| 路由决策开销 | < ms 级 | router 不是瓶颈 |

#### 6.4.3 调参 + image 决策矩阵

| 业务诉求 | 推荐配置 |
|---|---|
| **生产首选（吞吐 + cache hit）** | **v3 (SMG image + tuned)** ⭐⭐ |
| **流式输出（inter-token 平稳）** | **v4-B (SMG + max-tree=256MB)** ⭐ |
| 不想换 image，仅调参 | v2 (sglang image + tuned) |
| ❌ 不要瞎调 router 内部参数 | v4-A 实测 `eviction=600s + tree=256MB` 双改最差，拖慢 30%、降 hit 5pp |
| ❌ 单独 `--eviction-interval 600` 也是负向 | v4-B vs v4-A 证明 60s 默认更优 |
| ⚠️ 不要混合 grpc 后端 + dp-aware | SMG bug，必崩 |
| ⚠️ inter-token latency 看 SGLang 后端不看 evalscope | evalscope multi-turn TPOT 失真 |

#### 6.4.4 router 三个核心参数的源码语义（必读）

`cache_aware.rs:368-376` 是路由决策的中心：

```rust
let is_imbalanced = max_load.saturating_sub(min_load) > balance_abs_threshold
    && (max_load as f32) > (min_load as f32 * balance_rel_threshold);
//              ↑ AND 关系
if is_imbalanced {
    return select_worker_min_load();  // 完全不查 prefix tree
}
// 否则进 cache 路径，再看 cache_threshold
let selected = if match_rate > cache_threshold {
    workers.iter().position(|w| w.url() == tenant_url)  // 走 last_tenant 亲和
} else {
    healthy_indices.iter().min_by_key(|&&idx| workers[idx].load())  // fallback shortest-queue
};
```

| 参数 | 源码事实 |
|---|---|
| `cache_threshold` | 严格大于 (`>`)，不是 `>=`；match_rate 必须 **超过**阈值才走亲和 |
| `balance_abs_threshold` | `worker.load()` 是 router 维护的**瞬时 in-flight** 请求数（不是队列长度，不是 KV 占用） |
| `balance_rel_threshold` | 同上，比的是 in-flight 请求数比例 |
| **优先级** | `is_imbalanced`（abs AND rel）**优先于** `cache_threshold` |

##### 死活线（关键发现）

在 `parallel=N` 场景下，per-DP in-flight ≈ N/8，max-min ≤ N。当 `balance_abs > N` 时：
- `is_imbalanced` 永远为 false
- `balance_abs/rel` 两个参数**完全失效**
- 真正的旋钮只剩 `cache_threshold`

实测：v3 配方 `balance_abs=16` 在 `parallel=10` 场景下完全是死参数（max-min ≤ 10 < 16）。这解释了为什么 v3 默认参数已是 sweet spot — 因为 balance_* 没起作用，调它无用。

##### 不同并发档的 balance 推荐

| parallel | balance_abs | balance_rel | cache_threshold | 备注 |
|---|---|---|---|---|
| ≤ 10 | 16 (死参数) | 1.2 (死参数) | **0.5** | balance_* 失效，唯一旋钮 cache_threshold |
| 16 | 8 | 1.2 | 0.5 | balance_* 在峰值偶尔触发 |
| 24 | 6 | 1.15 | 0.5 | 严格分流 |
| 32 | 4 | 1.1 | 0.5 | 接近 V4 KV 上限（per slice max_running_requests=32） |
| 40+ | 4 | 1.1 | 0.45 | 松一档 cache_threshold 让 fallback 救命 |

##### "看似合理但负向"的组合（实测验证）

| 组合 | 为什么负向 |
|---|---|
| `cache_threshold=1.0` | `>` 永远 false → 永远 fallback → 比直接 `--policy round_robin` 还差（tree 仍 insert 浪费 CPU） |
| `balance_abs=4 + balance_rel=1.05`（parallel=10） | rel 几乎必触发 → 退化纯 round_robin → cache hit ~1-3% |
| `cache_threshold=0.3 + balance_abs=4 + balance_rel=1.05` | 兼顾的直觉 → 实际 100% 走 shortest-queue（imbalance 优先级高）→ cache 路径基本不走 |
| **`cache_threshold=0.6`（v6 实测 2026-04-28）** ❌ | 原以为提高阈值能逼"差强人意"匹配走 fallback 提升精度，**实测 cache hit −5.3pp（61.6%→56.4%）**；原本 50-60% 重叠请求被甩到 fallback → 路由到 in-flight 最少 DP（无该 prefix 历史 KV）→ 全 cold prefill → DP0/DP7 命中从 60%+ 跌到 40% |

#### 6.4.5 cold-spillover / 排队的精确机制（澄清）

**不是"同一 conv 的多个 turn 同时到达"**：evalscope multi-turn 串行 — 每个 worker 跑 turn-1 → 等响应 → turn-2 → ...，任意时刻每 worker 最多 1 个 in-flight。`parallel=10` = 10 个 worker = 10 个不同 conversation 各自的某个 turn。

**真正机制：多个不同 conv 因 first-pioneer 凝固被错误归并到同一 DP**：

```
worker-1 (conv-A turn-3) → router char-walk → 命中节点 last_tenant=DP0 → 路由 DP0
worker-2 (conv-B turn-2) → router char-walk → 该路径节点 last_tenant 也是 DP0 → 路由 DP0  ← 不同 conv 但归并
worker-3 (conv-C turn-4) → router → DP7
...
→ DP0 收到 conv-A + conv-B 当前 turn = 1 running + 1 queue
→ 其他 6 DP 各 1 conv = 8 in-flight 总数
```

random_multi_turn 数据集 conv 间理论无 prefix overlap，但字符级 trie 在短 prefix（如 system_prompt 的首字符 collision）上仍会归并，触发 first-write-wins 缓存（`tree.rs:570-598`）。

#### 6.4.6 关键洞察

1. **真正的瓶颈不是 router 本身，而是 DP7 cold prefill**（v1/v2 时）
   - cache_aware 把"找不到 prefix worker"的请求集中到一个 worker → 它独自吞所有 200K tokens 的 cold prefill
   - 后端 TTFT P99 ≥ 400-500s 全部来自 DP7 排队
2. **v2 调参的本质是分流 cold 请求**
   - `--cache-threshold 0.3 → 0.5`：必须 50% 以上前缀重叠才走亲和（更严格）
   - `--balance-rel-threshold 1.5 → 1.2`：prefix worker 比平均忙 1.2× 就分流（更早触发）
   - 结果：cold 不再独自给一个 worker，分摊到多个；DP7 cache hit 从 1% → 19%
3. **cache hit 没下降反而升**：触发亲和的条件更严格，命中的都是真高质量命中
4. **v2 → v3 换 image 后 TPOT 抖动消失**：sglang image 里 sglang_router 是 Python wrapper（响应处理慢导致 inter-token jitter）；SMG image 是 native Rust，请求 pipeline 全无锁实现 → TPOT P99 从 57s → 7.96s（−86%）
5. **DP7 cache hit 1% → 19% → 65%**：v3 的 router 不仅分流 cold，还能让 cold worker 也快速进入"温暖"状态
6. **生产部署强烈建议直接用 SMG image**，性能远超 sglang image 里的 sglang_router

##### v3 完整生产配方（推荐）

```bash
# 后端不变
sudo docker run -d --name deepseek-v4-pro \
  --gpus all --ipc=host --shm-size 32g --network host \
  -v /lssd/models:/lssd/models:ro -v /lssd/cache:/root/.cache \
  -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
  -e NCCL_IB_DISABLE=1 \
  lmsysorg/sglang:deepseek-v4-blackwell \
  python3 -m sglang.launch_server \
    --tp 8 --dp 8 --enable-dp-attention \
    --moe-a2a-backend deepep \
    --max-running-requests 256 --cuda-graph-max-bs 64 \
    --mem-fraction-static 0.82 \
    --enable-metrics --enable-metrics-for-all-schedulers \
    --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
    --host 0.0.0.0 --port 8088 ...

# Router 用 SMG 专用 image
sudo docker run -d --name sglang-router \
  --network host -v /lssd/models:/lssd/models:ro \
  lmsysorg/sgl-model-gateway:latest \
    --worker-urls http://127.0.0.1:8088 \
    --model-path /lssd/models/DeepSeek-V4-Pro \
    --tokenizer-path /lssd/models/DeepSeek-V4-Pro \
    --reasoning-parser deepseek-v4 --tool-call-parser deepseek \
    --dp-aware --policy cache_aware \
    --cache-threshold 0.5 --balance-rel-threshold 1.2 --balance-abs-threshold 16 \
    --prometheus-port 29000 --host 0.0.0.0 --port 8000

# 客户端打 :8000，evalscope --port 8000
```

⚠️ **不要试 grpc 后端 + dp-aware**：SMG dp-aware 探测硬编码 HTTP，与 grpc:// scheme 不兼容（详见 `feedback_smg_grpc_dpaware_incompatible.md`）

#### 6.4.7 parallel=8 vs parallel=10 公平对比（seed=42, 2026-04-28）

> **动机**：v3 baseline 用 `parallel=10`，10 个 worker / 8 DP slice 必然存在 2 个 DP 同时承担 2 个 in-flight 的"交替排队"现象，引入随机性。换 `parallel=8`（1:1 对应）应消除该噪音并让 cache_aware 收敛到更稳态。本轮用 `--seed 42 --number 100` 固定 conv 集合，公平对比两种并发。

##### 测试参数（A vs B 唯一差异：parallel）

```bash
evalscope perf \
  --model DeepSeek-V4-Pro --url http://127.0.0.1:8000/v1/chat/completions --api openai \
  --dataset random_multi_turn \
  --min-prompt-length 30000 --max-prompt-length 200000 --max-tokens 500 \
  --multi-turn --min-turns 5 --max-turns 5 \
  --number 100 --seed 42 \
  --parallel <8 | 10>
# Router: v3 配方 (cache_threshold=0.5 / balance_abs=16 / balance_rel=1.2)
```

##### 整体性能对比

| 维度 | v3 baseline<br>(p=10, noseed) | **vA (p=8, seed=42)** | **vB (p=10, seed=42)** | A vs B |
|---|---:|---:|---:|---|
| **耗时** | 995 s | **1096 s** ⭐ | 1329 s | A 快 18% |
| 完成 / 失败 | 100 / 0 | 99 / 1 ⚠️ | 100 / 0 | B 更稳 |
| **RPS** | 0.10 | **0.090** | 0.075 | A +20% |
| Avg latency | — | 82.5 s | 122.4 s | A 快 33% |
| Avg input tokens / req | ~280K | 316,942 | 347,096 | seed 同但 multi-turn 路径异 |
| **TTFT P99** | 286 s | 352 s | 348 s | 几乎持平 |
| TPOT P50 (evalscope) | — | 0.22 s | 0.27 s | A 快 18% |
| **TPOT P99 (evalscope)** ⚠️ | 7.96 s | **0.53 s** ⭐ | 2.10 s | A 快 4× |
| **真实 cache hit** (counter Δ) | **61.63%** | 59.09% | 56.14% | A +2.95pp vs B |
| evalscope approx hit ❌ | 61.63% | 68.81% | 69.16% | (Approx 不可信，见 §6.4.2 备注) |

##### 三个关键发现

1. **parallel=8 全面优于 parallel=10**（A vs B 同 seed 公平比）：耗时 −18%、RPS +20%、cache hit +2.95pp、TPOT P99 快 4×。这印证了"10:8 不整除引入的 2 DP 交替排队拖累全局指标"假设。
2. **TPOT P99 跌幅最显著**（7.96s → 0.53s，即 −93%）：parallel 从 10 降到 8 后，long-decode 不再被新到达的 prefill 抢占 — 因为每 DP 严格 1 个 in-flight，不存在抢占可能。
3. **vA/vB cache hit 都低于 v3 baseline (59.09% / 56.14% vs 61.63%)**：seed=42 让 conv 内容重新洗牌，输入 token 量 (316K-347K vs ~280K) 也不同 → conv 集合不可比，**不能据此说 seed 拖累 cache hit**，只能证 vA 与 vB 之间的差是 parallel 引起的真正信号。

##### vA 的 1 个失败 conv（2026-04-28 prom 双源诊断）

**指标证据**（Prometheus + sglang 后端日志双源对照）：

| 时间 (UTC) | DP6 num_queue | DP6 num_running | DP6 token_usage | 全 8 DP 平均 token_usage |
|---|---:|---:|---:|---:|
| 01:00 | 0 | 1 | 0.756 | 0.55 |
| 01:01 | 0 | 1 | 0.756 | 0.50 |
| **01:02** | **11 (突涨)** | **0** | **0** | **0** ← 集群进入低活动 |
| 01:05 | 14 | 0 | 0 | 0 |
| 01:10 | 19 | 0 | 0 | 0 |
| 01:15 | 24 | 0 | 0 | 0 |
| 01:20 | 29 | 0 | 0 | 0 |
| 01:22:26 | 31 | 0→24 | 0 | — |
| 01:22:27 | 0 | 24 | 0.03 | — |

**日志证据（01:02-01:22 共 20 分钟）**：
- 全 8 DP 每分钟仅输出 1 条 batch log（chunked prefill 256 token chunk）
- DP7 在 01:00 持续 decode：`gen throughput (token/s): 0.03` → 暗示在跑 400K+ tokens 输入的超长 conv
- DP6 在该 20 分钟内**完全无 batch log 输出**，但 prom 显示 num_queue 单调涨到 31
- 01:22:27 一次性吞 31 个 (8+8+8+7) 请求（vA 主流程结束 client 释放压力后才 burst 处理）
- 1 个请求未在 client deadline 前完成 → `sglang:num_aborted_requests_total=1`

**根因链路**（5 步）：

1. **vA 测试期间存在超长 prompt conv**（DP7 gen throughput 0.03 tok/s 暗示 400K+ token 输入正在 decode）
2. **DP attention + DeepEP all-to-all 强制 8 DP 每 step 同步**：DP7 forward pass 时间被超长 prompt 拉到秒级，整个集群 step 速率从 ~50/s 跌到 ~1/min
3. **evalscope 8 个 worker 持续送请求**：每收到响应立即送下一 turn，到达 router 的请求速率不变
4. **cache_aware first-pioneer 把请求归并到 DP6**：某条早期长 conv 把 DP6 节点锁成 hot tenant，后续相似 prefix 全甩给 DP6
5. **DP6 waiting_queue 接收速率 > 处理速率**：单调累 31 个；vA 主流程结束 client 不再压入新请求后，DP6 才有机会 burst 处理；其中 1 个超时被 client abort

**关键洞察**：

- **这不是 "DP6 单独卡死"，是集群级慢化 + cache_aware 路由热点的复合效应**。
- DP attention 把 8 DP 的 schedule loop 耦合在一起：单个 DP 跑超长 prompt → 全集群慢化。
- cache_aware 在集群慢化期间持续向 hot DP 累队列 → 形成"队列堆积漏斗"。
- v1 时代 DP7 cache hit 1% (cold spillover) 是一种表现；vA 时代 DP6 queue 堆 31 是另一种表现，根因相同：**cache_aware 把 cold/hot 请求集中到一个 DP 而集群无法横向分摊负载**。

**生产风险**：
- 长稳测试 / 真实生产流量必有 long-tail prompt（>200K tokens），任何时刻进入这种"集群慢化 + 路由热点"模式都会导致**几十个请求超时被 abort**。
- p=8 vA 表面比 p=10 vB 优秀（cache hit +2.95pp，TPOT P99 快 4×），但**容错性更差**（vA 99/100 vs vB 100/0），因为 8:8 的 1:1 路由让任何 DP 一旦慢化就没有冗余可分摊。

**对应 §6.4.5 cold-spillover 机制的延伸**：first-pioneer 凝固在 cluster 健康时表现为 cache hit 优势，在 cluster 慢化时表现为队列堆积 → 这是同一机制的两面性。

**待解决方向**（subagent 调研中）：
1. 限制单 conv prompt 长度上限以避免 DP attention 步长被单个长 prompt 拉爆
2. router 层加 max-queue-per-worker 防止单 DP 队列无限堆积
3. 集群慢化检测 + cache_aware 临时降级到 round_robin
4. evict 策略 / KV pool 大小调整避免 cache_aware 把 hot 凝固到单 DP

##### ⚠️ 真根因订正：evalscope multi-turn 累计 context 暴涨到 700K

**原推断错误**：之前的 §6.4.7 和初版方案推测"超长 prompt = 400K+"是从 `gen throughput 0.03 tok/s` 间接推算，违反"禁止间接推断"原则。

**实测证据**（evalscope perf 输出 input tokens 分布，2026-04-28）：

| Percentile | vA Input tokens | vB Input tokens |
|---|---:|---:|
| P50 | 319,085 | 366,279 |
| P95 | 618,883 | 640,719 |
| **P99** | **665,809** | **703,671** |
| Avg | 316,942 | 347,096 |

**真根因**（subagent 源码确认 + 5 conv × 5 turn 实测）：
- `evalscope/perf/multi_turn_benchmark.py:130-215`：每个 worker 维护 `context: List[Dict]`，**每轮硬编码把 user message + 上一轮真实 assistant response 全量 append**
- `random_multi_turn` 数据集：每轮 user 长度独立从 `[min_prompt_length, max_prompt_length]` 均匀采样，**与 turn 无关**
- 累计公式：`turn_N_input ≈ N × U + (N-1) × max_tokens` ≈ `N × U`
- vA `--min 30K --max 200K --turns 5` → turn-5 input ≈ 5 × 200K = **1M（理论上限）**，实测 P99 = 666K-704K **完全吻合**

**实测验证**（5K-10K 单轮 × 5 turn × 5 conv）：
| conv | turn-1 | turn-2 | turn-3 | turn-4 | turn-5 |
|---|---:|---:|---:|---:|---:|
| 0 | 8,286 | 16,338 | 22,118 | 28,804 | 34,759 |
| 1 | 6,214 | 13,028 | 21,988 | 28,865 | 38,125 |
| ... | ... | ... | ... | ... | ... |

严格单调递增 ≈ 5×，确认累计行为。

**evalscope 没有任何控制累计 context 的参数**：
- 无 `--max-history-tokens` / `--max-context` / `--rolling-window` / `--keep-only-last-k-turns`
- runner 硬编码 append 全量 context（`multi_turn_benchmark.py:155, 213`），不可配置
- 唯一杠杆：从输入端调小 `--max-prompt-length` 或减少 `--max-turns`

##### 修正后的方案重排

| 优先级 | 方案 | 对症 | 工程代价 |
|---|---|---|---|
| **#1（首选）** | **缩小 evalscope 单轮长度**：`--max-prompt-length 35000`（让 5 turn 累计 P99 < 200K） | ✅ 真正切断根因，且保持长上下文 multi-turn 测试语义 | 1 个参数变更，零代码 |
| **#2（备用）** | **减少 turn 数**：`--max-turns 3` + 单轮 30K-60K | ✅ multi-turn 深度减半但保持单轮长 prompt | 2 个参数变更 |
| **#3（生产侧）** | **业务侧 multi-turn 滚动 truncation**：客户端在 turn-N 之前裁掉早期历史 | ✅ 真生产环境唯一可行方案 | 业务代码改造 |
| ⚠️ #4 不推荐 | SGLang `--context-length 200000` | ❌ multi-turn 场景下会 reject 50% 以上后期 turn，等于禁掉测试 | — |
| ⚠️ #5 不对症 | `SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1` | ❌ vA 时 num_running 远未到 256，永远不触发 | — |

**首选方案 A 复测命令 + 实测结果（vC, 2026-04-28）**：
```bash
evalscope perf --model DeepSeek-V4-Pro \
  --url http://127.0.0.1:8000/v1/chat/completions --api openai \
  --dataset random_multi_turn \
  --min-prompt-length 5000 --max-prompt-length 35000 \
  --max-tokens 500 \
  --multi-turn --min-turns 5 --max-turns 5 \
  --number 100 --parallel 8 --seed 42
```

##### vC 实测结果（vA 同 seed/parallel/turns，唯一变化：单轮 prompt 200K → 35K）

| 维度 | vA (200K单轮) | **vC (35K单轮)** | 改善 |
|---|---:|---:|---|
| 耗时 | 1096 s | **228.8 s** | **快 4.8×** ⭐⭐⭐ |
| RPS | 0.090 | **0.437** | **+386%** |
| 完成 / 失败 | 99 / 1 ⚠️ | **100 / 0** | 0 失败 |
| Avg input tokens | 316,942 | 60,569 | -81% |
| Avg latency | 82.5 s | **18.1 s** | -78% |
| **TTFT P99** | **352 s** | **8.93 s** | **-97.5%** ⭐⭐⭐ |
| TPOT P50 | 0.22 s | 0.10 s | -55% |
| TPOT P99 | 0.53 s | 0.56 s | 持平 |
| **真实 cache hit** | 59.09% | **64.19%** | **+5.1pp** |
| Total tok/s | 28,645 | 26,532 | -7%（基本持平）|

##### vC 期间 prom 验证（02:32-02:36 UTC, step=15s）

| DP | queue_max | running_max | token_max |
|---:|---:|---:|---:|
| 0 | **0** | 1 | 0.12 |
| 1 | **0** | 2 | 0.33 |
| 2 | **0** | 1 | 0.14 |
| 3 | **0** | 1 | 0.16 |
| 4 | **0** | 2 | 0.12 |
| 5 | **0** | 1 | 0.15 |
| 6 | **0** | 1 | 0.13 |
| 7 | **0** | 1 | 0.13 |

**完美健康状态**：所有 DP queue 全程为 0，没有任何凝固 / 堆积 / 集群慢化。

##### 根因彻底确认

| 假设 | 验证结果 |
|---|---|
| **真根因 = multi-turn 累计 context 700K 拖爆 DPA 集群** | ✅ vC 缩小累计到 < 200K 后 TTFT P99 从 352s 暴跌到 8.93s（-97.5%）|
| 假根因 #1：cache_aware first-pioneer 凝固 | ❌ vC 健康集群下 cache_aware 工作良好（cache hit +5.1pp）|
| 假根因 #2：parallel=8 容错性差 | ❌ vC 同 p=8 但 0 失败 |
| 假根因 #3：DP6 单独有问题 | ❌ vC 8 DP 全部健康 |

**结论**：cache_aware 凝固是**集群慢化时的二级表现**，根本治理点在输入侧——不让单 conv 累计 context 把 DeepEP barrier 拉爆。

##### 生产建议（最终版）

1. **测试侧**：评估 multi-turn 性能时必须用 `N × U_max < model_context_target` 的输入约束（详见 `reference_evalscope_multiturn_accumulation.md`）
2. **业务侧**：multi-turn 业务必须做滚动 truncation 把单 conv 累计 context 控制在合理范围（实测 V4-Pro 8×B200 < 200K 健康，> 500K 集群塌缩）
3. **监控侧**：Grafana 加 `histogram_quantile(0.99, sum(rate(sglang:prompt_tokens_bucket[5m])) by (le))` panel + 阈值告警 200K
4. **不要做**：再尝试 SGLang/router 层缓解（已证 SchedulerEnhancer 不触发、cb 是被动恢复、`--context-length` 在 multi-turn 业务下不可用）

##### 生产配方决策更新

| 业务情况 | parallel 推荐 | 理由 |
|---|---|---|
| 已知并发 ≤ 8 | **parallel=8** ⭐ | 与 8 DP 1:1 对应，消除排队噪音；cache hit / 吞吐 / TPOT P99 全优 |
| 必须 parallel=10 | parallel=10 | 接受 ~5pp cache hit 损失 + 4× TPOT P99 抖动 |
| 高并发 16/24/32 | 见 `reference_router_concurrency_param_combos.md` | balance_abs/rel 必须按档调整 |

> ⚠️ **核心洞察**："parallel 设为 dp_size 的整数倍"不是装饰性建议，而是 cache_aware + dp-aware 路由的硬性约束。10:8 这种非整除会让 cache_aware 在两种"凝固模式"间来回震荡，是 v3 baseline 看似优秀但实际还有 18% 吞吐空间的根因。

#### 6.4.8 用 evalscope `custom_multi_turn` + 真实数据精准控制累计 context（2026-04-28）

随机数据 + `random_multi_turn` 的两个问题：(1) 内容不真实；(2) 累计 input 不可控（5 turn × 200K → P99 700K）。本节介绍如何用 `custom_multi_turn` 数据集 plugin + 自定义 JSONL 实现"真实数据 + 精准累计长度"。

##### 数据集 plugin 源码事实

`evalscope/perf/plugin/datasets/custom.py:32-129` `CustomMultiTurnDatasetPlugin`：
- 从本地 JSONL 读取，每行 = 一个 conv = OpenAI message dict 数组
- runner 丢弃 JSONL 里的 assistant 占位，只用模型实际响应累计 context
- 累计公式：`turn_N_input = Σ(turn_1..N user_len) + Σ(turn_1..N-1 model_output_len)`

`evalscope/perf/multi_turn_benchmark.py:23` 文档明确：**`--number` 是总 turn 数（API 请求数），不是 conv 数**。100 conv × 5 turn = `--number 500`。

##### V4-Pro tokenizer chat_template patch（必备）

V4-Pro tokenizer 不带 chat_template，evalscope `base.py:115` 的 `check_prompt_length` 会调 `apply_chat_template` 报错。最小 patch：

```python
import json
cfg_path = "/tmp/v4-tokenizer-with-chat-template/tokenizer_config.json"
# 拷贝 /lssd/models/DeepSeek-V4-Pro/tokenizer.json + tokenizer_config.json 到 /tmp/v4-tokenizer-with-chat-template/
with open(cfg_path) as f: cfg = json.load(f)
cfg["chat_template"] = '{% for m in messages %}{{ m["role"] }}: {{ m["content"] }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}'
with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
```

template 仅用于 client 端 token 长度估算，不影响发到 server 的 messages payload（OpenAI API 接收的是原始 messages 数组）。

##### 真实数据素材（B200 上 LMCache 测试遗留）

`/lssd/multi_turn_bench/data/`：
- **真实长文本**：`pg*.txt` (Project Gutenberg 经典文学，16.4 MiB) / `corpus_50m/100m/200m.txt` / `large_corpus.txt` (2.1 GiB)
- **真实多轮对话**：`sharegpt_conv_200.json` (1.8 MiB, 200 conv, 已 OpenAI 格式) / `sharegpt_raw.json` (907 MiB)

##### "200K 真实小说 base + 10K 真实问答 incr" JSONL 生成器

```python
import json, random, os
from transformers import AutoTokenizer
random.seed(42)

# PG 真实长文本 base
corpus = ""
for fp in sorted([f"/lssd/multi_turn_bench/data/{x}" for x in os.listdir("/lssd/multi_turn_bench/data") if x.startswith("pg") and x.endswith(".txt")]):
    with open(fp) as f: corpus += f.read() + "\n\n=== NEW BOOK ===\n\n"

# ShareGPT 真实短问题
sg = json.load(open("/lssd/multi_turn_bench/data/sharegpt_conv_200.json"))
sg_users = [m["content"].strip() for it in sg for m in it.get("messages", [])
            if m.get("role") == "user" and 200 < len(m.get("content","").strip()) < 3000]

tok = AutoTokenizer.from_pretrained("/lssd/models/DeepSeek-V4-Pro", trust_remote_code=True)

def take_corpus(off, n_tok):
    raw = corpus[off:off + int(n_tok * 4.5)]
    return tok.decode(tok.encode(raw, add_special_tokens=False)[:n_tok], skip_special_tokens=True)

def real_users(n_tok):
    parts, total = [], 0
    pool = sg_users[:]; random.shuffle(pool)
    for t in pool:
        parts.append(t); total += len(tok.encode(t, add_special_tokens=False))
        if total >= n_tok: break
    return tok.decode(tok.encode("\n\n".join(parts), add_special_tokens=False)[:n_tok], skip_special_tokens=True)

OUT = "/lssd/datasets/multi_turn_real_pg_sg.jsonl"
TURN_LENS = [200000, 10000, 10000, 10000, 10000]
offsets = random.sample(range(0, max(1, len(corpus)-2_000_000)), 100)
with open(OUT, "w") as f:
    for ci in range(100):
        msgs = [{"role":"user","content":take_corpus(offsets[ci], TURN_LENS[0])}, {"role":"assistant","content":""}]
        for tl in TURN_LENS[1:]:
            msgs += [{"role":"user","content":real_users(tl)}, {"role":"assistant","content":""}]
        f.write(json.dumps(msgs[:-1], ensure_ascii=False) + "\n")
```

##### 实测验证（vE diag, 5 turn × 1 conv × parallel=1）

| turn | prompt_tokens | completion | delta vs prev |
|---:|---:|---:|---:|
| 1 | 200,004 | 131 | base |
| 2 | 210,138 | 500 | +10,134 |
| 3 | 220,642 | 500 | +10,504 |
| 4 | 231,146 | 166 | +10,504 |
| 5 | 241,315 | 349 | +10,169 |

完美按 ~10.4K 单调递增（10K user + completion + chat_template overhead）。0 失败，端到端 64s。

##### 全量复现"DP 打满"测试命令

```bash
ssh maxwellx_google_com@forrest-b200-1 "evalscope perf \
  --model DeepSeek-V4-Pro \
  --tokenizer-path /tmp/v4-tokenizer-with-chat-template \
  --url http://127.0.0.1:8000/v1/chat/completions --api openai \
  --dataset custom_multi_turn \
  --dataset-path /lssd/datasets/multi_turn_real_pg_sg.jsonl \
  --min-prompt-length 1000 --max-prompt-length 250000 \
  --max-tokens 500 \
  --multi-turn --max-turns 5 \
  --number 500 --parallel 8 --seed 42"
```

预期：单 conv 累计 200K-244K（介于 vA 700K 与 vC 70K 之间），8 worker 并行 → 同时 in-flight 8 conv，KV 累计 ~2M tokens（V4 容量 3.66M，55% 占用）。预计 10-15 分钟。

##### 关键 takeaway

1. **真实数据测试必走 `custom_multi_turn` 路径**，`random_multi_turn` 是 token 噪音不真实
2. **V4-Pro tokenizer 必须 patch chat_template**（最小模板即可）
3. **`--number` 在 multi-turn 下是总 turn 数**，不是 conv 数（容易踩坑）
4. **`--max-prompt-length` 必须 >= 单 turn 最大 user 长度**，否则 conv 全被 length filter 过滤掉
5. **`--apply-chat-template` 是 client 端长度估算开关，不影响 server payload**（server 接收原 messages 数组）
6. **复用 LMCache 时遗留的 `/lssd/multi_turn_bench/data/`**：PG 经典文学 + ShareGPT 真实问答，零下载、即时可用

### 6.4.9 vE 真实数据多轮压测：parallel × balance_abs 必须匹配（2026-04-28）

#### 背景

vE 系列用 `multi_turn_real_pg_sg.jsonl`（100 conv × 5 turn，真实 PG+ShareGPT，200K base + 10K incr 累计到 P99 242K）跑 V4-Pro，目的：
1. 验证 §6.4.8 的真实数据 200K+10K incr 模式是否触发 DPA 集群慢化
2. 调出 router `balance_abs/rel` 与 client `parallel` 的最佳组合
3. 给生产推荐参数

#### 实测对比表（同数据集，同 seed=42）

| 版本 | parallel | abs | rel | DP 占用观察 | 备注 |
|---|---|---|---|---|---|
| **vE-v1** | 8 | **16** | 1.2 | 仅 DP1/2/6（3/8）凝固，DP6 token 80%、5 个 DP 完全空闲 | 凝固模式（手动停） |
| **vE-v2** | 8 | **4** | 1.2 | 5-7/8 DP 在跑，max-min=1-3，sum 多在 6-8（< parallel=8） | abs 起作用但 parallel=8 给的 in_flight 不足以填满 8 DP（手动停） |
| **vE-v3** | **16** | **8** | 1.2 | **持续 7-8/8 DP 在跑，sum 12-16 接近 parallel；max-min 多在 2-5 < abs=8** | ✅ 完整跑完 33.9 min |

**vE-v3 最终结果**：

| 指标 | 值 |
|---|---|
| Time taken | 2032.11s (33.87 min) |
| Success rate | 500/500 (100%) |
| RPS | 0.246 req/s |
| Avg input tokens | 220,191（P99 242,020 = 200K base + 4×10K incr）|
| Avg output tokens | 361.3（max=500, avg turns/req=2.95）|
| Total throughput | 54,267 tok/s |
| Output throughput | 88.91 tok/s |
| **Avg latency** | **64.83s** |
| **P99 latency** | **151.19s** |
| **Avg TTFT** | **9.73s** |
| **P99 TTFT** | **67.97s** |
| **Avg TPOT** | **0.155s (155 ms)** |
| **P99 TPOT** | **0.392s (392 ms)** |
| evalscope approx cache hit | 95.55%（prefix 估算）|
| **SGLang counter 真实 cache hit** | **78.01%**（GPU 实测）|

**真实 cache hit 计算**（counter delta 法）：
- T0: prompt=160,880,219 / cached=101,399,296（TS=1777350670）
- T1: prompt=270,975,558 / cached=187,284,992（TS=1777352796）
- ΔPrompt = 110,095,339；ΔCached = 85,885,696
- **真实 cache hit = 85.89M / 110.10M = 78.01%**

**evalscope 95.55% vs counter 78.01% 差距来源**：evalscope 用 prefix 长度估算（不知道 evict），counter 是 schedule 时 GPU 实际复用，差 17.5pp 说明长上下文累积下大量 KV 被 evict 出 pool。

#### 核心结论

1. **`balance_abs ≤ parallel × 0.5` 才能在多 DP 间真正触发 fallback**（abs=8 配 parallel=16，max-min=8 时触发）。abs 等于或大于 parallel 等价于"凝固模式"。
2. **`parallel` 必须是 `dp_size` 整数倍**（vE-v3 用 16=2×8 完美 8 DP 填满）。非整除（如 10:8）会让 cache_aware 在凝固模式间震荡。
3. **真实 200K base + 10K incr 数据下，DPA 集群没有触发明显慢化**：P99 TTFT 68s 在可接受范围（vA 在凝固+长 context 累计到 700K 时 P99 TTFT 达 352s+）。说明 DPA cluster slowdown 主要由"单 DP queue 堆积导致 DeepEP barrier 拖累"触发，把负载分散到 8 个 DP 后该效应消失。
4. **EAGLE/MTP 完全没生效**：`Avg decoded tokens per iter = 0.99`，`Spec accept rate = 0.0`。需另查（可能 router/SMG 路径下 EAGLE 不工作，或 V4-Pro 配方未启用）。

#### 推荐参数组合（生产）

| 场景 | parallel | abs | rel | cache_threshold | 备注 |
|---|---|---|---|---|---|
| 低并发 + 强亲和 | 8 | 4 | 1.2 | 0.5 | parallel=8 时 sum < 8 是常态，用 8 个 client 难填满 8 DP |
| **均衡（推荐）** | **16** | **8** | **1.2** | **0.5** | 8 DP 持续 7-8 全占满 + 真实 cache hit 78% |
| 高并发 | 24-32 | 6-4 | 1.15-1.1 | 0.3-0.5 | 触发 abs 更激进 + 部分 KV cache evict 接受 |

**重启 router 命令**（已验证）：

```bash
sudo docker stop sglang-router && sudo docker rm sglang-router && \
sudo docker run -d --name sglang-router --restart no --network host \
  -v /lssd/models:/lssd/models:ro \
  lmsysorg/sgl-model-gateway:latest \
  --worker-urls http://127.0.0.1:8088 \
  --model-path /lssd/models/DeepSeek-V4-Pro \
  --tokenizer-path /lssd/models/DeepSeek-V4-Pro \
  --reasoning-parser deepseek-v4 --tool-call-parser deepseek \
  --dp-aware --policy cache_aware \
  --cache-threshold 0.5 \
  --balance-abs-threshold 8 \
  --balance-rel-threshold 1.2 \
  --prometheus-port 29000 --host 0.0.0.0 --port 8000
```

### 6.4.10 vE-v5 客户端 dp_rank 路由：数据集多样性陷阱（2026-04-28）

#### 背景

vF 短数据测试已证明客户端 hash(session)→data_parallel_rank 在 V4-Pro DPA+DeepEP 下 cache hit 22%→77%（3.49×）。vE-v5 想用 wrapper 跑完整 200K real data + parallel=16 + 100 conv，期望复制甚至超过 vE-v3 router cache_aware 的 78% cache hit + DP 均衡度。

#### 实测结果

| 指标 | vE-v3 router cache_aware (abs=8) | **vE-v5 客户端 dp_rank** |
|---|---|---|
| Time taken | 33.9 min | **79.1 min (+2.34×)** |
| Success | 500/500 | 500/500 |
| RPS | 0.246 | 0.105 (−57%) |
| Avg latency | 64.83s | 150.32s (+2.32×) |
| **P99 latency** | 151.19s | **421.17s (+2.79×)** |
| Avg TTFT | 9.73s | **59.16s (+6.08×)** |
| **P99 TTFT** | 67.97s | **266.62s (+3.92×)** |
| Avg TPOT | 0.155s | 0.269s (+73%) |
| P99 TPOT | 0.392s | **1.044s (+2.66×)** |
| Avg ITL | 0.152s | 0.252s |
| Total throughput | 54,267 tok/s | 23,210 tok/s |
| **Real cache hit (counter)** | **78.0%** | **50.66%** |
| evalscope approx cache hit | 95.55% | 95.55% |

**vE-v5 实际比 vE-v3 慢 2.34×、cache hit 低 27pp**。

#### 真因：数据集 messages[0] 多样性不足

```
$ python3 -c 'hash check on /lssd/datasets/multi_turn_real_pg_sg.jsonl'
total conv: 100
unique FULL messages[0] content: 17     ← PG base passage 重复
content_len: avg 810,127 bytes
hash(FULL content) %8 distribution: {1:18, 2:12, 3:24, 4:11, 5:24, 6:11}
unique hash count: 6                    ← DP0/DP7 完全没分到
```

100 conv 共享 17 个 PG base passage（生成时选材有限），即使 hash 整个 800K 字节也只有 17 个不同 hash → 落 6 bin → DP3/DP5 各扛 24 conv 严重排队 → P99 TTFT 飙升。

监控期 vE-v5 server 端 num_running 持续显示 DP3/DP5 高占用 + DP0/DP7 长期 queue 堆积，与 atexit 报告 dp_rank 分布完美对应。

#### 对照 wrapper 链路验证（vE-v5 之前的 diag wrapper 测试）

双层 patch 加 process_request 验证 wire body 100% 带 `data_parallel_rank` 字段，server stream chat 路径也读字段（短测试 cache hit 39% > round_robin 22%）。**wrapper 工作完全正常**，问题完全在数据集 hash 分布。

#### 生产建议

| 场景 | 推荐方案 |
|---|---|
| **生产真实流量**（用户 conv 内容多样） | ⭐ 客户端 hash dp_rank（最便宜，零路由组件）|
| **Benchmark / 数据集 conv 来源单一** | ⭐⭐ router cache_aware (vE-v3 配方)，不依赖 hash 均匀性 |
| 需要 wrapper 又怕 hash 不均 | 跑前 sanity check：`hash(messages[0]) % dp_size` 必须散到全部 8 bin |

#### 数据集修复脚本（重生成 100 个不同 base）

未实施，参考方向：
```python
# 每 conv 用不同 PG offset，确保 messages[0] 全 100 unique
for i in range(100):
    base_offset = (i * 7919) % len(pg_corpus)  # 大质数避免 cluster
    base_text = pg_corpus[base_offset:base_offset + 200_000_chars]
    # ... build conv with this unique base
```

### 6.5 注意事项

- evalscope 默认不传该字段（用 round_robin），做 cache 对比测试用 B200 上已有的 wrapper：`/home/maxwellx_google_com/code/b200_vllm_opt/run_evalscope_dp.py`（monkey-patch `OpenaiPlugin.build_request`，无需改 evalscope 源码）
- ⚠️ **跑前必须 sanity check 数据集 messages[0] hash %dp_size 散到全部 8 bin**，否则会落少数 DP 严重排队（vE-v5 反例：100 conv 仅 17 种 messages[0] → 6 bin → 慢 2.34×）
- 同 session 钉同 slice 会让该 slice 负载偏高（如某 session 是热点）；可加二级散列降尖峰
- 所有 LoadBalanceMethod (round_robin / shortest_queue / decode_round_robin / minimum_tokens) 都被这个字段优先短路
- ⚠️ **验证字段是否生效必须用批量 ≥20 conv + counter delta**，不能 curl 单 shot 看 num_running（DPA barrier + chunking 让瞬时不准）

---

## 7. 总结判断

### 7.1 在不改拓扑前提下能否让 8 个 slice 共享 prefix cache？

**源码层面不存在让 8 个 slice 共享 GPU prefix cache 的路径**：
- RadixTree 是进程内 Python 对象（§1.3）
- 路由层不查 prefix（§2.1）
- KV pool 是 per-process GPU 显存（§1.2）

唯一"曲线救国"是开 `--enable-hierarchical-cache --hicache-storage-backend=...`，让 host pool 经外部 KV store 间接共享，但：
- 不是真共享，有拷贝开销
- 需要外置存储
- V4 走 HiSparseCoordinator，host pool 硬编码 `device.size_full / 4`，能否如预期 spill 到外部存储待验证

### 7.2 提升 multi-turn cache hit 的根本路径

1. **router separate + `--dp-aware`**（⭐⭐ 生产首选）
   - 加 1 个 router 容器（无 GPU），后端配方完全不变
   - router 自动探测后端 `dp_size`，cache_aware 在虚拟 worker 间真实 token-level prefix tree
   - 跨 session 共享 system prompt 的 prefix 也能命中
   - **客户端零改动**，直接打 router 端口
   - 实测 cache hit 2.86% → 46.85%

2. **客户端 session 亲和路由**（次选，**有适用条件**）
   - 在请求体里按 `hash(session_id) % dp_size` 设 `data_parallel_rank`
   - 走 `maybe_external_dp_rank_routing` (`data_parallel_controller.py:500-505`)
   - **零服务端改动**，把同会话所有 turn 钉同一 slice，本地 RadixTree 直接命中
   - 短数据/合成 session 实测 cache hit 22% → 77%（vF）
   - ⚠️ **vE-v5 反例**：messages[0] 多样性不足（100 conv 共享 17 种）时，hash 落少数 bin → cache hit 50.66% / 慢 router cache_aware 2.34×；生产真实流量通常 OK，benchmark 数据集易掉坑
   - **适用条件**：messages[0] (或 hash 输入字段) 的 unique 数 ≥ dp_size，理想 unique 数 ≥ 5×dp_size
   - 不能跨 session 命中

3. **加 hicache + storage backend**
   - `--enable-hierarchical-cache --hicache-storage-backend file`
   - 靠外部 KV store 跨 slice 互相 prefetch
   - 可缓解但拷贝开销高
   - V4 路径需验证 HiSparseCoordinator 是否会调用 storage backend

4. **放弃 DP attention，改 N 个独立 server + sglang/router cache_aware**
   - 拓扑级重构，开发与运维成本最高
   - 现已不必要（路径 1 已解决）

### 7.3 生产建议

**multi-turn / agent / 长 system prompt 场景**：
- ⭐⭐ **首选 router cache_aware (vE-v3 配方)**：完整参数见 §6.4.9 表格 + §7.4 命令
  - `--dp-aware --policy cache_aware --cache-threshold 0.5`
  - `--balance-abs-threshold 8 --balance-rel-threshold 1.2`（对应 client `parallel=16`）
  - 实测 cache hit 78%, P99 TTFT 68s, 33.9 min, 0 失败 — 对数据集多样性鲁棒
- ⭐ **次选客户端 hash dp_rank**：仅当 messages[0] hash 散到全 dp_size bin 时（生产真实流量通常 OK）
  - benchmark 数据集易掉坑，必须先 sanity check（详见 §6.4.10）
- benchmark 时必须传字段或经过 router，否则 cache hit 数据没意义

**单 turn / 无亲和需求场景**：
- 默认 round_robin 即可，不必引入额外复杂度

**Client `parallel` × router `balance_abs` 配档**（详见 [reference_router_concurrency_param_combos.md]）：
- `parallel=16` → `abs=8 + rel=1.2`（vE-v3 实测最优）
- `parallel=24` → `abs=6 + rel=1.15`
- `parallel=32` → `abs=4 + rel=1.1`
- `parallel ≤ 10` → balance 永远不触发，只调 cache_threshold
- 硬性约束：`balance_abs ≤ parallel × 0.5` + `parallel` 必须是 `dp_size` 整数倍

### 7.4 router 部署速查（vE-v3 推荐配方，2026-04-28 实测最优）

```bash
sudo docker run -d --name sglang-router --restart no --network host \
  -v /lssd/models:/lssd/models:ro \
  lmsysorg/sgl-model-gateway:latest \
  --worker-urls http://127.0.0.1:8088 \
  --model-path /lssd/models/DeepSeek-V4-Pro \
  --tokenizer-path /lssd/models/DeepSeek-V4-Pro \
  --reasoning-parser deepseek-v4 --tool-call-parser deepseek \
  --dp-aware --policy cache_aware \
  --cache-threshold 0.5 \
  --balance-abs-threshold 8 \
  --balance-rel-threshold 1.2 \
  --prometheus-port 29000 \
  --host 0.0.0.0 --port 8000
```

**关键参数说明**：
- `lmsysorg/sgl-model-gateway:latest`（SMG image, native Rust）vs `lmsysorg/sglang:...` 的 Python wrapper：实测耗时 −27%、cache hit +9pp（详见 §6.4.2）
- `--dp-aware`：自动探测 backend `dp_size` 并展开 8 个虚拟 worker
- `--policy cache_aware`：跨 worker 的 token-level radix prefix tree
- `--cache-threshold 0.5`：multi-turn sweet spot（v6 实测 0.6 反降 5pp）
- `--balance-abs-threshold 8 --balance-rel-threshold 1.2`：对应 client `parallel=16`，`abs ≤ parallel × 0.5` 才能触发 fallback

router 启动 ~1.5 秒（无模型加载）；后端配方不变；evalscope/客户端改打 `:8000` 即可。

---

## 附：源码位置速查

| 模块 | 文件 | 关键行 |
|---|---|---|
| DP 进程启动 | `managers/data_parallel_controller.py` | 393-489 |
| 客户端 dp_rank 短路 | `managers/data_parallel_controller.py` | 500-505 |
| round_robin 调度 | `managers/data_parallel_controller.py` | 507-515 |
| DP attention world 计算 | `layers/dp_attention.py` | 227-235 |
| Scheduler init / KV pool | `managers/scheduler.py` | 546-693 |
| Prefix lookup | `managers/schedule_batch.py` | 860 |
| RadixCache 实现 | `mem_cache/radix_cache.py` | 252-569 |
| NSA/MLA KV pool | `mem_cache/memory_pool.py` | 1813-1888 |
| Evict 触发 | `mem_cache/common.py` | 230-253 |
| Hierarchical cache | `mem_cache/hiradix_cache.py` | 35-117 |
| ep_size 强制覆盖 | `srt/server_args.py` | 1860-1865 |
| OpenAI 协议字段 | `entrypoints/openai/protocol.py` | 261, 538 |
| OpenAI 字段透传 | `entrypoints/openai/serving_chat.py` | 214 |
| OpenAI 字段透传 | `entrypoints/openai/serving_completions.py` | 119 |
| Engine 校验 | `entrypoints/engine.py` | 232-244 |
| SMG launch_server | `sgl-model-gateway/.../launch_server.py` | 86-95 |
| cache_hit_rate gauge bug | `scheduler_metrics_mixin.py` | 378 |

## 附：实测脚本

`multi_turn_dp_aware.py`（仓库 root）：mode = `round_robin` / `dp_rank` 双模式压测，固定 prompt 生成保证两轮可比性。
