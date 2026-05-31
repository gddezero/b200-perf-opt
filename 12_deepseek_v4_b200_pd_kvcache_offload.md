# DeepSeek V4-Pro PD 分离 + KV Offload 全家桶压测（B200 × 8）— vLLM Mooncake 版

> **测试时间**：2026-05-30 ~ 2026-05-31
> **硬件**：GCP A4 B200 SXM × 8 × 2 机（prefill `forrest-b200-01` + decode `forrest-b200-02`，183 GB HBM3e/卡，driver 590.48.01）
> **软件**：vLLM `0.21.1rc1.dev323+g1fc2cee50`，镜像 `vllm-v4-mooncake:5_27-pytest-pd` (`sha256:6efe9c06...`)；Mooncake `0.3.10.post2`
> **全家桶** = **PD 分离 + Prefix Cache + Mooncake Store（L2 CPU pool）+ Mooncake Transfer（P→D RDMA）**
> **配方**：`docker_vllm_fp8_{prefill,decode}_pd_mooncake.sh` + `docker_vllm_router.sh`（MultiConnector，DP=8 EP，MTP 关）
> **数据集**：`shared_prefix_{60,100,200,400}k_30turn.jsonl`（40k 字节级共享前缀 + nonce + filler，30 turn）
> **压测工具**：`batch_multi_turn_pd.sh`（evalscope perf custom_multi_turn，`X-Session-ID` consistent_hash 亲和，WARM=1 冷热各一）
> **关联文档**：[11_..._sglang_hicache.md](11_deepseek_v4_b200_multi_turn_sglang_hicache.md)（SGLang HiCache 单机方案，物理对照组）、[10_..._multi_turn.md](10_deepseek_v4_b200_multi_turn.md)（无 offload 基线）

---

# 第一部分：测试结果

## 1. 总览（16 档 4 dataset × 4 并发，ID #267-283）

| dataset | parallel 跑了 | 安全并发上限 | thrashing 点 | 全家桶状态 |
|---|---|:---:|:---:|---|
| 60K  | 8,16,32,64 | **64**（dec-L1 93%，仍健康）| 未撞（4.0M < 墙）| 全档 prefill 接力健康 |
| 100K | 8,16,32,64 | **32**（p64 dec-L1 降到 62%）| 未撞（6.4M）但 p64 起 decode-L1 受压 | 同上 |
| 200K | 8,16,32,64 | **32** | **p64 = 12.8M 🔴** | p64 decode 退化为全程 RDMA load |
| 400K | 8,16,24,32 | **24**（=9.6M 安全）| **p32 = 12.8M 🔴**（PARTIAL）| p24 健康 / p32 崩 |

**核心结论（6 条，详细见第三部分）**：

1. ⭐⭐ **Thrashing 容量墙 ≈ working set 12.8M token·conv**（working set = 并发 × 上下文K）。撞墙档 wall 爆 ~6×、decode-L1 hit 崩到 7-10%；安全最高点 400K×24 = 9.6M（wall 1158s，decode-L1 95.8%）。墙落在 **9.6M（安全）~ 12.8M（崩）** 之间。
2. ⭐⭐ **崩盘机制是 GPU-L1/HBM 活跃集装不下，不是 Mooncake Store 容量耗尽**：Store 总 3.43 TB 仅用 ~2.0 TB（未满）。working set 超过 8×B200 HBM 能 hold 的 decode 活跃 KV → decode-L1 崩 → decode 每 step 全程从 Store 走 RDMA load → Store-I/O 占 wall 40%+ → wall 爆。
3. **全家桶 prefill 缓存接力**：每长度内随并发上升，prefill GPU-L1 hit 单调降、Mooncake-Store ext hit 单调升（100K：L1 94→20% / ext 17→95%），合计命中维持高位 —— 这是 Store 组件在高并发下的核心价值（避免重算长 prompt）。
4. **decode-L1 由「并发数」与「working set」双驱动**：同 6.4M working set，100K×64（并发 64）decode-L1 掉到 62%，而 200K×32 / 400K×16 仍 95% —— 隔离出并发数本身对 decode KV 的额外压力。
5. ⭐ **KV offload 价值高度依赖 workload**（对照 #284-287）：prefill-bound（长 prompt 短输出高复用）下 offload 是 wall 主引擎（p64 −40%）；decode-bound（长多轮对话）下被 decode 时间稀释（≤10%）。不是「作用有限」，是「按 workload 定」。
6. **MTP 与 prefix cache 互斥**（0.22 实测）：开 MTP 则 L1+Store 全 0% 命中、每轮重算，multi-turn 净亏 → **生产 = no-MTP**。

## 2. 完整数据矩阵

**编号映射**：60K #271-274 · 100K #267-270 · 200K #275-278 · 400K #279/280/283(p24) · #281 = 400K p32（thrashing，PARTIAL 跑到 72% 止损留底）

### 2.1 wall 时长（冷 warmup / 热 warm，秒）

| 长度＼并发 | p8 | p16 | p32 | p64 |
|---|---|---|---|---|
| 60K  | 480/465 | 589/599 | 779/760 | 1099/1069 |
| 100K | 599/499 | 641/621 | 828/835 | 1424/1277 |
| 200K | 594/574 | 790/735 | 1020/961 | **6273/7457** 🔴 |
| 400K | 809/702 | 1071/950 | **6093** 🔴(p32, PARTIAL) | (p24) 1158/1286 ✅ |

### 2.2 prefill 缓存命中（冷，GPU-L1 hit / Mooncake-Store ext hit，%）

| 长度＼并发 | p8 | p16 | p32 | p64 |
|---|---|---|---|---|
| 60K  | 94.2 / 29.5 | 83.4 / 75.9 | 67.8 / 87.6 | 24.0 / 94.5 |
| 100K | 94.3 / 16.9 | 86.8 / 72.2 | 63.5 / 90.0 | 20.4 / 95.4 |
| 200K | 92.7 / 49.7 | 76.7 / 87.7 | 54.0 / 93.7 | 18.8 / 96.2 |
| 400K | 91.6 / 62.6 | 66.8 / 93.5 | 34.8 / 95.5 | (p24) 53.4 / 97.5 |

→ **接力规律**：L1 单调降、ext 单调升，合计始终高位。GPU-L1（HBM）装不下的 prefix 被 Mooncake-Store（CPU pool）接住。

### 2.3 decode-L1 hit（冷，%）/ decode 聚合吞吐（冷，tok/s）

| 长度＼并发 | p8 | p16 | p32 | p64 |
|---|---|---|---|---|
| 60K  | 94.8 / 195 | 95.0 / 268 | 94.8 / 398 | 93.0 / 566 |
| 100K | 95.2 / 104 | 95.5 / 274 | 95.4 / 327 | **62.3** / 452 |
| 200K | 95.7 / 131 | 95.7 / 233 | 94.8 / 251 | **10.4** / **99** 🔴 |
| 400K | 95.8 / 115 | 95.3 / 157 | **7.2** / **34** 🔴(p32) | (p24) 95.8 / 199 |

→ **decode-L1 是 thrashing 的真判据**：安全档稳定 93-96%；撞墙档（200K×64、400K×32）暴跌到 7-10%，decode 吞吐随之塌（99、34 tok/s）。

### 2.4 横向规律

1. **wall 在安全区随并发亚线性增长**（吞吐换延迟），撞墙后**超线性爆 6×**（200K×32 → ×64 = 1020 → 6273s）。
2. **prefill GPU-L1 hit 随并发单调降、Store-ext 单调升**，且**上下文越长接力起点越早**（400K p8 ext 已 62.6%，60K p8 仅 29.5%）。
3. **decode-L1 在安全区与长度/并发无关，恒 93-96%**；只有 working set ≥ 12.8M 或并发≥64 才掉。
4. **decode 吞吐安全区随并发单调升**（60K：195→566 t/s，p8→p64 +2.9×）→ decode 在安全区**未饱和**。
5. **冷热收益不强单调**：低并发热收益明显（100K p8 −17%），中高并发被 decode 饱和稀释甚至反转（200K p64 +19%，thrashing 高方差）。

> ## ⚠️ 方法论 caveat — 输出 token 未固定（影响冷热精确数字，不影响容量墙）
> 本轮 16 档 `--max-tokens 500` 是**上限**，脚本**未设 `ignore_eos`**，模型自然 stop → 实测每轮输出 **257–458 tok 波动**（小样本档 p8 尤甚：#267 p8 cold 257 / warm 458；大样本 p64 cold/warm 335/337 几乎一致）。
> - ❌ **不影响**：容量墙 / thrashing / HBM 活跃集机制 / Mooncake 源码结论（判据均不依赖输出长度）。
> - ⚠️ **影响**：**冷热 wall 收益的精确数字含采样噪声**，小样本档尤甚。`§10` prefill-bound 对照档输出≈固定（接近 max50 上限）相对干净，−40% vs −10% 的 4× 差距远超噪声、定性成立。
> - 彻底解耦需 `ignore_eos=true` + 固定输出重跑（`batch_multi_turn_pd.sh` 已加 `IGNORE_EOS` 支持，2026-05-31，见 §16.1）。

---

# 第二部分：可复现的测试方法与步骤

## 3. 硬件与模型

### 3.1 硬件（双机 PD）

| 角色 | 主机 | GPU | 网络 |
|---|---|---|---|
| Prefill (kv_producer) | `forrest-b200-01` | 8× B200 183 GB | 8× CX-7 mlx5 RDMA（物理 400 Gbps/卡）|
| Decode (kv_consumer) | `forrest-b200-02` | 8× B200 183 GB | 同 |
| Router | `forrest-b200-01:8080` | — | consistent_hash cache-aware |

主机网络优化（实测在生效，见 memory `reference_gdr_nvidia_peermem_pd` / `reference_gcp_sysctl_irq_mlnx_tune`）：GDR `nvidia_peermem` / sysctl BBR+fq+128MB rmem-wmem / IRQ per-NUMA pin / mlnx_tune / 8 RDMA active ports。

### 3.2 模型 + KV pool

DeepSeek-V4-Pro FP8（1.6T total / 49B active，sparse MLA + lightning indexer + MoE）。两端各 `--data-parallel-size 8 --enable-expert-parallel`（DP=8 EP），`--kv-cache-dtype fp8`，`--block-size 256`，`--max-model-len 1048576`，`--gpu-memory-utilization 0.85`。

⚠️ **block_size 必须 256**：V4 nvidia `model.py` 硬编码 256 假设，512 撞 model storage layout bug（详见 memory `reference_mooncake_tuning_sweep`）。

## 4. PD 部署：MultiConnector 全家桶（本次配方）

### 4.1 架构

```
client → vllm-router (b200-01:8080, consistent_hash + X-Session-ID 亲和)
  ├─ phase 1 (prefill):      → b200-01:8001 (kv_producer)
  ├─ phase 2 (KV transfer):  MooncakeConnector P2P RDMA → b200-02
  │      并行: MooncakeStoreConnector kv_both 写 CPU pool (跨 turn 持久化)
  └─ phase 3 (decode):       → b200-02:8002 (kv_consumer)
```

connector 组合（vLLM 官方 doc 推荐 PD）：
```
MultiConnector
  ├─ MooncakeConnector       (P→D direct RDMA transfer，单请求 KV 接力)
  └─ MooncakeStoreConnector  (CPU pool L2，跨 turn 共享 prefix 复用)
```

### 4.2 配方文件 + 关键参数

`gpu/b200/inference/recipes/deepseek_v4_pro/pd_disaggregation/`：

| 文件 | 角色 | 端口 | kv_role |
|---|---|---|---|
| `docker_vllm_fp8_prefill_pd_mooncake.sh` | prefill | 8001 | kv_producer |
| `docker_vllm_fp8_decode_pd_mooncake.sh` | decode | 8002 | kv_consumer |
| `docker_vllm_router.sh` | router | 8080 | — |

两端共同关键参数：

| 参数 | 值 | 备注 |
|---|---|---|
| `--kv-transfer-config` | `MultiConnector(MooncakeConnector + MooncakeStoreConnector)` | 两端 kv_role 与外层一致 |
| `--enable-prefix-caching` | 开（两端）| prefill 端典型 0% hit（producer release）但 honor |
| `--moe-backend` | `deep_gemm_mega_moe` | |
| `--attention_config.use_fp4_indexer_cache` | `True` | V4 sparse indexer |
| `--served-model-name` | `DeepSeek-V4-Pro` | ⛔ 必须显式（否则 404 + evalscope 崩，见 §16.2）|
| `-e MC_IB_PCI_RELAXED_ORDERING` | `1` | Mooncake 唯一净 win env（双端，见 memory `reference_mooncake_4env_sweep`）|
| `--speculative-config` (MTP) | **不加** | MTP 与 prefix cache 互斥（见 §13）|
| decode 额外 | `--max-cudagraph-capture-size 512` | 防 cudagraph capture OOM |

### 4.3 Mooncake 配置

`configs/mooncake/mooncake_config.json`（bind mount 进两端容器）：
- `protocol: rdma`，8× mlx5 1-to-1 GPU，`metadata_server: P2PHANDSHAKE`
- `master_server_address: forrest-b200-01:50063`（独立 mooncake-master 进程）
- `global_segment_size: 200 GB × 16 worker = 3.43 TB` 总容量（来源：config + master metric `master_total_capacity_bytes`）
- `local_buffer_size: 16 GB`

### 4.4 镜像（注意版本边界）

主 16 档 sweep（#267-283）跑在 **5/27 custom image** `vllm-v4-mooncake:5_27-pytest-pd`（vLLM 0.21.1rc1.dev323，= 0.22−19commits）。

⚠️ **现生产已升级 0.22 正式 release**：`vllm-v022-pytest`（FROM `vllm/vllm-openai:v0.22.0` + pip pytest；官方 image 已含 mooncake 0.3.10.post2）。0.22 增量小但含 #43719（connector spec-decode 修复）。容量墙/接力等机制结论与版本无关，0.22 仍适用。MTP 互斥（§13）+ MULTI_STREAM A/B（§14）在 0.22 上验证（详见 memory `reference_vllm_022_upgrade_mtp_cache_exclusive`）；**prefill-bound 对照（§10 / #284-287）与主 16 档同属 0.21 5/27 image**（与主 sweep 同期跑的 workload 对照，非 0.22）。

### 4.5 启动顺序 + 健康检查 + smoke

⛔ **顺序强制 decode → prefill → router**（NIXL/Mooncake handshake 依赖，任一节点重启必须整链路重启，见 §16.5）：

```bash
# 1. decode (b200-02) 先起
ssh maxwellx_google_com@forrest-b200-02 'bash ~/inference/recipes/deepseek_v4_pro/pd_disaggregation/docker_vllm_fp8_decode_pd_mooncake.sh'
ssh maxwellx_google_com@forrest-b200-02 'until curl -sf http://localhost:8002/health; do sleep 5; done'

# 2. prefill (b200-01)，~170s ready
ssh maxwellx_google_com@forrest-b200-01 'bash ~/inference/recipes/deepseek_v4_pro/pd_disaggregation/docker_vllm_fp8_prefill_pd_mooncake.sh'
ssh maxwellx_google_com@forrest-b200-01 'until curl -sf http://localhost:8001/health; do sleep 5; done'

# 3. router (b200-01)
ssh maxwellx_google_com@forrest-b200-01 'bash ~/inference/recipes/deepseek_v4_pro/pd_disaggregation/docker_vllm_router.sh'

# 4. smoke (打 router 全链路 PD)
curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4-Pro","messages":[{"role":"user","content":"用一句话介绍你自己"}],"max_tokens":60}'
# 验证 id 含 ___prefill_addr_...___decode_addr_... → PD 链路真通
```

## 5. 数据集（shared_prefix 30turn）

`gen_shared_prefix_datasets.sh` 生成到远程 `~/inference/datasets/`（不进 git）。每 conv 30 turn：
- **turn-1** = 40K 字节级共享前缀（所有 conv 一致，触发跨 conv prefix 复用）+ nonce（防完全去重）+ filler 到目标长度（60K/100K/200K/400K）
- **turn 2-30** 每轮 +1000~10000 tok 增量

| 数据集 | turn-1 长度 | 文件 |
|---|---:|---|
| `shared_prefix_60k_30turn.jsonl`  | 60K  | `~/inference/datasets/` |
| `shared_prefix_100k_30turn.jsonl` | 100K | 同 |
| `shared_prefix_200k_30turn.jsonl` | 200K | 同 |
| `shared_prefix_400k_30turn.jsonl` | 400K | 同 |

⛔ **datasets/ 是 pull/push 盲区**：不在 `sync_inference.sh push` 的 protect 列表 + 不进 git → push 的 `--delete` 会删远程数据集。压测期间**绝不跑 push**（见 §16.3）。

## 6. batch_multi_turn_pd.sh（PD 版批量压测）

每档自动：flush → prom snapshot before → 启 sampler → evalscope perf → kill sampler → snapshot after → finalize（`metrics/prom_<NNN>_<TAG>_p<P>.json` + `_summary.txt`）。

```bash
ID_START=267 PLIST="8 16 32 64" \
DATASET=~/inference/datasets/shared_prefix_100k_30turn.jsonl \
MAX_PROMPT=360000 TAG=mooncake_100k WARM=1 \
bash shared/multi_turn_bench/scripts/batch_multi_turn_pd.sh
```

| 参数 | 含义 |
|---|---|
| `WARM=1` | 每档跑两遍：冷 warmup + 热 warm，均落盘（冷 = `_warmup` summary）|
| `number=parallel` | 纯并发（evalscope 注入 `X-Session-ID`，consistent_hash session 亲和）|
| `MAX_TURNS` / `MAX_TOKENS` | 默认 30 / 500 |
| `IGNORE_EOS=1`（新增） | `--min-tokens MAX_TOKENS + extra-args {"ignore_eos":true}` 固定输出长度（见 §16.1）|

## 7. 编号映射 + 端到端时间线

| 编号 | dataset | parallel | tag |
|:---:|:---:|---|---|
| #267-270 | 100K | 8,16,32,64 | `mooncake_100k` |
| #271-274 | 60K  | 8,16,32,64 | `mooncake_60k` |
| #275-278 | 200K | 8,16,32,64 | `mooncake_200k` |
| #279,280,283 | 400K | 8,16,24 | `mooncake_400k` |
| #281 | 400K | 32 | PARTIAL（thrashing 跑到 72% 止损）|
| #284-287 | 100K | 8,16,32,64 | `prefillbound_100k`（§10 对照）|

⚠️ #282 跳号合规（编号一次性）：400K 实跑 p8(#279) / p16(#280) / p32(#281, PARTIAL) / p24(#283) 共 4 档，#282 为原计划 p64 槽位（标准 PLIST `8 16 32 64` 第四档，撞 thrashing 后未跑、改补 p24=#283）。

---

# 第三部分：分析、总结、踩坑

## 8. ⭐⭐ 容量墙机制：GPU-L1/HBM 活跃集，不是 Store 容量

### 8.1 thrashing 实证（working set ≈ 12.8M）

| 档位 | working set | wall | decode-L1 | decode Store-I/O | 判定 |
|---|---:|---:|---:|---:|:---:|
| 200K×32 | 6.4M | 1020s | 94.8% | — | ✅ 健康 |
| **200K×64** | **12.8M** | **6273s** (6.1×) | **10.4%** | **2482s (40% wall)** | 🔴 thrashing |
| **400K×32** | **12.8M** | **6093s** (PARTIAL 72%) | **7.2%** | **2546s (42%)** | 🔴 thrashing |
| 400K×24 | 9.6M | 1158s | 95.8% | — | ✅ 安全最高点 |

→ 墙落在 **9.6M（安全）~ 12.8M（崩）** 之间。

### 8.2 机制链条

```
working set 超过 8×B200 HBM 能 hold 的 decode 活跃 KV
  → decode-L1 hit 崩到 7-10%（活跃集被挤出 HBM）
  → decode 每个 step 都要从 Mooncake-Store 走 RDMA load 历史 KV
  → Store-I/O 占 wall 40%+（decode 被 RDMA 延迟拖死）
  → prefill GPU idle 排队 → wall 超线性爆 6×
```

### 8.3 ⚠️ Store 容量未满 + eviction 不是判据

- **Store 总 3.43 TB 仅用 ~2.0 TB**（#278 末态快照），**未满** → 排除「Store 容量耗尽」假说。
- **eviction（`master_key_count` 负 delta）不是 thrashing 判据**：健康档 400K×24 = −75350、60K×32 = −61423 的净 evict **比 thrashing 档（−60806 / −22695）更负**。evict 是 Store 常态容量管理，与 thrashing 无关。
- ✅ **真判据**：`wall 爆 ~6× + decode-L1 < 15% + decode Store-I/O 占比 ~40%` 三者同现。

> （本结论经 fact-check sub-agent 第 1 轮纠正：我一度把 evict 当 thrashing 信号、把机制归因为 Store 容量；订正为 HBM 活跃集墙。）

## 9. Mooncake KV 访问路径源码确认（为什么 decode 走 RDMA 而非本地 PCIe）

**问题**：decode 端本地也有 Mooncake CPU pool segment，为何 thrashing 时不走本地 PCIe 取 KV，反而全程 RDMA？

读 Mooncake v0.3.10 源码（commit `59c9f90`，subagent 调研 + 本机 `/tmp/mooncake_src` 抽验关键行）三点：

1. **Placement 无本地亲和**：默认 `RandomAllocationStrategy`（`mooncake-store/include/allocation_strategy.h:201,602`）随机对称分配到全部 16 segment（两机）—— **全树无 consistent_hash/ketama/hash_ring**（订正旧 memory 的「对称 hash」措辞）。写入不看「谁来 get」，`prefer_alloc_in_same_node`/`preferred_segment` 默认关；GET 取第一个 COMPLETE 副本、不优选本地 → 约半数 KV 物理落在远程机。
2. **本地 segment 也走 RDMA loopback（经本机 NIC），不是 memcpy/PCIe**：`MultiTransport::selectTransport`（`multi_transport.cpp:343-365`）只按 `protocol` 选 transport、**无 same-node 分支**；同机自连走 loopback mode 仍 `doSetupConnection` 建**真实 RDMA QP**（`rdma_endpoint.cpp:120-127`），数据经 NIC。
3. **唯一能绕 NIC 的开关 `MC_STORE_MEMCPY`**（env，默认 off，`transfer_task.cpp:413`）：开启且单对象 `submit()` 路径 + 目标 IP==本机 IP 时才走 `std::memcpy` 绕 NIC；`submit_batch()` 路径**永远 RDMA**。

→ reconcile 了「~100% 经 NIC」观察：随机 placement 使约半数物理跨机，本地半数也走 loopback 经 NIC。
→ ⭐ **locality 优化（如 MC_STORE_MEMCPY）不治本**：只省「本地 loopback」一档，远程仍 RDMA；根本矛盾是 **CPU 层 vs HBM 层带宽差 ~100×**。**解药是把 working set 压回 HBM（≤9.6M）让 KV 不下沉**，而非优化下沉后的取数路径。

## 10. ⭐ KV offload 价值 = workload 依赖（prefill-bound vs decode-bound，#284-287）

为验证「Mooncake Store/offload 在 prefill-bound 才香」，跑 100K 同数据集、改 `MAX_TURNS=3 MAX_TOKENS=50`（强 prefill-bound）对照 #267-270（decode-bound 30turn×500tok）：

| 并发 | decode-bound 冷热收益 | prefill-bound 冷热收益 | prefill-bound 热 Store-ext |
|---|:---:|:---:|:---:|
| p8 | −17% | −24% | 69% |
| p16 | −3% | −14% | 83% |
| p32 | +1% | −12.5% | 92% |
| p64 | **−10%** | **−40%** | **95%** |

（prefill-bound wall 冷/热秒：p8 25/19 · p16 29/25 · p32 40/35 · p64 92/55，全 exit=0 无 thrashing）

**结论**：
1. prefill-bound 冷热收益**系统性大于** decode-bound，中高并发尤甚。
2. ⭐ **p64 决定性铁证**：两种 workload Store-ext hit 都 ~95%（offload 同样大量接力），但 wall 收益 **prefill-bound −40% vs decode-bound −10%（差 4×）** —— 同样的 Store 命中，prefill-bound 直接省 wall，decode-bound 被 decode 时间淹没。精确隔离出「offload 价值 = workload 是否 prefill-bound」。
3. prefill-bound p64：prefill GPU-L1 仅 8→31%，−40% wall 几乎全靠 **Store-ext 86→95% 接力**贡献 → offload 是 wall 收益主引擎。
4. prefill-bound 全档不 thrashing（decode 活跃集小）→ 印证 thrashing 是 **decode-bound 专属**。

→ **KV offload 不是「作用有限」，而是价值高度依赖 workload**：prefill-bound（RAG / 长文档问答 / 代码补全，长 prompt 短输出高复用）下是主力（p64 −40%）；decode-bound（长多轮对话）下被稀释（≤10%）。**选型按 workload 定**。

## 11. decode-L1 双驱动（并发数 + working set）

- 同 working set 6.4M 三点：100K×64 decode-L1 = **62.3%** vs 200K×32 = **94.8%** vs 400K×16 = **95.3%**。
- 后两者（并发 ≤32）decode-L1 满血，唯独 100K×64（并发 64）掉 32pp → 隔离出**并发数本身的额外压力**（更多并发 seq 争 decode GPU KV）。
- 但 100K×64 wall = 1424s 仅为 100K×32（828s）的 1.7×，**未 thrashing**。即：**高并发先压 decode-L1，working set 达 12.8M 才真正 thrashing**（两个独立维度）。

## 12. 冷热（cache 跨遍）收益分析

- 多数档热略快 −3%~−17%（100K p8 **−17%** / 400K p8 −13% / 200K p16 −7%）。
- 但有持平/反转：60K p16 +2%、100K p32 +1%、400K p24 +11%、**200K p64 +19%**（thrashing 最严重档热反而更慢）。
- 解读：cache 跨遍收益在**低并发明显**（省 prefill 直接缩 wall），中高并发被 decode 饱和稀释，thrashing regime 高方差甚至反转。
- ⭐ 每档 `_warmup` summary 即冷启动真实指标，已全部落盘留底（用户要求的「第一次冷指标」）。

## 13. ⭐⭐ MTP 与 prefix cache 互斥（0.22 单变量对照坐实）

0.22 #43516 让 MTP 不再 corrupt（推理正确，接受率 **62.5%**），但**仍与 prefix caching 互斥**。单变量对照（100K 2conv×8turn，同 0.22 image，IGNORE_EOS=1，唯一变量 MTP on/off）：

| 指标 | MTP on (#292) | no-MTP (#293) |
|---|:---:|:---:|
| decode L1 hit | 0% | **85.76%** |
| decode Store ext hit | 0% | **99.23%** |
| Store save bytes | 0 | **2.07 GB** |
| master_key | 0 | **20953** |

- **机制**：MTP KV layout（extended，含 draft tokens）与 prefix-cache block hash 不兼容 → 开 MTP 则 prefill+decode 的 L1+Store 全失效、每轮重算；Store operation 只剩 lookup_exists（bytes=0）。
- multi-turn **净亏**（prefix cache 死 >> MTP decode 加速）→ ⭐ **生产 = 0.22 no-MTP**。
- 仅单轮 / 低复用 / decode-bound 且 prefill 非瓶颈场景可开 MTP。

## 14. V4 kernel 优化：MULTI_STREAM_GEMM（默认 1024 最优）

`VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD`（V4 专属 decode shared-expert GEMM ↔ 主干 overlap，#41526）**默认 1024，已默认开**（5/27 与 0.22 image 源码 `envs.py` 均 = 1024；PR #41526 标题「Tune default value」merged 2026-05-03，早于 5/27 image，故 **#267-283 本身就是 1024 数据**）。

A/B 实测（#294-299，100K 8turn IGNORE_EOS=500，0 vs 1024）：

| 并发 | decode tps (1024 / 0) | 提升 |
|---|:---:|:---:|
| p8  | 224 / 149 | **+50%** |
| p16 | 358 / 318 | +13% |
| p32 | 547 / 467 | +17% |

- wall 默认快 11-33%，**低并发收益最大**（decode batch token ≤ 并发 « 1024，全程 overlap；高并发 SM 自然饱和，overlap 空间小）。
- ⭐ **默认 1024 全档最优，不该关**。中间值（2048/4096）对 decode 无差别（decode token ≤64 « 1024）；唯一能改变 prefill 的是 8192（= max-num-batched-tokens），但 prefill GEMM 已大、overlap 通常无收益（#41526 默认 1024 即此原因）。

其他 V4 kernel（0.22，大多默认已开）：`VLLM_USE_DEEP_GEMM=1` / `VLLM_MOE_USE_DEEP_GEMM=1` / FlashInfer blockscale FP8 / FULL_AND_PIECEWISE cudagraph。⛔ 不开：`VLLM_BATCH_INVARIANT`（确定性非性能，慢）；`TOKENSPEED_MLA`（#41778，V4 sparse 不兼容，validate reject）。

## 15. 精度门准入（0.22 no-MTP PASS）

换 0.22 大版本属精度高风险变更，重测 GSM8K + MMLU（lm-eval-harness，打 router:8080）：

| 评测 | 实测 | base | floor | 判定 |
|---|:---:|:---:|:---:|:---:|
| GSM8K 8-shot EM | **93.78%** | 92.6 | 91.6 | ✅ PASS |
| MMLU 5-shot gen EM | **89.67%** | 90.1 | 89.1 | ✅ PASS |

两项过 floor → 0.22 no-MTP 数值无退化、可上生产。基线表见 `shared/docs/deepseek_v4_authoritative_accuracy_scores.md`。

## 16. 生产并发上限建议

经验律：**把 working set（并发 × 上下文K）控制在 ≤ 9.6M**。

| 上下文 | 安全并发上限 | 依据 |
|---|:---:|---|
| 60K  | 64（实测 dec-L1 93% 仍健康，上限或更高未测）| #274 |
| 100K | **32**（p64 时 decode-L1 降到 62%，建议止于 32）| #269/#270 |
| 200K | **32**（p64 = 12.8M thrashing）| #277/#278 |
| 400K | **24**（p32 = 12.8M thrashing；p24 = 9.6M 安全）| #283/#281 |

→ **上下文越长，安全并发上限越低**，乘积上限 ≈ 9.6M。超过即撞 HBM 活跃集墙、decode 退化为全程 RDMA load。

## 17. 踩坑记录

### 17.1 输出 token 未固定（已加 IGNORE_EOS）
本轮 `--max-tokens 500` 是上限非固定，模型自然 stop → 每轮输出 257-458 tok 波动，污染冷热精确数字（不影响容量墙）。已给 `batch_multi_turn_pd.sh` + SGLang `batch_multi_turn.sh` 加 `IGNORE_EOS=1`（`--min-tokens MAX_TOKENS + extra-args {"ignore_eos":true}`），后续重跑可彻底解耦 cache 效应与输出长度。

### 17.2 recipe 缺 `--served-model-name` → 404 + evalscope 崩
不加则 vLLM 用模型路径 `/lssd/models/DeepSeek-V4-Pro` 当 model id：① 客户端用 `DeepSeek-V4-Pro` 请求 404；② evalscope `--outputs-dir` 按 `os.path.join(dir, ts, model)` 建目录，model 带 `/` → 绝对路径覆盖前缀 → 写只读挂载 `/lssd` 崩。**Fix**：两端 recipe 必须显式 `--served-model-name DeepSeek-V4-Pro`。

### 17.3 `sync_inference.sh push` 会删远程 `datasets/`
`datasets/` 不在 push 的 protect 列表（只 protect `logs/ outputs/ benchmark_result/`）+ 不进 git → push 的 `--delete` 把远程刚生成的数据集全删（dry-run 实测 `*deleting datasets/shared_prefix_*`）。**Fix**：压测期间**绝不跑 push**（pull 安全）；推 recipe 用单文件 `rsync`。彻底解：给 sync_inference.sh protect 加 `datasets/`（待修）。`push -n` dry-run 看 `deleting` 行是铁律。

### 17.4 docker build pip install 失败（bridge 无 pypi）
0.22 image 加 pytest 时 `docker build` 默认 bridge 网络无 pypi 访问 → pip exit 1。**Fix**：`docker build --network=host`。

### 17.5 PD 节点重启必须整链路重启（NIXL/Mooncake UUID 缓存）
任一 PD 节点（decode/prefill/router）重启 → 对侧缓存的旧 agent UUID 失效 → KV transfer `NIXL_ERR_NOT_FOUND` 或返回 garbled 训练噪声。**Fix**：整链路全部重启，顺序 **decode → prefill → router**（见 §4.5）。详见 memory `feedback_pd_node_reset_must_restart_peer`。

### 17.6 vLLM recipe inline 注释吞 args
`docker run \` 续行命令中间加 `#` 注释 → bash 把后续 args 全丢失（如 `--port` 没传 → vLLM listen 默认端口 → router 转发 connection refused，但容器 `docker ps` 显示 Up）。**Fix**：注释只能在 docker run 命令外。详见 memory `feedback_vllm_recipe_inline_comment_bug`。

## 18. 下一步工作

### 18.1 短期
- [ ] IGNORE_EOS=500 固定输出重跑核心档（100K/200K p8-p32），出干净冷热收益（解耦 cache 效应 vs 输出长度）。
- [ ] 200K/400K 用 `--max-num-seqs` 显式限并发，看能否在 thrashing 边界平滑降级而非硬崩。

### 18.2 中期
- [ ] 同 prompt 的 vLLM Mooncake PD vs SGLang HiCache（doc 11）物理对比：cost basis + latency + 容量墙位置。
- [ ] Mooncake `submit_batch` lookup_exists 瓶颈（80ms master 单线程串行）等上游 PR #2221（shard-group BatchExistKey）合并后升级重测。

### 18.3 长期
- [ ] Mooncake L3 SSD backend（解决 CPU pool 上限 + 把容量墙从 HBM 推到更高层）。
- [ ] GB200/B300（更大 HBM）后 working set 墙位置上移对 200K+ 高并发的影响。

---

## 附录 A：参考资料

- 主数据源（#267-283 sweep，fact-check 2 轮 PASS）：`benchmark_result/267-283_20260531_DeepSeek-V4-Pro_pd_mooncake_sweep.md`
- 容量墙结论：memory `reference_vllm_pd_mooncake_capacity_wall_2026_05_31.md`
- 0.22 升级 + MTP 互斥 + MULTI_STREAM A/B：memory `reference_vllm_022_upgrade_mtp_cache_exclusive_2026_05_31.md`
- PR #41526 元信息（「Tune default value」/ merged 2026-05-03）+ 5/27 与 0.22 image `envs.py` 默认值均 = 1024：本会话 `gh api repos/vllm-project/vllm/pulls/41526` + `docker run … grep envs.py` 实测（2026-05-31）
- 压测环境坑：memory `feedback_vllm_pd_mooncake_bench_setup_traps_2026_05_30.md`
- 真 PD B 方案 5 轮调优：memory `reference_pd_mooncake_b_plan_tuning_2026_05_28.md`
- session affinity（consistent_hash + X-Session-ID）：memory `reference_session_affinity_consistent_hash_win_2026_05_29.md`
- GDR / sysctl / IRQ 网络优化：memory `reference_gdr_nvidia_peermem_pd_2026_05_28.md` + `reference_gcp_sysctl_irq_mlnx_tune_2026_05_28.md`
- Mooncake env sweep（RO 双端唯一 win）：memory `reference_mooncake_4env_sweep_2026_05_28.md`
- SGLang HiCache 对照组：[11_deepseek_v4_b200_multi_turn_sglang_hicache.md](11_deepseek_v4_b200_multi_turn_sglang_hicache.md)
- 31 份 prom 快照：`benchmark_result/metrics/prom_{267-283}_mooncake_*.json` + `_summary.txt`
- prefill-bound 对照 prom：`benchmark_result/metrics/prom_{284-287}_prefillbound_100k_*_summary.txt`

## 附录 B：本次产物清单

| 类型 | 路径 |
|---|---|
| Recipe (3) | `recipes/deepseek_v4_pro/pd_disaggregation/docker_vllm_fp8_{prefill,decode}_pd_mooncake.sh` + `docker_vllm_router.sh` |
| 数据集 (4) | `~/inference/datasets/shared_prefix_{60,100,200,400}k_30turn.jsonl`（远程，不进 git）|
| 压测工具 | `shared/multi_turn_bench/scripts/batch_multi_turn_pd.sh`（+ IGNORE_EOS 支持）|
| Mooncake config | `configs/mooncake/mooncake_config.json` |
| 主报告 | `benchmark_result/267-283_20260531_DeepSeek-V4-Pro_pd_mooncake_sweep.md` |
| prom 快照 | `benchmark_result/metrics/prom_{267-287}_*.json` + `_summary.txt` |
| 精度门 | `eval_results/#010_gsm8k` + `#011_mmlu`（0.22 no-MTP PASS）|

---

> 评审：经 fact-check sub-agent 两轮独立核实 — 第 1 轮对 16 档矩阵 + 容量墙 + Mooncake 源码 + prefill-bound 对照等数字**逐格 100% 吻合**，指出 2 处版本归属 / 编号注解需订正（#282 槽位、prefill-bound 版本边界）；第 2 轮复核订正到位，**PASS**。数据真值源 = `267-283_..._pd_mooncake_sweep.md`（本身亦经 2 轮 fact-check PASS）。
