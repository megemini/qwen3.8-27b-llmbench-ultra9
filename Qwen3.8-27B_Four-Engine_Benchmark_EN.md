# Qwen3.8-27B Local Inference Engine Benchmark

> Test platform: Intel Core Ultra 9 285H / 64 GB / Windows 11 Pro 24H2 (NucBox EVO-T1)
> Test tool: [llmbench](https://github.com/megemini/llmbench) — a self-developed LLM API benchmarking tool (TUI, targeting OpenAI-compatible endpoints).
> Engines under test:
> - Bionic (LM Studio 5.4.1)
> - Herdsman (v0.5.4-beta1)
> - OpenVINO Model Server (ovms, two groups: default device + explicit CPU)
> - Unsloth (v0.1.804-beta)

> Note: all data and screenshots in this article were collected by the author engine by engine; the article was written jointly by the author and AI.

---

## 0. TL;DR

This run collected four metrics: **Dec TPS (pure decode throughput), ITL (inter-token latency), Cache% (prefix cache hit rate), and Server Env (server framework)**. It is precisely because of Cache% and ITL that two "looks great" numbers were exposed:

| # | Surface impression | What is actually going on |
|---|---|---|
| 1 | Herdsman's prefill reaches 3750 tok/s, the fastest of all | Its prompt tokens are **92–100% served from prefix cache** — this measures cache replay speed, not real compute |
| 2 | Herdsman's UI says "Engine: llama.cpp" | The HTTP layer self-reports **ollama v0.6.4**; scheduling, caching and memory policy are all decided by ollama |
| 3 | Bionic's prefill of 3172 tok/s is also #1 | This one is #1 in the real (cache-free) sense (Cache% = 0), but the price is the worst decode and the choppiest output |
| 4 | Average decode speed differs by 2× across engines | Once ITL is factored in, the perceived gap doubles again: Bionic's ITL p95 reaches **1357 ms** |

> Note: llmbench was later upgraded to add Cache% and ITL collection.

### Overall ranking

| Rank | Engine | Pure decode (tok/s) | Long-prompt TTFT | ITL p95 | One-liner |
|---|---|---|---|---|---|
| 🥇 | **OVMS** (default device) | **5.03 / 4.95 / 4.85** | 6.93 s | 553 ms | Best and most stable decode; short-prompt TTFT of 871 ms is unbeatable; smooth output |
| 🥈 | **Herdsman** | 4.59 / 4.81 / 3.91 | 2.57 s ⚠️ | 679 ms | Fast decode, but TTFT benefits from a **92–100% prefix cache**; it self-reported insufficient memory before startup (over-committed), and long-context degradation is 15% |
| 🥉 | **Bionic** | 2.16 / 1.84 / 1.91 | **3.04 s** | **1357 ms** | **Fastest real prefill (3172 tok/s)**, but slowest decode and choppiest output (a batch only every 1.36 s) |
| 4 | **Unsloth** | 3.82 / 3.80 / 3.32 | **159.2 s** | 416 ms | **Prefill disaster** (59 tok/s, 23–54× slower); but the smoothest output and middling decode |

> ⭐ Not ranked: **OVMS-CPU** (the same OVMS pinned to CPU). decode 2.5 tok/s, median TTFT 17–224 s, long tier not completed (timed out during the first run) — its entire gap to OVMS comes from device selection, see §6.5.

### Most important findings

1. **Herdsman's prefill number is questionable.** `Cache%` shows its prompt tokens are **92.3% / 99.7% / 99.96% served from the server-side prefix cache** — meaning that pretty 3750 tok/s prefill (multi-run median, same below) measures "cache replay" rather than "real prefill compute". Every other engine reports 0% cache. This is consistent with the compute-ceiling check (roofline) in §10.1: for a dense 27B model it would need 202 TFLOPS (2×27B×3750 tok/s), whereas the Intel Arc 140T iGPU in this platform peaks at only 77 INT8 TOPS (≈19 TFLOPS FP32, Intel official spec), and the whole platform including CPU/NPU is just ~99 TOPS.
2. **Only OVMS and Unsloth truly "stream".** ITL **mean/median** ratio: Unsloth 1.05, OVMS 1.09 (close to 1 = smooth); while **Bionic 1166, Herdsman 921** — the latter two accumulate a batch of tokens and flush them to the client at once, so it feels like "freeze for a moment, then a burst". Bionic's ITL p95 is as high as **1357 ms**.
3. **Herdsman ran the whole test in an over-committed memory state.** Before startup it popped up its own warning: "estimated 43.5 GB required, 39.3 GB currently available, insufficient; continuing may cause startup failure and system crash". This explains why its long-context decode degrades by **-14.8%** (worst of the four).
4. **OVMS's speed has a prerequisite: it must land on the accelerator.** The additional OVMS-CPU control group (explicitly pinned to CPU) shows: decode drops to 2.5 tok/s, prefill degrades to **5–6 tok/s in later runs (run 1 still ~1000 tok/s, see §6.5)**, and the **long tier (9.46k tokens) did not complete (timed out during the first run, no run data)**. OVMS's 5 tok/s and 871 ms TTFT on the default device should be credited mainly to the accelerator — **verified: except for OVMS-CPU, all four groups (OVMS default / Bionic / Herdsman / Unsloth) run on the Arc 140T iGPU**.

---

## 1. Background and objectives

### 1.1 Why run this test

Small models keep getting stronger — can we break free from cloud services and run agents entirely locally?

The recently released Qwen3.8-27B is a 27B-class model. In the cloud it feels like it absolutely shreds; many basic capabilities can be handed off to a small model like this. So how does local deployment feel? I happen to have a NucBox EVO-T1 (Ultra 9 285H + 64 GB, no discrete GPU). Can it run this model, and how fast:

- GGUF or OpenVINO IR for the quantization format?
- Keep weights in CPU memory, or offload them to the Arc 140T iGPU?
- Enable speculative decoding / MTP or not?
- Enable server-side prefix cache and continuous batching or not?

The mainstream local inference tools on the market have **completely different defaults** for the questions above, and **the numbers they report use inconsistent definitions** — some include prefill in the throughput denominator, some enable prefix cache by default. You cannot compare the advertised tok/s figures across them directly.

So the goal of this test is not to "get a score", but to answer three concrete questions:

1. **On the same machine with the same weights, how much do the four engines' real decode speeds differ?** (measured with Dec TPS, which excludes prefill)
2. **With long prompts (9.4k tokens), who hits the wall first?** (prefill compute vs KV cache management vs memory capacity)
3. **What is the actual interactive experience?** (TTFT + whether output is smooth, ITL)

### 1.2 Workload design

Using llmbench's built-in preset prompts, covering three typical input lengths to simulate a realistic call distribution:

| Tier | Input length | Real-world scenario |
|---|---|---|
| short | ~92 tokens | Short Q&A, agent tool calls, code completion |
| medium | ~1.28k tokens | Single-file code analysis, medium-length document Q&A |
| long | ~9.46k tokens | Long-document RAG, whole-technical-document summarization |

The output cap is fixed at 512 tokens — a workload that "wants a complete answer" rather than "stops after a few sentences" — so that differences in the decode phase carry enough weight in the total time.

### 1.3 Test parameters

| Parameter | Value | Notes |
|---|---|---|
| Concurrency | `c = 1` | Single-stream latency only |
| Output cap | `t = 512` | |
| Runs | `--runs 3` | Three runs executed **serially** (awaited round by round). Definitions used in the body: **Dec TPS / prefill TPS are multi-run aggregates** (Dec uses `decode_throughput` — same source as `dec_tps` in the CSV, with no `_mean` suffix; prefill uses `prompt_throughput_median`, see §4.1 / §4.3), **TTFT uses the median**, supplemented by p95 |
| Sampling | `temperature = 1.0`, no fixed seed | Output length is not fixed |
| Interface | OpenAI-compatible `/v1/chat/completions`, `stream = true` | Same for all four |

> ⚠️ **Scope of this test**: only `concurrency = 1` was measured, mainly because a small machine has limited capability.

### 1.4 Subjects under test

All four engines are local Windows inference solutions, but with very different positioning:

| Engine | Positioning | Role played here |
|---|---|---|
| **Bionic** (LM Studio 5.4.1) | GUI tool for ordinary users, one-click GGUF loading | The "out-of-the-box + iGPU offload" default |
| **Herdsman** (v0.5.4-beta1) | A domestic local model management and inference platform | "GGUF with MTP + ollama scheduling" |
| **OVMS** (OpenVINO Model Server) | Intel's official inference server, requires IR conversion + NNCF quantization | "Server-grade + INT4 + continuous batching" |
| **Unsloth** (v0.1.804-beta) | Desktop app known for its quantization / fine-tuning ecosystem | "Self-built llama.cpp build" |

---

## 2. Test environment

### 2.1 Hardware platform

| Item | Spec |
|---|---|
| Device | NucBox EVO-T1 |
| CPU | Intel Core Ultra 9 285H @ 2.90 GHz, 16 logical cores (6P + 8E + 2LP-E, no hyper-threading) |
| iGPU | Intel Arc 140T GPU (32 GB shared memory) |
| Memory | 64.0 GB (63.5 GB usable) |
| OS | Windows 11 Pro 24H2, build 26100.3476 |
| Discrete GPU | None |

> ⚠️ This is a **CPU-only + iGPU** machine. So-called "GPU acceleration" is essentially competing with system memory for bandwidth, and this single fact drives many of the conclusions below.

### 2.2 Engines under test (including the real backends probed)

| Engine | Endpoint | Engine shown in UI | **Real backend self-reported via API** | Model / quantization | Size | **Actual inference device** |
|---|---|---|---|---|---|---|
| Bionic | `localhost:1234` | Vulkan (llama.cpp) Windows | Not reported (LM Studio 5.4.1) | `qwen3.8-27b` / GGUF Q4_K_M | 16.2 GB | **Arc 140T iGPU** |
| Herdsman | `localhost:8080` | llama.cpp | **ollama v0.6.4-0-ga95fb2fb** | `Qwen3.8-27B` / **GGUF Q4_K_M** | 16.2 GB | **Arc 140T iGPU** |
| OVMS | `localhost:8000` | — | Not reported (OpenVINO Model Server) | `OpenVINO/Qwen3.8-27B-int4-ov` / INT4 (NNCF) | 14.8 GB | **Arc 140T iGPU** |
| OVMS-CPU ⭐ | `localhost:8000` | — | Not reported (OpenVINO Model Server) | `OpenVINO/Qwen3.8-27B-int4-ov` / INT4 (NNCF), **CPU device explicitly specified** | 14.8 GB | **CPU only (the only such group)** ❗ |
| Unsloth | `127.0.0.1:8888` | GGUF inference engine / Vulkan | Not reported | `Qwen3.8-27B-GGUF:Q4_K_M` | 16.2 GB | **Arc 140T iGPU** |

> 📌 **Device matrix**: of the five groups, **only OVMS-CPU runs on CPU alone**; the other four (Bionic / Herdsman / OVMS / Unsloth) **all run on the Arc 140T iGPU**.
> This is crucial: it means **performance differences between engines cannot be attributed to "having a GPU or not"**, but to **engine / backend implementation and offload strategy** on the very same iGPU (see §6.5 and §9).

> ⭐ **OVMS-CPU is an additional control group** (`ovms_cpu/` directory, 2026-09-09): the same OVMS service re-tested with the inference device **explicitly pinned to CPU**. Its difference from the main OVMS group is exactly "the effect of device choice on OVMS" — and it is the first time this report covers the real performance of OVMS's CPU path.
>
> ⚠️ **Only Herdsman reported its framework and version** (`ollama / v0.6.4`). Its UI says "Engine: llama.cpp", but the HTTP layer is ollama — ollama itself is built on llama.cpp, so the two do not contradict each other, but **scheduling, caching and memory policy are all decided by ollama**, which explains its unique prefix cache behavior.

---

## 3. Metric definitions

Besides the usual TTFT / throughput, this run also collected the following advanced metrics (all emitted directly by [llmbench](https://github.com/megemini/llmbench)) — most conclusions in this report rest on them:

| Metric | Meaning | Why it matters |
|---|---|---|
| **Dec TPS** (`decode_throughput`) | Pure decode throughput = `completion_tokens / (duration − TTFT − latency)`, **excluding prefill** | Conventional `gen TPS` includes TTFT in the denominator, which badly underestimates engines with slow prefill. Dec TPS is what you should actually look at when choosing |
| **ITL** (`itl_mean/median/p95_ms`) | Inter-Token Latency, the arrival interval between two adjacent tokens | Determines the "typewriter" feel. Same mean but different distribution means a completely different experience |
| **Cache%** (`cache_hit_ratio`) | Share of prompt tokens served from the server-side prefix cache | Directly exposes whether TTFT / prefill numbers are polluted by cache |
| **Server Env** (`server_framework` / `version`) | Probes server framework and version | Unmasked Herdsman's real backend |
| `peak_kv_cache_usage_perc` / `peak_num_requests_waiting` | Server-side KV cache and queue peaks | All empty in this run (requires the server to expose vLLM-style `/metrics`) |

---

## 4. Core data overview

### 4.1 Pure decode throughput (Dec TPS, tok/s) — prefill excluded

| Engine | short (92) | medium (1.28k) | long (9.46k) | Long-context decay |
|---|---:|---:|---:|---:|
| **OVMS** | **5.03** | **4.95** | **4.85** | **-3.6%** ✅ |
| Herdsman | 4.59 | 4.81 | 3.91 | -14.8% ❌ |
| Unsloth | 3.82 | 3.80 | 3.32 | -13.1% |
| OVMS-CPU | 2.60 | 2.51 | **not completed** ❌ | — |
| Bionic | 2.16 | 1.84 | 1.91 | -11.6% |

> ⭐ **OVMS-CPU: with CPU explicitly specified, decode is only ~2.5 tok/s** (about half of OVMS on the default device), and the long (9.46k token) tier **did not complete (timed out during the first run, no run data left)** — OVMS on CPU cannot handle long prompts (see §6.5).

### 4.2 TTFT median

| Engine | short | medium | long |
|---|---:|---:|---:|
| OVMS | **871 ms** | 4078 ms | 6927 ms |
| Herdsman | 3695 ms ⚠️ | 2172 ms ⚠️ | 2567 ms ⚠️ |
| Bionic | 4126 ms | **2704 ms** | **3039 ms** |
| Unsloth | 1736 ms | 16344 ms | **159168 ms** |
| OVMS-CPU | **16943 ms** ❌ | **224023 ms** ❌ | **not completed** ❌ |

⚠️ = under this value, 92–100% of prompt tokens hit the prefix cache; it is not real prefill time.
❌ OVMS-CPU's median TTFT for short / medium is **19× / 55×** slower than OVMS on the default device; the long tier did not complete (timed out during the first run).

### 4.3 Prefill throughput (Prompt TPS, multi-run median) + cache hit rate

| Engine | short | medium | long | **Cache%** |
|---|---:|---:|---:|---:|
| Bionic | 23 | **484** | **3172** | 0% |
| OVMS | **112** | 318 | 1376 | 0% |
| Herdsman | 14 ⚠️ | 587 ⚠️ | 3750 ⚠️ | **92% / 99.7% / 99.96%** |
| Unsloth | 53 | 78 | **59** | 0% |
| OVMS-CPU | 5.4 | 5.7 | **not completed** | 0% |

> 📌 **Definition note**: this table (and the whole body text) uniformly uses the **multi-run median** of `prompt_throughput` (the source code emits this field directly) — the most robust choice for OVMS / OVMS-CPU, which show "fast first run, degraded later runs". Appendix A pastes the CSV's **mean** values; the difference between the two is exactly the run-to-run jitter (see §7.6). Taking OVMS medium as an example, the mean of 931 tok/s is inflated by the 0.65 s first run, while the median is only 318.
> ⭐ OVMS-CPU's prefill **median is only ~5–6 tok/s** (multi-run median), the same order of magnitude as its 2.5 tok/s decode (a healthy engine's prefill is usually tens of times its decode). But note: this median only reflects the **later runs** — its run-1 prefill actually reached ~1000 tok/s (medium tier: mean 358 = (1063 + 5.7 + 5.7) / 3, min_ttft only 1.26 s, see §7.6), showing the CPU path **still has batching capability on run 1 and degrades to ~5 tok/s from run 2 on**; the cause of the degradation is unverified. A long prompt at the degraded speed needs thousands of steps per run and eventually times out.

### 4.4 ITL (Inter-Token Latency)

| Engine | short mean/median/p95 (ms) | long mean/median/p95 (ms) | Output shape |
|---|---|---|---|
| **Unsloth** | 267 / 254 / 386 | 303 / 292 / 416 | ✅ **Smooth** (median ≈ mean) |
| **OVMS** | 219 / 201 / 468 | 230 / 207 / 553 | ✅ **Smooth** |
| Herdsman | 219 / 0.24 / 554 | 258 / 0.20 / 679 | ⚠️ **Batched spikes** |
| Bionic | 470 / 0.40 / **1242** | 527 / 0.36 / **1357** | ❌ **Severe batched spikes** |
| OVMS-CPU ⭐ | 405 / 384 / 545 | medium: 441 / 398 / 787 (long not completed) | ✅ Smooth (still token-by-token on CPU) |

---

## 5. Charts

### 5.1 Pure decode throughput (prefill excluded)

![Pure decode throughput comparison](https://origin.picgo.net/2026/09/09/01_decode_tps2e9f211c309580aa.png)

> **OVMS on the default device wins all three tiers** (5.03 / 4.95 / 4.85), and is the only engine keeping long-context decay under 5%.
> Bionic is last throughout (1.84–2.16), less than 40% of OVMS.
> ⭐ OVMS-CPU (light green bars) manages only ~2.5 tok/s and its **long tier did not complete (×, timed out during the first run)** — pure CPU drags OVMS down to a level close to Bionic (2.5 vs 1.91, **CPU is even slightly higher**).

### 5.2 Time to first token, TTFT (log scale)

![TTFT comparison](https://origin.picgo.net/2026/09/09/02_ttft2e88076c08dcbb80.png)

> Short prompts: OVMS at **871 ms** is the only one under 1 second.
> Long prompts: Unsloth at **159,168 ms (159 s)** is **52×** slower than the runner-up (Bionic, 3039 ms) and **23×** slower than OVMS.
> ⭐ OVMS-CPU pushes short/medium TTFT to **16.9 s / 224 s** (on a log scale it simply "takes off"); the long tier did not complete (×, timed out during the first run).
> ⚠️ All three Herdsman bars include a 92–100% prefix cache benefit and are not on the same scale as the other three.

### 5.3 Streaming smoothness (ITL)

![ITL comparison](https://origin.picgo.net/2026/09/09/03_itl00b15febaee0fe53.png)

> **Criterion**: smooth streaming ⟹ `ITL median ≈ mean ≈ 1/decode_tps`.
> - OVMS median 207 ms / mean 230 ms, Unsloth median 292 ms / mean 303 ms → **true token-by-token streaming**.
> - Herdsman median **0.20 ms**, Bionic median **0.36 ms** → the vast majority of tokens arrive "in the same batch"; the server is accumulating and flushing.
> - Bionic's ITL p95 reaches **1357 ms**: you wait about 1.36 seconds to see a batch of text, the worst experience of the four.
> - ⭐ OVMS-CPU is absent from this chart because its long tier did not complete, but its short / medium ITL mean/median ratios are **1.06 / 1.11** (§4.4 table, 405.37/383.71 and 440.79/397.55), showing the CPU path also stays token-by-token, just slower.

### 5.4 Prefill throughput and cache pollution

![Prefill throughput comparison](https://origin.picgo.net/2026/09/09/04_prefillda31b91f805919dd.png)

> The diagonally hatched Herdsman bars mean **92–100% of prompt tokens come from the prefix cache** — cache replay rather than real compute, so they **do not participate in the ranking**.
> Excluding it: Bionic has the fastest real prefill (3172 tok/s at the 9.46k tier), OVMS is second (1376), and Unsloth stays at 59 tok/s with no speed-up at all as prompts get longer.
> ⭐ The near-zero-height OVMS-CPU bar (multi-run median 5–6 tok/s, × = long not completed) reflects the degraded later runs: its run-1 prefill measured ~1000 tok/s (see §4.3 note and §7.6) — the CPU path still has batching capability on the first run, and degrades severely afterward; the cause is unverified.
> 📌 Both the chart and the body text use the **multi-run median** of `prompt_throughput`; the CSV rows in Appendix A use the mean, and the difference comes from run-to-run jitter (§7.6).

### 5.5 Long-context stability

![Long-context decay](https://origin.picgo.net/2026/09/09/05_context_decay525fc24d9e15ebf0.png)

> OVMS at **-3.6%** stands alone; Herdsman at **-14.8%** is last, directly related to the memory over-commit it self-reported before startup (see §7.5).
> ⭐ OVMS-CPU's long tier did not complete (timed out during the first run) and is marked **TIMEOUT** — decay cannot be computed; it never even got the chance to "slow down".

### 5.6 End-to-end latency (TTFT + generating 512 tokens)

![End-to-end latency](https://origin.picgo.net/2026/09/09/06_end_to_end84d2c47c40218958.png)

> OVMS stays steady at 103–113 seconds across all three prompt tiers, the only engine that does not slow down markedly as prompts grow.
> Bionic and Unsloth need 271 s and 313 s respectively on long prompts.
> ⭐ OVMS-CPU needs **214 / 428 seconds** even for short / medium under the same definition (its median TTFT already dominates), and the long tier did not complete (×) — even looking only at end-to-end, the CPU configuration is completely unusable.

---

## 6. Per-engine deep dive

### 6.1 OVMS (OpenVINO Model Server, default device) — best overall

**Implementation**: the model is quantized to INT4 with NNCF and converted to OpenVINO IR; the server uses continuous batching + paged attention with 16 unary threads. This group uses the **default device configuration** (no inference device explicitly specified) and **actually executes on the Arc 140T iGPU** — the same as Bionic / Herdsman / Unsloth; the only pure-CPU group is the OVMS-CPU control in §6.5.

**Strengths**
- **Best and most stable pure decode**: 5.03 / 4.95 / 4.85, with under 4% variation across the three tiers.
- **Long-context decay of only -3.6%**, the only one of the four to keep it under 5% — the Dec TPS metric quantifies paged attention's benefit clearly.
- **Short-prompt TTFT of 871 ms**, half of the runner-up (Unsloth, 1736 ms). A decisive advantage for high-frequency short requests like Q&A / agent tool calls.
- **Smooth output**: ITL mean 219 / median 201 / p95 468 ms, median ≈ mean — genuine token-by-token streaming.
- The only one with server-grade capabilities: continuous batching, gRPC + REST, Prometheus, hot model updates.

**Weaknesses / blockers**
- **Long-prompt prefill is only 1376 tok/s** (multi-run median), 43% of Bionic (3172).
- **TTFT fluctuates 7× across the three runs**: the long-prompt runs are **0.98 s / 6.93 s / 6.97 s**, with the first run clearly faster. `Cache% = 0` rules out prefix cache; but "fast first run" also appears in Bionic's / Herdsman's short tier and in OVMS-CPU, so it is not unique to OVMS (§7.6) — "KV block pool recycling / dynamic shape compile cache" therefore remains a **hypothesis to verify**, pending OVMS logs.
- No speculative decoding / MTP, so the decode ceiling is locked by single-step forward latency.
- Highest deployment barrier.

---

### 6.2 Herdsman (v0.5.4-beta1) — pretty numbers, but with water in them

**Implementation**: a local model management and inference platform written in Go; **the backend is actually ollama v0.6.4** (the UI labels it "Engine: llama.cpp", the API self-reports `ollama`), and **inference runs on the Arc 140T iGPU**.

**Strengths**
- decode 4.59–4.81 (short/medium), second only to OVMS.
- **Prefix cache hit rate 92–100%** — this is a real capability and a genuine advantage for "fixed system prompt + multi-turn conversation".
- TTFT 2.17–3.70 s, low throughout.

**Weaknesses / blockers**
- ⚠️ **Prefill numbers are not comparable**: 92–100% of prompt tokens come from cache; 3750 tok/s (multi-run median) is a "cache replay" speed. Before the formal test, llmbench first sends one `max_tokens=1` probe request with the full prompt, so **Herdsman hits the cache from run 1**.
- ❌ **Running over-committed on memory**: before startup it popped up "estimated 43.5 GB required, 39.3 GB currently available, insufficient; continuing may cause startup failure and system crash". This directly causes its **long-context decode drop of -14.8% (worst of the four)** — long prompts need a bigger KV cache, and paging starts once memory runs short.
- ⚠️ **Batched output**: ITL median 0.20–0.24 ms, mean 219–258 ms, p95 554–679 ms. It feels like "freeze for half a second, then a burst".
- Short-prompt TTFT (3695 ms) is still higher than medium (2172 ms), with a fixed overhead of about 1.5 s (independent of prompt length).
- Its tokenizer differs from the other three (it reports 52 tokens for the short prompt, while the other three all report 92).

---

### 6.3 Bionic (LM Studio 5.4.1 + GGUF/Vulkan) — king of prefill, last in decode

**Implementation**: LM Studio 5.4.1 with the `GGUF` + `Vulkan (llama.cpp) Windows` extensions installed; weights and KV cache placed on the Arc 140T iGPU.

**Strengths**
- 🏆 **Fastest real prefill**: with Cache% = 0 it still reaches **3172 tok/s** on long prompts, 2.3× OVMS and 54× Unsloth. Its 484 tok/s at the medium tier is also #1.
- **Long-prompt TTFT of 3.04 s**, the lowest among engines without cache.

**Weaknesses / blockers**
- ❌ **Slowest pure decode**: 1.84–2.16 tok/s, only 40% of OVMS.
- ❌ **Choppiest output**: ITL p95 **1357 ms**, median 0.36 ms. In other words you wait about 1.36 seconds to see a batch of text — the worst experience of the four.
- Short-prompt TTFT (4126 ms) is higher than medium (2704 ms), with about 1.4 s of length-independent fixed overhead.

---

### 6.4 Unsloth (desktop) — prefill disaster, the rest is acceptable

**Implementation**: the Unsloth desktop app with its own bundled llama.cpp build (a fairly early version), GGUF Q4_K_M, Vulkan device Arc 140T.

**Strengths**
- ✅ **Smoothest output**: ITL mean 267 / median 254 / p95 386 ms, `p95/mean = 1.4`, the best streaming experience of the four.
- **decode 3.32–3.82 tok/s**, 1.7–2.1× Bionic — both use **llama.cpp + Vulkan on the same Arc 140T iGPU** with the same Q4_K_M weights, so the gap can only come from **build version and batching / offload configuration**. Conversely: **the iGPU is not the decode bottleneck, the software stack is**.
- Long-context decay -13.1%, slightly better than Herdsman.
- Short-prompt TTFT 1736 ms, second only to OVMS.

**Weaknesses / blockers**
- ❌ **Prefill pinned at 59 tok/s, 23–54× slower**. TTFT for a 9.46k prompt is as high as **159.2 s** (2 min 39 s).
- ❌ **Prefill throughput does not improve as prompts grow**: 53 / 78 / 59 (the other three are 23→3172 and 112→1376). This is the decisive fingerprint of **prefill not being batched**.
- Suspected root cause: an outdated llama.cpp build, too small a ubatch size, or a degraded Vulkan kernel path (this group runs on the Vulkan iGPU).

---

### 6.5 OVMS-CPU (explicitly pinned to CPU) — control group: device choice is OVMS's lifeline

"Is OVMS fast because of the engine or because of the device?" — the same OVMS service re-tested with the inference device **explicitly pinned to CPU**.

| Metric | short (92) | medium (1.28k) | long (9.46k) |
|---|---:|---:|---:|
| Dec TPS | 2.60 | 2.51 | **not completed** ❌ (timed out during the first run, no data) |
| TTFT median | 16.94 s | 224.02 s | — |
| Prefill throughput | 5.4 tok/s | 5.7 tok/s | — |
| ITL mean / median / p95 | 405 / 384 / 545 ms | 441 / 398 / 787 ms | — |
| Cache% | 0% | 0% | — |

**Conclusion 1: the CPU path wipes out every OVMS advantage.** decode falls from 5.0 to **2.5 tok/s** — note it is still **higher than Bionic's 1.91 on the iGPU**; short-prompt TTFT rises from 871 ms to **16.9 s** (about 19×); medium reaches **224 s** (about 55×).

**Conclusion 2: the CPU path's prefill still has batching capability on run 1, but later runs degrade severely.** On the default device, prefill speeds up sharply as prompts grow (112→1376 tok/s, multi-run median), showing tokens are fed to the GPU in batches; the CPU group's run 1 is the same (medium tier run-1 ~1063 tok/s with TTFT of only 1.26 s; short tier run-1 ~79 tok/s), showing batching still exists on CPU; but from run 2 on, prefill degrades to **~5–6 tok/s** (the multi-run median is dragged down to this level, only about 1× faster than decode, whereas a healthy engine's prefill is usually tens of times its decode) — the cause of the degradation (KV cache / memory / scheduling) is unverified. A 9.46k long prompt at the degraded speed would need **~30 minutes** per run to produce the first token; in the actual test the first run timed out (no run data left).

**Conclusion 3: device choice = OVMS's lifeline, and no amount of parameter tuning can save that.** On the accelerated path (Arc 140T iGPU) the default device delivers 5 tok/s decode + batched prefill; pinned to CPU it drops to 2.5 tok/s. For OVMS, "which device it runs on" matters far more than "which parameters you tune".

**Why this control matters**: it decouples "engine differences" from "device differences". First, a device table needs to be clarified — **in this run only OVMS-CPU lands on CPU alone; the other four groups are all on the same Arc 140T iGPU** (§2.2):

| Group | Inference device | decode (long tier) |
|---|---|---:|
| OVMS (default) | **Arc 140T iGPU** | 4.85 |
| Herdsman | **Arc 140T iGPU** | 3.91 |
| Unsloth | **Arc 140T iGPU** | 3.32 |
| **OVMS-CPU** ⭐ | **CPU only** | **2.51** (short/medium; long timed out in the first run) |
| Bionic | **Arc 140T iGPU** | **1.91** |

From this, two more accurate and stronger conclusions follow:

1. **Same device ≠ same performance.** All four groups are on the same iGPU, yet decode differs by **2.6×** (4.85 → 1.91). This gap can no longer be explained by "having a GPU or not"; it can only be attributed to **backend kernel implementation and batching / offload strategy** (the OpenVINO GPU plugin is good, llama.cpp Vulkan is not).
2. **Offloading to the iGPU can be slower than pure CPU.** Bionic on the iGPU reaches only **1.91 tok/s, lower than OVMS-CPU's 2.51 on pure CPU**. In other words LM Studio's Vulkan offload is a **net negative** on this "shared memory + iGPU" machine — it gains no GPU compute benefit while paying extra latency for accesses across the ring bus. This is far more precise than "the device is better so it's faster", and agrees with §6.3 and §9.

---

## 7. Attribution of key differences

### 7.1 Prefill: only comparisons among the three with cache=0 are meaningful

| Engine | short | medium | long | Cache% | Verdict |
|---|---:|---:|---:|---:|---|
| **Bionic** | 23 | **484** | **3172** | 0% | Fastest real prefill |
| **OVMS** | **112** | 318 | 1376 | 0% | Fastest on short prompts, middling on long |
| Unsloth | 53 | 78 | **59** | 0% | No batching at all |
| Herdsman | 14 | 587 | 3750 | **92–100%** | ⚠️ Cache replay speed, not ranked |
| OVMS-CPU ⭐ | 5.4 | 5.7 | **not completed** | 0% | No batching + pure CPU, worst |

- Bionic leads OVMS on medium/long thanks to **the Vulkan backend's mature batched matmul** (batch matrix operations happen to be an iGPU strength).
- OVMS (default device) is **verified to run on the Arc 140T iGPU** and gets 112→1376 tok/s batched prefill (multi-run median); **pinned to CPU (OVMS-CPU) it immediately drops to 5–6 tok/s** — proving the "compute" behind this curve comes from the device, not the engine.
- **Unsloth's 59 tok/s, independent of length**, is the only genuine anomaly.

### 7.2 Decode: device + kernel determine per-step latency

| Engine | decode (long) | Relative to OVMS | Key attribution |
|---|---:|---:|---|
| OVMS (default device) | 4.85 | 1.00× | OpenVINO INT4 + paged attention; **running on the same Arc 140T iGPU as the other three, with the best kernel implementation** |
| Herdsman | 3.91 | 0.81× | The ollama backend is decent, but memory over-commit drags down long context |
| Unsloth | 3.32 | 0.68× | llama.cpp Vulkan is normal (**also on the iGPU**, not a CPU path) |
| OVMS-CPU ⭐ | not completed | — | The same OVMS pinned to CPU reaches only 2.51 (short/medium; long timed out in the first run) — decode is all about the device |
| Bionic | 1.91 | **0.39×** | iGPU Vulkan is a **negative optimization** for batch=1 decode (it is **lower than OVMS-CPU's 2.51 on pure CPU** — the negative optimization is confirmed) |

> 📌 **MTP's contribution cannot be isolated from this data.** Herdsman is labeled "MTP-enabled" (speculative decoding) and its decode (4.59) is 20% higher than Unsloth's (3.82); but **OVMS has no MTP at all yet takes the top 5.03**, showing a backend difference is enough to explain the gap. A definitive answer requires an A/B test with MTP on/off.

### 7.3 ITL: who is really streaming

| Engine | mean/median ratio | Shape | Experience |
|---|---:|---|---|
| Unsloth | 1.05 | Smooth | A token every ~260 ms, a steady typewriter |
| OVMS | 1.09 | Smooth | A token every ~200 ms, a steady typewriter |
| Herdsman | ~921 | Batched | A batch every ~0.6 s |
| Bionic | ~1166 | Batched | A batch every ~1.36 s, the choppiest |

> Criterion: a smooth streaming engine has ITL **mean/median ≈ 1** (Unsloth 1.05, OVMS 1.09); for batching engines the median is pulled down to ~0.2–0.4 ms by intra-batch intervals, so the ratio explodes to 900–1200.

This is the item with **the biggest impact on interactive experience** and the one most easily hidden by "average throughput" —
**the decode gap between Bionic and Unsloth (1.91 vs 3.32) is already sizable, but once ITL is added the perceived gap doubles again.**

### 7.4 Prefix cache: Herdsman's "sweet trap"

- llmbench first sends one probe request with the full prompt (`probe_tokens`), and Herdsman uses this to **hit the cache from run 1 (92%+)**.
- For real workloads: if the load is "long system prompt + multi-turn follow-ups", this cache is a **genuine, huge advantage**; if it is "a brand-new long document every time", it is of no use at all.
- The other three all report Cache% = 0 — but note: **0% may mean caching is off, or may mean the server does not report the `prompt_tokens_details.cached_tokens` field**. Since Bionic's / Unsloth's TTFT shows no improvement across three runs, the former is more likely.
- 🔍 **A consistent detail**: in all three tiers Herdsman misses exactly ~12 prompt tokens of cache per run (short: 156 − 144 = 12; medium: 3723 − 3711 = 12; long: 28254 − 28242 = 12, i.e. about 4 per run) — showing ollama **reuses a fixed prefix but always recomputes the trailing ~4 tokens**. That is why its hit rate never reaches 100% (99.96% on long), and why "92–100%" rather than "100%" is accurate in the body text.

### 7.5 Memory: Herdsman's over-commit is the main cause of long-context slowdown

| Engine | Long-context decay | Note |
|---|---:|---|
| OVMS | -3.6% | Paged attention, smallest decay |
| Bionic | -11.6% | Decay comes from the iGPU's **shared video memory** |
| Unsloth | -13.1% | Older llama.cpp build |
| **Herdsman** | **-14.8%** ❌ | **Strongly correlated**: it self-reported insufficient memory before startup (43.5 GB needed / 39.3 GB available) |

### 7.6 Run-to-run jitter: "fast first run, later runs degrade" is not unique to OVMS

After checking each group's "first-run TTFT" (older tool definition: `min_ttft` takes run 1) against the TTFT median, the conclusion must be **softened**:

| Engine | short | medium | long |
|---|---:|---:|---:|
| OVMS | 0.62 → 0.87 s (1.4×) | 0.65 → 4.08 s (**6.3×**) | 0.98 → 6.93 s (**7.1×**) |
| OVMS-CPU | 1.22 → 16.9 s (**13.9×**) | 1.26 → 224 s (**177.9×**) | not completed |
| Herdsman | 1.84 → 3.69 s (2.0×) | 1.92 → 2.17 s (1.1×) | 2.30 → 2.57 s (1.1×) |
| Bionic | 2.39 → 4.13 s (1.7×) | 2.93 → 2.70 s (**0.9×, the first run is actually the slowest**) | 2.80 → 3.04 s (1.1×) |
| Unsloth | 1.69 → 1.74 s (1.0×) | 16.34 → 16.34 s (1.0×) | 159.17 → 159.17 s (1.0×) |

- The most striking are OVMS / OVMS-CPU (6–178×), but **Bionic's and Herdsman's short tiers also show a faster first run (1.7–2.0×), while Bionic's medium tier has its first run as the slowest** — the direction is not consistent, showing this is **not a stable behavior unique to one engine**; it is more likely platform-level / scheduling-level incidental factors stacking up (iGPU driver throttling, shared memory bandwidth contention, etc.).
- **Unsloth's three runs being identical** is strong evidence: jitter is unrelated to "whether batching/pooling is done".
- The raw JSON of OVMS-CPU's medium tier shows a run-1 prefill of about **1063 tok/s** (TTFT of only 1.26 s), while the three-run mean is 358 / median 5.7 — i.e. "normal first run, degraded later runs" holds for prefill too, and even more extremely than the TTFT jitter (§6.5 Conclusion 2).

---

## 8. Test screenshots

### 8.1 Herdsman

![Herdsman insufficient-resources warning before startup](https://origin.picgo.net/2026/09/09/-2026-09-08-202116b58ffbbd40d458ef.png)

> Pre-startup popup: "estimated 43.5 GB required, 39.3 GB currently available, insufficient. Continuing may cause startup failure and system crash."
> I clicked `Start anyway` and forced it up. This is the most likely reason for its -14.8% long-context slowdown (worst of the four) (§7.5).

![Herdsman resource usage during the run](https://origin.picgo.net/2026/09/09/-2026-09-08-205222463a6d3da584dec9.png)

### 8.2 Bionic

![Bionic installed extensions](https://origin.picgo.net/2026/09/09/-2026-09-08-22070860c3e251d167b375.png)

> `Installed extensions`: `GGUF` + `Vulkan (llama.cpp) Windows` — confirming Bionic uses the llama.cpp Vulkan backend with GGUF weights.

![Bionic resource usage during the test](https://origin.picgo.net/2026/09/09/-2026-09-08-2206571d44e165b9db299d.png)

### 8.3 OVMS

![OVMS server status during the test](https://origin.picgo.net/2026/09/09/-2026-09-08-2230228d2b1e800116f011.png)

### 8.4 Unsloth

![Unsloth client status during the test](https://origin.picgo.net/2026/09/09/-2026-09-08-212735b655f36650aa039d.png)

### 8.5 OVMS-CPU: the long-tier timeout (not completed)

![OVMS-CPU test timeout screenshot](https://origin.picgo.net/2026/09/09/-2026-09-09-1232264c0db86457f86411.png)

> With CPU explicitly pinned, the TUI finished only the short / medium tiers; for long (9.46k tokens) it **sent just the first-run request before erroring out** (an error appears in the status bar), producing no run data at all — hence the long tier has no entry in the export file, and the body text uniformly describes it as "not completed". Short / medium TTFT medians still deteriorated to 16.9 s / 224 s.

---

## 9. Bottleneck and blocker checklist

| Engine | Primary blocker | Secondary blocker | Fixable? |
|---|---|---|---|
| **OVMS** | Long-prompt prefill 1376 tok/s (43% of Bionic's, and dependent on iGPU offload) | TTFT jitter of 7× across runs (cause unverified, §7.6); no speculative decoding | 🟡 Prefill can improve via tuning; jitter needs a look at the KV block pool; ⚠️ falls off a cliff if the device falls back to CPU (§6.5) |
| **Herdsman** | **Memory over-commit** (self-reported before startup: 43.5 GB needed / 39.3 GB available) | Prefill number includes a 92–100% cache benefit; batched output | 🟢 Already Q4_K_M (no need to change tier); a smaller quantization or more memory would improve it significantly |
| **Bionic** | **iGPU Vulkan offload kills decode** (1.91 tok/s, slower than OVMS-CPU's 2.51 on pure CPU) | Batched output, ITL p95 1357 ms | 🟢 Unsloth on the same iGPU is 1.7–2.1× faster → this is a build/config problem: upgrading llama.cpp or adjusting the offload layer count (`-ngl`) would help |
| **Unsloth** | **Prefill not batched (59 tok/s)** | Outdated llama.cpp build | 🟢 Upgrade the build and set `-b/-ub`; an order-of-magnitude improvement is expected |
| **OVMS-CPU** ⭐ | **Running 27B on pure CPU: prefill degrades to ~5 tok/s in later runs (run 1 ~1000) + halved decode** | Long tier not completed (timed out in the first run) | 🔴 Not a parameter problem: switching to the iGPU (default device) solves it, see §6.5 |

### Platform-level bottlenecks (engine-independent)

1. **Memory bandwidth locks the decode ceiling.** A 27B model needs ~16 GB of weights even at INT4/Q4, so the measured 3.3–5.0 tok/s is roughly the right value on that curve. **The only way to go faster is speculative decoding / MTP; switching engines brings limited gains.**
2. **Device is not the dividing line — the software stack is.** In this run **only OVMS-CPU ran on pure CPU**; the other four groups were all on the same Arc 140T iGPU (§2.2). On that same iGPU, decode ranges from **4.85 (OVMS) to 1.91 (Bionic), a 2.6× spread**; while the only pure-CPU group, **OVMS-CPU at 2.51 tok/s, is actually higher than Bionic on the iGPU**. Conclusion: on this "shared memory + iGPU" machine, **offloading to the GPU is not inherently faster** — what determines success is the engine kernel and the batching / offload configuration (the OpenVINO GPU plugin is good, llama.cpp Vulkan is not).

---

## 10. Selection advice

| Use case | Recommendation | Why |
|---|---|---|
| **Single-stream · short Q&A / agent tool calls** | **OVMS** | TTFT 871 ms (the only sub-second one), decode 5.03, smooth output |
| **Single-stream · long documents / RAG** | **OVMS** | decode 4.85 + decay -3.6%, shortest end-to-end at 112.5 s |
| **Fixed system prompt + multi-turn conversation** | **Herdsman** | The 92–100% prefix cache hit rate is a real advantage, TTFT steady at 2.2–2.6 s |
| **Multi-concurrency serving** (needs extra testing first) | **OVMS** | The only server-grade option with continuous batching + paged attention |
| **Output smoothness matters most** | **Unsloth ≈ OVMS** | ITL median ≈ mean, a true typewriter experience |
| **Short prompts dominate and you are willing to tune** | **Unsloth** | Once prefill is fixed: decode 3.8 + smoothest output, high price/performance |
| **Not recommended for now** | **Bionic** | On this platform iGPU offload is a negative optimization: worst decode (1.91) + choppiest output; Unsloth on the same iGPU is 2× faster, and **even OVMS-CPU on pure CPU (2.51) beats it** |
| **Avoid for long-prompt scenarios** | **Unsloth / OVMS-CPU** | Unsloth makes you wait 159 s for the first token; OVMS pinned to CPU cannot use the long tier (times out in the first run) |

> ⚠️ All engines in the table above run on the **Arc 140T iGPU** by default (**verified**; only OVMS-CPU is the CPU-pinned control group). The OVMS-CPU control proves that once it falls back to CPU, decode halves, prefill degrades to 5–6 tok/s in later runs (run 1 ~1000), and the long tier becomes unusable (§6.5). When deploying, still specify `target_device=GPU` explicitly to avoid an automatic fallback to CPU after a service restart.

### End-to-end latency reference (generating 512 tokens)

| Scenario | OVMS | Herdsman | Bionic | Unsloth |
|---|---:|---:|---:|---:|
| Short Q&A (92 in) | **102.7 s** | 115.2 s | 241.0 s | 135.7 s |
| Medium document (1.28k in) | **107.5 s** | 108.6 s | 281.0 s | 151.1 s |
| Long-document RAG (9.46k in) | **112.5 s** | 133.5 s | 271.1 s | 313.4 s |

> Note 1: Herdsman's TTFT includes the prefix cache benefit; if the cache misses, recompute based on its actual prefill capability.
> Note 2: OVMS-CPU is not listed — its long tier did not complete (timed out in the first run); its short / medium end-to-end times are 91 s / 205 s (single run), still worse than OVMS on the default device.
> Note 3: this table is an **estimate** (TTFT median + 512 / Dec TPS), not measured per-run wall time. In practice short-Q&A requests often **finish early** (Bionic short generated only 262 tokens and its measured first run took 124.2 s, far less than the 241 s estimated for 512 tokens), so this column is a pessimistic upper bound.

**One-line summary**

> **OVMS is the overall winner** — its decode is the fastest and most stable, its output the smoothest, and it is the only one breaking the 1-second barrier on short-prompt TTFT, all without any cache benefit. **But winning presupposes running on the Arc 140T iGPU**: pin it to CPU and decode halves, prefill degrades to 5–6 tok/s in later runs (run 1 ~1000), and the long tier becomes unusable (§6.5).
> **Herdsman's numbers need to be taken with a discount** — the 92–100% prefix cache hit makes its prefill incomparable, and the memory over-commit at startup drags it down on long context; but for "fixed system prompt, multi-turn conversation" it is still the best choice.
> **Bionic and Unsloth are two sides of the same mirror-image problem** — the former is strongest at prefill and weakest at decode, the latter decent at decode with a prefill disaster; **both run on the same Arc 140T iGPU** (§2.2), so their weaknesses can only be **software configuration problems** (offload strategy / batching parameters / build version), not hardware problems, and both deserve another tuning round. And OVMS-CPU provides a yardstick: pure CPU running 27B INT4 is **2.5 tok/s** — **faster than Bionic's 1.91 on the iGPU** — showing that on this shared-memory machine, offloading weights to the iGPU is not inherently advantageous; when the software stack is not tuned right, the iGPU is a liability rather than an asset.

---

## Appendix A: raw CSV data

<details>
<summary>Expand</summary>

```
# Bionic  (localhost:1234, qwen3.8-27b, latency 58.03ms, Cache 0%)
label,tok,runs,gen_med,dec_tps,prompt_tps,ttft_med,ttft_p95,itl_mean,itl_med,itl_p95,cache,dur,compl
short,92,3,2.1103,2.1619,27.99,4.1263,4.2676,470.25,0.4034,1241.59,0.0,124.21,262
medium,1281,3,1.8219,1.8363,472.17,2.7036,2.9323,546.28,0.4723,1271.27,0.0,271.86,512
long,9458,3,1.8367,1.9085,3242.53,3.0394,3.0979,526.91,0.3576,1357.31,0.0,172.70,317

# Herdsman  (localhost:8080, Qwen3.8-27B, latency 55.82ms, backend ollama v0.6.4)
short,52,3,4.5182,4.5924,19.12,3.6949,3.7957,219.15,0.2379,554.18,0.9231,106.59,487
medium,1241,3,4.7442,4.8082,606.51,2.1716,2.2381,208.56,0.2406,546.06,0.9968,107.98,512
long,9418,3,3.8040,3.9090,3893.24,2.5671,2.5765,257.52,0.2012,678.84,0.9996,90.46,339

# OVMS  (localhost:8000, OpenVINO/Qwen3.8-27B-int4-ov, latency 52.46ms, Cache 0%)
short,92,3,4.9453,5.0298,128.96,0.8707,0.8748,219.14,201.128,468.09,0.0,100.21,497
medium,1281,3,4.7677,4.9547,931.54,4.0775,4.0975,216.30,202.940,440.22,0.0,104.07,512
long,9458,3,4.5536,4.8500,4327.94,6.9268,6.9703,229.61,207.398,552.63,0.0,106.64,512

# Unsloth  (127.0.0.1:8888, Qwen3.8-27B-GGUF:Q4_K_M, latency 7.00ms, Cache 0%)
short,92,3,3.7260,3.8237,53.32,1.7360,1.7709,266.52,254.217,386.38,0.0,48.92,186
medium,1281,3,3.4086,3.8001,78.53,16.3437,16.4315,264.40,259.436,380.05,0.0,150.12,512
long,9458,3,1.6311,3.3221,59.40,159.1685,160.7369,302.73,291.595,415.61,0.0,312.16,512

# OVMS-CPU (added 2026-09-09, explicitly pinned to CPU; localhost:8000, Cache 0%)
label,tok,runs,gen_med,dec_tps,prompt_tps,ttft_med,ttft_p95,itl_mean,itl_med,itl_p95,cache,dur,compl
short,92,3,2.2590,2.5997,29.67,16.9435,19.4817,405.37,383.71,544.99,0.0,91.43,239
medium,1281,3,1.1963,2.5149,358.25,224.0229,225.6461,440.79,397.55,786.81,0.0,205.32,512
long,9458,——not completed: the first run timed out / the connection dropped, the test aborted, no run data left——
```

Units: TTFT / duration in seconds; ITL in milliseconds; cache = hit rate (0–1).


