# 【性能测试】Qwen3.8-27B 本地推理引擎性能对比报告

> 测试平台：Intel Core Ultra 9 285H / 64 GB / Windows 11 Pro 24H2（NucBox EVO-T1）
> 测试工具：[llmbench](https://github.com/megemini/llmbench) — 自己开发的 LLM API 性能基准测试工具（TUI，面向 OpenAI 兼容端点）。
> 被测引擎：
> - Bionic（LM Studio 5.4.1）
> - Herdsman（牧马人 v0.5.4-beta1）
> - OpenVINO Model Server（ovms，默认设备 + 显式 CPU 两组）
> - Unsloth（v0.1.804-beta）

> 说明：本文中所列数据与截图由作者逐个引擎测试并采集，文章由作者与 AI 共同撰写。

---

## 0. TL;DR

本次测试采集了 **Dec TPS（纯解码吞吐）、ITL（token 间延迟）、Cache%（prefix 缓存命中率）、Server Env（服务端框架）** 四项指标。正因为有了 Cache% 与 ITL，才得以识破两个"看起来很好"的数字：

| # | 表面现象 | 实际情况 |
|---|---|---|
| 1 | Herdsman 的 prefill 高达 3750 tok/s，全场最快 | 它的 prompt token **92–100% 来自 prefix cache**，测的是缓存回放速度，不是真实算力 |
| 2 | Herdsman 界面写着 "Engine: llama.cpp" | HTTP 层自报 **ollama v0.6.4**，调度 / 缓存 / 内存策略都由 ollama 决定 |
| 3 | Bionic 的 prefill 3172 tok/s 也是全场第一 | 这个是真实（无缓存）口径的第一（Cache% = 0），但代价是 decode 垫底 + 输出最卡 |
| 4 | 各家平均解码速度差 2 倍 | 加上 ITL 后，体感差距还要再放大一倍：Bionic 的 ITL p95 达 **1357 ms** |

> 说明：后续升级了 llmbench 工具，添加了 Cache% 和 ITL 的采集。

### 综合排名

| 排名 | 引擎 | 纯 decode (tok/s) | 长 prompt TTFT | ITL p95 | 一句话 |
|---|---|---|---|---|---|
| 🥇 | **OVMS**（默认设备） | **5.03 / 4.95 / 4.85** | 6.93 s | 553 ms | decode 全场第一且最稳，短 prompt TTFT 871 ms 无人能敌，输出平滑 |
| 🥈 | **Herdsman** | 4.59 / 4.81 / 3.91 | 2.57 s ⚠️ | 679 ms | decode 快，但 TTFT 有 **92–100% prefix cache 红利**；启动前自报内存不足（超配运行），长上下文掉速 15% |
| 🥉 | **Bionic** | 2.16 / 1.84 / 1.91 | **3.04 s** | **1357 ms** | **真实 prefill 最快（3172 tok/s）**，但 decode 最慢、输出最卡（1.36 s 才吐一批） |
| 4 | **Unsloth** | 3.82 / 3.80 / 3.32 | **159.2 s** | 416 ms | **prefill 灾难**（59 tok/s，慢 23–54 倍）；但输出最平滑、decode 中规中矩 |

> ⭐ 未上榜：**OVMS-CPU**（同一 OVMS 锁 CPU）。decode 2.5 tok/s、TTFT 中位数 17–224 s、long 档未完成（首轮即超时中断）——它与 OVMS 的差距全部来自设备选择，见 §6.5。

### 最重要的发现

1. **Herdsman 的 prefill 数字"含金量"存疑。** `Cache%` 显示它的 prompt token **92.3% / 99.7% / 99.96% 来自服务端 prefix cache** —— 也就是说它那漂亮的 3750 tok/s prefill（多轮中位数口径，下同），测的是"缓存回放"而非"真实 prefill 算力"。其余几家 cache 全为 0%。这与 §10.1 的算力上限校验（roofline）结论一致：按 dense 27B 计算它需要 202 TFLOPS（2×27B×3750 tok/s），而本平台 Intel Arc 140T 核显的 INT8 峰值只有 77 TOPS（≈19 TFLOPS FP32，Intel 官方 spec），平台含 CPU/NPU 合计也不过 ~99 TOPS。
2. **只有 OVMS 和 unsloth 在真正"流式"输出。** ITL **均值/中位数**比值：unsloth 1.05、OVMS 1.09（接近 1 = 平滑）；而 **Bionic 1166、Herdsman 921** —— 后两者是把攒够一批 token 再一次性刷给客户端，实际体感是"卡一下、吐一串"。Bionic 的 ITL p95 高达 **1357 ms**。
3. **Herdsman 在内存超配状态下跑完了全程。** 启动前它自己弹出告警："预估需要 43.5 GB，当前可用 39.3 GB，不足，继续可能导致启动失败和系统崩溃"。这解释了它为什么长上下文 decode 掉速 **-14.8%**（四家最差）。
4. **OVMS 的快有前提：要命中加速设备。** 追加的 OVMS-CPU 对照（显式锁定 CPU）显示：decode 掉到 2.5 tok/s、prefill 后续轮次退化到 **5–6 tok/s（首轮仍有 ~1000 tok/s，见 §6.5）**、**long 档（9.46k token）未完成（首轮即超时中断，无轮次数据）**。OVMS 默认设备的 5 tok/s / 871 ms TTFT 应主要归功于加速设备 —— **已核实：除 OVMS-CPU 外，四组（OVMS 默认设备 / Bionic / Herdsman / Unsloth）全部运行在 Arc 140T 核显上**。

---

## 1. 测试背景与目的

### 1.1 为什么要测

现在 `小模型` 的能力越来越强，是否可以脱离云端的服务，完全将 Agent 运行在本地？

前段时间发布的 Qwen3.8-27B 是一个 27B 级别的模型，云端体验的感觉是 `嘎嘎乱杀`，很多基础的能力完全可以交给这个小模型了，那么，在本地部署的感觉如何？刚好，手上有一台 NucBox EVO-T1（Ultra 9 285H + 64 GB，无独显），它能不能跑得动、能跑多快：

- 量化格式走 GGUF 还是 OpenVINO IR？
- 权重放 CPU 内存，还是 offload 到 Arc 140T 核显？
- 要不要启用投机解码 / MTP？
- 服务端要不要开 prefix cache、连续批处理？

市面上几款主流本地推理工具对上述问题的**默认取向完全不同**，而且它们报出的数字**口径也不一致** —— 有的把 prefill 算进吞吐分母，有的默认开启 prefix cache。直接看各家宣传的 tok/s 无法横向比较。

所以这次测试的目的不是"跑个分"，而是回答三个具体问题：

1. **在同一台机器、同一批权重上，四款引擎的真实解码速度差多少？**（用剔除 prefill 的 Dec TPS 衡量）
2. **长 prompt（9.4k token）场景下，谁会先撞墙？**（prefill 算力 vs KV cache 管理 vs 内存容量）
3. **实际交互体验如何？**（首字延迟 TTFT + 输出是否平滑 ITL）

### 1.2 负载设计

使用 llmbench 内置的 preset 提示词，覆盖三种典型输入长度，模拟真实调用分布：

| 档位 | 输入长度 | 对应真实场景 |
|---|---|---|
| short | ~92 token | 短问答、Agent 工具调用、代码补全 |
| medium | ~1.28k token | 单文件代码分析、中等长度文档问答 |
| long | ~9.46k token | 长文档 RAG、整篇技术文档总结 |

输出上限固定 512 token，属于"要完整回答"而非"生成几句就停"的负载 —— 这样 decode 阶段的差异才能在总耗时里占足够权重。

### 1.3 测试参数

| 参数 | 取值 | 说明 |
|---|---|---|
| 并发 | `c = 1` | 只测单路延迟 |
| 输出上限 | `t = 512` | |
| 重复轮数 | `--runs 3` | 三轮**串行**执行（逐轮 await）。正文取值口径：**Dec TPS / prefill TPS 为多轮聚合值**（Dec 取 `decode_throughput`——CSV 中 `dec_tps` 同源，无 `_mean` 后缀；prefill 取 `prompt_throughput_median`，见 §4.1 / §4.3），**TTFT 取中位数**并辅以 p95 |
| 采样 | `temperature = 1.0`，未固定 seed | 输出长度不固定 |
| 接口 | OpenAI 兼容 `/v1/chat/completions`，`stream = true` | 四家统一 |

> ⚠️ **本次测试的边界**：只测了 `concurrency = 1`。主要是因为小机器的能力有限。

### 1.4 被测对象

四款引擎都是 Windows 下的本地推理方案，但定位差异很大：

| 引擎 | 定位 | 本次扮演的角色 |
|---|---|---|
| **Bionic**（LM Studio 5.4.1） | 面向普通用户的 GUI 工具，一键加载 GGUF | "开箱即用 + 核显 offload"的默认取向 |
| **Herdsman**（牧马人 v0.5.4-beta1） | 国产本地模型管理与推理服务平台 | "带 MTP 的 GGUF + ollama 调度" |
| **OVMS**（OpenVINO Model Server） | Intel 官方推理服务器，需转 IR + NNCF 量化 | "服务端化 + INT4 + 连续批处理" |
| **Unsloth**（v0.1.804-beta） | 以量化 / 微调生态见长的桌面应用 | "自建 llama.cpp 构建" |

---

## 2. 测试环境

### 2.1 硬件平台

| 项目 | 规格 |
|---|---|
| 设备 | NucBox EVO-T1 |
| CPU | Intel Core Ultra 9 285H @ 2.90 GHz，16 逻辑核（6P + 8E + 2LP-E，无超线程） |
| 核显 | Intel Arc 140T GPU（32 GB 共享内存） |
| 内存 | 64.0 GB（63.5 GB 可用） |
| 系统 | Windows 11 Pro 24H2，build 26100.3476 |
| 独立 GPU | 无 |

> ⚠️ 这是一台**纯 CPU + 核显**的机器。所谓"GPU 加速"本质上是在和系统内存抢带宽，这一点直接决定了后文的很多结论。
>

### 2.2 被测引擎（含探测到的真实后端）

| 引擎 | 端点 | 界面标注的引擎 | **API 自报的真实后端** | 模型 / 量化 | 体积 | **实际推理设备** |
|---|---|---|---|---|---|---|
| Bionic | `localhost:1234` | Vulkan (llama.cpp) Windows | 未上报（LM Studio 5.4.1） | `qwen3.8-27b` / GGUF Q4_K_M | 16.2 GB | **Arc 140T 核显（GPU）** |
| Herdsman | `localhost:8080` | llama.cpp | **ollama v0.6.4-0-ga95fb2fb** | `Qwen3.8-27B` / **GGUF Q4_K_M** | 16.2 GB | **Arc 140T 核显（GPU）** |
| OVMS | `localhost:8000` | — | 未上报（OpenVINO Model Server） | `OpenVINO/Qwen3.8-27B-int4-ov` / INT4 (NNCF) | 14.8 GB | **Arc 140T 核显（GPU）** |
| OVMS-CPU ⭐ | `localhost:8000` | — | 未上报（OpenVINO Model Server） | `OpenVINO/Qwen3.8-27B-int4-ov` / INT4 (NNCF)，**显式指定 CPU 设备** | 14.8 GB | **纯 CPU（唯一一组）** ❗ |
| Unsloth | `127.0.0.1:8888` | GGUF inference engine / Vulkan | 未上报 | `Qwen3.8-27B-GGUF:Q4_K_M` | 16.2 GB | **Arc 140T 核显（GPU）** |

> 📌 **设备矩阵**：五组测试中**只有 OVMS-CPU 一组跑在纯 CPU**，其余四组（Bionic / Herdsman / OVMS / Unsloth）**全部运行在 Arc 140T 核显**上。
> 这一点至关重要：它意味着**各家之间的性能差异不能归因于"有没有 GPU"**，而是同样一颗核显上的**引擎 / 后端实现与 offload 策略**差异（详见 §6.5、§9）。

> ⭐ **OVMS-CPU 是追加的一组对照测试**（`ovms_cpu/` 目录，2026-09-09）：同一 OVMS 服务把推理设备**显式锁定为 CPU** 后重测。它与主表 OVMS 组的差异即"设备选择对 OVMS 的影响"——也是本报告第一次覆盖到的 OVMS CPU 路径真实性能。
>
> ⚠️ **只有 Herdsman 上报了框架与版本**（`ollama / v0.6.4`）。它的 UI 上写的是 "Engine: llama.cpp"，但 HTTP 层是 ollama —— ollama 本身基于 llama.cpp，所以两者不矛盾，但**调度、缓存、内存策略都由 ollama 决定**，这解释了它独有的 prefix cache 行为。

---

## 3. 指标定义

本次测试除常规的 TTFT / 吞吐外，还采集了下面几项进阶指标（均由 [llmbench](https://github.com/megemini/llmbench) 直接输出）—— 本报告的多数结论都建立在它们之上：

| 指标 | 含义 | 为什么重要 |
|---|---|---|
| **Dec TPS**（`decode_throughput`） | 纯解码吞吐 = `completion_tokens / (duration − TTFT − latency)`，**不含 prefill** | 常规 `gen TPS` 的分母包含 TTFT，prefill 慢的引擎会被严重低估。Dec TPS 才是选型真正该看的指标 |
| **ITL**（`itl_mean/median/p95_ms`） | Inter-Token Latency，相邻两个 token 的到达间隔 | 决定"打字机"体感。均值相同、分布不同，体验天差地别 |
| **Cache%**（`cache_hit_ratio`） | 命中服务端 prefix cache 的 prompt token 占比 | 直接暴露 TTFT / prefill 数字是否被缓存污染 |
| **Server Env**（`server_framework` / `version`） | 探测服务端框架与版本 | 揭穿了 Herdsman 的真实后端 |
| `peak_kv_cache_usage_perc` / `peak_num_requests_waiting` | 服务端 KV cache 与排队峰值 | 本次全部为空（需服务端暴露 vLLM 类 `/metrics`） |

---

## 4. 核心数据总览

### 4.1 纯 decode 吞吐（Dec TPS，tok/s）— 不含 prefill

| 引擎 | short (92) | medium (1.28k) | long (9.46k) | 长上下文衰减 |
|---|---:|---:|---:|---:|
| **OVMS** | **5.03** | **4.95** | **4.85** | **-3.6%** ✅ |
| Herdsman | 4.59 | 4.81 | 3.91 | -14.8% ❌ |
| Unsloth | 3.82 | 3.80 | 3.32 | -13.1% |
| OVMS-CPU | 2.60 | 2.51 | **未完成** ❌ | — |
| Bionic | 2.16 | 1.84 | 1.91 | -11.6% |

> ⭐ **OVMS-CPU：显式指定 CPU 后 decode 只剩 ~2.5 tok/s**（约 OVMS 默认设备的一半），且 long（9.46k token）档**未完成（首轮即超时中断，无轮次数据留下）**——CPU 上的 OVMS 处理不了长 prompt（详见 §6.5）。

### 4.2 TTFT 中位数

| 引擎 | short | medium | long |
|---|---:|---:|---:|
| OVMS | **871 ms** | 4078 ms | 6927 ms |
| Herdsman | 3695 ms ⚠️ | 2172 ms ⚠️ | 2567 ms ⚠️ |
| Bionic | 4126 ms | **2704 ms** | **3039 ms** |
| Unsloth | 1736 ms | 16344 ms | **159168 ms** |
| OVMS-CPU | **16943 ms** ❌ | **224023 ms** ❌ | **未完成** ❌ |

⚠️ = 该数值下 92–100% 的 prompt token 命中 prefix cache，非真实 prefill 耗时。
❌ OVMS-CPU 组 short / medium 的 TTFT 中位数比 OVMS 默认设备慢 **19× / 55×**；long 档未完成（首轮即超时中断）。

### 4.3 prefill 吞吐（Prompt TPS 多轮中位数）+ 缓存命中率

| 引擎 | short | medium | long | **Cache%** |
|---|---:|---:|---:|---:|
| Bionic | 23 | **484** | **3172** | 0% |
| OVMS | **112** | 318 | 1376 | 0% |
| Herdsman | 14 ⚠️ | 587 ⚠️ | 3750 ⚠️ | **92% / 99.7% / 99.96%** |
| Unsloth | 53 | 78 | **59** | 0% |
| OVMS-CPU | 5.4 | 5.7 | **未完成** | 0% |

> 📌 **口径说明**：本表（及全文正文）统一为 `prompt_throughput` 的**多轮中位数**（源码直接输出该字段）——对"第 1 轮快、后续轮次退化"的 OVMS / OVMS-CPU 最稳健。附录 A 贴的是 CSV 的 **mean 口径**，两者差异即轮间抖动（见 §7.6）；以 OVMS medium 为例，mean 931 tok/s 是被首轮的 0.65 s 拉高，median 只有 318。
> ⭐ OVMS-CPU 的 prefill **中位数只有 ~5–6 tok/s**（多轮中位数口径），与 decode 的 2.5 tok/s 同一数量级（健康引擎的 prefill 通常是 decode 的几十倍）。但需注意：这个中位数只反映**后续轮次**——其第 1 轮 prefill 实际达到 ~1000 tok/s（medium 档：mean 358 = (1063 + 5.7 + 5.7) / 3，min_ttft 仅 1.26 s，见 §7.6），说明 CPU 路径**首轮仍有批处理能力，从第 2 轮起退化到 ~5 tok/s**，退化原因待查；长 prompt 按退化后的速度每轮要跑几千步，最终超时未完成。

### 4.4 ITL（Inter-Token Latency）

| 引擎 | short 均值/中位/p95 (ms) | long 均值/中位/p95 (ms) | 输出形态 |
|---|---|---|---|
| **Unsloth** | 267 / 254 / 386 | 303 / 292 / 416 | ✅ **平滑**（中位≈均值） |
| **OVMS** | 219 / 201 / 468 | 230 / 207 / 553 | ✅ **平滑** |
| Herdsman | 219 / 0.24 / 554 | 258 / 0.20 / 679 | ⚠️ **攒批突刺** |
| Bionic | 470 / 0.40 / **1242** | 527 / 0.36 / **1357** | ❌ **严重攒批突刺** |
| OVMS-CPU ⭐ | 405 / 384 / 545 | medium: 441 / 398 / 787（long 未完成） | ✅ 平滑（CPU 上仍逐 token 输出） |

---

## 5. 图表

### 5.1 纯 decode 吞吐（已剔除 prefill）

![纯 decode 吞吐对比](https://origin.picgo.net/2026/09/09/01_decode_tps2e9f211c309580aa.png)

> **OVMS 默认设备三档全胜**（5.03 / 4.95 / 4.85），且是唯一把长上下文衰减压在 5% 以内的引擎。
> Bionic 全程垫底（1.84–2.16），不到 OVMS 的 40%。
> ⭐ OVMS-CPU（浅绿柱）只有 ~2.5 tok/s 且 **long 档未完成（×，首轮超时中断）** —— 纯 CPU 把 OVMS 拉到与 Bionic 相近的水平（2.5 vs 1.91，**纯 CPU 甚至略高**）。

### 5.2 首 token 延迟 TTFT（对数刻度）

![TTFT 对比](https://origin.picgo.net/2026/09/09/02_ttft2e88076c08dcbb80.png)

> 短 prompt：OVMS **871 ms** 是唯一进入 1 秒内的。
> 长 prompt：unsloth **159,168 ms（159 秒）** 比第二名（Bionic 3039 ms）慢 **52 倍**，比 OVMS 慢 **23 倍**。
> ⭐ OVMS-CPU 把短/中 prompt 的 TTFT 拉到 **16.9 s / 224 s**（对数刻度下直接"起飞"），long 档未完成（×，首轮即超时中断）。
> ⚠️ Herdsman 的三根柱子均含 92–100% prefix cache 红利，不与其余三家同尺度比较。

### 5.3 流式输出平滑度（ITL）

![ITL 对比](https://origin.picgo.net/2026/09/09/03_itl00b15febaee0fe53.png)

> **判据**：平滑流式 ⟹ `ITL 中位数 ≈ 均值 ≈ 1/decode_tps`。
> - OVMS 中位 207 ms / 均值 230 ms、unsloth 中位 292 ms / 均值 303 ms → **真·逐 token 流式**。
> - Herdsman 中位 **0.20 ms**、Bionic 中位 **0.36 ms** → 绝大多数 token 是"同一批"一起到达的，服务端在攒批刷送。
> - Bionic 的 ITL p95 达 **1357 ms**：要等约 1.36 秒才看到一批字，是四家里体感最差的。
> - ⭐ OVMS-CPU 因 long 档未完成未出现在本图中，但其 short / medium 的 ITL 均值/中位比值分别为 **1.06 / 1.11**（§4.4 表，405.37/383.71、440.79/397.55），说明 CPU 路径也保持逐 token 流式，只是速度更慢。

### 5.4 prefill 吞吐与缓存污染

![prefill 吞吐对比](https://origin.picgo.net/2026/09/09/04_prefillda31b91f805919dd.png)

> 斜线填充的 Herdsman 柱子代表 **92–100% 的 prompt token 来自 prefix cache**，属于缓存回放而非真实算力，**不参与排名**。
> 排除后：Bionic 真实 prefill 最快（9.46k 档 3172 tok/s），OVMS 次之（1376），unsloth 停留在 59 tok/s 且完全不随 prompt 变长提速。
> ⭐ 近零高度的 OVMS-CPU 浅绿柱（多轮中位数 5–6 tok/s，× = long 未完成）反映的是退化后的后续轮次：其第 1 轮 prefill 实测 ~1000 tok/s（见 §4.3 注、§7.6）——CPU 路径首轮尚有批处理能力，后续轮次退化严重，退化原因待查。
> 📌 图与正文均为 `prompt_throughput` **多轮中位数**口径；附录 A 的 CSV 行是 mean 口径，两者差异源自轮间抖动（§7.6）。

### 5.5 长上下文稳定性

![长上下文衰减](https://origin.picgo.net/2026/09/09/05_context_decay525fc24d9e15ebf0.png)

> OVMS **-3.6%** 一枝独秀；Herdsman **-14.8%** 垫底，与其启动前自报的内存超配直接相关（见 §7.5）。
> ⭐ OVMS-CPU 的 long 档未完成（首轮即超时中断），图上以 **TIMEOUT** 标注、无法计算衰减——它连"降速"的机会都没有。

### 5.6 端到端耗时（TTFT + 生成 512 tokens）

![端到端耗时](https://origin.picgo.net/2026/09/09/06_end_to_end84d2c47c40218958.png)

> OVMS 在三档 prompt 上都稳定在 103–113 秒，是唯一"不随 prompt 变长而明显变慢"的引擎。
> Bionic 与 unsloth 在长 prompt 上分别需要 271 秒和 313 秒。
> ⭐ OVMS-CPU 在同口径下 short / medium 就要 **214 / 428 秒**（其 TTFT 中位数已占大头），long 档未完成（×）——即使只看端到端，CPU 配置也完全不可用。

---

## 6. 逐引擎深度剖析

### 6.1 OVMS（OpenVINO Model Server，默认设备）— 综合第一

**实现**：模型经 NNCF 做 INT4 权重量化并转 OpenVINO IR；服务端 continuous batching + paged attention，16 unary threads。本组为**默认设备配置**（未显式指定推理设备），**实际执行在 Arc 140T 核显**——与 Bionic / Herdsman / Unsloth 相同；本次唯一的纯 CPU 组是 §6.5 的 OVMS-CPU 对照。

**优点**
- **纯 decode 全场第一且最稳**：5.03 / 4.95 / 4.85，三档波动 < 4%。
- **长上下文衰减仅 -3.6%**，四家唯一把衰减压进 5% 的 —— paged attention 的收益被 Dec TPS 指标清晰地量化了出来。
- **短 prompt TTFT 871 ms**，是第二名（unsloth 1736 ms）的一半。对短问答 / Agent 工具调用这类高频短请求是决定性优势。
- **输出平滑**：ITL 均值 219 / 中位 201 / p95 468 ms，中位数≈均值，是真正的逐 token 流式。
- 唯一具备服务端化能力：continuous batching、gRPC + REST、Prometheus、模型热更新。

**缺点 / 卡点**
- **长 prompt prefill 只有 1376 tok/s**（多轮中位数），是 Bionic（3172）的 43%。
- **TTFT 三轮之间抖动 7 倍**：long prompt 三轮分别为 **0.98 s / 6.93 s / 6.97 s**，第 1 轮明显偏快。`Cache% = 0` 排除了 prefix cache 解释；但"首轮偏快"同样见于 Bionic / Herdsman 的 short 档与 OVMS-CPU，并非 OVMS 独有（§7.6），因此"KV block 池回收 / 动态 shape 编译缓存"目前只是**待验证假设**，需抓 OVMS 日志确认。
- 无投机解码 / MTP，decode 上限被单步前向延迟锁死。
- 部署门槛最高。
---

### 6.2 Herdsman（牧马人 v0.5.4-beta1）— 数字漂亮但有水分

**实现**：Go 编写的本地模型管理与推理平台，**后端实际是 ollama v0.6.4**（UI 标注 "Engine: llama.cpp"，API 自报 `ollama`），**推理跑在 Arc 140T 核显**上。

**优点**
- decode 4.59–4.81（短/中），仅次于 OVMS。
- **prefix cache 命中率 92–100%** —— 这是真实能力，对"固定 system prompt + 多轮对话"场景是实打实的优势。
- TTFT 2.17–3.70 s，全程处于低位。

**缺点 / 卡点**
- ⚠️ **prefill 数字不可比**：92–100% 的 prompt token 来自缓存，3750 tok/s（多轮中位数）是"缓存回放"速度。llmbench 在正式测试前会先用完整 prompt 发一次 `max_tokens=1` 探测请求，导致 **Herdsman 从第 1 轮就命中缓存**。
- ❌ **内存超配运行**：启动前它自己弹出"预估需要 43.5 GB，当前可用 39.3 GB，不足，继续可能导致启动失败和系统崩溃"。这直接导致 **长上下文 decode 掉速 -14.8%（四家最差）** —— 长 prompt 需要更大的 KV cache，内存不够就开始换页。
- ⚠️ **输出攒批**：ITL 中位数 0.20–0.24 ms，均值 219–258 ms，p95 554–679 ms。实际体感是"卡半秒、吐一串"。
- 短 prompt TTFT（3695 ms）仍高于 medium（2172 ms），固定开销约 1.5 s（与 prompt 长度无关）。
- tokenizer 与其他三家不一致（short prompt 报 52 tokens，其他三家都是 92）。

---

### 6.3 Bionic（LM Studio 5.4.1 + GGUF/Vulkan）— prefill 之王，decode 之末

**实现**：LM Studio 5.4.1，已安装扩展 `GGUF` + `Vulkan (llama.cpp) Windows`，权重与 KV 放在 Arc 140T 核显。

**优点**
- 🏆 **真实 prefill 最快**：Cache% = 0，长 prompt 仍做到 **3172 tok/s**，是 OVMS 的 2.3 倍、unsloth 的 54 倍。medium 档 484 tok/s 也是第一。
- **长 prompt TTFT 3.04 s**，在无缓存的引擎里最低。

**缺点 / 卡点**
- ❌ **纯 decode 最慢**：1.84–2.16 tok/s，只有 OVMS 的 40%。
- ❌ **输出最卡**：ITL p95 **1357 ms**，中位数 0.36 ms。也就是说要等约 1.36 秒才看到一批字，是四家里体感最差的。
- 短 prompt TTFT（4126 ms）高于 medium（2704 ms），有 ~1.4 s 与长度无关的固定开销。

---

### 6.4 Unsloth（桌面版）— prefill 灾难，其余尚可

**实现**：Unsloth 桌面应用，内置自己的 llama.cpp 构建（较早期版本），GGUF Q4_K_M，Vulkan 设备 Arc 140T。

**优点**
- ✅ **输出最平滑**：ITL 均值 267 / 中位 254 / p95 386 ms，`p95/均值 = 1.4`，四家最好的流式体验。
- **decode 3.32–3.82 tok/s**，是 Bionic 的 1.7–2.1 倍 —— 两者**同为 llama.cpp + Vulkan、同样跑在 Arc 140T 核显**、权重也是同一份 Q4_K_M，因此差距只能来自**构建版本与批处理 / offload 配置**。反过来说：**核显并不是 decode 的瓶颈，软件栈才是**。
- 长上下文衰减 -13.1%，略好于 Herdsman。
- 短 prompt TTFT 1736 ms，仅次于 OVMS。

**缺点 / 卡点**
- ❌ **prefill 恒定 59 tok/s，慢 23–54 倍**。9.46k prompt 的 TTFT 高达 **159.2 s**（2 分 39 秒）。
- ❌ **prefill 吞吐不随 prompt 变长而提升**：53 / 78 / 59（其他三家是 23→3172、112→1376）。这是 **prefill 未被批处理** 的决定性指纹。
- 根因判断：llama.cpp 构建过旧，或 ubatch size 过小，或 Vulkan kernel 路径存在劣化（本组跑在 Vulkan 核显上）。

---

### 6.5 OVMS-CPU（显式锁定 CPU）— 控制组：设备选择是 OVMS 的生命线

"OVMS 的快，到底是引擎好还是设备好？"——把同一 OVMS 服务的推理设备**显式锁到 CPU** 后重测。

| 指标 | short (92) | medium (1.28k) | long (9.46k) |
|---|---:|---:|---:|
| Dec TPS | 2.60 | 2.51 | **未完成** ❌（首轮即超时中断，无数据） |
| TTFT 中位数 | 16.94 s | 224.02 s | — |
| prefill 吞吐 | 5.4 tok/s | 5.7 tok/s | — |
| ITL 均值 / 中位 / p95 | 405 / 384 / 545 ms | 441 / 398 / 787 ms | — |
| Cache% | 0% | 0% | — |

**结论一：CPU 路径把 OVMS 的所有优势清零。** decode 从 5.0 掉到 **2.5 tok/s**——注意它仍**高于跑在核显上的 Bionic（1.91）**；short 的 TTFT 从 871 ms 涨到 **16.9 s**（约 19 倍）；medium 更是到 **224 s**（约 55 倍）。

**结论二：CPU 路径的 prefill 首轮尚有批处理能力，但后续轮次严重退化。** 默认设备的 prefill 随 prompt 变长而大幅提速（112→1376 tok/s，多轮中位数），说明它在按 token 批量喂 GPU；CPU 组的第 1 轮同样如此（medium 档首轮 ~1063 tok/s、TTFT 仅 1.26 s；short 档首轮 ~79 tok/s），说明批处理能力在 CPU 上依然存在；但从第 2 轮起 prefill 退化到 **~5–6 tok/s**（多轮中位数被拉到这个量级，只比 decode 快约 1 倍，而健康引擎的 prefill 通常是 decode 的几十倍），退化原因（KV cache / 内存 / 调度）待查。9.46k 的 long prompt 若按退化后的速度，每轮需要 **~30 分钟** 才能出第一个字，实测首轮即超时中断（无轮次数据留下）。

**结论三：设备选择 = OVMS 性能的生命线，这不是参数微调能救的。** 默认设备跑在加速路径（Arc 140T 核显）上是 5 tok/s 的 decode + 批量 prefill；锁到 CPU 后跌到 2.5 tok/s。对 OVMS 而言，"在什么设备上跑"远比"调什么参数"重要。

**对照意义**：本组把"引擎差异"与"设备差异"解耦。需要先澄清一张设备表——**本次只有 OVMS-CPU 落在纯 CPU，其余四组都在同一颗 Arc 140T 核显上**（§2.2）：

| 组别 | 推理设备 | decode（long 档） |
|---|---|---:|
| OVMS（默认） | **核显 Arc 140T** | 4.85 |
| Herdsman | **核显 Arc 140T** | 3.91 |
| Unsloth | **核显 Arc 140T** | 3.32 |
| **OVMS-CPU** ⭐ | **纯 CPU** | **2.51**（short/medium；long 首轮超时中断） |
| Bionic | **核显 Arc 140T** | **1.91** |

由此可得出两个更准确、也更强的结论：

1. **同设备 ≠ 同性能。** 四组都在同一颗核显上，decode 却相差 **2.6 倍**（4.85 → 1.91）。这个差距不能再用"有没有 GPU"解释，只能归于**后端 kernel 实现与批处理 / offload 策略**（OpenVINO GPU plugin 好，llama.cpp Vulkan 差）。
2. **offload 到核显也可能比纯 CPU 更慢。** 跑核显的 **Bionic 只有 1.91 tok/s，反而低于纯 CPU 的 OVMS-CPU（2.51）**。也就是说 LM Studio 的 Vulkan offload 在这台"共享内存 + 核显"的机器上是**净负收益**——它既没有拿到 GPU 算力红利，又额外承担了跨环形总线的访问延迟。这比"设备好所以快"的说法精确得多，也与 §6.3、§9 的判断一致。

---

## 7. 关键差异归因

### 7.1 prefill：只在 cache=0 的三家之间比较才有意义

| 引擎 | short | medium | long | Cache% | 结论 |
|---|---:|---:|---:|---:|---|
| **Bionic** | 23 | **484** | **3172** | 0% | 真实 prefill 最快 |
| **OVMS** | **112** | 318 | 1376 | 0% | 短 prompt 最快，长 prompt 中等 |
| Unsloth | 53 | 78 | **59** | 0% | 完全无批处理 |
| Herdsman | 14 | 587 | 3750 | **92–100%** | ⚠️ 缓存回放速度，不参与排名 |
| OVMS-CPU ⭐ | 5.4 | 5.7 | **未完成** | 0% | 无批处理 + 纯 CPU，最差 |

- Bionic 在 medium/long 上领先 OVMS，靠的是 **Vulkan 后端成熟的 batched matmul**（iGPU 的强项恰好是批量矩阵运算）。
- OVMS（默认设备）**已核实跑在 Arc 140T 核显**，获得 112→1376 tok/s 的批量 prefill（多轮中位数）；**而锁到 CPU（OVMS-CPU）后立刻掉到 5–6 tok/s**——证明这条曲线的"算力"来自设备而非引擎本身。
- **unsloth 的 59 tok/s 且与长度无关**，是唯一真正的异常。

### 7.2 decode：设备 + kernel 决定单步延迟

| 引擎 | decode (long) | 相对 OVMS | 关键归因 |
|---|---:|---:|---|
| OVMS（默认设备） | 4.85 | 1.00× | OpenVINO INT4 + paged attention；**与其他三家同跑 Arc 140T 核显，kernel 实现最优** |
| Herdsman | 3.91 | 0.81× | ollama 后端尚可，但内存超配拖累长上下文 |
| Unsloth | 3.32 | 0.68× | llama.cpp Vulkan 正常（**同跑核显**，非 CPU 路径） |
| OVMS-CPU ⭐ | 未完成 | — | 同一 OVMS 锁 CPU 后仅 2.51（short/medium，long 首轮超时中断），解码全看设备 |
| Bionic | 1.91 | **0.39×** | iGPU Vulkan 对 batch=1 decode 是**负优化**（**低于纯 CPU 的 OVMS-CPU 2.51**，负优化坐实） |

> 📌 **MTP 的贡献无法从本次数据中分离**。Herdsman 标注为 "MTP-enabled"（投机解码），其 decode（4.59）比 unsloth（3.82）高 20%；但 **OVMS 完全没有 MTP 却拿到最高的 5.03**，说明后端差异足以解释这个差距。要定论必须做"开/关 MTP"的 A/B 对照。

### 7.3 ITL：谁在真正流式输出

| 引擎 | 均值/中位 比值 | 形态 | 体感 |
|---|---:|---|---|
| Unsloth | 1.05 | 平滑 | 每 ~260 ms 出一个字，稳定打字机 |
| OVMS | 1.09 | 平滑 | 每 ~200 ms 出一个字，稳定打字机 |
| Herdsman | ~921 | 攒批 | 每 ~0.6 s 吐一批 |
| Bionic | ~1166 | 攒批 | 每 ~1.36 s 吐一批，最卡 |

> 判据：平滑流式引擎的 ITL **均值/中位 ≈ 1**（unsloth 1.05、OVMS 1.09）；攒批引擎中位数被批内间隔拉到 ~0.2–0.4 ms，比值暴涨到 900–1200。

这是**对交互体验影响最大**、却又最容易被"平均吞吐"掩盖的一项 ——
**Bionic 与 unsloth 的 decode 差距（1.91 vs 3.32）已经不小，但加上 ITL 后体感差距还要再放大一倍。**

### 7.4 prefix cache：Herdsman 的"甜蜜陷阱"

- llmbench 会先用完整 prompt 发一次探测请求（`probe_tokens`），Herdsman 借此**从第 1 轮就命中缓存（92%+）**。
- 对真实业务：如果工作负载是"长 system prompt + 多轮追问"，这个缓存是**真实的巨大优势**；如果是"每次都是全新长文档"，则完全用不上。
- 其余三家 Cache% 全为 0 —— 但需要注意：**0% 可能是没开缓存，也可能是服务端不上报 `prompt_tokens_details.cached_tokens` 字段**。从 Bionic/unsloth 三轮 TTFT 毫无改善来看，更可能是前者。
- 🔍 **一个一致的细节**：Herdsman 三档每轮都恰好漏缓存 ~12 个 prompt token（short：156 − 144 = 12；medium：3723 − 3711 = 12；long：28254 − 28242 = 12，即每轮约 4 个）——说明 ollama 是**固定复用前缀、但尾部 ~4 token 始终重算**。因此它的命中率永远到不了 100%（long 档 99.96%），正文用"92–100%"而不是"100%"是准确的。

### 7.5 内存：Herdsman 的超配是长上下文掉速的主因

| 引擎 | 长上下文衰减 | 说明 |
|---|---:|---|
| OVMS | -3.6% | paged attention，衰减最小 |
| Bionic | -11.6% | 衰减来自 iGPU **共享显存** |
| Unsloth | -13.1% | llama.cpp 构建较旧 |
| **Herdsman** | **-14.8%** ❌ | **强相关**：启动前已自报内存不足（需 43.5 GB / 可用 39.3 GB） |

### 7.6 轮间抖动："首轮快、后续退化"不是 OVMS 独有

逐项核对每组的"首轮 TTFT"（旧版工具口径，`min_ttft` 取第 1 轮）与 TTFT 中位数后，结论要**降级**：

| 引擎 | short | medium | long |
|---|---:|---:|---:|
| OVMS | 0.62 → 0.87 s（1.4×） | 0.65 → 4.08 s（**6.3×**） | 0.98 → 6.93 s（**7.1×**） |
| OVMS-CPU | 1.22 → 16.9 s（**13.9×**） | 1.26 → 224 s（**177.9×**） | 未完成 |
| Herdsman | 1.84 → 3.69 s（2.0×） | 1.92 → 2.17 s（1.1×） | 2.30 → 2.57 s（1.1×） |
| Bionic | 2.39 → 4.13 s（1.7×） | 2.93 → 2.70 s（**0.9×，首轮反而最慢**） | 2.80 → 3.04 s（1.1×） |
| Unsloth | 1.69 → 1.74 s（1.0×） | 16.34 → 16.34 s（1.0×） | 159.17 → 159.17 s（1.0×） |

- 最显著的是 OVMS / OVMS-CPU（6–178×），但 **Bionic、Herdsman 的 short 档也出现首轮偏快（1.7–2.0×），而 Bionic 的 medium 档首轮反而最慢**——方向并不统一，说明它**不是某个引擎独有的稳定行为**，更可能是平台级/调度级的偶发因素（核显驱动节流、共享内存带宽竞争等）叠加。
- **Unsloth 三轮完全一致**是强证据：抖动与"有没有做批处理/池化"无关。
- OVMS-CPU 的 medium 档原始 JSON 显示第 1 轮 prefill 约 **1063 tok/s**（TTFT 仅 1.26 s），而三轮 mean 358 / median 5.7 —— 即"首轮正常、后续退化"在 prefill 上同样成立，且比 TTFT 的抖动更极端（§6.5 结论二）。

---

## 8. 测试截图

### 8.1 Herdsman

![Herdsman 启动前资源不足告警](https://origin.picgo.net/2026/09/09/-2026-09-08-202116b58ffbbd40d458ef.png)

> 启动前弹窗："预估需要 43.5 GB，当前可用 39.3 GB，不足。继续可能导致启动失败和系统崩溃。"
> 我点了 `Start anyway` 强行拉起。这是它长上下文掉速 -14.8%（四家最差）最可能的原因（§7.5）。

![Herdsman 运行期资源占用](https://origin.picgo.net/2026/09/09/-2026-09-08-205222463a6d3da584dec9.png)

### 8.2 Bionic

![Bionic 已安装扩展](https://origin.picgo.net/2026/09/09/-2026-09-08-22070860c3e251d167b375.png)

> `已安装的扩展`：`GGUF` + `Vulkan (llama.cpp) Windows` —— 确认 Bionic 走的是 llama.cpp Vulkan 后端、GGUF 权重。

![Bionic 测试期资源占用](https://origin.picgo.net/2026/09/09/-2026-09-08-2206571d44e165b9db299d.png)

### 8.3 OVMS

![OVMS 测试期服务端状态](https://origin.picgo.net/2026/09/09/-2026-09-08-2230228d2b1e800116f011.png)

### 8.4 Unsloth

![Unsloth 测试期客户端状态](https://origin.picgo.net/2026/09/09/-2026-09-08-212735b655f36650aa039d.png)

### 8.5 OVMS-CPU：long 档超时中断现场（未完成）

![OVMS-CPU 测试超时中断截图](https://origin.picgo.net/2026/09/09/-2026-09-09-1232264c0db86457f86411.png)

> 显式锁定 CPU 后，TUI 只跑完了 short / medium 两档；long（9.46k token）**仅发出第 1 轮请求即报错中断**（图中状态栏出现 error），未能产生任何轮次数据——因此导出文件中没有 long 档条目，正文也一律表述为"未完成"。short / medium 的 TTFT 中位数仍分别恶化到 16.9 s / 224 s。

---

## 9. 性能瓶颈与卡点清单

| 引擎 | 首要卡点 | 次要卡点 | 可修复性 |
|---|---|---|---|
| **OVMS** | 长 prompt prefill 1376 tok/s（Bionic 的 43%，且依赖核显 offload） | TTFT 轮间抖动 7×（成因待查，§7.6）；无投机解码 | 🟡 prefill 靠调参改善；抖动需查 KV block 池；⚠️ 设备回退到 CPU 即跌档（§6.5） |
| **Herdsman** | **内存超配**（启动前自报需 43.5 GB / 可用 39.3 GB） | prefill 数字含 92–100% 缓存红利；输出攒批 | 🟢 已是 Q4_K_M（无需改档位）；换更小量化或加内存即可显著改善 |
| **Bionic** | **iGPU Vulkan offload 拖死 decode**（1.91 tok/s，比纯 CPU 的 OVMS-CPU 2.51 还慢） | 输出攒批，ITL p95 1357 ms | 🟢 同为核显的 Unsloth 快 1.7–2.1 倍 → 属构建/配置问题：升级 llama.cpp 或调整 offload 层数（`-ngl`）可改善 |
| **Unsloth** | **prefill 无批处理（59 tok/s）** | llama.cpp 构建过旧 | 🟢 升级构建 + `-b/-ub` 参数，预期数量级改善 |
| **OVMS-CPU** ⭐ | **纯 CPU 跑 27B：prefill 后续轮次退化至 ~5 tok/s（首轮 ~1000）+ decode 减半** | long 档未完成（首轮超时中断） | 🔴 非参数问题：改用核显（默认设备）即解决，见 §6.5 |

### 平台级瓶颈（与引擎无关）

1. **内存带宽锁死 decode 上限。** 27B 模型即便 INT4/Q4 也要 ~16 GB 权重，实测 3.3–5.0 tok/s 基本是这条曲线上的合理值。**想再快只能靠投机解码 / MTP，换引擎收益有限。**
2. **设备不是分水岭，软件栈才是。** 本轮**只有 OVMS-CPU 跑在纯 CPU**，其余四组都在同一颗 Arc 140T 核显上（§2.2）。同一颗核显上 decode 从 **4.85（OVMS）到 1.91（Bionic）相差 2.6 倍**；而唯一的纯 CPU 组 **OVMS-CPU 2.51 tok/s 反而高于跑核显的 Bionic**。结论：在这台"共享内存 + 核显"的机器上，**offload 到 GPU 并不天然更快**——决定成败的是引擎 kernel 与批处理 / offload 配置（OpenVINO GPU plugin 优、llama.cpp Vulkan 差）。

---

## 10. 选型建议

| 使用场景 | 推荐 | 理由 |
|---|---|---|
| **单并发 · 短问答 / Agent 工具调用** | **OVMS** | TTFT 871 ms（唯一 sub-second）、decode 5.03、输出平滑 |
| **单并发 · 长文档 / RAG** | **OVMS** | decode 4.85 + 衰减 -3.6%，端到端 112.5 s 最短 |
| **固定 system prompt + 多轮对话** | **Herdsman** | 92–100% prefix cache 命中是真实优势，TTFT 稳定 2.2–2.6 s |
| **多并发服务**（需先补测） | **OVMS** | 唯一有 continuous batching + paged attention 的服务端方案 |
| **看重输出流畅度** | **Unsloth ≈ OVMS** | ITL 中位数≈均值，真·打字机体验 |
| **短 prompt 为主、且愿意调优** | **Unsloth** | 修好 prefill 后 decode 3.8 + 最平滑输出，性价比高 |
| **暂不推荐** | **Bionic** | 在本平台 iGPU offload 是负优化：decode 垫底（1.91）+ 输出最卡；同为核显的 Unsloth 快 2 倍，**纯 CPU 的 OVMS-CPU（2.51）都比它快** |
| **长 prompt 场景避免** | **Unsloth / OVMS-CPU** | Unsloth 要等 159 s 才出第一个字；OVMS 锁 CPU 后 long 档不可用（首轮即超时中断） |

> ⚠️ 上表中所有引擎默认都运行在**核显 Arc 140T** 上（**已核实**；唯 OVMS-CPU 为显式锁 CPU 的对照）。OVMS-CPU 对照证明：一旦回退到 CPU，decode 减半、prefill 后续轮次退化到 5–6 tok/s（首轮 ~1000）、long 档不可用（§6.5）。部署时仍建议显式指定 `target_device=GPU`，避免服务重启后被自动回退到 CPU。

### 端到端耗时参考（生成 512 tokens）

| 场景 | OVMS | Herdsman | Bionic | Unsloth |
|---|---:|---:|---:|---:|
| 短问答（92 in） | **102.7 s** | 115.2 s | 241.0 s | 135.7 s |
| 中等文档（1.28k in） | **107.5 s** | 108.6 s | 281.0 s | 151.1 s |
| 长文档 RAG（9.46k in） | **112.5 s** | 133.5 s | 271.1 s | 313.4 s |

> 注 1：Herdsman 的 TTFT 含 prefix cache 红利，若缓存未命中需按其 prefill 能力重算。
> 注 2：OVMS-CPU 未列入——long 档未完成（首轮即超时中断）；short / medium 端到端分别需 91 s / 205 s（单轮），仍是劣于 OVMS 默认设备的配置。
> 注 3：本表为**估算口径**（TTFT 中位数 + 512 / Dec TPS），非实测轮次耗时。实测中 short 问答类请求往往**提前结束**（Bionic short 只生成 262 token、实测首轮耗时 124.2 s，远小于按 512 token 估算的 241 s），因此该列是高估的悲观值。

**一句话总结**

> **OVMS 是综合赢家** —— 它的 decode 最快、最稳、输出最平滑，短 prompt TTFT 唯一破 1 秒，且不靠任何缓存红利。**但赢的前提是跑在 Arc 140T 核显上**：显式锁 CPU 后 decode 减半、prefill 后续轮次退化到 5–6 tok/s（首轮 ~1000）、long 档不可用（§6.5）。
> **Herdsman 的数字要打折扣看** —— 92–100% 的 prefix cache 命中让它的 prefill 数字失去可比性，而启动时的内存超配又在长上下文上把它拖了下来；但在"固定 system prompt 多轮对话"场景它依然是最优解。
> **Bionic 和 Unsloth 是同一个镜像问题的两面** —— 前者 prefill 最强、decode 最弱，后者 decode 尚可、prefill 灾难；**两者跑在同一颗 Arc 140T 核显上**（§2.2），短板因此只能是**软件配置问题**（offload 策略 / 批处理参数 / 构建版本），不是硬件问题，都值得再调一轮。而 OVMS-CPU 提供了一把尺子：纯 CPU 跑 27B INT4 是 **2.5 tok/s** —— 它**比跑在核显上的 Bionic（1.91）还快**，说明在这台共享内存的机器上，把权重 offload 到核显并非天然占优，软件栈没调好时 iGPU 反而是负资产。

---

## 附录 A：原始 CSV 数据

<details>
<summary>展开</summary>

```
# Bionic  (localhost:1234, qwen3.8-27b, latency 58.03ms, Cache 0%)
label,tok,runs,gen_med,dec_tps,prompt_tps,ttft_med,ttft_p95,itl_mean,itl_med,itl_p95,cache,dur,compl
short,92,3,2.1103,2.1619,27.99,4.1263,4.2676,470.25,0.4034,1241.59,0.0,124.21,262
medium,1281,3,1.8219,1.8363,472.17,2.7036,2.9323,546.28,0.4723,1271.27,0.0,271.86,512
long,9458,3,1.8367,1.9085,3242.53,3.0394,3.0979,526.91,0.3576,1357.31,0.0,172.70,317

# Herdsman  (localhost:8080, Qwen3.8-27B, latency 55.82ms, 后端 ollama v0.6.4)
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

# OVMS-CPU（追加 2026-09-09，显式锁定 CPU；localhost:8000, Cache 0%）
label,tok,runs,gen_med,dec_tps,prompt_tps,ttft_med,ttft_p95,itl_mean,itl_med,itl_p95,cache,dur,compl
short,92,3,2.2590,2.5997,29.67,16.9435,19.4817,405.37,383.71,544.99,0.0,91.43,239
medium,1281,3,1.1963,2.5149,358.25,224.0229,225.6461,440.79,397.55,786.81,0.0,205.32,512
long,9458,——未完成：首轮即超时/连接中断，测试中断，无轮次数据留下——
```

单位：TTFT / duration = 秒；ITL = 毫秒；cache = 命中率（0–1）。
