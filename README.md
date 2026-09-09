# Qwen3.8-27B 本地推理引擎横评

在一台 **无独显** 的小主机（Intel Core Ultra 9 285H + Arc 140T 核显 + 64 GB）上，用同一份 Qwen3.8-27B 权重横向对比四款本地推理引擎，并额外加了一组"锁 CPU"对照组，用来回答一个问题：

> **本地跑 27B 到底能跑多快？各引擎报出来的 tok/s 能不能直接比？**

## 📄 报告

| 语言 | 链接 |
|---|---|
| 中文（主报告） | [Qwen3.8-27B\_四引擎性能对比报告\_发布版.md](./Qwen3.8-27B_%E5%9B%9B%E5%BC%95%E6%93%8E%E6%80%A7%E8%83%BD%E5%AF%B9%E6%AF%94%E6%8A%A5%E5%91%8A_%E5%8F%91%E5%B8%83%E7%89%88.md) |
| English | [Qwen3.8-27B_Four-Engine_Benchmark_EN.md](./Qwen3.8-27B_Four-Engine_Benchmark_EN.md) |

## 结论速览

**综合排名（单并发，纯 decode 吞吐 tok/s）**

| 排名 | 引擎 | decode（短/中/长） | 长 prompt TTFT | ITL p95 | 一句话 |
|---|---|---|---|---|---|
| 🥇 | **OVMS**（OpenVINO Model Server，核显） | **5.03 / 4.95 / 4.85** | 6.93 s | 553 ms | 最快且最稳，短 prompt TTFT 871 ms 唯一破 1 秒 |
| 🥈 | **Herdsman** | 4.59 / 4.81 / 3.91 | 2.57 s ⚠️ | 679 ms | 数字含 92–100% prefix cache 红利，需打折看 |
| 🥉 | **Bionic**（LM Studio） | 2.16 / 1.84 / 1.91 | **3.04 s** | **1357 ms** | 真实 prefill 最快（3172 tok/s），但 decode 垫底、输出最卡 |
| 4 | **Unsloth** | 3.82 / 3.80 / 3.32 | **159.2 s** | 416 ms | prefill 未被批处理（59 tok/s），长 prompt 基本不可用 |
| — | **OVMS-CPU**（对照组，锁 CPU） | 2.60 / 2.51 / 未完成 | 16.9 s / 224 s / — | 545 ms | 纯 CPU 跑 27B 的参照系，long 档超时中断 |

![纯 decode 吞吐对比](https://origin.picgo.net/2026/09/09/01_decode_tps2e9f211c309580aa.png)

**四条关键结论**

1. **别只看 tok/s。** Herdsman 的 3750 tok/s prefill 里 **92–100% 的 prompt token 来自服务端 prefix cache**，测的是缓存回放而非真实算力（其余三家 Cache% 均为 0）。
2. **只有 OVMS 和 Unsloth 在真正流式输出。** ITL 均值/中位比值 1.09 / 1.05（≈1 即平滑）；Bionic 1166、Herdsman 921 —— 后两者攒够一批再吐，体感"卡一下、吐一串"。
3. **同设备 ≠ 同性能。** 五组里只有 OVMS-CPU 跑纯 CPU，其余四组都在同一颗 Arc 140T 核显上，decode 却相差 **2.6 倍**（4.85 → 1.91），差异来自 kernel 实现与 offload 策略，不是"有没有 GPU"。
4. **offload 到核显未必更快。** 跑核显的 Bionic 只有 1.91 tok/s，**低于纯 CPU 的 OVMS-CPU（2.51）** —— 在这台共享内存的机器上，软件栈没调好时 iGPU 反而是负资产。

## 测试平台

| 项目 | 规格 |
|---|---|
| 设备 | NucBox EVO-T1 |
| CPU | Intel Core Ultra 9 285H @ 2.90 GHz（6P + 8E + 2LP-E，无超线程） |
| 核显 | Intel Arc 140T GPU（32 GB 共享内存），INT8 峰值 77 TOPS |
| 内存 | 64.0 GB（63.5 GB 可用） |
| 系统 | Windows 11 Pro 24H2（build 26100.3476） |
| 独显 | 无 |

> 这台机器是"CPU + 核显共享系统内存"结构，所谓 GPU 加速本质上是在和系统内存抢带宽 —— 这一点决定了报告里的大部分结论。

## 被测引擎

| 引擎 | 版本 | 后端 | 推理设备 |
|---|---|---|---|
| **Bionic** | LM Studio 5.4.1 | llama.cpp + Vulkan | Arc 140T 核显 |
| **Herdsman** | v0.5.4-beta1 | ollama v0.6.4（UI 标注 llama.cpp） | Arc 140T 核显 |
| **OVMS** | OpenVINO Model Server | INT4（NNCF）+ OpenVINO IR | Arc 140T 核显 |
| **OVMS-CPU** ⭐ | 同上，显式锁 CPU | 同上 | **纯 CPU（唯一一组）** |
| **Unsloth** | v0.1.804-beta | 内置 llama.cpp 构建 + Vulkan | Arc 140T 核显 |

五组加载的均为 **Q4_K_M（或等价 INT4）**，体积 14.8–16.2 GB，横向可比。

## 关键指标

除常规 TTFT / 吞吐外，报告的核心结论依赖这三个指标（均由测试工具直接输出）：

- **Dec TPS** — 纯解码吞吐 = `completion_tokens / (duration − TTFT − latency)`，**不含 prefill**。常规 `gen TPS` 把 prefill 算进分母，会严重低估 prefill 慢的引擎。
- **ITL**（Inter-Token Latency）— 相邻 token 的到达间隔，决定"打字机"体感。均值相同、分布不同，体验天差地别。
- **Cache%** — 命中服务端 prefix cache 的 prompt token 占比，直接暴露 TTFT / prefill 数字是否被缓存污染。

## 复现测试

测试工具为自研的 [llmbench](https://github.com/megemini/llmbench)（TUI，面向 OpenAI 兼容端点）。统一参数：`-c 1` 单并发、`-t 512` 输出上限、`--runs 3` 三轮串行，接口为 `/v1/chat/completions` + `stream = true`。

```bash
# Herdsman (localhost:8080)
llmbench -u http://localhost:8080/v1 --runs 3 -m Qwen3.8-27B

# Unsloth (127.0.0.1:8888)
llmbench -u http://127.0.0.1:8888/v1 --runs 3 -m Qwen3.8-27B-GGUF:Q4_K_M --api-key <your-api-key>

# Bionic / LM Studio (localhost:1234)
llmbench -u http://localhost:1234/v1 --runs 3 -m qwen3.8-27b

# OVMS (localhost:8000)
llmbench -u http://localhost:8000/v1 --runs 3
```

> 负载分三档：short ~92 token / medium ~1.28k token / long ~9.46k token，输出上限 512 token。

## 目录结构

```
.
├── README.md
├── Qwen3.8-27B_四引擎性能对比报告_发布版.md   # 中文报告（主）
├── Qwen3.8-27B_Four-Engine_Benchmark_EN.md   # 英文报告
├── charts/                                   # 6 张对比图（decode / TTFT / ITL / prefill / 衰减 / 端到端）
├── Bionic/  Herdsman/  ovms/  ovms_cpu/  unsloth/
│                                             # 各引擎的 benchmark_*.json / .csv / .md 原始数据 + 测试截图
└── upload_picgo.py                           # 把报告图片批量上传图床并替换引用的脚本
```

## 说明

- 数据与截图由作者逐个引擎实测采集，文章由作者与 AI 共同撰写。
- 本次只测了 `concurrency = 1`，多并发场景的结论仍需补测。
- 报告中所有数字口径、字段定义与复核过程均记录在正文与附录中，欢迎 issue 指正。
