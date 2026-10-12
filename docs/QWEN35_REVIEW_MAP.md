# 原版到Qwen适配：代码review入口

本仓库直接继承原Strata提交与历史；根`src/`和`include/`就是原Strata。`vendor/llama.cpp/`是固定成熟模型图依赖，先提交其原版源码，再提交我们修改的文件。发布源码从65完成后/68开工前的不可变31文件快照复制并逐一SHA核验，不从正在施工的工作树拼接。

| 边界 | 文件/入口 | review关注点 |
|---|---|---|
| 构建与原程序入口 | `CMakeLists.txt`、`src/program/generate.cpp` | Qwen35选项、ABI/ggml实现、sm89，保留原Flash-Next默认路线 |
| 成熟模型图加载与状态 | `src/program/qwen35_model.cpp`、`include/strata/program/qwen35_model.hpp` | 读取真实GGUF几何，成熟attention/GDN/KV/shared expert，物理微批与context独立，原runtime生命周期 |
| 模型图expert边界 | `src/core/qwen35_bridge.cpp`及同名header | device输入、decode/prefill分派与依赖，避免隐式CPU-only替代；不是独立模型runner |
| 原GPU专家流水线适配 | `src/core/qwen35_prefill.cu`、`include/strata/core/qwen35_prefill.hpp`、`src/prefill/prefill.cpp` | regroup、count_desc同步重建所有索引、group16、resident/cold MMQ、gather/scatter、workspace借还 |
| 原cache/source泛化 | `src/core/expert_cache.cpp`、`src/core/expert_source.cpp`及headers | 变量blob/真实payload、容量不被profile截断、mmap三tensor slice、实际驻留生命周期 |
| cold权重producer | `src/prefill/cold_stager.cpp`、`include/strata/prefill/cold_stager.hpp` | 有界pin/ring、2producer、取消/drain、ready/used与copyFIFO协议、WC默认off |
| CPU/native量化支持 | `src/kernels/cpu/{expert_layout,native_expert,pool}.cpp`与headers、`src/kernels/cuda/iq_kernels.cu` | IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS真实类型、数学修正和几何泛化，不能替换整套kernel |
| 成熟图的接缝 | `vendor/llama.cpp/src/{llama-context.cpp,llama-graph.cpp,models/qwen35moe.cpp}` | routedexpert hook、device生产回调、保持Qwen原数学 |
| backend/external producer依赖 | `vendor/llama.cpp/ggml/{include/ggml-backend.h,src/ggml-backend.cpp,src/ggml-cuda/ggml-cuda.cu}` | 输入/输出同步与stream所有权，不只“链接了库” |
| 小型检查/默认关诊断 | `src/program/qwen35_expert_check.cpp`、replay/capture/events相关分支 | known failure保留，诊断不计为正式速度或全部数值认证 |

适配功能差异为23个Strata文件和6个llama.cpp文件；两份GDN源/header在65快照仍原版未改。包含了此前验收用的默认关诊断，不能把所有新增行都称性能优化。`evidence/source-snapshot65.json`列逐文件原始字节SHA；仓库checkout受Git换行策略影响时以来源SHA与`.gitattributes`说明为准。

GitHub用`upstream-dependencies...main`查看适配提交；本地用`git diff upstream-dependencies..main -- src include CMakeLists.txt vendor/llama.cpp`。完整与原版差异另包含依赖引入和新增讨论文档，故建议分开阅读。

## 更新（2026-10-12）

- 适配文件现为 **25 个 Strata 文件 + 8 个 llama.cpp 文件**（逐文件 SHA 见 `evidence/source-snapshot-20261012.json`）。
- 65 快照之后新增/变化的源码：`include/strata/prefill/moe_mmq.hpp`、`src/prefill/moe_mmq.cu`（任务 74 direct-to-group 掩码 gather，默认关）；`vendor/llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu` 与 `.cuh`（任务 68 GDN eager 计时，默认关）——65 快照时这两份 GDN 源/header 仍为原版。
- 默认关诊断 gate 清单（本快照内含）：A_SPLIT、SEAM_GAP、OP_TIMING、PREFILL_PHASES、GPU_EVENTS、STAGER_STATS、LAYER_CAPTURE、ROUTE_FP、GGML_MOE 等；交付 exe 中默认全关，OFF 等价已核。
- **施工中任务对 `norm.cu` 的改动未复核，不包含在本快照。**
