# 外部审核指南

先读 [阶段报告](DECODE_STAGE_REPORT_20261012.md)。目标是审核完整T=1专家流水线和短JSON响应，不把profile命中或micro数字当整体加速。

## 源码与审查入口

`codex/decode-stage-review-20261012`相对`qwen35-stage-20261012`：

- `src/program/qwen35_model.cpp`：EOS、逐token时钟、base扣除、JSON闭合时间、STOP-only保护；M0曾构建但全业务质量/Prefill gate未完成。
- `src/core/expert_source.cpp`、`include/strata/core/expert_source.hpp`：default-OFF TIER_SPLIT分量；GU/Q/DOWN与drain嵌套，不能相加。检查gate静态缓存、对象重开/关闭计数生命周期。
- `vendor/llama.cpp/ggml/src/ggml-cuda/{ggml-cuda.cu,norm.cu}`：151默认OFF融合（引擎命中0）；不要当已生效优化。
- `src/kernels/cuda/native_mmvq.cu`仅EOL归一，审阅用`git diff --ignore-space-at-eol`消除噪声。
- 既有 `src/core/qwen35_bridge.cpp`：seam_exec(T==1)→expert_native_decode_layer；原Strata generate和共享runtime仍为入口。

`codex/upload-run-batch-review-20261012`相对上一review分支仅172两文件：连续miss rank、host_miss紧凑行映射、H2D字节/stream、ev_miss/work等待、hit行不覆盖、默认OFF。CPU镜像/回放通过不等于CUDA验证，尚未编译。其他patch仅存档，不混入同一构建。

## 复核命令

```sh
git fetch origin
git diff upstream-dependencies...qwen35-stage-20261012 -- src include vendor/llama.cpp
git diff qwen35-stage-20261012...origin/codex/decode-stage-review-20261012 -- src include vendor/llama.cpp
git diff origin/codex/decode-stage-review-20261012...origin/codex/upload-run-batch-review-20261012 -- src include
```

构建沿用[已有指南](../QWEN35_BUILD.md)和锁定依赖，在本阶段Windows构建还需`-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared`避免static/shared重复链接。共享runtime修复是配置参数，不是新增功能源码。新机器重新编译不承诺复现原binary SHA或Gate；无需下载模型就可审核依赖/逻辑。模型与私有业务fixtures未发布。

## 最希望得到的反馈

1. 实际40层T=1的router回读/activation等待/CPU冷矩阵/H2D/GPU热专家/结合中，哪个等待边可缩短且不破坏stream或图依赖？
2. 冷miss与热hit的互斥关键路径如何验证？不要把host timer等同CPU或把包含式区间相加。
3. 已有原runtime缓存借还/恢复、跨请求state/KV与Graph兼容是否有缺陷？
4. 短输出下动态缓存不回本后，哪些结构方向有明显大于噪声的上限？
5. 为长驻同进程请求设计一个有限对照，保住EOS/政策/风险等级和下一请求Prefill，不无限扩大样本。

请给函数/行、证据、最小改动、收益上限和退出条件。未测方案标未测；微小措辞许可不涵盖风险等级/NaN/CUDA/OOM。
