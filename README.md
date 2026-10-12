# mini-strata: Strata Qwen35 / IQ3_S adaptation

基于 [Niko1221/Strata](https://github.com/Niko1221/Strata) **v0.1.41** 的实验性 Qwen3.6-35B-A3B 适配。原始 Strata 提交和完整祖先历史保留；成熟 Qwen attention/GDN/shared expert/KV 来自固定 llama.cpp，routed experts 接入 Strata 原共享专家服务。

当前发布的是 **2026-10-12 阶段已复核源码快照**（任务 69–150），对应本机 RTX4060 Laptop 8GiB 的固定 3–4k 中文字符内容审核实验。交付配置：物理微批 4096、cache 3200、冷专家 producer 4、count_desc 排序。**官方口径（单进程完整 prefill、大样本）han 风险/正常样本 1121.00 / 1114.01 token/s（1000+ 达成），chars 两样本 992.75 / 983.36（未达 1000）；长驻服务形态（n=30）四样本 1429.96 / 1428.43 / 1377.34 / 1362.17。** 不是质量或跨机器保证；口径与边界见阶段结论。

- [阶段结论 2026-10-12](docs/QWEN35_STAGE_20261012.md)
- [优化方案与证据限制](docs/QWEN35_OPTIMIZATIONS.md)
- [相对原版的源码改动及 review 入口](docs/QWEN35_REVIEW_MAP.md)
- [构建、参数与复现范围](docs/QWEN35_BUILD.md)
- [数值差异和已知未闭合问题](docs/QWEN35_KNOWN_LIMITS.md)
- [高上限优化计划（2026-10-10）](docs/QWEN35_HIGH_CEILING_PLAN.md)
- [原始 Strata README](docs/UPSTREAM_STRATA_README.md)
- [公开性能汇总（67 时点）](evidence/business67-summary.json) ／ [阶段汇总 2026-10-12](evidence/stage-20261012.json)

## 直接看我们改了什么

```sh
git diff upstream-strata-v0.1.41..main -- CMakeLists.txt include src
git diff upstream-dependencies..main -- vendor/llama.cpp
git diff upstream-dependencies..main --stat
```

`upstream-strata-v0.1.41` 就是原始提交 `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`。`upstream-dependencies` 仅增加未修改的 llama.cpp `988190680d5a89fce97de3c20df2c2813731fd61`。后续适配提交使模型与专家执行路径的修改可单独 review。原始许可证均保留。

模型权重、真实业务原文、token 文件、机器构建产物和 OpenCode 请求日志保留在本地。仓库提供来源锁定、源码、哈希、指标、调用计数和已知问题；公开汇总不足以逐项重放未公开的业务样本。

本快照不含施工中任务的未复核改动；服务形态数字为并列口径，官方口径（单进程完整 prefill）为准入基线。下一阶段方向：以长驻服务 1600+ 为目标、优化 llama 图侧 GDN/投影/融合，冷专家低位宽（O5/O6）独立实验并行——均未测/不承诺。
