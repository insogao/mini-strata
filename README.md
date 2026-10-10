# mini-strata: Strata Qwen35 / IQ3_S adaptation

基于 [Niko1221/Strata](https://github.com/Niko1221/Strata) **v0.1.41** 的实验性 Qwen3.6-35B-A3B 适配。原始 Strata 提交和完整祖先历史保留；成熟 Qwen attention/GDN/shared expert/KV 来自固定 llama.cpp，routed experts 接入 Strata 原共享专家服务。

当前发布的是 **65/67 已复核源码快照**，对应本机 RTX4060 Laptop 8GiB 的固定 3–4k 中文字符内容审核实验。微批2048、双冷专家 producer、可选按 expert token 数降序分组：风险样本 **954.560 token/s / TTFT4.055s**，正常样本 **911.174 token/s / TTFT4.241s**，各一次。尚未达到1000+，未做三次中位数确认，不是质量或跨机器保证。

- [可讨论的优化方案与证据限制](docs/QWEN35_OPTIMIZATIONS.md)
- [相对原版的源码改动及 review 入口](docs/QWEN35_REVIEW_MAP.md)
- [构建、参数与复现范围](docs/QWEN35_BUILD.md)
- [数值差异和已知未闭合问题](docs/QWEN35_KNOWN_LIMITS.md)
- [原始 Strata README](docs/UPSTREAM_STRATA_README.md)
- [公开性能汇总](evidence/business67-summary.json)

## 直接看我们改了什么

```sh
git diff upstream-strata-v0.1.41..main -- CMakeLists.txt include src
git diff upstream-dependencies..main -- vendor/llama.cpp
git diff upstream-dependencies..main --stat
```

`upstream-strata-v0.1.41` 就是原始提交 `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`。`upstream-dependencies` 仅增加未修改的 llama.cpp `988190680d5a89fce97de3c20df2c2813731fd61`。后续适配提交使模型与专家执行路径的修改可单独 review。原始许可证均保留。

模型权重、真实业务原文、token 文件、机器构建产物和 OpenCode 请求日志保留在本地。仓库提供来源锁定、源码、哈希、指标、调用计数和已知问题；公开汇总不足以逐项重放未公开的业务样本。

68版 GDN计时已完成本地 Codex复核，**不包含在65/67发布源码快照**。默认关闭诊断的新风险样本单次为909.138 token/s、TTFT4.257s；正常样本未复测。不要将954.560作为68版速度或稳定吞吐。

用户已认可接近1000的阶段成果，下一阶段继续提高完整prefill上限。[后续优化方向与核定边界](docs/QWEN35_HIGH_CEILING_PLAN.md)记录已启动任务及待验证假设；探索1100–1200、进一步1300–1500+均不是实测或承诺。
