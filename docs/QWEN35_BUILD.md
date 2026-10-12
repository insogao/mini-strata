# 构建与复现范围

发布快照 2026-10-12 对应本机交付 exe SHA256 `0e00af8a11274c62789f47c1b5d21aed811f33adb9b79608ffb7df5525a906b6`。搬移源码与重新构建后不要求得到同一二进制SHA：编译器版本、源路径、PDB等会变。此仓库本轮未重新编译/测速，不能把本地旧构建Gate自动继承给任意新构建。

已用工具链：Windows/MSVC14.44/CUDA12.8.1/Ninja，C++20，sm89。来源锁见`UPSTREAM_QWEN35.json`。在VS Developer PowerShell、CUDA Toolkit已配置的环境，按原CMake选项构建（这里是可移植路径表达，不是本轮重新验证过的命令）：

```powershell
$llama = (Resolve-Path './vendor/llama.cpp').Path
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_QWEN35=ON `
  -DSTRATA_QWEN35_TOOLS=ON "-DSTRATA_GGML_DIR=$llama" `
  -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_CUDA=ON -DGGML_OPENMP=OFF `
  -DSTRATA_BUILD_TESTS=OFF
cmake --build build --target strata
```

真实本机配置额外GGML_AVX/AVX2/AVX512/SSE42/FMA/F16C等CPU开关见`evidence/build65-cache-options.json`，要求匹配时逐项传入。这只是源码搬移，未做新的平台支持承诺。

模型：`Qwen3.6-35B-A3B-UD-IQ3_S.gguf`，13676723168 bytes，SHA256 `66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`。expert pack目录需要与该GGUF严格匹配的`native_experts.txt`索引；Qwen路线原模型图加载dense部分，冷专家从GGUF三tensor slice读。不能直接拿原Flash-Next pack或其它量化的offset复用。

下面假定本机已有`models`下上述模型、正确native索引，fixtures下独立校准profile、固定审核token文件。仓库未发布业务文本，以下是参数表达而非一键公开精确重放：

```powershell
$env:STRATA_QWEN35_ROUTER_WEIGHTS_DEVICE = '1'
$env:STRATA_QWEN35_PACKED_IDS = '1'
$env:STRATA_QWEN35_EXPERT_ORDER = 'count_desc'
# 清除继承的STRATA/ GGML/ LLAMA_GRAPH诊断覆盖；其余诊断off，CUDA graphs保持默认；producer/ring默认4/16。
./build/strata.exe --qwen35-model models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf `
  --qwen35-pack models --tokens-file fixtures/risk.prompt.tokens `
  --max-new 1 --max-context 4096 --qwen35-threads 8 `
  --qwen35-cache-slots 3200 --qwen35-profile fixtures/calibration-expanded.profile `
  --qwen35-ubatch 4096
```

perf应通过Python subprocess直接继承stdout/stderr文件句柄；PowerShell消费者重定向可能改变wall，不能把不同sink当产品优化。加载另列，记录prefill wall/token/s、TTFT、actualGPU峰值、payload、hit/miss/cold/MMQ计数、loan恢复、输出格式/审核结论。物理ub4096与上下文c4096独立；cache 3200是本阶段交付标准（3700可再快约1.65%但余量仅26–29MiB，不作为默认）。

不必为审阅代码下载模型。先从README的diff命令和review map讨论具体路径；各候选后续实测仍由唯一GPU所有者有界串行进行。
