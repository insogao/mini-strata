# mini-strata Phase 2.2 r1｜Strata PR #1402：Qwen3.6 MTP 推测解码专项研发方案（夜间进展补订）

> **用途**：将本文件原样交给负责 `mini-strata` 的主 OpenCode / AI Agent，作为一个**独立、可持续执行**的研发专题。**单独提供本文即可启动工作，不依赖 ChatGPT 聊天记录；但仍必须以本机最新 `AGENTS.md`、工作树及实验账本为准。**
>
> **制定日期**：2026-10-13（Asia/Tokyo）。**本文件不是已完成的移植报告，也不授权直接合并上游 PR。**
>
> **优先目标**：在 RTX 4060 Laptop 8 GiB、Qwen3.6-35B-A3B IQ3_S、约 3.8K token 中文审核输入的现有环境里，验证 PR #1402 的 **MTP（Multi-Token Prediction）+ 多 Token 主模型验证**能否降低真正的短 JSON E2E 延迟，并使 128–256 token 的解释性输出获得稳定净收益；同时保护冻结的 Prefill、数学正确性、审核质量、EOS 与显存安全。

---

## 0. Agent 接到文件后的第一件事

1. 完整阅读本地 `AGENTS.md`、已有的 Phase 2.1 规划及项目构建说明，再阅读下方全部 **「一、已核验的上游事实」**和 **「二、mini-strata 现状」**。
2. 检查工作树、当前分支、已有后台任务/唯一 GPU owner、未合并 PR、模型和 exe 的实际哈希；不要假设公开 GitHub 快照与本地最新工作树相同。
3. **第一阶段只读研究**。将 PR #1402 与本地专用 Qwen35 路径做**逐函数差异矩阵**；确认 GGUF 是否带 MTP 权重与支持的 KV/GDN rollback 语义。完成 S0 可行性报告、S0.5 极小窗口验证成本审查后，才允许考虑进入独立实验分支。
4. 不设「只剩一轮」或「最多一个候选」的全局次数限制；但**每次只允许一个 GPU 实验执行者、一个构建/核心文件写入所有者**。每个候选单独设预算、失败即停、可回退、可复核。
5. 若前提不成立，给出证据与替代路径，不要为了交付一个“加速补丁”而破坏原基线。

## 一、已核验的上游事实：PR #1402 究竟测到了什么

### 1.1 官方链接（务必打开原件，以下使用当前审阅时的固定 commit）

| 资料 | 链接 | 核心用途 |
|---|---|---|
| **Strata PR #1402** | https://github.com/Niko1221/Strata/pull/1402 | 上游 PR 的完整说明、讨论、变更清单和最新状态 |
| **PR 源码固定 commit** | https://github.com/aflin/Strata/tree/cd1daa4e2291f2705a0bf23976233608cf2631cf | 重现实验与差异比较的固定版本，避免分支前进导致结果漂移 |
| **Qwen3.6 测试全文** | https://github.com/aflin/Strata/blob/cd1daa4e2291f2705a0bf23976233608cf2631cf/docs/QWEN36.md | 硬件、量化、上下文、MTP、8GB模拟、性能与局限 |
| **MTP 具体实现** | https://github.com/aflin/Strata/blob/cd1daa4e2291f2705a0bf23976233608cf2631cf/src/core/mtp.cpp | MTP草稿层、权重绑定、候选生成与状态处理 |
| **Verifier** | https://github.com/aflin/Strata/blob/cd1daa4e2291f2705a0bf23976233608cf2631cf/src/core/verify.cpp | 多 Token 验证、CPU/GPU专家、Attention/DeltaNet状态 |
| **生成器/显存管理** | https://github.com/aflin/Strata/blob/cd1daa4e2291f2705a0bf23976233608cf2631cf/src/program/generate.cpp | MTP启动、显存预留、专家缓存、自适应context |
| **上游专家调度** | https://github.com/aflin/Strata/blob/cd1daa4e2291f2705a0bf23976233608cf2631cf/src/core/expert_source.cpp | 同一上游的路由/CPU-GPU专家支持，不混同本地代码 |
| **上游模型格式** | https://github.com/aflin/Strata/blob/cd1daa4e2291f2705a0bf23976233608cf2631cf/include/strata/artifact/gguf_reader.hpp | GGUF读取、MTP张量布局判断 |
| **模型与MTP权重来源** | https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF | 作者实际选用的带MTP层的GGUF模型系列 |

> 截至 **2026-10-13 本次核验**：PR #1402 **open / 未合并**。固定源版本 `cd1daa4e2291f2705a0bf23976233608cf2631cf`，作者分支 `aflin/Strata:qwen36-35b-support`。后续如PR有更新，应在新快照下重新标注哈希，而不是静默混入变更。

### 1.2 可引用的数据与**不可推出**的结论

作者环境：**单张 RTX 4070 Ti 12 GB，Ryzen 9 5950X，62 GB RAM，Debian 12，CUDA 13.0，32K context，8-bit K/V，MTP `--spec 4` 开启**。所谓**“模拟8GB”**是让另一个进程占据同一张4070 Ti约4GiB显存；不是物理8GB的RTX4060或2080Ti，也不是双卡合算。

| 同一作者的测试 | 12GB可用 / Prefill | 12GB可用 / Decode | 模拟8GB / Prefill | 模拟8GB / Decode |
|---|---:|---:|---:|---:|
| UD-IQ3_S，21-token短输入 / 128-token生成 | — | **127 tok/s** | — | **89 tok/s** |
| UD-IQ3_S，8014-token输入 / 64-token生成 | **4284 tok/s** | **106 tok/s** | **3076 tok/s** | **70 tok/s** |
| UD-IQ4_XS，短输入 / 长输入 | 8K: 4460 tok/s | 短:105 / 8K:86 | 8K:2669 tok/s | 短:75 / 8K:68 |

作者还报告：上述若干测试中的 MTP 草稿**接受率约0.84–0.96**，每个验证窗口实际平均推进 **2.4–3.1 token**。这些数值来源于作者报告，**并非当前 mini-strata 实测**；接受率和每轮推进数是两个不同指标，不能用接受率直接乘现有TPS。上述 Decode 成绩包含 MTP 与硬件/实现差异，**不是纯 T=1 Decode 速度**。

作者的 Qwen3.6 支持包括：40层/每层256专家/每token选8、GDN（DeltaNet）与稠密Attention混合、MTP草稿层从GGUF `blk.40` 读取、`nextn.eh_proj.weight`（按 embedding/hidden 分拆）等。`--spec 4` 是**上游 PR 分支**的参数，不可未经确认直接加到本地 `strata generate` 的专用 Qwen35 runner。

**不允许**把70/89 tok/s或84%–96%当作本机性能承诺；也不允许把在4070 Ti上预留4GiB的情况说成“实测RTX4060 8GB”。

## 二、mini-strata 已知基线与限制

### 2.1 原工程与报告链接

| 文档/代码 | 地址 |
|---|---|
| 用户仓库 | https://github.com/insogao/mini-strata |
| Decode阶段完整报告 | https://github.com/insogao/mini-strata/blob/main/docs/phase2_decode/DECODE_STAGE_REPORT_20261012.md |
| 外部审核指南 | https://github.com/insogao/mini-strata/blob/main/docs/phase2_decode/EXTERNAL_REVIEW_GUIDE.md |
| PR #1：M0测量/质量诊断（草稿） | https://github.com/insogao/mini-strata/pull/1 |
| PR #2：冷专家结果行 H2D 合并（草稿，未编译测速） | https://github.com/insogao/mini-strata/pull/2 |
| Qwen35专用主执行器 | https://github.com/insogao/mini-strata/blob/main/src/program/qwen35_model.cpp |
| Qwen35图与专家衔接 | https://github.com/insogao/mini-strata/blob/main/src/core/qwen35_bridge.cpp |
| Decode冷/热专家路径 | https://github.com/insogao/mini-strata/blob/main/src/core/expert_source.cpp |
| 原有（其他架构使用的）Strata MTP代码 | https://github.com/insogao/mini-strata/blob/main/src/core/mtp.cpp |

### 2.2 不可破坏的冻结基线

- 硬件：RTX 4060 Laptop **8GiB**、Ryzen 9 7945HX、32GiB RAM，Windows；单GPU、`threads=8`。
- 目标模型：`models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`（以本地实物和 SHA 校准；公开报告给出的既有模型 SHA256：`66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66`）。**未核实该文件一定包含 MTP 层**。
- 冻结 exe SHA256：`0e00af8a11274c62789f47c1b5d21aed811f33adb9b79608ffb7df5525a906b6`，不得覆盖。
- Prefill参数：`--qwen35-ubatch 4096 --qwen35-cache-slots 3200`、producer4/ring16、`t8`；环境变量：`STRATA_QWEN35_ROUTER_WEIGHTS_DEVICE=1`、`STRATA_QWEN35_PACKED_IDS=1`、`STRATA_QWEN35_EXPERT_ORDER=count_desc`。
- **冻结Prefill**：官方冷进程 han-risk/benign **1121.00/1114.01 tok/s**，chars **992.75/983.36**；历史同进程稳态四组 **1429.96/1428.43/1377.34/1362.17 tok/s**。这是回退基线；任何新exe都**不自动继承**这些数据。
- 2026-10-12阶段报告截止：7个GPU批次、37次实际native、5次构建尝试；**0项确认的完整业务 Decode 提速**。**用户随后提供的夜间更新：另新增23次模型进程、9次完整构建尝试（6成3败），仍无新增已确认Decode提速**。不可把夜间任务数当候选数或自行把不同状态测数合并成稳定基线。最近有效同进程第二次完整短JSON观测为：16可见tokens、Decode 28.86–29.92 tok/s、TTFT 2.77–2.81秒、JSON可见3.29–3.32秒、EOS 3.32–3.34秒；**这些还不是稳定服务/p95成绩**。
- 业务样本：主体约3863–3869 prompt tokens（han 约3000–4000汉字），90%短审核/短JSON（约16可见token，实际长度需按统一输出固定），10%详细解释/画像等中长输出（128、256或512）。
- 质量缺口：某风险样本冻结实现与候选实现曾给出 `risk2/violation=true`，标签为3；需独立核查业务标签/模板/模型，不得在提速验收中掩盖或擅改。

### 2.3 架构差异：为何不能直接照搬PR

- **本地现有 Qwen35**：`src/program/qwen35_model.cpp` 使用**固定补丁 llama.cpp 图**完成注意力、GDN、共享专家、head，由 `qwen35_bridge.cpp` 在运行时回调 `expert_native_decode_layer()` 完成 T=1 routed MoE；T>1有单独的 Prefill 专家路径。
- **PR #1402**：在**Strata自己的原生生成器、MTP、Verifier及缓存/状态生命周期**里加入 `qwen35moe` 形状与执行支持，涉及几十个文件；它不是单独的 `mtp.cpp` 热补丁。
- 因此移植难点不是“读到下一token草稿”，而是 **主模型同时验证 K 个 token 的数学正确性**、**KV/GDN recurrent state 的临时写入、接受前缀提交、拒绝后恢复**，以及混合CPU/GPU专家执行和显存预算。
- 本地 `src/core/mtp.cpp` 已有其他模型使用的 MTP 代码，但**不代表 Qwen35 专用 runner 已接入**；必须通过函数调用图、参数解析与端到端用例证明。

### 2.4 夜间最新结果：不再重复已否决的微优化

以下来自用户最新夜间进度汇总，晚于上一阶段 GitHub 报告；主 Agent 须核对本地任务 ledger 和真实源代码状态：

| 方向 | 新证据 | Phase 2.2 处理 |
|---|---|---|
| 连续 miss 行上传合并 | `cudaMemcpyAsync` 提交3033→1361、字节/输出/借还一致；速度收益未证明 | 保留默认 OFF；不与 MTP 混测，不以 memcpy 次数冒充吞吐收益 |
| 同进程两次短 JSON | 真实 EOS、reset、cache 借还已可核验；稳态/p95未建立 | 按 S1 扩展多请求同进程测量，不把第二请求等同稳定服务 |
| CPU pool park/publish | 约0.020ms/token | 封存，不重复探索 |
| CPU activation量化 | 约0.048ms/token | 封存 |
| IQ3_S DOWN 专用内核复用 | CPU局部慢13%–16%，20组配对无胜出 | 该**具体复用**方案不用，不代表全部CPU优化无效 |
| 主线程固定核心0 | Decode下降19%–22%，JSON时延增加4%–5% | 负收益，禁止默认启用 |
| 四行一块输出词表投影 | Q6_K、2048×248320，候选有stride单位错误，未编译/验证/测速 | 隔离存档；不得顺手合进 MTP 实验 |

**工程纪律补充**：夜间有构建失败后自动续步、失败退出码被后续步骤掩盖、CPU镜像校验了错误实现的问题。Windows PowerShell 构建脚本要显式检测 `$LASTEXITCODE` / 失败抛出；跨平台脚本确保非零退出立即停止。每个测试必须把实际 exe SHA、源码 SHA、启用的gate、输入SHA绑定到原始记录；构建失败立即终止该轮、记录失败预算，禁止后续步骤“修正”为成功。

## 三、技术目标（分层，不许混淆）

**目标 A——可行性**：确认当前 GGUF 是否已有完整 MTP 权重；梳理 PR 的数据结构、主模型多token verifier、KV和GDN提交回滚、CPU/GPU专家、sampling/EOS、显存分配。做出可运行/不具备前提的判断。

**目标 B——正确性**：在同模型、相同 prompt/template/seed 与相同采样设置下，`spec=0` 和 `spec=K` 具有符合推测解码理论的输出分布；贪心情形应逐token一致（但浮点近并列须记录logits/精度偏差）。保证拒绝位置之后的 GDN/KV 状态不被错误保留，EOG和完成JSON不重复/截断。

**目标 C——性能**：提高**完整请求E2E与实际业务产出效率**，而非仅“单验证轮推进更多 token”。短审核优先 JSON闭合时刻/TTFT/p95；长解释优先 TPOT/DecodeTPS/总时延。

**目标 D——可回退**：功能默认关闭，旧 runner 原路径保持；不增加永久GPU压力导致3200槽缓存损坏、不因 graph/capture/decode宽度造成非法显存访问。

## 四、研发路径：六个阶段（含 S0.5），允许滚动推进

### S0：上游源码审核与模型权重清点（先做，**零GPU实验**）

1. 固定上游 `cd1daa4e...`，执行 `git diff`，不要直接 `git merge` / `cherry-pick` 57个文件；阅读 `docs/QWEN36.md`、`src/core/mtp.cpp`、`src/core/verify.cpp`、`src/program/generate.cpp`、`src/core/layout.cpp`、`src/core/expert_source.cpp`、`src/prefill/prefill.cpp`。
2. 建表：PR变更 → 上游函数/结构 → 本地对应函数/文件 → 数学/状态/显存/调用依赖 → 可复用或必须重写 → 风险级别。
3. 清点本地 GGUF 张量清单与元数据，**只读**检索 `blk.40`、`nextn.eh_proj.weight`、draft experts/norm/router等所需张量，核对数据类型、shape、量化格式、是否是附带draft层的完整模型。不要只按文件名 `UD-IQ3_S` 推测。
4. 检查 **目标模型在 llama.cpp、GGUF parser、现有补丁与缓存打包中** 是否保留 `blk.40`。作者提示 `tools/iq_pack.py` 特别将MTP层从主专家表分离；若本地pack丢弃该层，应修pack而非读取不存在的权重。
5. 若本地模型不含 MTP：**先报告缺口**。可选择从上游所用 [Unsloth MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) 获得独立的同权重实验副本；模型变更必须记录SHA、权重来源与是否仅新增draft，不能直接将两份GGUF生成速度作同条件因果对照。
6. 检查 NVIDIA sm_89、Windows MSVC/CUDA兼容、GPU可用显存和MTP层字节占用；不要借用4070Ti Linux编译通过作为Windows成功证据。

**交付**：`docs/phase2_mtp/PR1402_REVIEW.md`、`MODEL_MTP_INVENTORY.md`、`IMPLEMENTATION_MAP.md`、`GO_NO_GO_S0.md`。

**S0退路**：缺权重/状态验证不可行，可保留架构分析，先做独立参考引擎对照；不许用“随机草稿”或teacher forced token冒充MTP。

### S0.5：MTP 验证成本与独立上游对照（在全面移植前设硬门槛）

**工作包 A：极小 token 窗口成本，区分“能批处理”和“批处理真的便宜”**

1. 读本地 `qwen35_bridge.cpp` 与 `qwen35_model.cpp` 的实际 T>1 route，证明哪部分是小T原生验证、哪部分只是4096微批Prefill的专用供料。不得把Prefill在T=4096下的高吞吐直接外推到T=2–5。
2. 使用已经存在的可信测试入口或构造**隔离的有限最小实验**，分别分析/测试 T=1、2、3、4、5 的：总设备墙钟、GPU热门专家与CPU冷专家、路由ID回读、专家权重H2D/CPU结果H2D、每层同步、非MoE GDN/Attention状态、peak VRAM。只有证明临时State安全后才允许实际运行含完整主模型的多T请求。不存在正确验证入口时，先给下界/成本模型，**不可用teacher-force当作已实现的MTP性能**。
3. 给出公式与数值账：`T_round = T_draft + T_target_verify + T_accept_commit_or_rollback`；记录 `committed_tokens / wall_seconds` 与每次验证推进数，并用拒绝率敏感性分析给出回本阈值。短JSON需要16-token E2E净收益，不是微基准超过T=1就宣称成功。
4. 没有可信的GDN/KV rollback时，**允许只读分析或单层孤立micro测试，禁止在真实业务链路擅自运行错误的多T预测**。

**工作包 B：把 PR #1402 独立运行当参考，不是移植前必过的障碍**

1. 在另一份独立worktree/构建目录固定上游 commit，审查Windows RTX4060 sm_89的可编译性、同模型能否装入32GB RAM、显存与driver限制。上游作者是在Linux 4070Ti上验证，**Windows仅有构建记录而非作者实测推理**。
2. 若成本和模型条件允许，独立运行其 `spec=0` vs `spec=K`，**同一 GGUF/相同prompt/context/KV/slots**，得到同卡的MTP净增益；不可拿带MTP的外部70/89 tok/s去直接除以本机未开启MTP的速度。
3. 若移植工作量明显小于独立上游构建，或者先决条件缺失，可给出有证据的不可行/延期结论；**不得为完成“独立对照”无限消耗构建轮数**。

**S0.5 首轮交付**：`SMALL_T_VERIFY_COST.md`、`UPSTREAM_REFERENCE_DECISION.md`、`MTP_ROI_GATE.md`。报告需明确**能否进入S2，或仍需先修多Token verifier**。

### S1：建立可信的 MTP OFF 基线与测试器（优先保证可归因）

设计至少如下隔离对照：

| Arm | 模型文件 | 引擎/功能 | 用途 |
|---|---|---|---|
| A（原生产） | 冻结IQ3_S原文件 | 冻结exe、原Prefill/Decode | 保护历史产品基线，不用于不同模型的MTP直接因果归因 |
| B（同权重控制） | **确认带draft的同一GGUF** | 新实验exe，MTP **OFF** | 真正的 MTP性能与正确性对照 |
| C（MTP候选） | **与B逐字节相同的GGUF** | 同构建exe、MTP **ON** | 唯一主要变量为MTP开关与其显存布局 |
| D（可选上游参考） | 上游适配的MTP-GGUF | PR #1402原生引擎 | 验证作者逻辑/数据，不参与A/B/C百分比归因 |

若为了编译/加载改变主模型量化、KV类型或expert slots，所有变化须明示；B/C须使用**等价显存约束下真实部署能成立**的配置。若MTP迫使cache从3200下降，应单独报告性能的“完整方案收益”（含缓存下降）及“算法净收益”，不可只挑后者。

建立同进程、长驻、按业务分层的性能工具：

- S：原3.8K prompt风险/正常、最大输出32，合法EOS，JSON结构闭合；不得只跑 `max-new 1`。
- M：同一风险政策与完整理由（输出64/128）。
- L：长解释/证据引用（128/256/512），**context预留足够验证宽度和生成上限**；4K上下文接近输入长度，不许拿 `c4096` 的输出32与输出512做不等价对比。
- 每组记录真实生成 token、MTP尝试轮数、草稿token数、接受数、拒绝数、有效推进token/verify、草稿耗时、主验证耗时、回滚耗时、TPOT/TTFT/E2E/JSON闭合、resident hit/miss、显存峰值及Prefill回归。
- 原有171/174短序列受进程冷暖、磁盘/内存状态严重干扰：先固定预热、数据次序、日志与环境，记录后台负载；候选A/B交错或ABBA，但每臂必须用自己的干净**长驻进程**。先让自对照波动显著收敛，才评估小幅收益。
- debug级时间戳、CUDA事件、Nsight采样作为单独诊断批次，不和关闭诊断的产品吞吐混淆。

**交付**：`MTP_BENCH_SPEC.md`、可复算JSONL原始指标、可回放无私有业务内容的hash索引、`BASELINE_STABILITY.md`。

### S2：实现最小 MTP 核心（独立分支/默认OFF）

建议依赖顺序：

1. **GGUF draft reader**：验证并加载 `blk.40`、`eh_proj`等原始权重；实现shape/type/量化严格检查；拒绝缺失或错格式，不能静默用随机数据。
2. **draft model**：实现或适配单层草稿输出，草稿token顺序与对应embedding/hidden状态一致；不要把上游40层主模型当成draft。先只跑静态输入和单轮的人工oracle。
3. **主模型 multi-token verify**：让目标模型一次正确验证至多K个候选，保证 **T>1 在本地 qwen35 bridge 内实际走哪条路径** 有记录；验证窗口的专家路由数/缓冲尺寸/CPU-GPU kernel支持上限须明确。不要把批量Prefill路径“能运行”当成“能无误提交多Token”。
4. **状态事务**：在未确认候选前保留GDN/DeltaNet recurrent状态与Attention KV的可回滚checkpoint；按接受的**连续前缀**提交，拒绝后只留下合法token及其状态。检查每层KV位置、context剩余长度、graph对齐、RNG采样状态及EOS边界。**必须有拒绝首token、拒绝中间token、全部接受、遇到EOS、接近context末端等测试**。
5. **采样正确性**：greedy先做无歧义逐token对照；非greedy必须采用正确的概率校正/验收规则，不得仅逐位比较抽样token后直接输出而宣称保持采样分布。未做采样正确性时只能标 `greedy-only experimental`。
6. **VRAM**：MTP layer、draft caches、verify workspace、CUDA Graph相关缓冲先建显存账；保留可用余量和显式OOM失败路径。原产品 `cache3200` 是重要回退约束，**不能为了启动MTP静默降缓存/挤占桌面显存**。如需tradeoff，给出分开方案让用户评审。
7. 生成 `MTP_K=0,1,2,3,4`独立开关/CLI功能（具体名字以本地代码统一命名为准）。**`--spec 4` 只代表上游参数，不是本地现成开关**。

实现中的代码应按**权重加载 → 单草稿数学 → 主模型验证 → 状态回滚 → EOS/JSON → 性能**逐步增加，任何一步没有校验结果就不要直接进入吞吐测试。

**交付**：独立分支、默认关闭的最小代码差异、模型无关的单元测试、校验日志、状态事务测试、构建信息与回退方式。

### S3：三种测试，不可互相替代

**正确性门**（硬约束）：

- K=0与K>0在确定性测试用例中逐token一致；发现近并列的量化浮点差异，记录logits/概率间隙，区分本来就不确定的算子漂移与新的verify/rollback错误。
- 真实业务风险等级0–3、是否违规、原始策略约束、JSON Schema、EOS、quoted evidence的出处（不能只是出现引号）、输出无重复token、无越界、无NaN。
- 原始冻结版也有risk2 vs 标签3的质量缺口：单独解决测试fixture与模型能力区分，不允许修改标签、删审核政策、把“不确定”当质量PASS。性能诊断可继续，但**正式上线不能跨越质量门**。
- GPU资源：冷热交替、cache loan归还、下一个请求Prefill、长驻多请求、合法context尾部与停止；无CUDA illegal access/OOM/空vector UB/graph生命周期问题。

**机制门**：

- 按独立业务场景记录 `proposed`, `accepted`, `rejected`, `rounds`, `committed_per_verify`, `avg_draft_ms`, `avg_verify_ms`, `rollback_ms`。
- 理解为何接受率变化；尤其验证短JSON中的标点、数字风险等级和引号对草稿的影响。只允许真实模型采样轨迹，不能偷看未来token/teacher-force当加速。
- 作者的 84%–96% / 2.4–3.1是外部对照，不能假设本地同样成立。

**业务性能门**：

- **短JSON（90%）**：完整请求p50/p95、JSON闭合时刻、TTFT，重点避免 MTP启动+验证比剩余16tokens更贵；报告 **请求/s 与秒/请求**。
- **长解释（10%）**：完整输出成功率、Decode committed tok/s、TPOT p50/p95、E2E；明确截断/超上限不能算成功输出。
- **Prefill保护**：复验四类历史负载；同模型同进程常驻与冻结官方口径分别对照，不得把不同权重、不同K/V类型及冷暖混合成单一“提升”。
- 仅当同条件、无诊断产品配置重复测试的收益超出测量噪声且质量通过，才允许声称「有业务提速」；否则标INCONCLUSIVE/NO-GO。

### S4：优化K值，是否值得产品化

1. 从 `K=1/2` 最小verify路径开始，确认正确性后试 `K=3/4`；不要直接按上游 `spec4` 定死。
2. 对每个K记录成本模型：`T_cycle(K) = T_draft(K) + T_verify(K) + T_rollback(K) + T_wait(K)`，`TPS_committed ≈ E[committed_tokens_per_round] / E[T_cycle]`。**这些量应基于总数/总时间计算，不能只用各项均值相除以免产生偏差**。
3. 对短审核，另算 `T_E2E = T_prefill + T_first_output + T_decode_remaining + T_finish`，避免TPS改善而业务响应反而慢。
4. 若产品仅需10–20个token，允许自适应策略：默认K=0，只有基于**已知的**请求类型/长度预算等先验，并经A/B确认时才启用K>0；不能靠泄漏真实将来输出长度选择策略。
5. 长输出若有可靠收益，允许设为独立产品模式，短输出仍维持最稳健路径。

**Go标准**：数值/语义/显存/Prefill均过；至少一个真实目标业务获得可重复的**完整请求**改善，并给出收益量级、置信与成本。**No-Go**：欠缺草稿权重、状态正确性无法证明、额外显存使8GB不可稳定运行、长驻p95恶化或产生严重风险漏报。**Inconclusive**：样本太少/负载波动大、只有MTP推进数没有E2E改善。

## 五、与其他研究路线的关系：不重走失败路线

- 本项目之前**静态profile**可降低约24% CPU miss，但完整业务提速仍无法确认；**动态专家迁移/淘汰**在短16-token审核里不回本；`admit`满时no-op；CPU量化开销约0.62ms。这些不应成为MTP实现的隐含前置工作。
- **PR #2 冷专家连续行上传合并**是独立、默认关闭、尚未编译/测速的微优化，不与本MTP臂混合或归功。不要为了探索MTP先合入PR #2。
- **T=1 CPU/GPU关键路径审计与长驻稳定性**仍是基础，MTP并不会免费消除完整主模型的验证成本；要同时测出主模型**单token Decode**和**K-token verify**成本。对于本地 `T>1` 专家路径尤其重要。
- **NInfer**提供另一个MTP/offload实现参考，但本专题优先选择同代码体系的Strata PR #1402作为主参考；**KVMem/ActKV**解决长期上下文KV问题，当前4K审核不并行移植。
- 不因为外部PR数据很好就把冻结的Prefill、Cache参数、Strata/llama依赖版本整体替换；只在明确必要时小范围选择性移植并逐一验收。

## 六、交付结构与研发纪律

最终至少形成：

```text
 docs/phase2_mtp/
   PR1402_REVIEW.md             # 上游PR与本地路径差异，逐函数和链接
   MODEL_MTP_INVENTORY.md       # GGUF层/张量/type/shape/SHA + 能否用MTP
   IMPLEMENTATION_MAP.md        # draft → verify → commit/rollback → KV/GDN → EOS
   MTP_BENCH_SPEC.md            # A/B/C arms、输入/输出、统计与业务质量协议
   BASELINE_STABILITY.md        # 同进程稳态基线、自对照波动与干扰判断
   EXPERIMENT_LEDGER.md         # 每候选hypothesis/预算/构建/运行/哈希/结论
   RESULTS.md                   # raw数据路径、结果与NO-GO/INCONCLUSIVE/PASS
   HANDOFF.md                   # 当前分支、下一步、撤销方式、未闭合风险
```

每项实验必须记录：精确 `repo HEAD / PR upstream commit / GGUF SHA / exe SHA / GPU型号与空闲显存 / 驱动 / CUDA版本 / context / ubatch / slots / threads / K / 采样参数 / prompt SHA / 输出长度 / EOS / 正确性判定 / 首用或长驻 / A/B顺序`。

- **单 GPU owner**：无人能与实验同时使用同卡运行其他模型；若检测到资源竞争，停止并标记，绝不宣称有效加速。
- **单 writer**：核心代码修改在独立worktree，严格基于冻结基线与明确的本地检查点，不覆盖已审查的PR与原始证据。
- **Feature flag 默认关闭**，新exe与旧exe不同目录，回退能一步恢复。进入正式产品前另做低频非法显存访问的长驻压力测试。
- 测量结论分为：**事实验证 / 数学正确 / 机制有效 / 局部加速 / 完整业务加速 / 质量合格 / 可以上线**。不得混用。
- 除非用户另外要求，不要把私有原始聊天、模型权重、token文件、客户凭证、业务证据上传GitHub。

## 七、最终需要主Agent回答的五个问题

1. **现在这份GGUF是否真的有 `blk.40` MTP草稿层？** 附张量清单/类型/哈希。若没有，能否以**同主权重、仅增加草稿层**构成公平试验？
2. **当前 `qwen35_model.cpp + patched llama.cpp + bridge` 能否合法进行K-token一次验证、部分接受后恢复DeltaNet/KV状态？** 逐函数指出缺口与最小改动。
3. **8GiB显存还放不放得下？** draft层、verify workspace、额外K/V与缓存3200槽之间的精确预算；如果需要降缓存，完整方案是否净收益？
4. **在真实16-token短JSON与128/256-token长解释下，哪种K获得同条件可重复E2E收益？** 返回吞吐、延迟、验证轮成本、接受率、错误/拒绝率及不确定性。
5. **与PR #1402的70/89 tok/s差距有多少来自4070Ti计算/带宽、MTP接受率、专家调度和上下文长度？** 不允许用不同硬件测数直接按显存带宽换算本机理论上限。

**第一轮汇报请先给 S0 的可行性与实现映射，尤其 `blk.40` / `multi-token verifier` / GDN-KV rollback / VRAM账；并附 S0.5 的测试设计与ROI预判。没有证据前，禁止直接合并PR #1402或宣布MTP已移植。**

## 八、给主Agent的明确执行口令（单独转发本MD即可）

> 本文件是Phase 2.2的独立研发任务书。完整阅读后，从S0开始，不要立即构建/下载大模型或动GPU；先核对本地最新夜间实验账和工作树，完成MTP权重盘点、上游PR函数级差异、GDN/KV状态事务和显存预算；同时评估S0.5小T验证成本与独立上游对照是否可行。提交S0/S0.5证据后，根据GO/NO-GO再执行后续阶段。保留既有Prefill产品基线和PR #1/#2，不合并其未确认优化。单GPU owner、单writer、每阶段预算/正确性/错误退出传播严格执行。需要多轮持续优化，但不以实验次数代替业务净收益。
