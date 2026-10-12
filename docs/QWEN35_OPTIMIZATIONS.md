# 优化方案：哪些有收益，哪些只是基础能力

## 更新（2026-10-12，任务 69–150 已复核）

本阶段结论见 [QWEN35_STAGE_20261012.md](QWEN35_STAGE_20261012.md)。相对 65/67 时点的净变化：物理微批 4096（冷流送 −44%、实测 −262.8ms）与 producer 4（2→3 +7.9%、3→4 +3~6%）已交付，han 两样本官方口径达到 1000+（中心 ~1122）；cache 3200 作为交付标准（对 3700 付 −1.65% 换 600MB+ 余量）。已封闭不再投入：d2h（我们只占 5.59ms）、供给侧缺页（被 DMA pacing 吸收）、COLD_PREAD、扩缓存（量化到"临界"）、workspace 杠杆（≤−2.9%）。以下 65/67 时点原文保留为历史。

## 实验前提

RTX4060 Laptop 8GiB；Qwen3.6-35B-A3B-UD-IQ3_S GGUF；40层/H2048/FF512/256experts/top8，其中30GDN/10attention。固定风险/正常两组合审核样本分别3844/3819中文字符，实际3869/3863tokens，c4096/t8/out1。prefill token/s按实际输入tokens/完整prefill wall，加载排除，TTFT记录。无新增warmup；各方案单次比较，时间/页缓存/设备状态波动未完全消除。相同业务输入不等于自然单篇文档，两样本由完整业务对话组合，未靠重复填充达成长度。

## 当前推荐组合及收益证据

| 方案 | 做了什么 | 实测依据 | 证据强度/当前选择 |
|---|---|---|---|
| 物理微批扩大到2048 | 更多token先路由，再按expert分组批量MMQ，减少跨微批重复搬冷权重、分组与调度 | 同25b旧构建：ub512风险539.54/正常527.07；ub1024为614.16/629.32；ub2048为689.66/665.89tok/s。风险cold流送量约36357→21029→11697MiB，产品调用7044→4020→2194 | 历史各一次，支持方向但非稳定因果倍率；当前显式ub2048。微批不是上下文上限，c4096仍独立 |
| 双cold producer | 原有有界ColdStager环由2个CPU生产线程准备权重；claim/cancel/drain、拷贝流和GPU消费者协议不变 | task57正常同轮1producer743.370→2producer878.024tok/s（约18.1%）；风险2producer850.477，早先1producer668.034为跨轮参考 | 单次业务两样本通过；当前`STRATA_QWEN35_STAGER_THREADS=2`。第三producer风险846.692，没有超过既定门槛，未采用 |
| 按expert有效token数降序组队 | 非空expert按count降序、ID作tie；同步重建off/cursor、gather/scatter、coldjob索引，让每组16个expert的maxcnt更接近各expert有效行数 | task67同构建/双producer：risk控制732.547→排序954.560；排序normal911.174tok/s。旧task57参考850.477/878.024；控制偏慢，不能宣称30%稳定增益 | 两样本各一次业务守卫通过；保留默认off，可选`STRATA_QWEN35_EXPERT_ORDER=count_desc`。真实排序后MMQ tile变化未完整量化，代理指标不能当实测kernel加速 |

当前业务推荐参数是 **ub2048 + producer2 + count_desc + cache3700 + WCoff**。task67风险prefill4.0531744/TTFT4.0551644s；正常4.2395845/4.2412114s。距1000需要约4.76%/9.75%吞吐提升。每项不能将其单次百分比相加，或认为组合在所有输入上必然更快。

## 已接通并使用的基础能力（无独立A/B收益数字）

- 原Strata expert分组/成熟native IQ MMQ，而非逐token调用8个expert。expert分组早已存在；新的count_desc进一步调整组队，不是修复此前完全没regroup。
- 原ExpertCache真实GPU驻留：3700槽，实际专家payload3976.8MiB，槽位padding/reserve另计；独立profile扩展候选不会静默把容量卡在旧条目数。不是空分配，不是把整卡显存都报成expert payload。
- cold专家主要mmap GGUF gate/up/down三个slice，进入有界pin staging再异步送GPU，设备cold环16；不pin全部冷专家。
- 原workspace借还恢复：task67每次借262槽，恢复262/262，额外ownedworkspace0；不是永久侵占cache的独立arena。
- 成熟Qwen attention/GDN/shared expert/KV及原graph依赖，原Strata generate入口、共享expert service生命周期；CUDA graphs默认保持。
- 合并GPUgather已有fast路径，task62所有1097次均走fast；无独立A/B证明新增收益，不把它当刚修好的缺失机制。

## 未采用/不当作加速方案

- 整段ub4096：风险245.34、正常712.78tok/s，显存接近8187.5MiB，收益不稳定；没有继续扩大网格。
- 第3producer：846.692低于2producer850.477参考与5%采用门槛。
- write-combined pinned staging：风险809.289/正常875.710，控制异常偏慢，未证实WC收益；当前off。
- CPU批量预取/adaptive cache等仅有合同或分析，没有收益证据，不列为已实施优化。
- 默认关events/phase/rowhist：用于定位时间，不是加速；诊断速度不能当正式业务速度。

## 请其他AI优先讨论

1. count_desc是否在真实IQ MMQ tile/stream-K上降低浪费，如何用少量真实计数验证而不放大trace？
2. sorted路径的完整剩余4秒中，冷权重生产/传输与GPU计算是否有效重叠？host累计计时嵌套、两个线程时间重叠，不能相加作criticalpath。
3. GDN/attention/共享expert/图同步成本中，哪个最小改动能节省正常样本约0.377秒，同时保留成熟数学/状态布局？68刚交的GDN eager诊断仍待review，不能先下结论GDN是最大瓶颈。
4. 独立业务校准profile能否在相同真实cache容量内减少cold字节，避免用测试样本自身路由来作弊？

约束：原Strata流水线与成熟Qwen图、当前IQ3_S、固定主负载；实现由OpenCode承担；GPU串行，有界候选与失败即停，不以8k加长负载、局部kernel吞吐或最快一次宣布1000+达标。
