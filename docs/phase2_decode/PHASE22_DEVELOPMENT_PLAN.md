# Phase 2.2：Decode关键路径与长驻稳定性优先

2026-10-12用户直接给出的下一轮建议已采用；优先于v2.1旧缓存优先级。长期研发继续，每项有限预算，Codex架构/review/协调/验收，功能实现仅OpenCodeGo Flash/max。

## 目标与阶段

90%短审核以完整JSON可见/真实EOS完成请求延迟、风险等级正确性为主，DecodeTPS辅助；10%详细风险解释以128–256实际输出的TPOT/吞吐/完整性/尾延迟为主。30–35、40–45、50–60+token/s分别是探索里程碑，不是当前成绩或可达到承诺。

P0先建立同进程多个完整autoregressive JSON请求：loadonce、首请求/协议预热/稳态分列；模型context/专家cache/OS页热度/请求序列、CPU线程/频率、Graph覆盖状态有证据。现WARM_ROUND/MULTI_ROUND仅Prefill+单答案，不当多请求Decode；176给最小状态/测量合同后才派独立defaultOFF实现。初始功能闭环仅最少请求，不能给可靠p95；正式分位验证再注册固定请求数、重复进程数/预热次数/ABBA顺序/样本单位和退出门。内部请求数与native进程启动数分别记账。候选静态env只能在独立同配置进程设定，不中途改gate混AB。

P1建立40层T=1互斥或明确重叠时间线：CPU GU/DOWN/线程池等待、GPU驻留热专家、activation就绪、H2D及combine、非MoE GDN/Graph。约9ms/token IDs host窗口含上游GPU，不当router纯耗时；pool drain含GU/Q/DOWN，不相加。主机跨度含设备回压，CPU/GPU时长不是简单求和。

允许新的**有界Nsight Systems + CUDA事件/细粒度主机时间线**，覆盖少数真实Decode token/layer/request区间，需预先登记range、配置、trace次数/时长/体积、计时对象及扰动比较；不恢复旧全模型全量trace/G2/小时套件/性能网格。1Hz仅环境背景，不能解释40ms/token依赖；Graph capture计时与replay必须区分。先已有只读源码/轨迹判断覆盖与可辨识缺口，再最小采集，不为“定位”无限加测。

P2依真实关键路径选结构候选：CPU冷矩阵/SIMD/线程池、GPU热冷重叠或T1 GDN/Graph。每候选须给可缩短的依赖边、收益上限及缺口，独立baseline/source/binarySHA/文件接口所有权/defaultOFF/gate/累计构建native与时间预算/失败即停；正式GPU AB前基线可比。不能用miss/memcpy/kernel改善替代业务E2E。单项通过后组合另回归，不相加收益。

P3普通Decode基线可靠后研究MTP/DFlash，多token验证/acceptance率、draft与验证墙、额外KV/权重/峰值显存、专家cache损失和质量/状态回退成本；128–256解释先，短审核只接受完整E2E净收益。当前不实现/不加载draft模型。NInfer是参考，sm70/75、不同量化/权重布局与CPU/内存配置的数字不换算成本机IQ3_S/sm89。https://github.com/merbanan/ninfer/blob/qwen38-flash-next/docs/v100.md

## 路线判决

| 路线 | 决定 |
|---|---|
| admit缩减 | 封存（满cache下no-op） |
| 动态迁移/淘汰 | 短审核封存；长输出待P3独立成本模型 |
| CPU activation量化 | 封存（全输出约0.62ms） |
| 静态profile重标定 | 保留但暂停施工/再确认，基线稳定后真实E2E再评 |
| activation预取 | 暂缓，先辨11ms中真正可隐藏部分 |
| PR2连续miss行上传 | 一次1build/最多2native正确性验证；不是主线，不反复测速；计数诊断与未来性能AB分开 |
| CPU冷矩阵/调度 | 主线，先175依赖审查 |
| T1 GDN/Graph | 并行主线，179只读实际T1/覆盖合同 |
| MTP/DFlash | P3，当前不施工 |

减少miss约24%是机制事实，但现速度无法确认，不能推出缓存永远无用或已证明性能回退。174同基线17.48/25.48显示漂移，采样本身有扰动，根因unknown。

## 守卫与预算

冻结0e00af8a Prefill和M0573/生产profile保持；原固定负载、政策/template/context/t8不变，P c4096/out1与D合法context/实际EOS分列。每个newbinary须重新验证来源、构建选项、资源/数值/JSON/EOS与P回归，不继承Gate。完整JSON可见时刻、EOS采样结束、服务协议/queue/load与进程wall分列；TPS=(n-1)/(last-first)，TPOT=(last-first)/(n-1)，n<2无有效TPS。用户粗模型n/TPS仅估计，不替代首token实测。

风险2/true对label3独立调查，识违/等级/格式EOS/数值资源四门；正常合法诊断可继续，正式质量门未闭合禁止合并/产品加速宣称，不能改标签/答案。Prefill相关路径或state生命周期变动必须P强制回归、下一请求借还恢复与KV/GDN残留核对。

原producer/ring/pread/cache容量等封闭实验不重做。实验失败原件、超预算、部分输出/已知失败保留，不能新编号重置成本。一个GPU所有者，正式计时无CPU重活；独立实现隔离源码/构建/接口，主writer统一集成，read-only可并行。每15min一次有界推进/检查/报告，不持续监控。每次报告实际进展/最近有效条件与指标/未知/下一唯一最高价值动作/session与GPU；没有新测量如实说明。

## 本轮执行序列

175/176继续原只读范围，与新优先级一致，不中断重派。177为PR2一次正确性验证（独立candidate source、1configurebuild尝试、≤2native、60min、失败即停，不改数学/默认/冻结），两臂TIER_SPLIT=1仅读copy计数，UPLOAD_RUN_BATCH unset/1；没有产品性能AB、不称p95/Pgate。178编号保留给176交付后的P0独立实现准备，尚未派/无预算。179并行只读T1 GDN/Graph调用与trace规格，0build/native/GPU，不复读155的泛化地图、不直接搬Prefill563ms。
