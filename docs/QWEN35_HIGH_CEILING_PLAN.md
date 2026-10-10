# 高上限优化：计划与证据边界

2026-10-10：用户认可约1000的阶段成果，授权继续优化。主负载仍是冻结3000–4000中文字符审核文本（3869/3863实际tokens），c4096/t8/短审核输出，加载单列。1100–1200及1300–1500+是探索区间，未达到、无速度承诺。公开源码仍为reviewed65-business67标签；未验收功能候选不混入该快照。

已采用且有实验依据：物理微批2048（不是context上限）、两个冷专家producer、按专家有效行数降序分组。完整历史与不可相加的收益见QWEN35_OPTIMIZATIONS.md。

1. 主任务69：router weights保持GPU，移除GPU→CPU重排→GPU往返，保留实际stride与生命周期。同构建开关对照、真实4token combine索引检查及两个业务样本，累计最多4个native，失败停止；IDs与host冷调度等待仍保留。正在实现，未报收益。
2. 只读70已交付：packed Top8 IDs及GPU metadata/grouping契约。源码推导T2048的strided IDs约2MiB，可压为64KiB；实际stride/字节与wall需测。CPU仍需冷专家job信息，搬grouping不能宣称消除host barrier。与69同prefill文件，后续串行集成。
3. 只读71已交付：cold两段直接写MMQ group并乒乓复用，减少设备环→group拷贝。须保留producer就绪、H2D完成、compute读取结束、workspace借还；resident指针读取需要MMQ kernel接口改动，不能当简单替换。均未实施/测速。

Codex复核纠正：70的原始2次测试预算未明确覆盖A/B，实施前逐case另立预算；71与69/70存在文件交集，不能多writer同时改。旧严格mixed replay已有失败，不能假设它是候选可通过基线；新增路径用同构建OFF/ON同路由真实权重比较，保留旧失败，不扩大容差掩盖。

68本地诊断：GDN eager事件总526.57ms，60对/两个prompt shape；不是SM忙碌时间或关键路径，capture/error清理仍有限制。默认关闭计时风险样本909.138 token/s/TTFT4.257s各一次；正常68未跑。诊断有扰动，不把局部事件时间相加当完整wall。

thin专家CPU分流、whole-layer cache、FP8/新compute format、persistent/graph等需额外合同和实测。thin rowhist尚无residency交叉；全层缓存会挤出其他层驻留并涉及临时loan，槽数简单相除不能证明收益。外部vLLM/2080Ti成绩不能換算本机IQ3_S。

每个候选先有限单次比较，有收益或明确架构价值才另立确认预算。质量守卫覆盖finite、CUDA/OOM、乱码、格式、审核结论与实际cache/流送/借还；微小表达差异按用户许可处理，数学与资源错误仍须处理。GPU始终单所有者，功能实现由OpenCode Flash/max承担。
