# 当前方案清单：供外部讨论

截至2026-10-13 16:40（北京时间）。依据本地冻结实验与原始日志；268/269仍运行，本清单不把进行中的方案当作已成功。

## 当前结果和条件

RTX4060 Laptop 8GiB / Windows / threads8 / context4096 / actual ubatch4096 / 2700专家槽 / 固定正常审核输入3896 tokens。

实验模型为Qwen3.6-35B-A3B-MTP-GGUF UD-IQ3_S，SHA256 ab639a7f330f96c47d3e6c2dd2d6445182e7b763e17e6048dc850a71bbc9f27f。原生产模型与冻结Prefill构建保持不变。新模型有49项主模型张量量化类型不同，因此历史旧模型速度不能直接充当严格MTP OFF对照。

最新已核真实ON、K=1、紧凑缓存、同进程第二请求（267 control，诊断关闭）：

| 指标 | 实测 |
|---|---:|
| Prefill | 2.903400秒，约1342 token/s |
| 真首token flush TTFT | 2.911624秒 |
| 完整JSON可见 | 3.565759秒 |
| EOS完成 | 3.566007秒 |
| Decode：15/(末token flush−首token flush) | 22.931 token/s |
| 引擎窗口 | 0.660812秒；不含完整Prefill |
| 整卡显存峰值 | 7445.2MiB |

输出16个可见token，完整risk0/violation=false JSON，EOS248046；8轮全部接受。与此前264约22.83TPS/EOS3.565秒接近，尚未证明稳定Decode提速。风险等级、真实拒绝回滚、128–256长解释均未完成验收。

## 已采用、已经实际运行的方案

| 方案 | 做法 | 判定 |
|---|---|---|
| 同家族MTP实验模型副本 | 独立加载blk.40全部20项真实草稿权重，主模型40层专家pack另建 | 模型和真实草稿能运行；不是仅打开开关或伪草稿 |
| 新量化GPU适配 | 开启MMQ/K-quants/Q6专家支持，修正真实Q6 DOWN几何2048/512 | 消除此前整模型Prefill CPU回退；属于兼容修复，没有独立量化内核优化收益证明 |
| CPU/GPU混合专家与2700槽 | GPU热专家、CPU冷专家；沿用固定profile和现有Strata调度 | 为8GB实验配置，生产默认3200不变 |
| 紧凑静态专家缓存 | 每槽按所属层实际blob分配，避免全部按最大blob固定步长；补slot span与借还防护 | arena4535.2→3199.2MiB，节省1335.93MiB已实测；ON整卡峰降低。速度仍需更多同状态对照 |
| Greedy K=1 MTP | 每轮预测一个草稿；主模型T=2验证pending+draft，取两行logits与hidden | 真实完整JSON/EOS/第二请求已跑；当前约23TPS，未达预期 |
| Hidden深拷贝及接受后草稿KV重放 | draft接收上一次目标hidden；接受后用真实token+目标row0 hidden补草稿KV，下一pending用row1 | 是正确性必要接线；还未做省重放或重放融合 |
| 连续前缀事务与EOS预算 | target outputs_max2/n_rs_seq1；拒绝分支seq_rm；引擎统一拥有seed/draft/bonus/corrective输出 | 正常全接受样本和EOS通过；实际拒绝路径未触发，不称完整事务通过 |
| 同进程清理再请求 | clear、完整重新Prefill、新seed、再跑真实ON | 正常样本成立；首请求与后续请求分列 |
| 真输出时钟及阶段诊断 | fflush后记录每token时钟；诊断默认关闭，另跑同步控制 | 完整E2E现已可测；267阶段计数有重叠，主管已指出，不能采用精确83%占比或负残差clamp |

当前全接受K=1的成本结构：一次草稿T1 + 一次目标T2 + 一次接受后草稿T1重放，通常推进两个可见token。100%接受仍不必然加速。目标T2验证约582ms/8轮是较大窗口线索，但不是精确互斥GPU耗时。

## 正在尝试

**268：目标Verifier串行T1候选。** 保持真正MTP ON和真实草稿，default-OFF独立开关。目标先解码pending取t0/h0；接受时再解码draft取t1/h1，拒绝时不写投机未来行。与现有T2批量验证作同binary、同负载、两进程各两请求对照。目的是检验现有T1优化路径是否比小T2更高效；它不是已成功的加速方案，也不等于关闭MTP。预算2构建/2GPU进程/4内部请求，失败保原件。

**269：固定源码T2路径反证。** 只读核对T2是否绕开已有native Decode路径、进入何种专家/GDN执行；不改268正在施工的源，不占GPU。尚无最终报告。

## 前一Decode阶段已评估的路线

| 路线 | 证据与判定 |
|---|---|
| 缩减admit同步 | 缓存已满，整段约0.5–0.9ms，收益不足，封存 |
| 短审核动态专家迁移/淘汰 | 离线命中可提升，但迁移成本不回本，短审核NO-GO；长解释未实测 |
| 静态Decode profile重标定 | 正常miss3033→2309，风险3076→2213；机制有效，但完整Decode速度受波动影响无法确认，暂停施工 |
| CPU activation量化优化 | 整个输出量化仅约0.618ms，不作为主攻；不代表CPU矩阵/调度已无空间 |
| 连续miss行上传合并 | 3028行可合并1359段，默认关闭patch已准备；没有完整GPU速度收益证明，不算成功或实测失败 |
| activation预取 | 只做上限估计，未实施性能A/B |
| 图侧ADD/RMS/MUL融合 | 局部计算有改善，但实际图命中0，没有端到端收益 |

旧生产Prefill已保留的有效路线包括：ubatch4096、count_desc专家顺序、router weights/packed IDs设备路径、producer4与ring16及批量专家流水线。历史旧GGUF稳态Prefill约1362–1430TPS，仅作为冻结成果，不能替代当前新模型完整审核数据。

## 尚未实施，不应列作失败

- K=2–4多步真实草稿及T=3–5完整Verifier循环。
- 接受后重放与下一次草稿融合。
- 省掉接受后草稿重放的不使用logits head。这里指draft replay，不是目标t1 bonus。
- MTP循环专门CUDA Graph覆盖、草稿与验证重叠。
- DFlash或其他推测解码算法。
- 紧凑布局下3200槽真实ON测试、128–256-token长解释验收。
- 新GGUF专项MMQ tile/专家分组优化。

## 希望外部评审回答的问题

1. 本地小T2是否比两次T1更慢？应检查哪个实际函数/同步依赖，如何最小化修复？
2. K=1每两token付出T2目标加两次T1草稿，是否应优先扩K，还是先融合重放？请给出正确hidden/KV与拒绝恢复合同。
3. 草稿replay无用输出head如何安全关闭？vendor API是否支持真正零输出，而非helper false仍输出最后行？
4. 在8GB下，长解释接受率、Verifier成本和额外显存如何影响净E2E？不能把短JSON全接受推广到长解释。
5. 如何建立同状态ON/OFF公平对照，解释首请求约9秒与后续约2.9秒的Prefill差异？

外部建议最好包含：具体函数/等待边、收益上限、最小改动、状态与文件所有权、对照方式、预算和停止条件。当前最新MTP实验尚未在本次核实为GitHub main内容，审核应使用对应冻结源码，不把旧公开快照当最新实现。

证据入口：results/phase2_mtp/review267/verified.json及frozen/；prototype267原始日志；review264/verified_raw.json；前期docs/phase2_decode/DECODE_STAGE_REPORT_20261012.md。所有计数、失败与历史报告保留，不以任务编号数量当优化实验数量。

## 串行Verifier首筛与smallT原生MoE隔离准备 2026-10-13T16:56:33.721296+08:00

本轮一次实际check268CLI30248仍live、无GPU/编译/fence，不另开核心writer或追加268GPU。实际两native均exit0/tree0/16ID×4/EOG一致，7source/pre-run+exe2678417b/rawSHA已独立冻结review268。第二reqT2 20.652TPS/P2.992089/EOS3.727243，对串行T1 23.341TPS/P3.000233/EOS3.652200，只一次约13%Decode/2%E2E不称稳定；peak7586.7/7616.0，正常全接受0reject。2682build/2native4req实际已用，staticchecker>8 sites偏差待最终真实日志，不假称预算通过。269CLI0只读链完成：T2promptMMQ+GDN_CH，T1native+GDN_AR，现场decode0/prefill720对1280/80吻合；串行snapshot消费者未证。P271独立副本唯一bridge小T2–5逐行native service原型defaultOFF，保持targetbatch/GDNCH数学、实际stride/failclosed/行间work+copy生命周期fence，≤1functional/8CPU一次10s/0buildGPU，不碰268live。待源码review才集成/实际ON，不承诺收益。M268当前唯一GPU预约直到自然checkpoint交付。已核模型79/build29/MTPGPU20/启动21、P此前2单列；生产冻结/ACTIVE25不变。review CODEX_REVIEW_268_269_SMALLT_NEXT.md。
