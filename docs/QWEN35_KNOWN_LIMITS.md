# 已知限制：不要把业务通过写成数学完全认证

- task63/65同一真实单层replay严格逐位比较仍exit1，549265/4194304个float逐位不同；原guard没有放宽、失败没有抹除。
- Codex独立量化：全部finite，maxabs9.909272e-6，overall relativeL2 3.854647e-6。token1086的2048channels全有逐位差异，该token局部relativeL2 1.438805e-4（0.0144%）；其它token maxabs2.6077e-8。不能概括为全部few ULP。
- 分组会改变MMQ形状/stream-K partial reduction顺序；激活再量化阈值放大也可能解释集中token，但完整因果未证明，旧34数值root未闭合。没有把“所有regroup必然不同”当定理。
- 用户接受微小表达/数值差异。task67两个固定业务样本结论、格式、UTF-8、finite/CUDA/OOM、cache借还均通过，支持有界性能比较，不是所有审核文本的准确率认证。
- 两样本各一次、设备/pagecache状态漂移，风险控制732.547明显慢于此前参考；因此954.560只是已测候选值，不能宣称长期稳定或30%稳定因果增益。
- TTFT从prefill计时边界到首输出；模型加载已排除，task57/67加载单列未暴露，记unknown；不拿总进程wall减prefill冒充load。
- 当前尚未达到两样本各3次中位数1000+。没有8k主负载或多请求/跨GPU保证。
- 公开仓库不包含私有业务文本/固定prompt tokens/模型/二进制，所以公开指标可审阅但精确相同业务重放需要另行取得本地fixtures、独立profile与模型。
- 68计时扩展已交付、未完成Codex代码复核，未合入本次快照；报告中的GDN eager时间不是SMbusy/criticalpath，也不涵盖任意capture/replay。
