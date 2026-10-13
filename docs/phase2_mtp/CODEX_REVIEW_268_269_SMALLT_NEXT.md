# 268实际首筛 / 269路径反证

268 GPU两个arm已实际exit0/全树零/fenceclean；同exe2678417b，正常16IDs×4/EOG一致，7源hash与pre-run匹配。串行T1第二请求23.341TPS/EOS3.652200/P3.000233；T2第二请求20.652TPS/EOS3.727243/P2.992089。一次对照约13%Decode但E2E约2%，不同冷热首请求不配对，不称稳定收益。整卡peak7586.7/7616.0变动保留。实际build2/native2/internal4，最终CLI交付另核；check268.py明显多于8 textual check sites，实际执行数待最终日志，不继承预算PASS。

269静态链已核frozen桥：T2 ne1>1进qwen_prefill_experts，T1进expert_native_decode_layer；GDN CH/AR也不同。现场summary T2 decode_layers0/prefill_layers720，串行decode1280/prefill80，与链吻合。不是只能从函数名猜，不把host等待算纯kernel。串行仍保n_rs_seq1但reject无投机future/不消费rollback；全接受未证拒绝消费者。

下一P271只做隔离small-T MoE原型patch：保持目标批量T2及GDN_CH/事务，在bridge的T=2（可支持≤5）走逐token native service，非把整target改serial。每行真实stride x/ids/weight/output，借用stage需cudaStreamSynchronize(work)完成后才复用host_miss/parts/event，防异步host staging覆写；不可仅把T2指针当T1一次用。大Prefill与T1默认路径保持；defaultOFF，0build/GPU，主管review后才集成。这样可区分MoE管线与GDN/两次调度的贡献。不期望该原型必快，多同步成本也要实测。M268live不碰core，P271不改live而用固定bridge副本。
