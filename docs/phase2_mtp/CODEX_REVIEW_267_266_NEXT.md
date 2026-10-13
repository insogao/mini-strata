# Codex review267/266与下一候选

267 CLI0，两实际GPU exit0/tree0/fenceclean，source7hash/raw2hash/真实16ID×4/EOG248046已独立复核。2build含首次C2065失败与修复；2native4internal。模型77/build27/MTPGPU18/启动19，P此前2GPU单列。266只读报告完成，不重做265被拒Temp操作。

实际ON control第二请求 P2.903400/TTFT2.911624/JSON3.565759/EOS3.566007；真flush TPS见verified.json约22.93。诊断第二请求P2.925728/EOS3.597030，非提速。33同步调用raw sync_wait_ms=0，不能编造2ms/call或69ms。

时钟确定缺陷：verify打开早于draft close；hcopy打开早于verify close；hcopy包括已经emit计数的round写；done写在engine wall之后仍计入emit。−18ms不是可忽略时钟粒度，clamp不能当归因。verify大窗约582ms为线索，不精确互斥占比。unused-head指draft replay，不是target t1；全接受不是K>1 NO-GO证据。故不按错误报告首选重放融合或封闭K。

下一268研究候选为真实ON target verifier串行T1对原T2。同样draft/greedy最终序列，T1先取t0/h0，匹配时再T1取t1/h1；不匹配时无投机target未来行，无需删不存在的行。目标是检验既有T1优化路径是否优于小T2路径，不承诺收益，不称批量Verifier已优化。只normal首筛仍不关闭risk/reject门。成本计数边界先纠错、性能两个arm均COST_SPLIT/SYNC关闭。269只读固定267快照与固定expert文件定位T2实际路径，不读取268live，不占GPU。

266: 3200多596.36MiB，余145.94只粗算，不能安全保证；256需要合法context>=实际输入+输出/额外边界，c8192是候选配置而不是数学上唯一所需值。暂无3200/长输出GPU授权。
