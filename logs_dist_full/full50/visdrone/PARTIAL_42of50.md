# VisDrone 分布式 42/50 部分结果（保留）

- 运行时间：2026-11-01 14:19:03 – 2026-11-02 02:19:03
- 请求轮数：50；实际完成：**42 轮**
- R42 server mAP@0.5：**36.53%**
- R42 client mAP@0.5：30.45% – 35.62%
- 失败原因：orchestrator 总超时 **43200 秒（12h）**，不是 OOM 或 DPFLA/分布式代码错误；R43 广播后 client0 断开，orchestrator 杀掉全部进程。
- 11 个 TB run（server + client0–9）每个 **42 个点**，位于仓库 `tb/` 与云平台 `/root/tf-logs/`。
- 没有 `result.json`、没有模型 checkpoint，**不能从 R42 续跑**；正式 50 轮需从头重跑。
- 日志：`logs_dist_full/full50/visdrone/minmax/dpfla/20261101_141903_00/`
- 说明：本目录保留为部分结果，不作为 50 轮 PASS 结论；重跑 50 轮应将总 timeout 设为 ≥18h，或把 client 评估改为 128 张。
