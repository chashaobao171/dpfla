# 04_Retracted.md — 已证伪 / 已推翻的结论

<!-- 最后更新：2026-10-10 -->
<!-- 目的：本记忆包历史上出现过错误结论。此文件集中记录，禁止再次作为决策依据。 -->
<!-- 规则：每条写清 原结论 → 出处 → 推翻证据 → 正确处理。 -->

> 读到这里请牢记：**下面每一条都曾被写入记忆并被当成事实。**

---

## R1【已证伪】"FedAIoT 的核心是 FedAvgM（服务端动量 0.9）"

- **原结论**：`05_FedAIoT_Reference_Report.md` 断言 FedAIoT 达到高 mAP 的核心是"服务端动量 0.9 + 服务端优化器 SGD lr=1.0 = FedAvgM"，并据此建议在本项目实现 FedAvgM。
- **推翻证据**（直接读 FedAIoT 源码）：
  - `FedAIoT-main/aggregators/base.py`：`server_momentum` 参数**被接收后完全未使用**——构建优化器时只传了 `lr=server_lr`，没传 `momentum`。
  - `step()` 里 `grad = param - avg`，`optimizer.step()` → `param -= lr*(param-avg)`；示例命令用 `--server_lr 1` → 结果 = `avg`，**数学上等价于纯 FedAvg**。
  - `FedAIoT-main/distributed_main.py` 虽传了 `server_momentum=0.9`，但因上一条，该值无效。
  - `commands/overview/fedavg/*.sh` 全部为 `--server_optimizer sgd --server_lr 1`（= 纯 FedAvg）。
- **正确处理**：FedAIoT 的收益**不在聚合**，在其训练配方（原生 ultralytics 训练、lr0=0.01 + warmup3 + CosineWarmRestarts、mosaic=1.0、600 轮、AMP、grad clip10）。**本项目模仿 FedAvgM 属错误借鉴对象**，已回退。详见 `07_ThirdParty.md`。

---

## R2【已证伪】"fl_core.py 子进程 val 每次返回 0%（当前卡点）"

- **原结论**：`00_QuickStart.md` 曾把"子进程 YOLO val 存在 ultralytics 内部机制冲突，**每次 val 返回 0%**"列为冻结红线第 7 条与当前卡点。
- **推翻证据**：
  - `proof_dynamic_20260907_1520.log`：**R1=3.24%**（非 0）。
  - `run_no_attack_baseline_20260916_1437.log`：**R1=3.35%**。
  - 与历史基线 R1≈3.36%（2026-06-10）一致 → 评估链工作正常。
  - 日志里出现的 `mAP@0.5: 0.00%` 是**训练前**对冷启动模型的评估（检测头刚随机初始化 + BN 为 COCO 统计量），属正常现象。
  - 09-15 的 `code=-9` 是 **OOM 被杀**（SIGKILL），不是代码 bug。
- **正确处理**：子进程 val 可用，**不是卡点、不是 bug**。真正的大头是联邦 mAP 上不去。

---

## R3【已证伪】"BN 统计量从加载时就错了"（found in `round3_collapse_analysis.md`）

- **原结论**：把 mAP=0 归因为"COCO 预训练 BN 统计量不适配 VisDrone，加载时就错了"，并推荐 `reset_running_stats()`。
- **推翻证据**：
  - `round3_collapse_deep_analysis.md` 指出：若 BN 加载即错，则 LR=2e-4 基线不可能到 23.7%；实测 running_var≈0.008 是 **LR=0.01 训练 1 epoch 后**的结果，不是加载时的值。
  - 实测 `reset_running_stats()` **有害**，commit `8554d4d` 已撤销。
- **正确处理**：BN 不是根因，**不要 reset BN**。当时 mAP=0 的主因是 **LR=0.01 过高 + LOCAL_EPOCHS=1**。

---

## R4【已证伪/已回退】"FedAvgM 硬编码 0.9 恒开 + 梯度裁剪 10.0 能提 mAP"

- **原结论**：按 R1 的误导，FedAvgM(0.9) + grad clip(10.0) 被硬编码常开。
- **推翻证据**：9/2 demo 10 轮仅 **2.55%**；同初始状态第 1 个 epoch 训练损失 **705→1010**（裁剪削弱有效学习率）。
- **正确处理**：两者改环境变量门控、**默认关闭**（`FL_SERVER_MOMENTUM=0`、`FL_GRAD_CLIP=0`）。

---

## R5【已失效】路径 `/root/chashaobao/DPFLA-master-lastest/DPFLA-master`

- **原结论**：旧 README / 09-15 日志 traceback 引用该路径。
- **推翻证据**：该目录**不存在**（`ls` 为空）。项目现址为 `/root/chashaobao/DPFLA-master`。
- **正确处理**：以 `05_Environment.md` 的路径为准。

---

## R6【已失效】标签目录 `labels_yolo_visdrone10/`

- **原结论**：旧文档称 YOLO 标签在 `labels_yolo_visdrone10/`。
- **推翻证据**：该目录**不存在**；实际评估走原始 VisDrone 的 `labels/`（与 `images/` 同级），由 `generate_visdrone_yaml.py` 生成 yaml。
- **正确处理**：以 `05_Environment.md` §数据 为准。

---

## R7【已失效】第三方工程 `YOLOv8_Federated_Learning-main/`

- **原结论**：`03_ThirdParty_Skeleton.md` 称参考工程在项目内 `YOLOv8_Federated_Learning-main/`。
- **推翻证据**：该目录**不存在**。
- **正确处理**：现存第三方工程只有 `FedAIoT-main/`，见 `07_ThirdParty.md`。

---

## R8【待定·未核验】"FedAIoT 达到 34.85% mAP@0.5"

- **来源**：`05_FedAIoT_Reference_Report.md`（agent 产出，且该报告已因 R1 被证伪）。
- **状态**：**未从一手来源核验**。
- **正确处理**：不得作为决策依据。若需引用，须先跑通/核验 FedAIoT 的原始实验。

---

## R9【已证伪】"AMP 混合精度（autocast + GradScaler）已合入训练循环"

- **原结论**：旧 ChangeLog / 新 `03_History.md` 记 `client.py` 已合入 AMP。
- **推翻证据**：2026-10-10 全仓核查（排除 `FedAIoT-main/`）：`grep -rn "autocast|GradScaler|ModelEMA"` **零命中**；[client.py](file:///root/chashaobao/DPFLA-master/federated_learning/client.py) 训练循环为纯 fp32 `loss.backward(); optimizer.step()`。
- **正确处理**：VisDrone 训练循环**没有 AMP、没有 EMA、没有 LR scheduler**。旧记载与实现不符，不得据其推断"已启用混合精度"。

