# 05_Environment.md — 环境 / 路径 / 数据 / 冻结红线

<!-- 最后更新：2026-10-10 -->
<!-- 本文件是路径、数据、命令的唯一事实源；其它记忆文件只引用，不重复定义。 -->

## §1 当前环境（2026-10-10 实测，有卡模式）

| 项 | 值 |
|----|----|
| 平台 | autodl 云主机，**有卡模式** |
| GPU | ✅ **NVIDIA GeForce RTX 4080 SUPER 32GB**（driver 595.71.05，compute cap 8.9）|
| CPU | 12 核（`nproc`=12） |
| 内存 | 503 GB |
| 磁盘 | `/root/autodl-tmp` 50G（可用 ~48G） |
| torch / CUDA | 2.5.1+cu124 / CUDA 12.4，`cuda.is_available()=True` |

> ⚠️ **无卡 vs 有卡差异**：无卡模式为 无 GPU / 1 核 / 754G；有卡模式为上表。切换后需重新确认。
>
> ⚠️ **沙箱拦截**：默认沙箱会拦截 `/dev/nvidia*`，导致 `nvidia-smi` 误报 "No devices were found"、`cuda.is_available()=False`。
> **跑训练/GPU 命令必须用非沙箱模式**，否则 GPU 不可见。

**环境验证命令**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：2.5.1+cu124  True  NVIDIA GeForce RTX 4080 SUPER
python -c "from ultralytics import YOLO; print('YOLO OK')"
nvidia-smi
```

## §2 路径映射（实测）

| 资源 | 路径 |
|------|------|
| 项目根 | `/root/chashaobao/DPFLA-master` |
| VisDrone 原始数据根 | `/root/autodl-tmp/data/visdrone` |
| 训练图 / 标签 | `.../visdrone/VisDrone2019-DET-train/VisDrone2019-DET-train/{images,labels,annotations}` |
| 验证图 / 标签 | `.../visdrone/VisDrone2019-DET-val/VisDrone2019-DET-val/{images,labels,annotations}` |
| YOLO 模型权重 | 项目根 `yolov8{s,n,m,l}.pt` |
| MNIST | `/root/autodl-tmp/data/MNIST` |
| CIFAR10 | `/root/autodl-tmp/data/cifar-10-batches-py` |
| 文档 / 对比图 | 项目根 `docs/`（`visdrone_fl_research_report.md`、`cifar_compare.png`、`mnist_compare.png`） |
| 历史日志 | 项目根 `logs_3/`（含 `demo_out.log`、`visdrone_all.log`、`oracle_1ep_*.log`） |

**实测数量**：train 图 6471 张 / val 图 548 张（均含 10 类原始标注）。

**代码中的路径优先级**（`visdrone_dataset.py` / `generate_visdrone_yaml.py`）：

```
VisDrone:  /root/autodl-tmp/data/visdrone  >  /home/featurize/data/visdrone  >  D:/Pycharmworkplace/visdrone
```

**评估用 yaml**：运行时由 `generate_visdrone_yaml.py` 生成 `visdrone_temp.yaml`：

```yaml
path: /root/autodl-tmp/data/visdrone
val:  VisDrone2019-DET-val/VisDrone2019-DET-val/images
nc: 10
```

**注意区分**：`/root/autodl-tmp/data/{images,labels}`（含 trainA~D / validation / test，train 1763 张）是**第三方工程的数据结构**，主项目**不使用**。

## §3 依赖安装（仅首次或依赖异常）

```bash
pip install -r requirements.txt
pip uninstall opencv-python -y
pip install --force-reinstall --no-deps numpy==1.26.4 opencv-python-headless==4.8.1.78
# ultralytics 会自动升级 numpy，需手动回退
```

## §4 运行命令

```bash
cd /root/chashaobao/DPFLA-master

# 无攻击基线（纯 FedAvg）
FL_ATTACHED=1 python run-test/visdrone/run_no_attack_baseline.py

# 无防御对照（FedAvg + 标签翻转）
python run-test/visdrone/run_no_defense_label_flipping.py

# DPFLA 防御（DPFLA + 标签翻转）—— 核心实验
python run-test/visdrone/run_dpfla_label_flipping.py

# TensorBoard
tensorboard --logdir runs/ --port 6006
```

- 默认 `nohup` 脱壳（关终端仍跑）；前台实时加 `FL_ATTACHED=1`；**tmux 启动脚本**：`./scripts/run_fl_tmux.sh <实验名> [ENV=VAL...]`（实时日志 + 断网不影响）。
- 日志：`logs_3/visdrone/<tag>_<北京时间>.log`；曲线：`runs/VisDrone_YOLO_<rule>_attack-<attack>_mr-<rate>/`

**常用环境变量覆盖**：`YOLO_MODEL_SIZE`、`FL_IMG_SIZE`(默认640，训练+评估口径)、`FL_GLOBAL_ROUNDS`、`FL_LOCAL_EPOCHS`、`FL_TRAIN_BATCH_SIZE`、`FL_NUM_WORKERS`、`FL_LOCAL_LR`、`FL_SERVER_MOMENTUM`(默认0)、`FL_GRAD_CLIP`(默认0)。

## §5 冻结红线（禁止擅改）

1. **DPFLA 主路径必须走 SVD+KMeans**：`rule=='DPFLA'` 时 `use_validation=False`，不用 loss 打分版作主结论。
2. **评估指标唯一**：主指标 = YOLO 原生 `val()` 的 `mAP@0.5`，**禁用 fallback 自定义 mAP**。
3. **聚合精度**：YOLO 场景统一 `float16_floats=False`（float32）。
4. **数据白名单**：`logs_3/mnist/`、`runs/MNIST_*/` 为实验数据保留目录，删除前需确认不在此列。
5. **基线配置（以当前代码为准，见 01_Facts §6）**：`TRAIN_BATCH_SIZE=64`、`LOCAL_EPOCHS=3`、`LOCAL_LR=2e-4`、`LR_SCHEDULE=constant`。禁用 cosine。
   > 历史记录里曾出现 `LOCAL_EPOCHS=10` 的"好配置"表述，但实测达到 R26-33≈23.7% 的 100 轮基线用的是 **EPOCH=3**；两者不冲突（BS=64 下 EPOCH=3 即可收敛）。
6. **单变量原则**：任何实验一次只改一个变量（87e3f7a 的教训）。
7. **数据分布固定 IID**：脚本不设 `dd_type`，走默认 `IID`（`arguments.py:58`）；**不考虑 Non-IID**（2026-10-10 用户确认）。

> 注：旧版"红线第 7 条：fl_core.py val bug"已删除——该结论已证伪，见 `04_Retracted.md` R2。

## §6 故障字典

| 现象 | 原因 | 处理 |
|------|------|------|
| `libGL.so.1` 加载失败 | ultralytics 装了 GUI 版 opencv | 卸载 opencv-python，装 opencv-python-headless |
| val 子进程 `code=-9` | 内存不足被 OOM 杀 | 降低并发/显存占用后重试（**非代码 bug**） |
| `pickle data was truncated` | 缓存损坏 | `rm cache/cache_yolo_val_*.pkl`（缓存可重建） |
| Round 0/1 mAP=0 | 换头后检测头随机初始化（冷启动） | 正常；R1 起 >0 即链路正常 |
| 训练看起来慢，GPU 未利用 | 前期在 CPU 做标注缓存 + Dirichlet 划分 | 等缓存生成完再观察 |
| 换聚合/改超参后 mAP 更差 | 一次改了多个变量，无法归因 | 回退，改单变量重跑 |
