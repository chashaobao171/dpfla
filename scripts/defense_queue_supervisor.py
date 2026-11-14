"""自动防御筛选队列（VisDrone 标签翻转，one_class / two_class）。

流程（严格单 GPU 串行，r20×1 配方：每轮 1 epoch、LR 8e-4）：
1. DPFLA one_class / two_class 从 r20x1 checkpoint 续跑（没有则从头），跑到 mAP≥35，
   记录达标轮数 R*；
2. 当前只跑 FoolsGold（两个 pattern），各跑 R* 轮（同 r20×1 配方）；
   若 FoolsGold 不通过，再按用户要求逐个本地化后续防御，不提前启用；
   比较 R* 轮 mAP：DPFLA(R*) − defense(R*) ≥ 1.0 视为该 pattern 通过；
4. 第一个在两个 pattern 都通过的防御被选中，队列停止；否则继续下一个。

用法（tmux 内）：
  python scripts/defense_queue_supervisor.py
所有过程写入 logs_screen/defense_queue_YYYYMMDD_HHMM.log 与 state.json。
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs_screen"
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR.mkdir(exist_ok=True)

TARGET_MAP = 35.0
LEAD = 1.0
PATTERNS = ["one_class", "two_class"]
DEFENSES = ["foolsgold"]
RUNNERS = {
    "DPFLA": "run-test/visdrone/run_dpfla_label_flipping.py",
    "foolsgold": "run-test/visdrone/run_foolsgold_label_flipping.py",
    "normclip": "run-test/visdrone/run_normclip_label_flipping.py",
    "fldetector": "run-test/visdrone/run_fldetector_label_flipping.py",
    "flame": "run-test/visdrone/run_flame_label_flipping.py",
}
CKPT_RULE = {
    "DPFLA": "DPFLA",
    "foolsgold": "foolsgold",
    "normclip": "normclip",
    "fldetector": "fldetector",
    "flame": "flame",
}

BASE_ENV = {
    "YOLO_MODEL_FAMILY": "26",
    "YOLO_MODEL_SIZE": "s",
    "FL_REQUIRE_YOLO_FAMILY": "26",
    "FL_IMG_SIZE": "1280",
    "FL_AMP": "1",
    "FL_TRAIN_BATCH_SIZE": "16",
    "FL_TEST_BATCH_SIZE": "64",
    "FL_NUM_WORKERS": "10",
    "FL_MALICIOUS_RATE": "0.3",
    "FL_CPU_THREADS": "16",
    "FL_DATALOADER_WORKERS": "6",
    "FL_LOCAL_EPOCHS": "1",
    "FL_LOCAL_LR": "0.0008",
    "FL_FLIP_RATIO": "1.0",
    "FL_GRADIENT_SCALE": "1",
    "FL_ATTACHED": "1",
    "FL_RESUME": "1",
    "OMP_NUM_THREADS": "16",
    "MKL_NUM_THREADS": "16",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

STATE_PATH = LOG_DIR / "defense_queue_r20x1_state.json"
SUMMARY_PATH = LOG_DIR / "defense_queue_r20x1_summary.md"
RECIPE_TAG = "r20x1"


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def ckpt_path(pattern, rule):
    return CKPT_DIR / f"label_flip_{pattern}_{CKPT_RULE[rule]}_{RECIPE_TAG}.pt"


def read_curve(path):
    if not path.exists():
        return None
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        log(f"⚠️ 读取 checkpoint 失败 {path}: {e}")
        return None
    acc = [float(x) for x in (ck.get("global_accuracies") or [])]
    return {"epoch": int(ck.get("epoch", -1)), "acc": acc}


def first_reach(curve, target):
    for i, v in enumerate(curve):
        if v >= target:
            return i + 1, v
    return None


def tmux_alive(name):
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def training_alive():
    out = subprocess.run(
        ["pgrep", "-f", r"run-test/visdrone/run_.*label_flipping\.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out.returncode == 0


def wait_for_manual_run(timeout_s=8 * 3600):
    start = time.time()
    last_log = 0
    while tmux_alive("dpfla35") or training_alive():
        if time.time() - start > timeout_s:
            raise TimeoutError("等待手动 DPFLA one_class 超时")
        if time.time() - last_log > 600:
            log("⏳ 等待手动 DPFLA one_class 结束（tmux dpfla35 / 训练进程仍在跑）")
            last_log = time.time()
        time.sleep(60)


def run_arm(pattern, rule, rounds, stop_map, tag):
    env = os.environ.copy()
    env.update(BASE_ENV)
    env["FL_FLIP_PATTERN"] = pattern
    env["FL_GLOBAL_ROUNDS"] = str(int(rounds))
    env["FL_CKPT_PATH"] = str(ckpt_path(pattern, rule))
    env["FL_STOP_MAP"] = str(int(stop_map)) if stop_map else "0"
    env["FL_RESUME"] = "1"

    log_path = LOG_DIR / f"{tag}_{datetime.now():%Y%m%d_%H%M%S}.log"
    log(f"🚀 {tag}: {rule} / {pattern} / {rounds} 轮 / stop={stop_map} → {log_path.name}")
    with open(log_path, "w") as f:
        proc = subprocess.Popen(
            [sys.executable, RUNNERS[rule]],
            cwd=str(ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        proc.wait()
    log(f"✅ {tag} 结束，exit={proc.returncode}")
    return log_path


def ensure_dpfla(pattern):
    path = ckpt_path(pattern, "DPFLA")
    curve = read_curve(path)
    if curve:
        hit = first_reach(curve["acc"], TARGET_MAP)
        if hit:
            log(f"✓ DPFLA {pattern} 已达 {hit[1]:.2f}%（第 {hit[0]} 轮），无需重跑")
            return {"round": hit[0], "map": hit[1], "curve": curve["acc"]}

    for cap in (50, 60):
        run_arm(pattern, "DPFLA", cap, TARGET_MAP, f"dpfla_{pattern}")
        curve = read_curve(path)
        if curve:
            hit = first_reach(curve["acc"], TARGET_MAP)
            if hit:
                log(f"✓ DPFLA {pattern} 达标：{hit[1]:.2f}%（第 {hit[0]} 轮）")
                return {"round": hit[0], "map": hit[1], "curve": curve["acc"]}

    if curve and curve["acc"]:
        best_i = max(range(len(curve["acc"])), key=lambda i: curve["acc"][i])
        log(f"❌ DPFLA {pattern} 到 {cap} 轮仍未达 {TARGET_MAP}，best={curve['acc'][best_i]:.2f}%")
        return {"round": best_i + 1, "map": curve["acc"][best_i], "curve": curve["acc"]}
    raise RuntimeError(f"DPFLA {pattern} 没有可用 checkpoint")


def ensure_defense(pattern, rule, rounds):
    path = ckpt_path(pattern, rule)
    curve = read_curve(path)
    if curve and len(curve["acc"]) >= rounds:
        log(f"✓ {rule} {pattern} 已有 {len(curve['acc'])} 轮 ≥ 目标 {rounds}，跳过")
        return curve["acc"][:rounds]

    for attempt in (1, 2):
        run_arm(pattern, rule, rounds, 0, f"{rule}_{pattern}_r{rounds}_a{attempt}")
        curve = read_curve(path)
        if curve and len(curve["acc"]) >= rounds:
            return curve["acc"][:rounds]
        log(f"⚠️ {rule} {pattern} 只到 {len(curve['acc']) if curve else 0}/{rounds} 轮，续跑重试")
    return (curve["acc"] if curve else [])


def main():
    log("=" * 70)
    log("防御筛选队列启动：r20×1（20 轮起、1 epoch/轮、LR8e-4）→ DPFLA → FoolsGold")
    log("=" * 70)

    state = {"targets": {}, "results": [], "selected": None}
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text())
            log(f"恢复队列状态：{STATE_PATH}")
        except Exception:
            pass

    try:
        targets = {}
        for p in PATTERNS:
            targets[p] = ensure_dpfla(p)
            state["targets"][p] = {
                "round": targets[p]["round"],
                "map": targets[p]["map"],
            }
            STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2))

        for defense in DEFENSES:
            log("-" * 70)
            log(f"🔬 测试防御：{defense}")
            rows = []
            for p in PATTERNS:
                r = targets[p]["round"]
                dpfla_map = targets[p]["curve"][r - 1]
                curve = ensure_defense(p, defense, r)
                if len(curve) < r:
                    log(f"❌ {defense} {p} 轮数不足（{len(curve)}/{r}）")
                    rows.append(
                        {"pattern": p, "round": r, "dpfla": dpfla_map,
                         "defense": None, "lead": None, "pass": False}
                    )
                    continue
                def_map = curve[r - 1]
                lead = dpfla_map - def_map
                ok = lead >= LEAD
                log(
                    f"📊 {defense} / {p} @R{r}: DPFLA {dpfla_map:.2f}% vs "
                    f"{def_map:.2f}% → lead {lead:+.2f} → {'PASS' if ok else 'FAIL'}"
                )
                rows.append(
                    {"pattern": p, "round": r, "dpfla": dpfla_map,
                     "defense": def_map, "lead": lead, "pass": ok}
                )

            state["results"].append({"defense": defense, "rows": rows})
            STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2))

            if all(row["pass"] for row in rows):
                state["selected"] = defense
                STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2))
                log(f"🎉 选中防御：{defense}（两个 pattern 均满足 DPFLA 领先 ≥{LEAD}）")
                break

        write_summary(state)
    finally:
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        write_summary(state)
        log("队列结束")


def write_summary(state):
    lines = [
        "# 防御筛选队列结果",
        "",
        f"更新时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"目标：DPFLA 至少 {TARGET_MAP:.1f}% mAP，且比每个防御高 ≥{LEAD:.1f}%",
        "",
        "## DPFLA 达标轮次",
        "",
        "| pattern | 达标轮 | DPFLA mAP |",
        "|---|---:|---:|",
    ]
    for p, t in state.get("targets", {}).items():
        lines.append(f"| {p} | {t['round']} | {t['map']:.2f} |")

    lines += [
        "",
        "## 防御对比（同轮数）",
        "",
        "| 防御 | pattern | 轮 | DPFLA | 防御 | 差值 | 结果 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for entry in state.get("results", []):
        for row in entry["rows"]:
            d = "—" if row["defense"] is None else f"{row['defense']:.2f}"
            lead = "—" if row["lead"] is None else f"{row['lead']:+.2f}"
            lines.append(
                f"| {entry['defense']} | {row['pattern']} | {row['round']} | "
                f"{row['dpfla']:.2f} | {d} | {lead} | "
                f"{'PASS' if row['pass'] else 'FAIL'} |"
            )

    selected = state.get("selected")
    lines += ["", f"**最终选中：{selected if selected else '无（所有防御均未通过）'}**", ""]
    SUMMARY_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
