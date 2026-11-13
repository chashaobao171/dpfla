"""标签翻转 1–4 类筛选 supervisor（DPFLA → TMean →（条件）MKrum）。

按 MEMORY/11_LabelFlip_50R_Plan.md §3 Phase C 的规则自动推进：
  1. 跑 DPFLA 20×1（LR 8e-4，训练量≈旧 5×3）；R20 < 30 → 不通过，换下一个类数。
  2. 跑 TMean 20×1；DPFLA R20 < TMean R20 + 1 → 跳过 MKrum，换下一个类数。
  3. TMean 过线才跑 MKrum；DPFLA R20 ≥ MKrum R20 + 1 → 该 pattern 通过，队列停止。

用法（tmux 内，前台串行，避免显存冲突）：
  python scripts/label_flip_screen_supervisor.py --patterns three_class four_class
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "logs_screen"
RUNNERS = {
    "DPFLA": "run-test/visdrone/run_dpfla_label_flipping.py",
    "TMean": "run-test/visdrone/run_tmean_label_flipping.py",
    "MKrum": "run-test/visdrone/run_mkrum_label_flipping.py",
}
ROUND_RE = re.compile(r"Round (\d+)/(\d+) - .*mAP@0\.5: ([0-9.]+)%")
THRESH_ABS = 30.0
THRESH_LEAD = 1.0
ROUNDS = 20          # 判定轮次：20 轮 ×1 epoch ≈ 旧 5 轮 ×3 epoch 的训练量
RECIPE_TAG = "r20x1"
RUN_TIMEOUT = 4 * 60 * 60

BASE_ENV = {
    "YOLO_MODEL_FAMILY": "26",
    "YOLO_MODEL_SIZE": "s",
    "FL_REQUIRE_YOLO_FAMILY": "26",
    "FL_IMG_SIZE": "1280",
    "FL_AMP": "1",
    "FL_TRAIN_BATCH_SIZE": "16",
    "FL_TEST_BATCH_SIZE": "64",
    "FL_GLOBAL_ROUNDS": str(ROUNDS),
    "FL_LOCAL_EPOCHS": "1",
    "FL_LOCAL_LR": "0.0008",
    "FL_FLIP_RATIO": "1.0",
    "FL_NUM_WORKERS": "10",
    "FL_CPU_THREADS": "16",
    "FL_DATALOADER_WORKERS": "6",
    "FL_ATTACHED": "1",
    "FL_RESUME": "0",
    "OMP_NUM_THREADS": "16",
    "MKL_NUM_THREADS": "16",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def parse_rounds(log_path):
    rounds = {}
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = ROUND_RE.search(line)
            if m:
                rounds[int(m.group(1))] = float(m.group(3))
    return rounds


def run_one(pattern, rule):
    env = os.environ.copy()
    env.update(BASE_ENV)
    env["FL_FLIP_PATTERN"] = pattern
    # 与旧 5×3 结果分开存，避免误续跑
    env["FL_CKPT_PATH"] = f"checkpoints/label_flip_{pattern}_{rule.lower()}_{RECIPE_TAG}.pt"
    OUT_DIR.mkdir(exist_ok=True)
    tag = f"{pattern}_{rule}_{datetime.now():%Y%m%d_%H%M%S}"
    log_path = OUT_DIR / f"{tag}.log"
    log(f"启动 {pattern} / {rule} → {log_path}")
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.Popen(
            [sys.executable, "-u", RUNNERS[rule]],
            cwd=str(ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        try:
            proc.wait(timeout=RUN_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            log(f"{pattern}/{rule} 超时（{RUN_TIMEOUT}s），已终止")
    elapsed = (time.time() - t0) / 60
    rounds = parse_rounds(log_path)
    rfinal = rounds.get(ROUNDS)
    log(f"完成 {pattern}/{rule}：{elapsed:.1f} min，rounds={rounds}，R{ROUNDS}={rfinal}")
    return {"rule": rule, "R": rfinal, "rounds": rounds, "minutes": round(elapsed, 1),
            "log": str(log_path), "returncode": proc.returncode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", nargs="+", default=["three_class", "four_class"])
    args = ap.parse_args()

    summary = {"started": datetime.now().isoformat(timespec="seconds"), "results": {}}
    passed = None

    for pattern in args.patterns:
        log(f"===== pattern: {pattern} =====")
        rec = {}
        summary["results"][pattern] = rec

        rec["DPFLA"] = run_one(pattern, "DPFLA")
        dpfla = rec["DPFLA"]["R"]
        if dpfla is None:
            rec["verdict"] = "ERROR_DPFLA"
            log(f"{pattern}: DPFLA 无 R{ROUNDS}（异常），队列停止")
            break
        if dpfla < THRESH_ABS:
            rec["verdict"] = f"FAIL_DPFLA_LOW({dpfla})"
            log(f"{pattern}: DPFLA R{ROUNDS}={dpfla} < {THRESH_ABS} → 不通过，换下一个类数")
            continue

        rec["TMean"] = run_one(pattern, "TMean")
        tmean = rec["TMean"]["R"]
        if tmean is None:
            rec["verdict"] = "ERROR_TMEAN"
            log(f"{pattern}: TMean 无 R{ROUNDS}（异常），队列停止")
            break
        gap_t = dpfla - tmean
        rec["gap_vs_tmean"] = round(gap_t, 2)
        if gap_t < THRESH_LEAD:
            rec["verdict"] = f"FAIL_VS_TMEAN({dpfla} vs {tmean})"
            log(f"{pattern}: DPFLA {dpfla} − TMean {tmean} = {gap_t:+.2f} < +1 → 跳过 MKrum，换下一个类数")
            continue

        rec["MKrum"] = run_one(pattern, "MKrum")
        mkrum = rec["MKrum"]["R"]
        if mkrum is None:
            rec["verdict"] = "ERROR_MKRUM"
            log(f"{pattern}: MKrum 无 R{ROUNDS}（异常），队列停止")
            break
        gap_k = dpfla - mkrum
        rec["gap_vs_mkrum"] = round(gap_k, 2)
        if gap_k >= THRESH_LEAD:
            rec["verdict"] = "PASS"
            passed = pattern
            log(f"{pattern}: DPFLA {dpfla} − MKrum {mkrum} = {gap_k:+.2f} ≥ +1 → **PASS**")
            break
        rec["verdict"] = f"FAIL_VS_MKRUM({dpfla} vs {mkrum})"
        log(f"{pattern}: DPFLA {dpfla} − MKrum {mkrum} = {gap_k:+.2f} < +1 → 不通过")

    summary["passed"] = passed
    summary["finished"] = datetime.now().isoformat(timespec="seconds")
    (OUT_DIR / "label_flip_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    lines = ["# 标签翻转筛选汇总", ""]
    for p, rec in summary["results"].items():
        lines.append(f"- **{p}**: {rec.get('verdict')}")
        for rule in ("DPFLA", "TMean", "MKrum"):
            if rule in rec:
                lines.append(f"  - {rule} R{ROUNDS}={rec[rule]['R']}（{rec[rule]['minutes']} min）")
    lines.append("")
    lines.append(f"结论：{'通过 ' + passed if passed else '无 pattern 通过'}")
    (OUT_DIR / "label_flip_summary.md").write_text("\n".join(lines) + "\n")
    log(f"队列结束：passed={passed}")


if __name__ == "__main__":
    main()
