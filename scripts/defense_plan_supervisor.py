#!/usr/bin/env python3
"""Defense plan supervisor（2026-11-25 用户定稿）。

流程（严格串行，单 GPU）：
1. DPFLA + MinMax：40 轮 ×1 epoch、LR8e-4，mAP 首次 >=35 停；
   若 40 轮未达 35，自动续到 50 轮。
2. NormClip + MinMax：同口径，跑到 DPFLA 停止的同一轮数，无提前停止；
   比较峰值，DPFLA 需领先 >=1；不达标 -> 直接转 FLDetector（跳过 NormClip 标签翻转）。
3. NormClip + two_class 标签翻转：41 轮，无提前停止；与 DPFLA two_class 峰值比较，
   领先 >=1 则选 NormClip；否则转 FLDetector。
4. FLDetector 兜底：先 two_class 标签翻转 41 轮，领先 >=1 后再跑 MinMax 到 DPFLA 轮数；
   两条都领先 >=1 则选 FLDetector，否则失败。

断点续跑：
- 每个实验使用独立 FL_CKPT_PATH；
- 启动时若 checkpoint 已存在则 FL_RESUME=1，否则 0；
- supervisor 崩溃/主机重启后重新运行本脚本，会按 checkpoint 轮数自动续跑；
- 所有实验固定 FL_LOCAL_EPOCHS=1、FL_LOCAL_STEPS=0（防止 50 轮时被脚本改成 12 step）。
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs_screen"
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

STATE_PATH = LOG_DIR / "defense_plan_state.json"
SUMMARY_PATH = LOG_DIR / "defense_plan_summary.md"
LOCK_PATH = LOG_DIR / "defense_plan.lock"

PYTHON = sys.executable
LEAD = 1.0
TARGET = 35.0
LABEL_ROUNDS = 41
MM_ROUNDS = 40
MM_EXT = 50

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
    "FL_LOCAL_STEPS": "0",
    "FL_LOCAL_LR": "0.0008",
    "FL_FLIP_RATIO": "1.0",
    "FL_GRADIENT_SCALE": "1",
    "FL_ATTACHED": "1",
    "OMP_NUM_THREADS": "16",
    "MKL_NUM_THREADS": "16",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

RUNNERS = {
    "server_attack": "run-test/visdrone/run_screen_server_attack.py",
    "dpfla_label": "run-test/visdrone/run_dpfla_label_flipping.py",
    "normclip_label": "run-test/visdrone/run_normclip_label_flipping.py",
    "fldetector_label": "run-test/visdrone/run_fldetector_label_flipping.py",
}

CKPTS = {
    "dpfla_minmax": CKPT_DIR / "minmax_DPFLA_r20x1.pt",
    "normclip_minmax": CKPT_DIR / "minmax_normclip_r20x1.pt",
    "normclip_label": CKPT_DIR / "label_flip_two_class_normclip_r20x1.pt",
    "normclip_full": CKPT_DIR / "label_flip_full_cycle_normclip_r20x1.pt",
    "dpfla_full_cycle": CKPT_DIR / "label_flip_full_cycle_DPFLA_r20x1.pt",
    "fldetector_label": CKPT_DIR / "label_flip_two_class_fldetector_r20x1.pt",
    "fldetector_minmax": CKPT_DIR / "minmax_fldetector_r20x1.pt",
    "dpfla_label_ref": CKPT_DIR / "label_flip_two_class_DPFLA_r20x1.pt",
}


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception as e:
            log(f"读取 state 失败，重新开始：{e}")
    return {"status": "running", "selected": None, "phases": {}, "note": ""}


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str))
    tmp.replace(STATE_PATH)


def read_ckpt(path: Path):
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        log(f"读取 checkpoint 失败 {path}: {e}")
        return None


def curve(ck) -> list:
    if not ck:
        return []
    return [float(x) for x in (ck.get("global_accuracies") or [])]


def first_reach(acc: list, target: float):
    for i, v in enumerate(acc):
        if v >= target:
            return i + 1, v
    return None


def peak_of(acc: list, rounds=None):
    vals = acc[:rounds] if rounds else acc
    return max(vals) if vals else None


def run_arm(name: str, script: str, ckpt: Path, rounds: int, stop_map: float, extra_env: dict):
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(extra_env or {})
    env["FL_GLOBAL_ROUNDS"] = str(int(rounds))
    env["FL_STOP_MAP"] = str(int(stop_map)) if stop_map else "0"
    env["FL_CKPT_PATH"] = str(ckpt)
    env["FL_RESUME"] = "1" if ckpt.exists() else "0"
    env["FL_ATTACHED"] = "1"

    log_path = LOG_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    log(
        f"启动 {name}: rounds={rounds}, stop={stop_map}, "
        f"ckpt={ckpt.name}, resume={env['FL_RESUME']}, log={log_path.name}"
    )
    with open(log_path, "w") as f:
        proc = subprocess.Popen(
            [PYTHON, script],
            cwd=str(ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        ret = proc.wait()
    log(f"结束 {name}: exit={ret}")
    return log_path, ret


def ensure_rounds(name: str, script: str, ckpt: Path, rounds: int, extra_env: dict, attempts: int = 2):
    """跑满指定轮数；若 checkpoint 已足够则跳过；中断后自动续跑。"""
    for attempt in range(1, attempts + 1):
        ck = read_ckpt(ckpt)
        if ck and int(ck.get("epoch", -1)) + 1 >= rounds:
            log(f"{name} 已有 {int(ck.get('epoch', -1)) + 1}/{rounds} 轮，跳过")
            return ck
        run_arm(name, script, ckpt, rounds, 0, extra_env)
    ck = read_ckpt(ckpt)
    got = int(ck.get("epoch", -1)) + 1 if ck else 0
    if got < rounds:
        log(f"警告：{name} 只到 {got}/{rounds} 轮")
    return ck


def ensure_dpfla_minmax() -> dict:
    """DPFLA + MinMax：到首次 >=35 停；40 轮未达则续到 50。"""
    ckpt = CKPTS["dpfla_minmax"]
    for cap in (MM_ROUNDS, MM_EXT):
        ck = read_ckpt(ckpt)
        acc = curve(ck)
        hit = first_reach(acc, TARGET)
        if hit:
            res = {
                "rounds": int(hit[0]),
                "peak": float(max(acc[: hit[0]])),
                "done": True,
                "curve_len": len(acc),
            }
            log(f"DPFLA MinMax 达标：R{hit[0]}={hit[1]:.2f}%，peak={res['peak']:.2f}%")
            return res

        got = int(ck.get("epoch", -1)) + 1 if ck else 0
        if got >= cap:
            log(f"DPFLA MinMax 已跑满 {cap} 轮仍未达 35，best={peak_of(acc)}")
            continue

        for _ in range(2):
            run_arm(
                "dpfla_minmax",
                RUNNERS["server_attack"],
                ckpt,
                cap,
                TARGET,
                {"FL_SERVER_ATTACK": "minmax", "FL_RULE": "DPFLA"},
            )
            ck = read_ckpt(ckpt)
            acc = curve(ck)
            hit = first_reach(acc, TARGET)
            if hit:
                res = {
                    "rounds": int(hit[0]),
                    "peak": float(max(acc[: hit[0]])),
                    "done": True,
                    "curve_len": len(acc),
                }
                log(f"DPFLA MinMax 达标：R{hit[0]}={hit[1]:.2f}%，peak={res['peak']:.2f}%")
                return res
            if ck and int(ck.get("epoch", -1)) + 1 >= cap:
                break

    ck = read_ckpt(ckpt)
    acc = curve(ck)
    return {
        "rounds": len(acc),
        "peak": peak_of(acc),
        "done": False,
        "curve_len": len(acc),
    }


def dpfla_label_peak() -> float:
    ck = read_ckpt(CKPTS["dpfla_label_ref"])
    if not ck:
        raise RuntimeError(f"缺少 DPFLA two_class 断点：{CKPTS['dpfla_label_ref']}")
    acc = curve(ck)
    if len(acc) < LABEL_ROUNDS:
        raise RuntimeError(
            f"DPFLA two_class 断点只有 {len(acc)} 轮，少于 {LABEL_ROUNDS} 轮"
        )
    return float(max(acc[:LABEL_ROUNDS]))


def compare(phase: str, dpfla_peak: float, def_peak, rounds: int):
    if def_peak is None:
        log(f"{phase}: 防御方无有效峰值，判失败")
        return None, False
    lead = round(float(dpfla_peak) - float(def_peak), 4)
    ok = lead >= LEAD
    log(
        f"{phase}: DPFLA peak={dpfla_peak:.2f}% vs 防御 peak={def_peak:.2f}% "
        f"@R{rounds} -> lead={lead:+.2f} -> {'PASS' if ok else 'FAIL'}"
    )
    return lead, ok


def write_summary(state: dict) -> None:
    rows = []
    for name, r in (state.get("phases") or {}).items():
        rows.append(
            f"| {name} | {r.get('rounds')} | {r.get('peak')} | "
            f"{r.get('lead')} | {r.get('pass')} |"
        )
    text = [
        "# Defense plan summary（2026-11-25）",
        "",
        f"- status: **{state.get('status')}**",
        f"- selected: **{state.get('selected')}**",
        f"- note: {state.get('note', '')}",
        "",
        "| phase | rounds | peak | lead vs DPFLA | pass |",
        "|---|---:|---:|---:|---|",
        *rows,
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(text))
    log(f"summary 已写入 {SUMMARY_PATH}")


def run_fld_fallback(state: dict, dpfla: dict, dpfla_rounds: int, reason: str):
    log(f"转入 FLDetector 兜底：{reason}")
    state["note"] = reason
    save_state(state)

    # 先标签翻转（two_class 41 轮）
    if not state["phases"].get("fldetector_label"):
        log("=== Phase 4a: FLDetector + two_class label flip ===")
        ck = ensure_rounds(
            "fldetector_label",
            RUNNERS["fldetector_label"],
            CKPTS["fldetector_label"],
            LABEL_ROUNDS,
            {"FL_FLIP_PATTERN": "two_class"},
        )
        acc = curve(ck)
        p = peak_of(acc, LABEL_ROUNDS)
        lead, ok = compare("fldetector_label", dpfla_label_peak(), p, LABEL_ROUNDS)
        state["phases"]["fldetector_label"] = {
            "rounds": LABEL_ROUNDS,
            "peak": p,
            "lead": lead,
            "pass": ok,
            "curve_len": len(acc),
        }
        save_state(state)

    if not state["phases"]["fldetector_label"].get("pass"):
        state["status"] = "failed"
        state["note"] = "FLDetector two_class 未达 DPFLA 领先 >=1"
        save_state(state)
        write_summary(state)
        return

    # 再 MinMax，跑到 DPFLA 的同一轮数
    if not state["phases"].get("fldetector_minmax"):
        log("=== Phase 4b: FLDetector + MinMax ===")
        ck = ensure_rounds(
            "fldetector_minmax",
            RUNNERS["server_attack"],
            CKPTS["fldetector_minmax"],
            dpfla_rounds,
            {"FL_SERVER_ATTACK": "minmax", "FL_RULE": "fldetector"},
        )
        acc = curve(ck)
        p = peak_of(acc, dpfla_rounds)
        lead, ok = compare(
            "fldetector_minmax", float(dpfla["peak"]), p, dpfla_rounds
        )
        state["phases"]["fldetector_minmax"] = {
            "rounds": dpfla_rounds,
            "peak": p,
            "lead": lead,
            "pass": ok,
            "curve_len": len(acc),
        }
        save_state(state)

    if state["phases"]["fldetector_minmax"].get("pass"):
        state["status"] = "done"
        state["selected"] = "fldetector"
        state["note"] = "FLDetector 两条线均满足 DPFLA 领先 >=1"
    else:
        state["status"] = "failed"
        state["note"] = "FLDetector MinMax 未达 DPFLA 领先 >=1"
    save_state(state)
    write_summary(state)


def run_post50_extension(state: dict, dpfla: dict, dpfla_rounds: int) -> None:
    """用户 2026-11-25 追加：MinMax 两臂补到 50 轮；达标则补 DPFLA full_cycle 50。"""
    log("=== Phase 5: MinMax 50 轮复核 ===")

    if not state["phases"].get("dpfla_minmax_50"):
        log("=== Phase 5a: DPFLA + MinMax 续到 50 轮 ===")
        ck = ensure_rounds(
            "dpfla_minmax_50",
            RUNNERS["server_attack"],
            CKPTS["dpfla_minmax"],
            50,
            {"FL_SERVER_ATTACK": "minmax", "FL_RULE": "DPFLA"},
        )
        acc = curve(ck)
        p = peak_of(acc, 50)
        state["phases"]["dpfla_minmax_50"] = {
            "rounds": 50,
            "peak": p,
            "curve_len": len(acc),
        }
        save_state(state)

    if not state["phases"].get("normclip_minmax_50"):
        log("=== Phase 5b: NormClip + MinMax 续到 50 轮 ===")
        ck = ensure_rounds(
            "normclip_minmax_50",
            RUNNERS["server_attack"],
            CKPTS["normclip_minmax"],
            50,
            {"FL_SERVER_ATTACK": "minmax", "FL_RULE": "normclip"},
        )
        acc = curve(ck)
        p = peak_of(acc, 50)
        state["phases"]["normclip_minmax_50"] = {
            "rounds": 50,
            "peak": p,
            "curve_len": len(acc),
        }
        save_state(state)

    dp = state["phases"]["dpfla_minmax_50"].get("peak")
    nc = state["phases"]["normclip_minmax_50"].get("peak")
    if dp is None or nc is None:
        state["status"] = "failed"
        state["note"] = "MinMax 50 轮峰值缺失"
        save_state(state)
        write_summary(state)
        return
    lead, ok = compare("minmax_50", float(dp), nc, 50)
    state["phases"]["minmax_50_compare"] = {
        "dpfla": dp,
        "normclip": nc,
        "lead": lead,
        "pass": ok,
    }
    save_state(state)

    if not ok:
        state["status"] = "failed"
        state["note"] = "MinMax 50 轮峰值未达 DPFLA 领先 >=1，停止"
        save_state(state)
        write_summary(state)
        return

    # 用户 2026-11-25 追加：NormClip full_cycle 也补到 50 轮，保证同轮峰值比较
    if not state["phases"].get("normclip_full_50"):
        log("=== Phase 5c: NormClip + full_cycle 续到 50 轮 ===")
        ck = ensure_rounds(
            "normclip_full_50",
            RUNNERS["normclip_label"],
            CKPTS["normclip_full"],
            50,
            {"FL_FLIP_PATTERN": "full_cycle"},
        )
        acc = curve(ck)
        p = peak_of(acc, 50)
        state["phases"]["normclip_full_50"] = {
            "rounds": 50,
            "peak": p,
            "curve_len": len(acc),
        }
        save_state(state)

    if not state["phases"].get("dpfla_full_cycle_50"):
        log("=== Phase 5d: DPFLA + full_cycle 50 轮 ===")
        ck = ensure_rounds(
            "dpfla_full_cycle_50",
            RUNNERS["dpfla_label"],
            CKPTS["dpfla_full_cycle"],
            50,
            {"FL_FLIP_PATTERN": "full_cycle"},
        )
        acc = curve(ck)
        p = peak_of(acc, 50)
        state["phases"]["dpfla_full_cycle_50"] = {
            "rounds": 50,
            "peak": p,
            "curve_len": len(acc),
        }
        save_state(state)

    nc_full = (state["phases"].get("normclip_full_50") or {}).get("peak")
    dp_full = state["phases"]["dpfla_full_cycle_50"].get("peak")
    if dp_full is None or nc_full is None:
        state["status"] = "failed"
        state["note"] = "full_cycle 50 峰值缺失"
        save_state(state)
        write_summary(state)
        return
    lead2, ok2 = compare("full_cycle_50", float(dp_full), nc_full, 50)
    state["phases"]["full_cycle_50_compare"] = {
        "dpfla": dp_full,
        "normclip": nc_full,
        "lead": lead2,
        "pass": ok2,
    }
    save_state(state)

    if ok2:
        state["status"] = "done"
        state["selected"] = "normclip"
        state["note"] = "MinMax 50 与 full_cycle 50 均满足 DPFLA 领先 >=1"
    else:
        state["status"] = "failed"
        state["selected"] = None
        state["note"] = "full_cycle 50 未达 DPFLA 领先 >=1"
    save_state(state)
    write_summary(state)


def main() -> None:
    state = load_state()
    if state.get("status") == "done":
        log(f"计划已完成，selected={state.get('selected')}；如需重跑请删除 state/checkpoint")
        return

    # Phase 1: DPFLA MinMax
    if not state["phases"].get("dpfla_minmax", {}).get("done"):
        log("=== Phase 1: DPFLA + MinMax（40 轮，≥35 停；未达则续 50）===")
        state["phases"]["dpfla_minmax"] = ensure_dpfla_minmax()
        save_state(state)

    dpfla = state["phases"]["dpfla_minmax"]
    if not dpfla.get("done"):
        state["status"] = "failed"
        state["note"] = "DPFLA MinMax 在 50 轮内未达到 35"
        save_state(state)
        write_summary(state)
        return

    dpfla_rounds = int(dpfla["rounds"])
    log(f"DPFLA MinMax 基准：R{dpfla_rounds}，peak={dpfla['peak']:.2f}%")

    # Phase 2: NormClip MinMax，跑满 DPFLA 的轮数
    if not state["phases"].get("normclip_minmax"):
        log(f"=== Phase 2: NormClip + MinMax（{dpfla_rounds} 轮，无提前停止）===")
        ck = ensure_rounds(
            "normclip_minmax",
            RUNNERS["server_attack"],
            CKPTS["normclip_minmax"],
            dpfla_rounds,
            {"FL_SERVER_ATTACK": "minmax", "FL_RULE": "normclip"},
        )
        acc = curve(ck)
        p = peak_of(acc, dpfla_rounds)
        lead, ok = compare("normclip_minmax", float(dpfla["peak"]), p, dpfla_rounds)
        state["phases"]["normclip_minmax"] = {
            "rounds": dpfla_rounds,
            "peak": p,
            "lead": lead,
            "pass": ok,
            "curve_len": len(acc),
        }
        save_state(state)

    nc_mm = state["phases"]["normclip_minmax"]
    if not nc_mm.get("pass"):
        run_fld_fallback(state, dpfla, dpfla_rounds, "NormClip MinMax 未达 DPFLA 领先 >=1")
        return

    # Phase 3: NormClip two_class label flip
    if not state["phases"].get("normclip_label"):
        log(f"=== Phase 3: NormClip + two_class（{LABEL_ROUNDS} 轮，无提前停止）===")
        ck = ensure_rounds(
            "normclip_label",
            RUNNERS["normclip_label"],
            CKPTS["normclip_label"],
            LABEL_ROUNDS,
            {"FL_FLIP_PATTERN": "two_class"},
        )
        acc = curve(ck)
        p = peak_of(acc, LABEL_ROUNDS)
        lead, ok = compare("normclip_label", dpfla_label_peak(), p, LABEL_ROUNDS)
        state["phases"]["normclip_label"] = {
            "rounds": LABEL_ROUNDS,
            "peak": p,
            "lead": lead,
            "pass": ok,
            "curve_len": len(acc),
        }
        save_state(state)

    if state["phases"]["normclip_label"].get("pass"):
        state["status"] = "done"
        state["selected"] = "normclip"
        state["note"] = "NormClip 两条线均满足 DPFLA 领先 >=1"
        save_state(state)
        write_summary(state)
        return

    # Phase 3b: NormClip + full_cycle（two_class 不达标后再试全量翻转）
    if not state["phases"].get("normclip_full"):
        log(f"=== Phase 3b: NormClip + full_cycle（{LABEL_ROUNDS} 轮，无提前停止）===")
        ck = ensure_rounds(
            "normclip_full",
            RUNNERS["normclip_label"],
            CKPTS["normclip_full"],
            LABEL_ROUNDS,
            {"FL_FLIP_PATTERN": "full_cycle"},
        )
        acc = curve(ck)
        p = peak_of(acc, LABEL_ROUNDS)
        lead, ok = compare("normclip_full", dpfla_label_peak(), p, LABEL_ROUNDS)
        state["phases"]["normclip_full"] = {
            "rounds": LABEL_ROUNDS,
            "peak": p,
            "lead": lead,
            "pass": ok,
            "curve_len": len(acc),
        }
        save_state(state)

    # 用户 2026-11-25 追加：不论 full_cycle 是否通过，先做 50 轮 MinMax 复核
    if state["phases"]["normclip_full"].get("pass"):
        log("NormClip full_cycle 已满足 DPFLA 领先 >=1；继续 50 轮复核")
    else:
        log("NormClip full_cycle 未达 DPFLA 领先 >=1；仍按新计划继续 50 轮复核")

    run_post50_extension(state, dpfla, dpfla_rounds)


def guard() -> None:
    if LOCK_PATH.exists():
        try:
            pid = int(LOCK_PATH.read_text().strip())
            os.kill(pid, 0)
            log(f"已有 supervisor 在运行（pid={pid}），退出")
            return
        except Exception:
            log("发现过期 lock，删除后继续")
            LOCK_PATH.unlink()
    LOCK_PATH.write_text(str(os.getpid()))
    try:
        main()
    finally:
        try:
            LOCK_PATH.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    guard()
