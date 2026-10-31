#!/usr/bin/env python3
"""VisDrone 50 轮队列结束后的自动续跑（不设自动停止）。

流程
----
1. 等 `queue/DONE` 且无训练进程（当前 full50 队列自然结束/截止）。
2. 补跑 VisDrone 中尚未完成的臂（含被 supervisor 跳过的 IPM TMean/MKrum/FedAvg）。
3. CIFAR 多映射攻击强度筛选：M2(20%) / M3(30%) / M5(100%) × DPFLA/TMean/MKrum，
   10 轮 × 24 step；选 `DPFLA − max(TMean, MKrum)` 最大的映射。
4. 用选中映射重跑 CIFAR 标签线 4 臂（FedAvg/DPFLA/MKrum/TMean，50 轮 ×24 step）。
5. 再重跑 CIFAR MinMax 4 臂（50 轮 ×24 step）。
6. 每条成功后归档同类型旧 TB、写汇总、commit + push dev。
   **不自动停止**：一直顺序执行；主机续费/重启后运行 start_after_vd.sh 可从 state 断点续跑。

约束
----
- 复用 `scripts/full50_supervisor.py` 的 job_env / run_job / TB 归档逻辑；
- 本文件的运行目录/状态放在 `queue_after_vd/`，不干扰正在跑的 `queue/`；
- 屏幕筛选（10 轮）的日志与 TB 在选完映射后删除；正式 50 轮结果保留。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONT = ROOT / "queue_after_vd"
CONT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))
import full50_supervisor as base  # noqa: E402

# 把 base 的路径切到 queue_after_vd，避免与正在运行的 queue/ 冲突
base.QUEUE = CONT
base.STATE_PATH = CONT / "state.json"
base.STATUS_PATH = CONT / "status.json"
base.RESULTS_PATH = CONT / "results.csv"
base.SUMMARY_PATH = CONT / "auto_results.md"
base.SUP_LOG = CONT / "supervisor.log"
base.CONSOLE_DIR = CONT / "console"
base.PID_PATH = CONT / "pid"
base.DONE_PATH = CONT / "DONE"
base.ARCHIVE_DIR = Path("/root/tb_old_after_vd")

STATE_PATH = CONT / "state.json"
DONE_PATH = CONT / "DONE"
SUMMARY_PATH = CONT / "auto_results.md"
SCREEN_PATH = CONT / "screen_results.md"
SUP_LOG = CONT / "supervisor.log"
PID_PATH = CONT / "pid"
ARCHIVE_DIR = Path("/root/tb_old_after_vd")

MAPPINGS = [
    ("M2", '{"5":3,"3":5}'),
    ("M3", '{"5":3,"3":5,"4":6}'),
    ("M5", '{"0":1,"1":2,"2":3,"3":4,"4":5,"5":6,"6":7,"7":8,"8":9,"9":0}'),
]
SCREEN_METHODS = [
    ("DPFLA", "run-test/cifar10/run_dpfla_label_flipping.py", "run_dpfla_label_flipping"),
    ("mkrum", "run-test/cifar10/run_mkrum_label_flipping.py", "run_mkrum_label_flipping"),
    ("tmean", "run-test/cifar10/run_tmean_label_flipping.py", "run_tmean_label_flipping"),
]
LABEL_ARMS = [
    ("fedavg", "run-test/cifar10/run_no_defense_label_flipping.py", "run_no_defense_label_flipping"),
    ("DPFLA", "run-test/cifar10/run_dpfla_label_flipping.py", "run_dpfla_label_flipping"),
    ("mkrum", "run-test/cifar10/run_mkrum_label_flipping.py", "run_mkrum_label_flipping"),
    ("tmean", "run-test/cifar10/run_tmean_label_flipping.py", "run_tmean_label_flipping"),
]
MINMAX_ARMS = ["fedavg", "DPFLA", "mkrum", "tmean"]


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(SUP_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    return base.load_json(STATE_PATH) or {"completed": [], "failed": [], "history": []}


def save_state(state: dict) -> None:
    base.atomic_write_json(STATE_PATH, state)


def training_running() -> bool:
    for p in Path("/proc").iterdir():
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except Exception:
            continue
        if "python" in cmd and "run-test/" in cmd:
            return True
    return False


def wait_for_full50() -> None:
    log("等待当前 full50 队列结束（queue/DONE 且无训练进程）...")
    while True:
        if (ROOT / "queue" / "DONE").exists() and not training_running():
            log("full50 队列已结束，开始续跑。")
            return
        time.sleep(60)


def resume_current(state: dict) -> None:
    cur = state.get("current")
    if not cur:
        return
    log_path = Path(cur.get("log", ""))
    total = int(cur.get("total", 0))
    parsed = base._parse_metrics(log_path, total) if log_path.exists() else {}
    if parsed.get("last_round") == total:
        log(f"恢复：{cur['id']} 已跑满 {total} 轮，记为 success")
        state["completed"].append(cur["id"])
    else:
        log(f"恢复：{cur['id']} 未跑满，记为 failed（可后续重跑）")
        state["failed"].append(cur["id"])
    state.pop("current", None)
    save_state(state)


def git_commit_push(msg: str) -> None:
    paths = ["logs_3", "tb", "queue_after_vd/auto_results.md"]
    if SCREEN_PATH.exists():
        paths.append("queue_after_vd/screen_results.md")
    try:
        subprocess.run(["git", "add", "-A", *paths], cwd=str(ROOT), timeout=180,
                       capture_output=True, text=True, check=False)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(ROOT), timeout=30)
        if diff.returncode == 0:
            return
        c = subprocess.run(["git", "commit", "-m", msg], cwd=str(ROOT), timeout=180,
                           capture_output=True, text=True)
        if c.returncode != 0:
            log(f"⚠️ git commit 失败: {c.stderr.strip()[-400:]}")
            return
        p = subprocess.run(["git", "push", "origin", "dev"], cwd=str(ROOT), timeout=300,
                           capture_output=True, text=True)
        if p.returncode == 0:
            log(f"✓ 已推送 dev: {msg}")
        else:
            log(f"⚠️ git push 失败（本地已提交）: {p.stderr.strip()[-400:]}")
    except Exception as e:
        log(f"⚠️ git 操作异常: {e}")


def write_summary(rows: list[dict]) -> None:
    lines = [
        "# VisDrone 结束后续跑 + CIFAR 多映射重跑",
        "",
        f"- 更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- 说明：主口径末 5 轮；标签线同时看 global 与 attacked_class。",
        "",
        "| # | 实验 | 数据集 | 攻击 | 方法 | 状态 | best | last | 末5 | 末10 | 耗时(min) | 日志 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        log_name = Path(str(r.get("log", ""))).name if r.get("log") else ""
        lines.append(
            f"| {i} | {r.get('id','')} | {r.get('dataset','')} | {r.get('attack','')} | "
            f"{r.get('rule','')} | {r.get('status','')} | {r.get('best','')} | {r.get('last','')} | "
            f"{r.get('last5','')} | {r.get('last10','')} | {r.get('duration_min','')} | {log_name} |"
        )
    lines.append("")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_final_job(job: dict, rounds: int, timeout: float, stall: float, state: dict) -> dict:
    result = base.run_job(job, rounds, "aftervd", state, timeout, stall)
    result["moved_old_tb"] = base.overwrite_old_tb(job, result) if result["status"] == "success" else []
    state["history"].append(result)
    if result["status"] == "success":
        state["completed"].append(job["id"])
    else:
        state["failed"].append(job["id"])
    save_state(state)
    rows = state["history"]
    write_summary(rows)
    log(f"{job['id']}: {result['status']} last={result.get('last')} last5={result.get('last5')}")
    return result


def screen_jobs() -> list[dict]:
    jobs = []
    for mname, mjson in MAPPINGS:
        for rule, script, tag in SCREEN_METHODS:
            jobs.append(dict(
                id=f"screen_{mname}_{rule}", dataset="cifar10", attack="label_flipping", rule=rule,
                script=script, tag=tag, extra={"FL_FLIP_MAPPING": mjson, "FL_RULE": rule},
                mapping=mname,
            ))
    return jobs


def label_jobs(mapping_json: str) -> list[dict]:
    jobs = []
    for rule, script, tag in LABEL_ARMS:
        jobs.append(dict(
            id=f"relabel_{rule}", dataset="cifar10", attack="label_flipping", rule=rule,
            script=script, tag=tag, extra={"FL_FLIP_MAPPING": mapping_json, "FL_RULE": rule},
        ))
    return jobs


def minmax_jobs() -> list[dict]:
    jobs = []
    for rule in MINMAX_ARMS:
        jobs.append(dict(
            id=f"cifar10_minmax_{rule}_rerun", dataset="cifar10", attack="minmax", rule=rule,
            script="run-test/cifar10/run_screen_server_attack.py",
            tag=f"run_screen_minmax_{rule}",
            extra={"FL_SERVER_ATTACK": "minmax", "FL_RULE": rule},
        ))
    return jobs


def visdrone_remaining(full_state: dict) -> list[dict]:
    done = set(full_state.get("completed", []))
    jobs = [j for j in base.JOBS if j["dataset"] == "visdrone" and j["id"] not in done]
    return jobs


def select_mapping(screen: dict) -> tuple[str, str, float]:
    best = None
    for mname, mjson in MAPPINGS:
        dp = screen.get((mname, "DPFLA"))
        tm = screen.get((mname, "tmean"))
        mk = screen.get((mname, "mkrum"))
        if not (dp and tm and mk):
            continue
        margin = float(dp.get("last5") or dp.get("last") or 0) - max(
            float(tm.get("last5") or tm.get("last") or 0),
            float(mk.get("last5") or mk.get("last") or 0),
        )
        log(f"筛选 {mname}: DPFLA={dp.get('last5')} TMean={tm.get('last5')} MKrum={mk.get('last5')} margin={margin:.2f}")
        if best is None or margin > best[2]:
            best = (mname, mjson, margin)
    if best is None:
        log("⚠️ 没有完整筛选结果，默认 M3")
        return "M3", dict(MAPPINGS)["M3"], 0.0
    return best


def main() -> int:
    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")
    wait_for_full50()
    state = load_state()
    state.setdefault("completed", [])
    state.setdefault("failed", [])
    state.setdefault("history", [])
    save_state(state)
    resume_current(state)

    full_state = base.load_json(ROOT / "queue" / "state.json") or {}
    rows = state["history"]
    write_summary(rows)

    # 1) VisDrone 剩余臂
    for job in visdrone_remaining(full_state):
        if job["id"] in state["completed"]:
            continue
        log(f"补跑 VisDrone: {job['id']} | 50 轮 × 12 step")
        result = run_final_job(job, 50, 360, 20, state)
        git_commit_push(f"exp(aftervd): {job['id']} {result['status']} last={result.get('last')}")

    # 2) CIFAR 映射筛选
    screen: dict[tuple[str, str], dict] = {}
    screen_rows = []
    for job in screen_jobs():
        log(f"筛选: {job['id']} | 10 轮 × 24 step")
        result = base.run_job(job, 10, "screen", state, 30, 10)
        if result["status"] == "success":
            screen[(job["mapping"], job["rule"])] = result
        screen_rows.append(result)
        base.cleanup_test_artifacts(job, result)
    mname, mjson, margin = select_mapping(screen)
    SCREEN_PATH.write_text(
        f"# CIFAR 多映射筛选（10 轮 × 24 step）\n\n选中：**{mname}** margin={margin:.2f}\n\n" +
        "\n".join([
            f"| {r['id']} | {r['status']} | last={r.get('last','')} last5={r.get('last5','')} |"
            for r in screen_rows
        ]) + "\n",
        encoding="utf-8",
    )
    log(f"选中映射 {mname}: {mjson}（margin={margin:.2f}）")

    # 3) CIFAR 标签线 4 臂（强攻击映射）
    for job in label_jobs(mjson):
        if job["id"] in state["completed"]:
            continue
        log(f"重跑 CIFAR 标签线: {job['id']} | 映射 {mname}")
        result = run_final_job(job, 50, 60, 10, state)
        git_commit_push(f"exp(aftervd): {job['id']} {mname} {result['status']} last={result.get('last')}")

    # 4) CIFAR MinMax 4 臂（不吃映射，刷新 50 轮）
    for job in minmax_jobs():
        if job["id"] in state["completed"]:
            continue
        log(f"重跑 CIFAR MinMax: {job['id']}")
        result = run_final_job(job, 50, 60, 10, state)
        git_commit_push(f"exp(aftervd): {job['id']} {result['status']} last={result.get('last')}")

    write_summary(state["history"])
    DONE_PATH.write_text("finished", encoding="utf-8")
    log("续跑队列完成（全部 job 已执行/记录）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
