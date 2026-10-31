#!/usr/bin/env python3
"""50 轮全量实验自动队列（MEMORY/09_Full_Run_Plan.md 阶段 A）。

流程
----
1. 体检（各 5 轮）：CIFAR `label_flipping + DPFLA + FL_LOCAL_STEPS=24`；
   VisDrone `MinMax + DPFLA + LOCAL_EPOCHS=1`。判定"有增长、不崩、不 NaN"。
2. 体检通过后**删除体检日志与体检 TB**（不删除历史正式产物）。
3. 串行跑 18 条 50 轮实验（9 CIFAR → 9 VisDrone，按优先级）。
4. 每条成功后：把同类型旧 TB 移到 /root/tb_old_full50（TB 界面只留新 50 轮），
   写 queue/auto_results.md，并自动 commit + push dev（可用 FULL50_AUTO_PUSH=0 关闭）。
5. 23 小时截止：剩余时间不足以完成下一条时不再启动，写 queue/DONE。

状态 / 产物
-----------
- queue/status.json       实时状态（当前 job / 轮次 / 指标 / ETA / GPU）
- queue/results.csv       每条实验的结果
- queue/auto_results.md   可读汇总（入库）
- queue/test_summary.md   体检结论（入库）
- queue/supervisor.log    队列自身日志
- queue/console/          每条实验的控制台输出（不入库）
"""

from __future__ import annotations

import csv
import fcntl
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "queue"
QUEUE.mkdir(exist_ok=True)
STATE_PATH = QUEUE / "state.json"
STATUS_PATH = QUEUE / "status.json"
RESULTS_PATH = QUEUE / "results.csv"
SUMMARY_PATH = QUEUE / "auto_results.md"
TEST_SUMMARY_PATH = QUEUE / "test_summary.md"
SUP_LOG = QUEUE / "supervisor.log"
CONSOLE_DIR = QUEUE / "console"
ARCHIVE_DIR = Path("/root/tb_old_full50")
LOCK_PATH = QUEUE / "supervisor.lock"
PID_PATH = QUEUE / "supervisor.pid"
DONE_PATH = QUEUE / "DONE"

ROUND_RE = re.compile(r"Round (\d+)/(\d+) - .*?(?:准确率|mAP@0\.5):\s*([0-9]+(?:\.[0-9]+)?)%")
FAIL_PATTERNS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
    "测试失败",
    "RuntimeError",
    "Segmentation fault",
    "Killed",
)

LOCK_HANDLE = None


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(SUP_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def gpu_mem() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    except Exception:
        return ""


# ---------------------------------------------------------------- 实验清单

JOBS = [
    # ---------- CIFAR10（先跑，快） ----------
    dict(id="cifar10_baseline", dataset="cifar10", attack="no_attack", rule="fedavg",
         script="run-test/cifar10/run_no_attack_baseline.py", tag="run_no_attack_baseline",
         extra={}),
    dict(id="cifar10_label_fedavg", dataset="cifar10", attack="label_flipping", rule="fedavg",
         script="run-test/cifar10/run_no_defense_label_flipping.py", tag="run_no_defense_label_flipping",
         extra={}),
    dict(id="cifar10_label_dpfla", dataset="cifar10", attack="label_flipping", rule="DPFLA",
         script="run-test/cifar10/run_dpfla_label_flipping.py", tag="run_dpfla_label_flipping",
         extra={}),
    dict(id="cifar10_label_mkrum", dataset="cifar10", attack="label_flipping", rule="mkrum",
         script="run-test/cifar10/run_mkrum_label_flipping.py", tag="run_mkrum_label_flipping",
         extra={}),
    dict(id="cifar10_label_tmean", dataset="cifar10", attack="label_flipping", rule="tmean",
         script="run-test/cifar10/run_tmean_label_flipping.py", tag="run_tmean_label_flipping",
         extra={}),
    dict(id="cifar10_minmax_fedavg", dataset="cifar10", attack="minmax", rule="fedavg",
         script="run-test/cifar10/run_screen_server_attack.py", tag="run_screen_minmax_fedavg",
         extra={"FL_SERVER_ATTACK": "minmax", "FL_RULE": "fedavg"}),
    dict(id="cifar10_minmax_dpfla", dataset="cifar10", attack="minmax", rule="DPFLA",
         script="run-test/cifar10/run_screen_server_attack.py", tag="run_screen_minmax_DPFLA",
         extra={"FL_SERVER_ATTACK": "minmax", "FL_RULE": "DPFLA"}),
    dict(id="cifar10_minmax_mkrum", dataset="cifar10", attack="minmax", rule="mkrum",
         script="run-test/cifar10/run_screen_server_attack.py", tag="run_screen_minmax_mkrum",
         extra={"FL_SERVER_ATTACK": "minmax", "FL_RULE": "mkrum"}),
    dict(id="cifar10_minmax_tmean", dataset="cifar10", attack="minmax", rule="tmean",
         script="run-test/cifar10/run_screen_server_attack.py", tag="run_screen_minmax_tmean",
         extra={"FL_SERVER_ATTACK": "minmax", "FL_RULE": "tmean"}),

    # ---------- VisDrone（后跑，慢；先保证 MinMax 线完整） ----------
    dict(id="visdrone_baseline", dataset="visdrone", attack="no_attack", rule="fedavg",
         script="run-test/visdrone/run_no_attack_baseline.py", tag="run_no_attack_baseline",
         extra={}),
    dict(id="visdrone_minmax_dpfla", dataset="visdrone", attack="minmax", rule="DPFLA",
         script="run-test/visdrone/run_screen_server_attack.py", tag="run_screen_minmax_DPFLA",
         extra={"FL_SERVER_ATTACK": "minmax", "FL_RULE": "DPFLA"}),
    dict(id="visdrone_minmax_fedavg", dataset="visdrone", attack="minmax", rule="fedavg",
         script="run-test/visdrone/run_screen_server_attack.py", tag="run_screen_minmax_fedavg",
         extra={"FL_SERVER_ATTACK": "minmax", "FL_RULE": "fedavg"}),
    dict(id="visdrone_minmax_tmean", dataset="visdrone", attack="minmax", rule="tmean",
         script="run-test/visdrone/run_screen_server_attack.py", tag="run_screen_minmax_tmean",
         extra={"FL_SERVER_ATTACK": "minmax", "FL_RULE": "tmean"}),
    dict(id="visdrone_minmax_mkrum", dataset="visdrone", attack="minmax", rule="mkrum",
         script="run-test/visdrone/run_screen_server_attack.py", tag="run_screen_minmax_mkrum",
         extra={"FL_SERVER_ATTACK": "minmax", "FL_RULE": "mkrum"}),
    dict(id="visdrone_ipm_dpfla", dataset="visdrone", attack="ipm", rule="DPFLA",
         script="run-test/visdrone/run_screen_server_attack.py", tag="run_screen_ipm_DPFLA",
         extra={"FL_SERVER_ATTACK": "ipm", "FL_RULE": "DPFLA"}),
    dict(id="visdrone_ipm_fedavg", dataset="visdrone", attack="ipm", rule="fedavg",
         script="run-test/visdrone/run_screen_server_attack.py", tag="run_screen_ipm_fedavg",
         extra={"FL_SERVER_ATTACK": "ipm", "FL_RULE": "fedavg"}),
    dict(id="visdrone_ipm_tmean", dataset="visdrone", attack="ipm", rule="tmean",
         script="run-test/visdrone/run_screen_server_attack.py", tag="run_screen_ipm_tmean",
         extra={"FL_SERVER_ATTACK": "ipm", "FL_RULE": "tmean"}),
    dict(id="visdrone_ipm_mkrum", dataset="visdrone", attack="ipm", rule="mkrum",
         script="run-test/visdrone/run_screen_server_attack.py", tag="run_screen_ipm_mkrum",
         extra={"FL_SERVER_ATTACK": "ipm", "FL_RULE": "mkrum"}),
]

TEST_CIFAR = dict(id="test_cifar10_label_dpfla", dataset="cifar10", attack="label_flipping", rule="DPFLA",
                  script="run-test/cifar10/run_dpfla_label_flipping.py", tag="run_dpfla_label_flipping",
                  extra={})
TEST_VISDRONE = dict(id="test_visdrone_minmax_dpfla", dataset="visdrone", attack="minmax", rule="DPFLA",
                     script="run-test/visdrone/run_screen_server_attack.py", tag="run_screen_minmax_DPFLA",
                     extra={"FL_SERVER_ATTACK": "minmax", "FL_RULE": "DPFLA"})


def job_env(job: dict, rounds: int) -> dict:
    env = os.environ.copy()
    env.pop("FL_LOCAL_STEPS", None)
    env.pop("FL_SERVER_ATTACK", None)
    env.pop("FL_RULE", None)

    env.update({
        "FL_ATTACHED": "1",
        "FL_GLOBAL_ROUNDS": str(rounds),
        "FL_NUM_WORKERS": "10",
        "FL_FRAC_WORKERS": "1.0",
        "FL_MALICIOUS_RATE": "0.3",
        "FL_GRADIENT_SCALE": "1",
        "FL_MALICIOUS_BEHAVIOR_RATE": "1.0",
        "TZ": "Asia/Shanghai",
        "PYTHONUNBUFFERED": "1",
    })

    if job["dataset"] == "cifar10":
        env.update({
            "CIFAR_MODEL": "ResNet18CIFAR",
            "FL_LOCAL_EPOCHS": "1",
            "FL_TRAIN_BATCH_SIZE": "64",
            "FL_TEST_BATCH_SIZE": "1000",
            "FL_LOCAL_LR": "0.01",
            "FL_LOCAL_STEPS": "24",
        })
    else:
        env.update({
            "YOLO_MODEL_SIZE": "m",
            "FL_IMG_SIZE": "1280",
            "FL_AMP": "1",
            "FL_LOCAL_EPOCHS": "1",
            "FL_TRAIN_BATCH_SIZE": "16",
            "FL_TEST_BATCH_SIZE": os.environ.get("FULL50_VD_TEST_BATCH", "64"),
            "FL_LOCAL_LR": "0.0002",
            "FL_CPU_THREADS": "16",
            "FL_DATALOADER_WORKERS": "6",
        })
    env.update(job.get("extra") or {})
    return env


def tb_prefix(job: dict) -> str:
    mr = "0.3" if job["attack"] != "no_attack" else "0.0"
    if job["dataset"] == "cifar10":
        return f"CIFAR10_ResNet18CIFAR_{job['rule']}_attack-{job['attack']}_mr-{mr}_img640_"
    return f"VisDrone_YOLO_{job['rule']}_attack-{job['attack']}_mr-{mr}_yolov8m_img1280_"


def same_prefix_dirs(prefix: str):
    dirs = []
    for base in (ROOT / "tb", Path("/root/tf-logs")):
        if not base.exists():
            continue
        for p in base.glob(prefix + "*"):
            if p.is_dir():
                dirs.append(p)
    return dirs


# ---------------------------------------------------------------- 单条实验


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            return
    try:
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _find_log(dataset: str, tag: str, before: set, start_ts: float, proc: subprocess.Popen, timeout: float = 240):
    log_dir = ROOT / "logs_3" / dataset
    deadline = time.time() + timeout
    while time.time() < deadline:
        candidates = [
            p for p in log_dir.glob(f"{tag}_*.log")
            if p not in before and p.stat().st_mtime >= start_ts - 5
        ]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
        if proc.poll() is not None:
            return None
        time.sleep(2)
    return None


def _parse_metrics(log_path: Path, total: int):
    rounds = {}
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = ROUND_RE.search(line)
                if m:
                    rounds[int(m.group(1))] = float(m.group(3))
    except Exception:
        pass
    if not rounds:
        return {}
    ordered = [rounds[k] for k in sorted(rounds)]
    n = len(ordered)
    return {
        "rounds_seen": n,
        "last_round": max(rounds),
        "first": ordered[0],
        "last": ordered[-1],
        "best": max(ordered),
        "last5": round(sum(ordered[-5:]) / min(5, n), 2),
        "last10": round(sum(ordered[-10:]) / min(10, n), 2),
        "metric": ordered[-1],
    }


def _update_status(phase: str, job: dict | None, round_no: int, total: int, metric, elapsed_min: float, state: dict) -> None:
    eta = ""
    if round_no and total:
        eta = round(elapsed_min / round_no * (total - round_no), 1)
    atomic_write_json(STATUS_PATH, {
        "phase": phase,
        "job": job["id"] if job else "",
        "dataset": job["dataset"] if job else "",
        "attack": job["attack"] if job else "",
        "rule": job["rule"] if job else "",
        "round": round_no,
        "total": total,
        "metric": metric,
        "elapsed_min": round(elapsed_min, 1),
        "eta_min": eta,
        "gpu": gpu_mem(),
        "completed": state.get("completed", []),
        "failed": state.get("failed", []),
        "skipped": state.get("skipped", []),
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


def run_job(job: dict, rounds: int, phase: str, state: dict, timeout_min: float, stall_min: float) -> dict:
    prefix = tb_prefix(job)
    before_logs = set((ROOT / "logs_3" / job["dataset"]).glob("*.log"))
    before_tb = set(same_prefix_dirs(prefix))

    CONSOLE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    console_path = CONSOLE_DIR / f"{job['id']}_{stamp}.log"
    env = job_env(job, rounds)

    log(f"启动 [{phase}] {job['id']} | {job['dataset']} {job['attack']} + {job['rule']} | {rounds} 轮")
    started = time.time()
    started_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(console_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            [sys.executable, "-u", job["script"]],
            cwd=str(ROOT), env=env,
            stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    state["current"] = {
        "id": job["id"], "phase": phase, "started_at": started_dt,
        "log": "", "total": rounds, "prefix": prefix,
        "console": str(console_path), "dataset": job["dataset"],
    }
    atomic_write_json(STATE_PATH, state)

    log_path = _find_log(job["dataset"], job["tag"], before_logs, started, proc)
    if log_path is None:
        tail = ""
        try:
            tail = console_path.read_text(encoding="utf-8", errors="ignore")[-2500:]
        except Exception:
            pass
        _kill_group(proc)
        log(f"✗ {job['id']} 未生成日志；console 末尾:\n{tail}")
        return dict(status="failed", reason="no_log", log="", console=str(console_path),
                    duration_min=round((time.time() - started) / 60, 1), started=started_dt,
                    ended=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    state["current"]["log"] = str(log_path)
    atomic_write_json(STATE_PATH, state)
    log(f"日志: {log_path}")

    pos = 0
    metrics: dict[int, float] = {}
    fail_reason = ""
    completed = False
    complete_ts = None
    last_growth = time.time()
    last_size = 0
    last_status = 0.0
    deadline = started + timeout_min * 60

    while True:
        now = time.time()
        size = log_path.stat().st_size if log_path.exists() else 0
        if size > last_size:
            last_size = size
            last_growth = now
            try:
                with open(log_path, encoding="utf-8", errors="ignore") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
            except Exception:
                chunk = ""
            for line in chunk.splitlines():
                m = ROUND_RE.search(line)
                if m:
                    r = int(m.group(1))
                    metrics[r] = float(m.group(3))
                    if r >= rounds:
                        completed = True
                        complete_ts = complete_ts or now
                for pat in FAIL_PATTERNS:
                    if pat in line:
                        fail_reason = pat

        if completed and proc.poll() is not None and not fail_reason:
            break
        if proc.poll() is not None and not completed:
            if not fail_reason:
                fail_reason = f"进程提前退出(code={proc.returncode})"
            break
        if completed and complete_ts and now - complete_ts > 900:
            fail_reason = fail_reason or "完成后 15 分钟未退出，强杀"
            _kill_group(proc)
            break
        if now > deadline:
            fail_reason = f"超时 {timeout_min:.0f} 分钟"
            _kill_group(proc)
            break
        if now - last_growth > stall_min * 60 and proc.poll() is None:
            fail_reason = f"日志 {stall_min:.0f} 分钟无增长"
            _kill_group(proc)
            break

        if now - last_status >= 10:
            last_status = now
            r = max(metrics) if metrics else 0
            _update_status(phase, job, r, rounds, metrics.get(r) if r else None,
                           (now - started) / 60, state)
        time.sleep(5)

    if proc.poll() is None:
        _kill_group(proc)
    duration = round((time.time() - started) / 60, 1)
    ended = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 读全量日志做汇总
    parsed = _parse_metrics(log_path, rounds)
    status = "success" if (completed and not fail_reason) else "failed"
    if status == "failed" and not fail_reason:
        fail_reason = f"未跑到 {rounds}/{rounds}"

    result = {
        "id": job["id"], "dataset": job["dataset"], "attack": job["attack"], "rule": job["rule"],
        "status": status, "reason": fail_reason, "rounds": rounds,
        "first": parsed.get("first", ""),
        "best": parsed.get("best", ""), "last": parsed.get("last", ""),
        "last5": parsed.get("last5", ""), "last10": parsed.get("last10", ""),
        "rounds_seen": parsed.get("rounds_seen", 0),
        "duration_min": duration, "started": started_dt, "ended": ended,
        "log": str(log_path), "console": str(console_path),
        "tb_new": [str(p) for p in same_prefix_dirs(prefix) if p not in before_tb],
        "tb_before": [str(p) for p in before_tb],
    }
    log(f"结束 {job['id']}: {status} | last={result['last']} best={result['best']} "
        f"duration={duration} min | {fail_reason or 'OK'}")
    state.pop("current", None)
    atomic_write_json(STATE_PATH, state)
    return result


# ---------------------------------------------------------------- 收尾动作


def cleanup_test_artifacts(job: dict, result: dict) -> list[str]:
    """删除体检日志与体检 TB（仅本次体检新生成的）。"""
    removed = []
    lp = Path(result.get("log") or "")
    if lp.exists():
        lp.unlink()
        removed.append(str(lp))
    for p in result.get("tb_new", []):
        p = Path(p)
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            removed.append(str(p))
    cp = Path(result.get("console") or "")
    if cp.exists():
        cp.unlink()
        removed.append(str(cp))
    log(f"体检产物已删除: {len(removed)} 项")
    return removed


def overwrite_old_tb(job: dict, result: dict) -> list[str]:
    """同类型旧 TB 移到归档目录，TB 界面只保留本次 50 轮新曲线。

    例外：名字含 `_first5` 的历史切片保留。
    """
    if result.get("status") != "success":
        return []
    new_dirs = {Path(p) for p in result.get("tb_new", [])}
    moved = []
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for old in [Path(p) for p in result.get("tb_before", [])]:
        if not old.exists():
            continue
        if "_first5" in old.name:
            continue
        src_kind = "tb" if str(old).startswith(str(ROOT / "tb")) else "tflogs"
        dst = ARCHIVE_DIR / f"{src_kind}__{old.name}"
        if dst.exists():
            dst = ARCHIVE_DIR / f"{src_kind}__{old.name}.{int(time.time())}"
        try:
            shutil.move(str(old), str(dst))
            moved.append(str(dst))
        except Exception as e:
            log(f"⚠️ 移动旧 TB 失败 {old}: {e}")
    if moved:
        log(f"旧 TB 已归档 {len(moved)} 项 → {ARCHIVE_DIR}")
    return moved


def write_results(all_results: list[dict]) -> None:
    with open(RESULTS_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "id", "dataset", "attack", "rule", "rounds", "status", "reason",
            "best", "last", "last5", "last10", "rounds_seen", "duration_min",
            "started", "ended", "log",
        ])
        w.writeheader()
        for r in all_results:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    lines = [
        "# 50 轮全量实验自动汇总（阶段 A）",
        "",
        f"- 更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 说明：主口径为末 5 轮均值；`last` 为第 50 轮；`best` 为 50 轮内最佳。",
        "",
        "| # | 实验 | 数据集 | 攻击 | 方法 | 状态 | best | last | 末5 | 末10 | 耗时(min) | 日志 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(all_results, 1):
        lines.append(
            f"| {i} | {r.get('id','')} | {r.get('dataset','')} | {r.get('attack','')} | "
            f"{r.get('rule','')} | {r.get('status','')} | {r.get('best','')} | {r.get('last','')} | "
            f"{r.get('last5','')} | {r.get('last10','')} | {r.get('duration_min','')} | "
            f"{Path(str(r.get('log',''))).name if r.get('log') else ''} |"
        )
    lines.append("")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def git_commit_push(msg: str) -> None:
    if os.environ.get("FULL50_AUTO_PUSH", "1") != "1":
        return
    paths = ["logs_3", "tb", "queue/auto_results.md", "queue/test_summary.md"]
    try:
        subprocess.run(["git", "add", "-A", *paths], cwd=str(ROOT), timeout=180, check=False,
                       capture_output=True, text=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(ROOT), timeout=30)
        if diff.returncode == 0:
            return
        c = subprocess.run(["git", "commit", "-m", msg], cwd=str(ROOT), timeout=180,
                           capture_output=True, text=True)
        if c.returncode != 0:
            log(f"⚠️ git commit 失败: {c.stderr.strip()[-500:]}")
            return
        p = subprocess.run(["git", "push", "origin", "dev"], cwd=str(ROOT), timeout=300,
                           capture_output=True, text=True)
        if p.returncode == 0:
            log(f"✓ 已推送 dev: {msg}")
        else:
            log(f"⚠️ git push 失败（本地已提交，可稍后手动推）: {p.stderr.strip()[-500:]}")
    except Exception as e:
        log(f"⚠️ git 操作异常: {e}")


# ---------------------------------------------------------------- 体检


def validate_test(job: dict, result: dict) -> tuple[bool, str]:
    if result["status"] != "success":
        return False, f"运行失败: {result.get('reason','')}"
    last = result.get("last")
    first = None
    if last in ("", None):
        return False, "没有解析到轮次指标"
    try:
        last = float(last)
    except Exception:
        return False, "末轮指标无法解析"
    # 从日志重新取首轮
    parsed_log = Path(result["log"])
    rounds = {}
    try:
        with open(parsed_log, encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = ROUND_RE.search(line)
                if m:
                    rounds[int(m.group(1))] = float(m.group(3))
    except Exception:
        pass
    first = rounds[min(rounds)] if rounds else None
    if first is None:
        return False, "没有解析到首轮指标"
    threshold = 20.0 if job["dataset"] == "cifar10" else 15.0
    if last < threshold:
        return False, f"末轮 {last}% < 阈值 {threshold}%"
    if last <= first:
        return False, f"末轮 {last}% 未高于首轮 {first}%"
    if last - first < 3:
        return False, f"增长仅 {last - first:.2f} 点（<3）"
    return True, f"R1={first}% → R5={last}%，增长 {last - first:.2f} 点"


def _looks_oom(result: dict) -> bool:
    for key in ("console", "log"):
        p = Path(result.get(key) or "")
        if not p.exists():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")[-2000:]
        except Exception:
            continue
        low = txt.lower()
        if "out of memory" in low or "oom" in low:
            return True
    return False


def run_tests(state: dict) -> bool:
    tests = [
        (TEST_CIFAR, 5, "CIFAR 体检（FL_LOCAL_STEPS=24）", 40, 10, 1.5),
        (TEST_VISDRONE, 5, "VisDrone 体检（LOCAL_EPOCHS=1）", 90, 15, 3.0),
    ]
    summary = ["# 50 轮实验前体检", "", f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    summary.append("| 项 | R1 | R5 | 判定 | 耗时(min) | 删除产物 |")
    summary.append("|---|---|---|---|---|---|")

    for job, rounds, desc, timeout, stall, setup_min in tests:
        log(f"===== {desc} =====")
        result = run_job(job, rounds, "test", state, timeout, stall)
        ok, why = validate_test(job, result)

        # 32GB 显存兜底：VisDrone 若评估 OOM，只降 test batch（不动模型/分辨率/训练 batch）
        if not ok and job["dataset"] == "visdrone" and _looks_oom(result):
            log("VisDrone 体检 OOM：按计划只降低 test batch 到 32，重试一次。")
            cleanup_test_artifacts(job, result)
            os.environ["FULL50_VD_TEST_BATCH"] = "32"
            result = run_job(job, rounds, "test", state, timeout, stall)
            ok, why = validate_test(job, result)
            if ok:
                state["visdrone_test_batch"] = 32
                atomic_write_json(STATE_PATH, state)
                log("test batch=32 通过，后续 VisDrone 队列沿用 32。")

        r1 = result.get("first", "")
        r5 = result.get("last", "")
        if ok:
            removed = cleanup_test_artifacts(job, result)
        else:
            removed = []
            log(f"体检失败，保留日志/TB 便于排查：{result.get('log')} | {result.get('tb_new')}")
        summary.append(f"| {job['id']} | {r1} | {r5} | {'✅ 通过' if ok else '❌ 失败'}：{why} | "
                       f"{result['duration_min']} | {len(removed) if ok else '保留'} 项 |")
        summary.append("")
        if not ok:
            TEST_SUMMARY_PATH.write_text("\n".join(summary), encoding="utf-8")
            log(f"✗ 体检失败：{why}")
            return False
        # 记录单轮耗时估算（用于截止判断）
        per_round = max(0.05, (result["duration_min"] - setup_min) / rounds)
        state.setdefault("estimates", {})[job["dataset"]] = round(per_round * 50 + setup_min, 1)
        atomic_write_json(STATE_PATH, state)
        log(f"✓ 体检通过：{why}；50 轮预计 {state['estimates'][job['dataset']]} 分钟")

    TEST_SUMMARY_PATH.write_text("\n".join(summary), encoding="utf-8")
    return True


# ---------------------------------------------------------------- 主流程


def _acquire_lock() -> bool:
    global LOCK_HANDLE
    LOCK_HANDLE = open(LOCK_PATH, "w")
    try:
        fcntl.flock(LOCK_HANDLE, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _fill_blank_results(all_results: list[dict], state: dict) -> None:
    """把此前已完成/失败/跳过的 job 补进汇总（用于重启后续跑）。"""
    done_ids = {r["id"] for r in all_results}
    for rec in state.get("history", []):
        if rec["id"] not in done_ids:
            all_results.append(rec)


def main() -> int:
    if DONE_PATH.exists():
        log("DONE 已存在，队列已结束。")
        return 0
    if not _acquire_lock():
        log("另一个 supervisor 正在运行，退出。")
        return 0

    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")
    state = load_json(STATE_PATH) or {}
    state.setdefault("started_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if "deadline" not in state:
        started = datetime.strptime(state["started_at"], "%Y-%m-%d %H:%M:%S")
        state["deadline"] = (started + timedelta(hours=23)).strftime("%Y-%m-%d %H:%M:%S")
    state.setdefault("completed", [])
    state.setdefault("failed", [])
    state.setdefault("skipped", [])
    state.setdefault("history", [])
    state.setdefault("estimates", {"cifar10": 30.0, "visdrone": 180.0})
    state.setdefault("tests_passed", False)
    state.setdefault("visdrone_test_batch", None)
    if state.get("visdrone_test_batch") == 32:
        os.environ["FULL50_VD_TEST_BATCH"] = "32"
    atomic_write_json(STATE_PATH, state)

    log("=" * 60)
    log(f"50 轮全量队列启动 | 开始 {state['started_at']} | 截止 {state['deadline']}")
    log("=" * 60)

    # 恢复：如果上次 supervisor 中断时留下 current，判断该条是否已经跑完
    cur = state.get("current")
    if cur:
        lp = Path(cur.get("log", ""))
        total = int(cur.get("total", 0))
        parsed = _parse_metrics(lp, total) if lp.exists() else {}
        if parsed.get("last_round") == total:
            log(f"恢复：{cur['id']} 日志已跑满 {total} 轮，记为成功。")
            rec = dict(id=cur["id"], dataset=cur.get("dataset", ""), attack="", rule="",
                       rounds=total, status="success", reason="recovered",
                       best=parsed.get("best", ""), last=parsed.get("last", ""),
                       last5=parsed.get("last5", ""), last10=parsed.get("last10", ""),
                       rounds_seen=parsed.get("rounds_seen", 0), duration_min="",
                       started=cur.get("started_at", ""), ended="", log=str(lp), console="")
            state["history"].append(rec)
            state["completed"].append(cur["id"])
        else:
            log(f"恢复：{cur['id']} 未跑满，按中断处理。")
            state["failed"].append(cur["id"])
        state.pop("current", None)
        atomic_write_json(STATE_PATH, state)

    # 体检
    if not state["tests_passed"]:
        if not run_tests(state):
            DONE_PATH.write_text("test_failed", encoding="utf-8")
            _update_status("done", None, 0, 0, None, 0, state)
            return 1
        state["tests_passed"] = True
        atomic_write_json(STATE_PATH, state)
        # 避免体检与正式实验落在同一分钟（日志 / TB 目录名都精确到分钟）
        now = datetime.now()
        wait = 70 - now.second if now.second < 70 else 0
        if wait > 0:
            log(f"等待 {wait}s 进入下一分钟，避免与体检日志/TB 撞名...")
            time.sleep(wait)

    all_results = []
    write_results(all_results)

    # 正式队列
    for job in JOBS:
        if job["id"] in state["completed"] or job["id"] in state["failed"] or job["id"] in state["skipped"]:
            continue
        if DONE_PATH.exists():
            break
        deadline_dt = datetime.strptime(state["deadline"], "%Y-%m-%d %H:%M:%S")
        remaining = (deadline_dt - datetime.now()).total_seconds() / 60
        est = float(state["estimates"].get(job["dataset"], 180.0))
        if remaining < est + 30:
            log(f"跳过 {job['id']}：剩余 {remaining:.0f} min < 预计 {est:.0f} + 30 min")
            state["skipped"].append(job["id"])
            atomic_write_json(STATE_PATH, state)
            continue

        timeout = 60 if job["dataset"] == "cifar10" else 360
        stall = 10 if job["dataset"] == "cifar10" else 20
        result = run_job(job, 50, "queue", state, timeout, stall)
        result["moved_old_tb"] = overwrite_old_tb(job, result)
        state["history"].append(result)
        if result["status"] == "success":
            state["completed"].append(job["id"])
        else:
            state["failed"].append(job["id"])
        atomic_write_json(STATE_PATH, state)
        all_results.append(result)
        write_results(all_results)
        git_commit_push(f"exp(full50): {job['id']} {result['status']} "
                        f"last={result['last']} last5={result['last5']}")
        if result["status"] == "failed" and result.get("reason") == "no_log":
            # 代码/环境级错误，继续下一条意义不大
            log("出现代码/环境级失败（no_log），停止队列。")
            DONE_PATH.write_text("failed_no_log", encoding="utf-8")
            break

    write_results(all_results)
    DONE_PATH.write_text("finished", encoding="utf-8")
    _update_status("done", None, 0, 0, None, 0, state)
    log("=" * 60)
    log(f"队列结束 | 完成 {len(state['completed'])} / 失败 {len(state['failed'])} / 跳过 {len(state['skipped'])}")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("收到中断，退出（未完成的实验按 current 记录）。")
        sys.exit(130)
