"""
VisDrone 入口脚本默认「脱壳」启动：关 SSH 后训练继续跑（nohup 子进程）。

默认行为：自动 nohup（不重定向第二份 *_nohup_*.log；主日志为 server.py loguru）。

显式保持前台（终端实时刷屏）：
  FL_ATTACHED=1 python run-test/visdrone/run_no_attack_baseline.py
  或: python run-test/visdrone/run_no_attack_baseline.py --attach

兼容旧变量：FL_AUTO_DETACH=0 等价于强制不自动脱壳（与 FL_ATTACHED=1 二选一即可）。

监控：tail -f logs_3/visdrone/<experiment_tag>_北京时间.log

仅 Unix；无 nohup 或 Windows 时保持前台运行。
"""

from __future__ import annotations

import os
import subprocess
import sys


def _build_monitor_cmd(argv: list[str]) -> str:
    script = os.path.basename(argv[0]) if argv else "run_no_attack_baseline.py"
    tag = os.path.splitext(script)[0]
    return (
        "tail -f \"$(ls -t logs_3/visdrone/"
        f"{tag}_*.log 2>/dev/null | head -1)\""
    )


def maybe_detach() -> None:
    if os.environ.get("FL_DETACHED") == "1":
        return
    force_attached = (
        os.environ.get("FL_ATTACHED", "").strip() == "1"
        or "--attach" in sys.argv
        or os.environ.get("FL_AUTO_DETACH", "").strip() == "0"
    )
    if force_attached:
        sys.argv = [a for a in sys.argv if a != "--attach"]
        return
    if sys.platform.startswith("win"):
        return
    argv = [a for a in sys.argv if a not in ("--detach", "--attach")]
    env = os.environ.copy()
    env["FL_DETACHED"] = "1"
    cmd = [sys.executable, "-u", argv[0], *argv[1:]]
    try:
        subprocess.Popen(
            ["nohup", *cmd],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=os.getcwd(),
            start_new_session=True,
        )
    except FileNotFoundError:
        print("nohup not found; continuing in foreground.", flush=True)
        return
    monitor_cmd = _build_monitor_cmd(argv)
    print(
        "Detached (nohup, default). Safe to close this terminal.\n"
        "Foreground next time: FL_ATTACHED=1 or --attach\n"
        "Run to monitor logs:\n"
        f"{monitor_cmd}",
        flush=True,
    )
    raise SystemExit(0)
