# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleop LAUNCHER — one command to start the whole pipeline.

Spawns both halves of the teleop stack, each inside its own conda env
(they must stay separate: WiLoR's torch 2.5 would clobber Isaac Lab's torch):

    teleop_sim.py        (env_isaaclab)  Isaac Sim driver, UDP subscriber
    perception_node.py   (wilor)         camera -> WiLoR -> retarget, UDP publisher

Run from ANY env (only stdlib is used):

    cd tools/teleop
    python teleop.py                     # sim GUI + realsense perception with overlay
    python teleop.py --no-show           # perception headless
    python teleop.py --source webcam --camera 10
    python teleop.py --sim-args "--debug" --perc-args "--beta 2.0"

Ctrl+C stops both. If either side exits on its own (e.g. pressing 'q' in the
overlay window), the other is shut down too. The Isaac side is never SIGKILLed
(that can corrupt the Isaac Sim install) — worst case you wait a few seconds
for its clean shutdown.
"""
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SIM_ENV = "env_isaaclab"
PERC_ENV = "wilor"


def _conda_base() -> Path:
    exe = os.environ.get("CONDA_EXE")
    if exe:
        return Path(exe).resolve().parent.parent
    for cand in (Path.home() / "miniconda3", Path.home() / "anaconda3", Path("/opt/conda")):
        if (cand / "etc/profile.d/conda.sh").exists():
            return cand
    raise SystemExit("[teleop] cannot find conda (set CONDA_EXE or install to ~/miniconda3)")


def _spawn(env_name: str, script: str, script_args: list[str]) -> subprocess.Popen:
    """Launch `python script args` inside a properly activated conda env.

    A plain envs/<name>/bin/python is NOT enough for Isaac Lab (activation
    scripts set required env vars), so go through `conda activate`. `exec`
    replaces the wrapper bash so signals reach python directly.
    """
    conda_sh = _conda_base() / "etc/profile.d/conda.sh"
    inner = (
        f"source {shlex.quote(str(conda_sh))} && "
        f"conda activate {shlex.quote(env_name)} && "
        f"exec python {shlex.quote(script)} " + " ".join(shlex.quote(a) for a in script_args)
    )
    return subprocess.Popen(["bash", "-c", inner], cwd=HERE)


def _shutdown(procs: dict[str, subprocess.Popen]) -> None:
    """Politely stop whatever is still running. Never SIGKILL the sim."""
    for name, p in procs.items():
        if p.poll() is None:
            print(f"[teleop] stopping {name} (SIGINT) ...")
            try:
                p.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
    deadline = time.time() + 10.0
    for name, p in procs.items():
        while p.poll() is None and time.time() < deadline:
            time.sleep(0.2)
        if p.poll() is None:
            print(f"[teleop] {name} still up, escalating to SIGTERM ...")
            try:
                p.terminate()
            except ProcessLookupError:
                pass
    # Isaac needs time for a clean exit; do NOT kill -9 it (corrupts the install).
    for name, p in procs.items():
        if p.poll() is None:
            print(f"[teleop] waiting for {name} to exit cleanly (no SIGKILL) ...")
            p.wait()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Start the full WiLoR -> Isaac Sim xhand teleop pipeline (both conda envs).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # shared UDP link
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=51234)
    # common perception knobs (forwarded to perception_node.py)
    ap.add_argument("--source", choices=["webcam", "realsense"], default="realsense")
    ap.add_argument("--camera", default="0", help="cv2 camera index/device (webcam source)")
    ap.add_argument("--rs-serial", default=None, help="RealSense serial/name substring")
    ap.add_argument("--hand", choices=["right", "left"], default="right")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--no-show", action="store_true",
                    help="run perception headless (default opens the overlay window)")
    # real hardware instead of Isaac sim
    ap.add_argument("--real", action="store_true",
                    help="drive the REAL xhand over serial (real_node.py) instead of Isaac Sim")
    ap.add_argument("--real-args", default="",
                    help="extra args for real_node.py, quoted string (e.g. \"--max-speed 1.5\")")
    ap.add_argument("--arm", action="store_true",
                    help="also start arm_node.py: WASD keyboard jog of the real xArm7 TCP")
    ap.add_argument("--arm-args", default="",
                    help="extra args for arm_node.py, quoted string (e.g. \"--ip 192.168.1.205\")")
    # escape hatches for anything else
    ap.add_argument("--sim-args", default="", help="extra args for teleop_sim.py, quoted string")
    ap.add_argument("--perc-args", default="", help="extra args for perception_node.py, quoted string")
    args = ap.parse_args()

    link = ["--host", args.host, "--port", str(args.port)]

    sim_args = link + shlex.split(args.sim_args)

    perc_args = link + ["--source", args.source, "--camera", str(args.camera), "--hand", args.hand]
    if args.rs_serial:
        perc_args += ["--rs-serial", args.rs_serial]
    if args.mirror:
        perc_args.append("--mirror")
    if args.flip:
        perc_args.append("--flip")
    if not args.no_show:
        perc_args.append("--show")
    perc_args += shlex.split(args.perc_args)

    real_args = link + shlex.split(args.real_args)

    arm_args = shlex.split(args.arm_args)

    if args.real:
        print(f"[teleop] REAL hand  ({PERC_ENV}): real_node.py {' '.join(real_args)}")
    else:
        print(f"[teleop] sim        ({SIM_ENV}): teleop_sim.py {' '.join(sim_args)}")
    if args.arm:
        print(f"[teleop] REAL arm   ({PERC_ENV}): arm_node.py {' '.join(arm_args)}")
    print(f"[teleop] perception ({PERC_ENV}): perception_node.py {' '.join(perc_args)}")

    # a SIGTERM to the launcher must also tear the children down cleanly
    def _on_sigterm(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_sigterm)

    procs: dict[str, subprocess.Popen] = {}
    exit_code = 0
    try:
        if args.real:
            procs["real"] = _spawn(PERC_ENV, "real_node.py", real_args)
        else:
            procs["sim"] = _spawn(SIM_ENV, "teleop_sim.py", sim_args)
        if args.arm:
            procs["arm"] = _spawn(PERC_ENV, "arm_node.py", arm_args)
        procs["perception"] = _spawn(PERC_ENV, "perception_node.py", perc_args)

        # babysit: if either side dies, tear the other down
        while True:
            for name, p in procs.items():
                rc = p.poll()
                if rc is not None:
                    print(f"[teleop] {name} exited (code {rc}); shutting down the other side ...")
                    exit_code = rc if rc not in (0, -signal.SIGINT) else 0
                    raise KeyboardInterrupt  # reuse the shutdown path below
            time.sleep(0.5)
    except KeyboardInterrupt:
        # Ctrl+C also delivers SIGINT to both children (same process group);
        # _shutdown() re-sends it to whoever is still alive and waits.
        print("\n[teleop] shutting down ...")
    finally:
        _shutdown(procs)
    print("[teleop] all stopped.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
