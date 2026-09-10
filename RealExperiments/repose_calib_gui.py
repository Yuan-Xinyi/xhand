#!/usr/bin/env python3
"""Draggable-slider calibration panel for the repose cube pose.

INCREMENTAL: the saved yaml is the BASE; sliders always start at zero and act
as a delta ON TOP of it (total = base + delta is what gets streamed on udp
9880). Press s to fold the delta into the base (saved + sliders re-zeroed), so
every session refines the previous calibration with the full slider range.

    s  save base+delta -> repose_manual_calib.yaml, sliders back to zero
    r  reset the current delta (back to the saved base)
    z  zero EVERYTHING (base and delta)
    q  quit

Run (needs GUI cv2, same trick as the tracker):
    conda activate env_isaaclab
    python RealExperiments/repose_calib_gui.py
"""
import os
import socket
import struct
import sys
import time

# demote Isaac's headless-cv2 prebundle (same as the tracker)
_demoted = [p for p in sys.path if "omni.pip.compute" in p]
sys.path = [p for p in sys.path if p not in _demoted] + _demoted

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

CALIB_UDP = ("127.0.0.1", 9880)
CALIB_FMT = "<6d"  # dx, dy, dz [m], rx, ry, rz [rad]
CALIB_COMMIT_FMT = "<7d"  # + flag 1.0: tell the control loop to re-run auto-center
YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repose_manual_calib.yaml")

WIN = "cube calib  (s=save  r=reset  q=quit)"
NAMES = ["dx mm", "dy mm", "dz mm", "rx 0.1deg", "ry 0.1deg", "rz 0.1deg"]
SPANS = [100, 100, 100, 300, 300, 300]  # trackbar range; value = pos - span/2


def values():
    out = []
    for name, span in zip(NAMES, SPANS):
        v = cv2.getTrackbarPos(name, WIN) - span // 2
        out.append(v / 1000.0 if "mm" in name else np.radians(v / 10.0))
    return out


def set_values(vals6):
    for name, span, v in zip(NAMES, SPANS, vals6):
        raw = int(round(v * 1000)) if "mm" in name else int(round(np.degrees(v) * 10))
        cv2.setTrackbarPos(name, WIN, int(np.clip(raw + span // 2, 0, span)))


def load_base():
    if os.path.exists(YAML_PATH):
        with open(YAML_PATH) as f:
            d = yaml.safe_load(f) or {}
        return np.array([d.get(k, 0.0) for k in ["dx", "dy", "dz", "rx", "ry", "rz"]])
    return np.zeros(6)


def main():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 560, 420)
    for name, span in zip(NAMES, SPANS):
        cv2.createTrackbar(name, WIN, span // 2, span, lambda _v: None)

    base = load_base()
    if np.any(base):
        print(f"[calib] base loaded from {YAML_PATH}: d={np.round(base[:3] * 1000, 1)}mm "
              f"r={np.round(np.degrees(base[3:]), 2)}deg (sliders = delta on top)")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    canvas = np.full((150, 560, 3), 28, np.uint8)
    while True:
        delta = np.array(values())
        total = base + delta
        sock.sendto(struct.pack(CALIB_FMT, *total), CALIB_UDP)
        img = canvas.copy()
        cv2.putText(img, f"base  d=({base[0]*1000:+.0f},{base[1]*1000:+.0f},{base[2]*1000:+.0f})mm "
                         f"r=({np.degrees(base[3]):+.1f},{np.degrees(base[4]):+.1f},{np.degrees(base[5]):+.1f})deg",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        cv2.putText(img, f"delta d=({delta[0]*1000:+.0f},{delta[1]*1000:+.0f},{delta[2]*1000:+.0f})mm "
                         f"r=({np.degrees(delta[3]):+.1f},{np.degrees(delta[4]):+.1f},{np.degrees(delta[5]):+.1f})deg",
                    (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(img, f"TOTAL d=({total[0]*1000:+.0f},{total[1]*1000:+.0f},{total[2]*1000:+.0f})mm "
                         f"r=({np.degrees(total[3]):+.1f},{np.degrees(total[4]):+.1f},{np.degrees(total[5]):+.1f})deg",
                    (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
        cv2.putText(img, "s=save(base+=delta)  r=delta->0  z=zero all  q=quit",
                    (12, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.imshow(WIN, img)
        k = cv2.waitKey(50) & 0xFF
        if k == ord("q") or k == 27:
            break
        if k == ord("r"):
            set_values([0.0] * 6)
        if k == ord("z"):
            base = np.zeros(6)
            set_values([0.0] * 6)
            print("[calib] base + delta zeroed")
        if k == ord("s"):
            base = base + delta
            d = dict(zip(["dx", "dy", "dz", "rx", "ry", "rz"], [float(v) for v in base]))
            with open(YAML_PATH, "w") as f:
                yaml.safe_dump(d, f)
            set_values([0.0] * 6)
            sock.sendto(struct.pack(CALIB_COMMIT_FMT, *base, 1.0), CALIB_UDP)
            print(f"[calib] saved -> {YAML_PATH}: {d} — control loop re-centers now "
                  "(keep the cube at rest!)", flush=True)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
