#!/usr/bin/env python3
"""Draggable-slider calibration panel for the repose cube pose.

Six trackbars (dx/dy/dz +-50 mm, rx/ry/rz +-15 deg) are streamed live to the
control loop (udp 9880), which applies them to every cube pose. Drag until the
cube in the MIRROR viewer sits exactly like the real one, then press:

    s  save to RealExperiments/repose_manual_calib.yaml (auto-loaded next runs)
    r  reset all sliders to zero
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


def main():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 520, 380)
    for name, span in zip(NAMES, SPANS):
        cv2.createTrackbar(name, WIN, span // 2, span, lambda _v: None)

    if os.path.exists(YAML_PATH):
        with open(YAML_PATH) as f:
            d = yaml.safe_load(f) or {}
        set_values([d.get(k, 0.0) for k in ["dx", "dy", "dz", "rx", "ry", "rz"]])
        print(f"[calib] loaded {YAML_PATH}: {d}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    canvas = np.full((110, 520, 3), 28, np.uint8)
    while True:
        vals = values()
        sock.sendto(struct.pack(CALIB_FMT, *vals), CALIB_UDP)
        img = canvas.copy()
        cv2.putText(img, f"d = ({vals[0]*1000:+.0f}, {vals[1]*1000:+.0f}, {vals[2]*1000:+.0f}) mm",
                    (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 1)
        cv2.putText(img, f"r = ({np.degrees(vals[3]):+.1f}, {np.degrees(vals[4]):+.1f}, "
                         f"{np.degrees(vals[5]):+.1f}) deg",
                    (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 1)
        cv2.putText(img, "drag until the MIRROR cube matches reality, then press s",
                    (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.imshow(WIN, img)
        k = cv2.waitKey(50) & 0xFF
        if k == ord("q") or k == 27:
            break
        if k == ord("r"):
            set_values([0.0] * 6)
        if k == ord("s"):
            d = dict(zip(["dx", "dy", "dz", "rx", "ry", "rz"], [float(v) for v in vals]))
            with open(YAML_PATH, "w") as f:
                yaml.safe_dump(d, f)
            print(f"[calib] saved -> {YAML_PATH}: {d}", flush=True)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
