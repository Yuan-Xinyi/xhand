# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera sources for the teleop perception node.

Uniform interface: ``read() -> (ok, bgr_uint8)`` and ``release()``. Two backends:

- ``CVCamera``       — any V4L2 webcam via OpenCV. MJPG + 1-frame buffer to cut
                       capture latency (a full buffer serves stale frames).
- ``RealSenseCamera``— Intel RealSense color stream at high FPS via pyrealsense2.
                       Cleaner, better-exposed, higher-rate frames than the webcam
                       -> fewer motion-blur detection drops. Depth is available on
                       the D435 and exposed via ``read_depth()`` for future use.
"""
from __future__ import annotations

import cv2
import numpy as np


class CVCamera:
    def __init__(self, index_or_path, width: int = 640, height: int = 480):
        cam = int(index_or_path) if str(index_or_path).isdigit() else index_or_path
        self._cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()


def list_realsense():
    """Return [(name, serial)] for every connected RealSense device."""
    import pyrealsense2 as rs

    out = []
    for d in rs.context().query_devices():
        out.append((d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)))
    return out


def _reset_device(rs, serial: str):
    """Hardware-reset ONLY the given device (never touch the others)."""
    for d in rs.context().query_devices():
        if d.get_info(rs.camera_info.serial_number) == serial:
            d.hardware_reset()
            return True
    return False


class RealSenseCamera:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 60,
                 align_depth: bool = False, serial: str | None = None):
        import time

        import pyrealsense2 as rs

        self._rs = rs
        self._align_depth = align_depth
        self._width, self._height, self._fps = width, height, fps
        self._last_depth = None
        self._serial = serial

        def _cfg():
            cfg = rs.config()
            if serial is not None:
                cfg.enable_device(serial)   # bind to a specific camera (D435 vs D435IF)
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            if align_depth:
                cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            return cfg

        self._align = rs.align(rs.stream.color) if align_depth else None
        # A stale handle from a crashed run leaves the device busy -> first frame
        # never arrives. Try once; on failure reset ONLY this device and retry.
        for attempt in range(2):
            try:
                self._pipe = rs.pipeline()
                profile = self._pipe.start(_cfg())
                if serial is None:
                    self._serial = profile.get_device().get_info(rs.camera_info.serial_number)
                for _ in range(5):  # let auto-exposure settle
                    self._pipe.wait_for_frames(10000)
                return
            except RuntimeError:
                try:
                    self._pipe.stop()
                except Exception:
                    pass
                if attempt == 0 and self._serial is not None:
                    _reset_device(rs, self._serial)
                    time.sleep(3)
                else:
                    raise

    def is_opened(self) -> bool:
        return True

    def read(self):
        try:
            frames = self._pipe.wait_for_frames(2000)
        except RuntimeError:
            return False, None
        if self._align is not None:
            frames = self._align.process(frames)
            depth = frames.get_depth_frame()
            self._last_depth = np.asanyarray(depth.get_data()) if depth else None
        color = frames.get_color_frame()
        if not color:
            return False, None
        return True, np.asanyarray(color.get_data())

    def read_depth(self):
        """Latest aligned depth frame (uint16 mm) if align_depth was set, else None."""
        return self._last_depth

    def release(self):
        self._pipe.stop()


def _is_if_variant(name: str) -> bool:
    """True for the D435IF (IR-filter) variant, whose color is IR-tinted."""
    return "d435if" in name.lower().replace(" ", "")


def pick_realsense_serial(prefer: str | None = None):
    """Choose a RealSense serial. If `prefer` (name substring or serial) matches,
    use it; else prefer a plain D435 over the D435IF (IR-filter variant, worse for
    WiLoR). Returns (serial, name) or (None, None).
    """
    devs = list_realsense()  # [(name, serial)]
    if not devs:
        return None, None
    if prefer:
        p = prefer.lower()
        for name, serial in devs:
            if p == serial.lower() or p in name.lower():
                return serial, name
    plain = [(n, s) for n, s in devs if not _is_if_variant(n)]
    name, serial = plain[0] if plain else devs[0]
    return serial, name


def open_camera(source: str, index, width: int, height: int, fps: int = 60, rs_serial: str | None = None):
    """Factory: source in {'webcam','realsense'}."""
    if source == "realsense":
        serial, name = pick_realsense_serial(rs_serial)
        if serial is None:
            raise SystemExit("[camera] no RealSense device found (check USB / re-plug)")
        print(f"[camera] RealSense: {name} (serial {serial})")
        if _is_if_variant(name):
            print("[camera][WARN] this is the D435IF (IR-filter) variant — its color is "
                  "IR-tinted and weaker for WiLoR. Plug in / select the plain D435.")
        return RealSenseCamera(width=width, height=height, fps=fps, serial=serial)
    return CVCamera(index, width=width, height=height)
