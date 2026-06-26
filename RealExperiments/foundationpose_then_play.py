#!/usr/bin/env python3
"""Initialize cube pose with FoundationPose, then launch IsaacLab RL play.

This follows the startup flow of FoundationPose's cube/live_demo.py:
capture one RGB-D frame, let the user draw the cube ROI, run registration,
save the resulting camera_T_cube pose, then start scripts/rl_games/play.py
with that pose injected as the Isaac cube pose.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOUNDATIONPOSE_ROOT = Path("/home/lqin/disk2/FoundationPose")
DEFAULT_CHECKPOINT = "logs/rl_games/pick_cube/0_2026-06-25_12-07-39/nn/pick_cube.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", default="Pick-Cube-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--play_script", default=str(REPO_ROOT / "scripts" / "rl_games" / "play.py"))
    parser.add_argument("--foundationpose_root", default=str(DEFAULT_FOUNDATIONPOSE_ROOT))
    parser.add_argument("--mesh_file", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--track_frames", type=int, default=5, help="Tracking frames to refine after registration.")
    parser.add_argument(
        "--pose_out",
        default=str(Path(tempfile.gettempdir()) / "foundationpose_cube_pose.npy"),
        help="Where to save the 4x4 pose passed to play.py.",
    )
    parser.add_argument(
        "--image_out",
        default=str(Path(tempfile.gettempdir()) / "foundationpose_init_frame.png"),
        help="Where to save the first RGB frame before ROI selection.",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="Cube ROI on the init image. Use this when OpenCV GUI/selectROI is unavailable.",
    )
    parser.add_argument("--preview", action="store_true", help="Show the registered pose overlay before launching Isaac.")
    parser.add_argument("--no_play", action="store_true", help="Only save pose; do not launch play.py.")
    parser.add_argument(
        "play_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to play.py. Put them after '--'.",
    )
    return parser.parse_args()


def load_live_demo(foundationpose_root: Path):
    cube_dir = foundationpose_root / "cube"
    if not cube_dir.exists():
        raise FileNotFoundError(f"FoundationPose cube directory not found: {cube_dir}")
    sys.path.insert(0, str(cube_dir))
    import live_demo  # type: ignore

    return live_demo


def _browser_roi(image_path: str) -> tuple[int, int, int, int]:
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"ROI image does not exist: {image_file}")

    done = threading.Event()
    state: dict[str, tuple[int, int, int, int] | Exception | None] = {"roi": None, "error": None}

    class RoiHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            if self.path.startswith("/image"):
                data = image_file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return

            html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>FoundationPose ROI</title>
  <style>
    body {{
      margin: 0;
      background: #202124;
      color: #e8eaed;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 14px;
      background: #111;
      border-bottom: 1px solid #3c4043;
    }}
    button {{
      border: 1px solid #5f6368;
      background: #303134;
      color: #e8eaed;
      padding: 7px 12px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
    }}
    button.primary {{
      background: #1a73e8;
      border-color: #1a73e8;
      color: white;
    }}
    #wrap {{
      padding: 16px;
    }}
    canvas {{
      display: block;
      background: #000;
      cursor: crosshair;
      image-rendering: auto;
      max-width: none;
      max-height: none;
    }}
    #status {{
      font-variant-numeric: tabular-nums;
      color: #bdc1c6;
    }}
  </style>
</head>
<body>
  <header>
    <button class="primary" id="confirm">Confirm ROI</button>
    <button id="reset">Reset</button>
    <span id="status">Drag a rectangle around the cube.</span>
  </header>
  <div id="wrap"><canvas id="canvas"></canvas></div>
  <script>
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const img = new Image();
    let dragging = false;
    let start = null;
    let roi = null;

    function normRect(a, b) {{
      const x = Math.round(Math.min(a.x, b.x));
      const y = Math.round(Math.min(a.y, b.y));
      const w = Math.round(Math.abs(a.x - b.x));
      const h = Math.round(Math.abs(a.y - b.y));
      return {{x, y, w, h}};
    }}

    function pointer(evt) {{
      const r = canvas.getBoundingClientRect();
      return {{
        x: (evt.clientX - r.left) * canvas.width / r.width,
        y: (evt.clientY - r.top) * canvas.height / r.height
      }};
    }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      if (roi) {{
        ctx.save();
        ctx.fillStyle = "rgba(26,115,232,0.18)";
        ctx.strokeStyle = "#00ff66";
        ctx.lineWidth = 2;
        ctx.fillRect(roi.x, roi.y, roi.w, roi.h);
        ctx.strokeRect(roi.x, roi.y, roi.w, roi.h);
        ctx.restore();
        statusEl.textContent = `ROI x=${{roi.x}} y=${{roi.y}} w=${{roi.w}} h=${{roi.h}}`;
      }}
    }}

    img.onload = () => {{
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      draw();
    }};
    img.src = "/image?ts=" + Date.now();

    canvas.addEventListener("mousedown", evt => {{
      dragging = true;
      start = pointer(evt);
      roi = {{x: start.x, y: start.y, w: 0, h: 0}};
      draw();
    }});
    window.addEventListener("mousemove", evt => {{
      if (!dragging) return;
      roi = normRect(start, pointer(evt));
      draw();
    }});
    window.addEventListener("mouseup", () => {{
      dragging = false;
      draw();
    }});
    document.getElementById("reset").onclick = () => {{
      roi = null;
      statusEl.textContent = "Drag a rectangle around the cube.";
      draw();
    }};
    document.getElementById("confirm").onclick = async () => {{
      if (!roi || roi.w < 2 || roi.h < 2) {{
        statusEl.textContent = "Draw a valid rectangle first.";
        return;
      }}
      const res = await fetch("/roi", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify(roi)
      }});
      if (res.ok) {{
        statusEl.textContent = "ROI submitted. You can return to the terminal.";
      }} else {{
        statusEl.textContent = "Failed to submit ROI.";
      }}
    }};
  </script>
</body>
</html>"""
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != "/roi":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                roi = tuple(int(payload[k]) for k in ("x", "y", "w", "h"))
                state["roi"] = roi
                self.send_response(204)
                self.end_headers()
                done.set()
            except Exception as exc:  # noqa: BLE001
                state["error"] = exc
                self.send_error(400, str(exc))
                done.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), RoiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/"
    print("[FoundationPose] OpenCV ROI window is unavailable; using browser ROI UI.")
    print(f"[FoundationPose] ROI UI: {url}")
    try:
        webbrowser.open(url, new=1)
    except Exception:
        pass

    try:
        while not done.wait(0.2):
            pass
    except KeyboardInterrupt:
        raise RuntimeError("ROI selection aborted.") from None
    finally:
        server.shutdown()
        server.server_close()

    if state["error"] is not None:
        raise RuntimeError(f"Browser ROI failed: {state['error']}")
    if state["roi"] is None:
        raise RuntimeError("Browser ROI returned no rectangle.")
    return state["roi"]


def _roi_to_mask(live_demo, color_rgb: np.ndarray, roi: list[int] | tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        raise ValueError(f"ROI width/height must be positive, got {(x, y, w, h)}")
    img_h, img_w = color_rgb.shape[:2]
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        raise ValueError(f"ROI {(x, y, w, h)} is outside image bounds {(img_w, img_h)}")
    return live_demo.grabcut(color_rgb[..., ::-1].copy(), (x, y, w, h)) > 0


def _prompt_roi(image_path: str) -> tuple[int, int, int, int]:
    print("[FoundationPose] OpenCV GUI is unavailable in this Python environment.")
    print(f"[FoundationPose] Open the saved image and enter ROI as: x y w h")
    print(f"[FoundationPose] image: {image_path}")
    while True:
        text = input("[FoundationPose] ROI x y w h (blank to abort): ").strip()
        if not text:
            raise RuntimeError("ROI selection aborted.")
        parts = text.replace(",", " ").split()
        if len(parts) != 4:
            print("[FoundationPose] Please enter exactly four integers: x y w h")
            continue
        try:
            return tuple(int(v) for v in parts)
        except ValueError:
            print("[FoundationPose] ROI must contain integers.")


def _select_mask(live_demo, color_rgb: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.roi is not None:
        return _roi_to_mask(live_demo, color_rgb, args.roi)
    try:
        mask = live_demo.select_mask(color_rgb)
        if mask is None:
            raise RuntimeError("ROI selection was cancelled.")
        return mask
    except cv2.error as exc:
        if "The function is not implemented" not in str(exc):
            raise
        try:
            roi = _browser_roi(args.image_out)
        except Exception as browser_exc:  # noqa: BLE001
            print(f"[FoundationPose] Browser ROI failed: {browser_exc}")
            roi = _prompt_roi(args.image_out)
        return _roi_to_mask(live_demo, color_rgb, roi)


def _safe_destroy_windows() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def capture_pose(args: argparse.Namespace) -> np.ndarray:
    import pyrealsense2 as rs

    foundationpose_root = Path(args.foundationpose_root).expanduser()
    live_demo = load_live_demo(foundationpose_root)

    mesh_file = args.mesh_file
    if mesh_file is None:
        mesh_file = str(foundationpose_root / "cube" / "mesh" / "textured.obj")

    live_demo.set_logging_format()
    live_demo.set_seed(0)
    est, _, to_origin, bbox, mt = live_demo.build_estimator(mesh_file)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)

    for _ in range(15):
        pipeline.wait_for_frames()

    def grab():
        frames = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        color = np.asarray(color_frame.get_data())
        depth = np.asarray(depth_frame.get_data()).astype(np.float32) * depth_scale
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
        return color, depth, K

    try:
        print("[FoundationPose] capturing init frame...")
        color, depth, K = grab()
        cv2.imwrite(args.image_out, color[..., ::-1])
        print(f"[FoundationPose] saved init image: {args.image_out}")
        print("[FoundationPose] drag a box around the cube, then press ENTER.")
        mask = _select_mask(live_demo, color, args)

        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
        for _ in range(max(0, args.track_frames)):
            color, depth, K = grab()
            pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)

        pose = np.asarray(pose, dtype=np.float32)
        np.save(args.pose_out, pose)
        t = pose[:3, 3]
        print(f"[FoundationPose] saved pose: {args.pose_out}")
        print(f"[FoundationPose] camera_T_cube xyz(m): {t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}")

        if args.preview:
            vis = live_demo.annotate(color, pose, K, to_origin, bbox, mt)
            try:
                cv2.imshow("FoundationPose registered pose - press any key", vis[..., ::-1])
                cv2.waitKey(0)
            except cv2.error:
                preview_path = str(Path(args.pose_out).with_suffix(".preview.png"))
                cv2.imwrite(preview_path, vis[..., ::-1])
                print(f"[FoundationPose] OpenCV GUI unavailable; saved preview image: {preview_path}")

        return pose
    finally:
        pipeline.stop()
        _safe_destroy_windows()


def launch_play(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        args.play_script,
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--checkpoint",
        args.checkpoint,
        "--external_cube_pose_npy",
        args.pose_out,
    ]
    extra_args = args.play_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd.extend(extra_args)

    print("[IsaacLab] launching:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    args = parse_args()
    start = time.time()
    capture_pose(args)
    print(f"[FoundationPose] pose initialization took {time.time() - start:.2f}s")
    if not args.no_play:
        launch_play(args)


if __name__ == "__main__":
    main()
