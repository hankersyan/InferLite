#!/usr/bin/env python
"""End-to-end test for the human-pose-estimation-0001 model on InferLite.

Runs the Open Model Zoo lightweight-OpenPose model (human-pose-estimation-0001)
against a running InferLite HTTP endpoint and dumps the results into a temp
output folder:
  - outputs.npz          raw output tensors (Mconv7_stage2_L1 PAF, Mconv7_stage2_L2 heatmaps)
  - keypoints.json       detected 18-keypoint peaks mapped back to the input image
  - pose_result.jpg      input image with keypoints + skeleton drawn
  - heatmaps_grid.jpg    the 18 keypoint heatmap channels (JET colormap)

Usage:
  # start the server first, e.g.:
  #   build\\inferlite.exe --model-repository=models --http-port=8000
  python scripts\\test_human_pose_estimation.py
  python scripts\\test_human_pose_estimation.py --server http://127.0.0.1:8000 \
      --image tests\\images\\man-pose.jpg --out temp\\human_pose
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = REPO_ROOT / "tests" / "images" / "man-pose.jpg"
DEFAULT_OUT = REPO_ROOT / "temp" / "human_pose"
DEFAULT_SERVER = "http://127.0.0.1:8100"

MODEL = "human-pose-estimation-0001"
INPUT_NAME = "data"
INPUT_SHAPE = (1, 3, 256, 456)  # NCHW
OUTPUTS = {  # name -> shape (NCHW)
    "Mconv7_stage2_L1": (1, 38, 32, 57),  # PAF: 19 limbs x 2
    "Mconv7_stage2_L2": (1, 19, 32, 57),  # heatmaps: 18 keypoints + background
}
HEATMAP_SHAPE = (32, 57)  # H, W of the heatmap output (input / stride 8)

KEYPOINT_NAMES = [
    "nose", "neck",
    "r-shoulder", "r-elbow", "r-wrist",
    "l-shoulder", "l-elbow", "l-wrist",
    "r-hip", "r-knee", "r-ankle",
    "l-hip", "l-knee", "l-ankle",
    "r-eye", "l-eye", "r-ear", "l-ear",
]
# Limb index pairs into KEYPOINT_NAMES (Open Model Zoo demo skeleton).
SKELETON = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17),
]

# Distinct BGR colors per limb for drawing.
_LIMB_COLORS = [
    (255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (0, 165, 255), (255, 100, 0), (100, 255, 0), (0, 100, 255), (200, 200, 0),
    (200, 0, 200), (0, 200, 200), (128, 0, 255), (128, 255, 0), (0, 128, 255),
    (255, 128, 0), (0, 255, 128), (128, 128, 255), (255, 128, 128),
]


def preprocess(image_bgr):
    """Resize keeping aspect ratio, pad to 456x256, NCHW FP32 in [0, 255].

    Mirrors the Open Model Zoo human_pose_estimation_demo preprocessing: the
    input image is scaled by min(H_in/h, W_in/w), padded with zeros on the
    bottom/right. NOTE: this human-pose-estimation-0001 variant expects raw
    [0, 255] pixels, NOT [0, 1] (dividing by 255 collapses the heatmaps to
    background). cv2 loads BGR, which is the channel order this model wants.
    """
    h_in, w_in = INPUT_SHAPE[2], INPUT_SHAPE[3]
    h_img, w_img = image_bgr.shape[:2]
    scale = min(h_in / h_img, w_in / w_img)
    resized = cv2.resize(image_bgr, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((h_in, w_in, 3), dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    blob = canvas.astype(np.float32)
    blob = blob.transpose((2, 0, 1))  # HWC -> CHW
    blob = blob[np.newaxis, ...]  # -> NCHW
    return blob, scale


def infer(server, model, blob):
    """POST the input blob to /v2/models/<model>/infer, return dict of outputs."""
    url = f"{server}/v2/models/{model}/infer"
    payload = {
        "inputs": [{
            "name": INPUT_NAME,
            "datatype": "FP32",
            "shape": list(blob.shape),
            "data": base64.b64encode(blob.tobytes()).decode("ascii"),
        }]
    }
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=30)
    latency_s = time.perf_counter() - t0
    if resp.status_code != 200:
        raise RuntimeError(
            f"infer failed with HTTP {resp.status_code}: {resp.text[:500]}")
    doc = resp.json()
    outputs = {}
    for o in doc.get("outputs", []):
        arr = np.frombuffer(base64.b64decode(o["data"]), dtype=np.float32)
        outputs[o["name"]] = arr.reshape([int(d) for d in o["shape"]])
    return outputs, latency_s, doc.get("trace_id", "")


def find_peaks(heatmap, threshold=0.1, nms_radius=1):
    """Greedy local-maxima extraction with 3x3 NMS (OMZ find_peaks)."""
    h = heatmap.astype(np.float32).copy()
    peaks = []
    while True:
        idx = int(np.argmax(h))
        y, x = divmod(idx, h.shape[1])
        score = float(h[y, x])
        if score < threshold:
            break
        peaks.append((x, y, score))
        y0, y1 = max(0, y - nms_radius), min(h.shape[0], y + nms_radius + 1)
        x0, x1 = max(0, x - nms_radius), min(h.shape[1], x + nms_radius + 1)
        h[y0:y1, x0:x1] = 0.0
    return peaks


def detect_keypoints(heatmaps, scale, threshold=0.1):
    """Extract the best peak per keypoint channel, mapped to input-image pixels."""
    hm = heatmaps[0]  # (19, 32, 57); channel 18 is the background
    stride_h = INPUT_SHAPE[2] / HEATMAP_SHAPE[0]   # 8.0
    stride_w = INPUT_SHAPE[3] / HEATMAP_SHAPE[1]   # 8.0
    keypoints = []
    for kp in range(len(KEYPOINT_NAMES)):
        peaks = find_peaks(hm[kp], threshold)
        best = max(peaks, key=lambda p: p[2], default=None)
        if best is None:
            keypoints.append({"name": KEYPOINT_NAMES[kp], "index": kp,
                              "x": None, "y": None, "score": None})
            continue
        x, y, score = best
        keypoints.append({
            "name": KEYPOINT_NAMES[kp],
            "index": kp,
            "x": round(x * stride_w / scale, 2),
            "y": round(y * stride_h / scale, 2),
            "score": round(score, 4),
        })
    return keypoints


def draw_skeleton(image_bgr, keypoints):
    img = image_bgr.copy()
    pts = [(kp["x"], kp["y"]) for kp in keypoints]
    for i, (a, b) in enumerate(SKELETON):
        pa, pb = pts[a], pts[b]
        if pa[0] is None or pb[0] is None:
            continue
        cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                 _LIMB_COLORS[i % len(_LIMB_COLORS)], 2, cv2.LINE_AA)
    for kp in keypoints:
        if kp["x"] is None:
            continue
        cv2.circle(img, (int(kp["x"]), int(kp["y"])), 4, (0, 0, 255), -1,
                   cv2.LINE_AA)
        cv2.circle(img, (int(kp["x"]), int(kp["y"])), 4, (255, 255, 255), 1,
                   cv2.LINE_AA)
    return img


def draw_heatmaps_grid(heatmaps):
    """Upscale and grid the 18 keypoint heatmap channels with a JET colormap."""
    hm = heatmaps[0][: len(KEYPOINT_NAMES)]  # drop background channel
    cell_w, cell_h = 114, 64
    cols = 6
    rows = (len(KEYPOINT_NAMES) + cols - 1) // cols
    grid = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)
    for i in range(len(KEYPOINT_NAMES)):
        ch = hm[i]
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6)
        ch = (ch * 255).astype(np.uint8)
        ch = cv2.resize(ch, (cell_w, cell_h), interpolation=cv2.INTER_CUBIC)
        color = cv2.applyColorMap(ch, cv2.COLORMAP_JET)
        r, c = divmod(i, cols)
        grid[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = color
        cv2.putText(grid, str(i), (c * cell_w + 4, r * cell_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return grid


def save_jpg(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    t_total0 = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"InferLite HTTP base URL (default: {DEFAULT_SERVER})")
    parser.add_argument("--model", default=MODEL,
                        help=f"model name (default: {MODEL})")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE),
                        help=f"input image (default: {DEFAULT_IMAGE})")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"output folder (default: {DEFAULT_OUT})")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="peak detection score threshold (default: 0.1)")
    args = parser.parse_args()

    # --- sanity check: server reachable -------------------------------------
    try:
        r = requests.get(f"{args.server}/v2/health/ready", timeout=5)
        r.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"ERROR: cannot reach InferLite at {args.server} ({e}).\n"
                 "Start the server first, e.g.:\n"
                 "  build\\inferlite.exe --model-repository=models --http-port=8000")

    # --- load + preprocess ---------------------------------------------------
    image_path = Path(args.image)
    if not image_path.is_file():
        sys.exit(f"ERROR: image not found: {image_path}")
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        sys.exit(f"ERROR: cannot decode image: {image_path}")
    blob, scale = preprocess(image_bgr)
    t_pre = time.perf_counter()
    print(f"image      : {image_path} ({image_bgr.shape[1]}x{image_bgr.shape[0]})")
    print(f"input blob : {blob.shape} fp32, scale={scale:.4f}")
    print(f"preprocess : {(t_pre - t_total0) * 1000:.1f} ms")

    # --- run inference -------------------------------------------------------
    outputs, latency_s, trace_id = infer(args.server, args.model, blob)
    print(f"inference  : {latency_s * 1000:.1f} ms  (trace_id={trace_id})")
    for name, arr in outputs.items():
        expected = OUTPUTS.get(name)
        if expected is not None and tuple(arr.shape) != expected:
            print(f"WARNING: {name} shape {tuple(arr.shape)} != expected {expected}")
        print(f"output     : {name} {tuple(arr.shape)}")

    heatmaps = outputs["Mconv7_stage2_L2"]
    paf = outputs["Mconv7_stage2_L1"]

    # --- postprocess + save ---------------------------------------------------
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(out_dir / "outputs.npz", **{"Mconv7_stage2_L1": paf,
                                         "Mconv7_stage2_L2": heatmaps})

    keypoints = detect_keypoints(heatmaps, scale, args.threshold)
    t_post = time.perf_counter()
    n_detected = sum(1 for kp in keypoints if kp["x"] is not None)
    print(f"keypoints  : {n_detected}/{len(KEYPOINT_NAMES)} detected "
          f"(threshold={args.threshold})")
    for kp in keypoints:
        if kp["x"] is not None:
            print(f"  {kp['name']:<11} ({kp['x']:7.2f}, {kp['y']:7.2f})  score={kp['score']}")

    with open(out_dir / "keypoints.json", "w") as f:
        json.dump({"model": args.model, "image": str(image_path),
                   "keypoints": keypoints}, f, indent=2)

    save_jpg(draw_skeleton(image_bgr, keypoints), out_dir / "pose_result.jpg")
    save_jpg(draw_heatmaps_grid(heatmaps), out_dir / "heatmaps_grid.jpg")
    t_end = time.perf_counter()

    print(f"\npostprocess: {(t_post - t_pre) * 1000:.1f} ms")
    print(f"total      : {(t_end - t_total0) * 1000:.1f} ms")
    print(f"\nresults written to {out_dir.resolve()}")
    for name in ("outputs.npz", "keypoints.json", "pose_result.jpg", "heatmaps_grid.jpg"):
        p = out_dir / name
        print(f"  {name:<18} {p.stat().st_size:>9,} bytes")


if __name__ == "__main__":
    main()
