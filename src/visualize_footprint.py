#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple video overlay:
- YOLOv8 segmentation
- draw segmentation mask
- draw bbox + ID
- draw ONLY bottom contact line (footprint), no BEV, no measurement, no CSV
- optional track ID association from JSON using IoU
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# -------------------------- Helpers --------------------------
def draw_label(img, text, org, color=(255, 255, 255), bg=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    x = max(0, min(img.shape[1] - tw - 8, x))
    y = max(th + 8, min(img.shape[0] - 4, y))
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), bg, -1)
    cv2.putText(img, text, (x + 3, y - 3), font, scale, color, thick, cv2.LINE_AA)


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def color_for_id(tid: Any):
    h = hash(str(tid))
    return (50 + (h & 0x7F), 50 + ((h >> 7) & 0x7F), 50 + ((h >> 14) & 0x7F))


def contact_points_from_mask(mask: np.ndarray, smooth_k: int = 5):
    """
    Footprint = only the lowest point of the mask for each x-column.
    This matches the old idea of the bottom contact line, not a large filled band.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    pts = []
    for x in np.unique(xs):
        ys_x = ys[xs == x]
        yb = int(np.max(ys_x))
        pts.append((float(x), float(yb)))

    if len(pts) < 2:
        return None

    arr = np.array(pts, dtype=np.float32)

    k = max(3, int(smooth_k))
    if k % 2 == 0:
        k += 1

    if len(arr) >= k:
        ys_sm = arr[:, 1].copy()
        pad = k // 2
        padded = np.r_[np.repeat(ys_sm[0], pad), ys_sm, np.repeat(ys_sm[-1], pad)]
        ys_s = np.convolve(padded, np.ones(k) / k, mode="valid")
        arr[:, 1] = ys_s

    return arr


# -------------------------- Track JSON helpers --------------------------
def extract_tracks(data: Any) -> List[Dict[str, Any]]:
    tracks: List[Dict[str, Any]] = []

    def walk(obj: Any):
        if isinstance(obj, dict):
            has_time = any(k in obj for k in ("timestamps", "frames"))
            has_box = any(k in obj for k in ("boxes", "bboxes"))
            if has_time and has_box:
                tracks.append(obj)
            else:
                for v in obj.values():
                    walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    return tracks


def get_track_fields(trk: Dict[str, Any]):
    tid = trk.get("id", trk.get("track_id", trk.get("tid", None)))
    timestamps = trk.get("timestamps", trk.get("frames", []))
    boxes = trk.get("boxes", trk.get("bboxes", []))
    return tid, timestamps, boxes


# -------------------------- Main --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="input video path")
    ap.add_argument("--out", required=True, help="output video path")
    ap.add_argument("--model", default="yolov8s-seg.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", nargs="*", default=["car"])
    ap.add_argument("--device", default=None, help="cpu / cuda:0 / mps")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--json", default=None, help="optional tracking JSON for ID association")
    ap.add_argument("--iou_match", type=float, default=0.30, help="IoU threshold for matching with JSON tracks")

    # visualization
    ap.add_argument("--draw_boxes", action="store_true")
    ap.add_argument("--draw_labels", action="store_true")
    ap.add_argument("--draw_masks", action="store_true")
    ap.add_argument("--draw_mask_edges", action="store_true")
    ap.add_argument("--mask_alpha", type=float, default=0.35)
    ap.add_argument("--smooth_k", type=int, default=5, help="smoothing kernel for bottom footprint line")
    ap.add_argument("--overlay_stats", action="store_true")

    args = ap.parse_args()

    class_set = set(args.classes) if args.classes else None

    # Optional tracker JSON
    bboxes_by_frame: Dict[int, List[Tuple[Any, Tuple[float, float, float, float]]]] = {}
    if args.json:
        with open(args.json, "r", encoding="utf-8") as f:
            data = json.load(f)
        tracks = extract_tracks(data)
        if not tracks:
            raise RuntimeError("No tracks found in JSON. Expected keys like 'timestamps' + 'boxes'.")
        for trk in tracks:
            tid, timestamps, boxes = get_track_fields(trk)
            if tid is None:
                continue
            for ts_f, bbox in zip(timestamps, boxes):
                if isinstance(bbox, list) and len(bbox) == 4:
                    bboxes_by_frame.setdefault(int(ts_f), []).append((tid, tuple(map(float, bbox))))

    print("Loading model:", args.model)
    model = YOLO(args.model)

    print("Opening video:", args.video)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    print(f"Output video: {out_path} ({W}x{H}@{fps:.2f}fps)")

    bar = None
    if tqdm is not None:
        bar = tqdm(total=total_frames if total_frames is not None else None, unit="f", dynamic_ncols=True)
        bar.set_description("Processing")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        predict_kwargs = {"conf": args.conf, "verbose": False}
        if args.device is not None:
            predict_kwargs["device"] = args.device
        if args.imgsz is not None:
            predict_kwargs["imgsz"] = args.imgsz
        if args.half:
            predict_kwargs["half"] = True

        res = model.predict(frame, **predict_kwargs)[0]

        cands = []
        if res.boxes is not None and res.masks is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            names = [res.names[int(c)] for c in clss]
            masks = res.masks.data.cpu().numpy()

            for i in range(len(boxes)):
                name = names[i]
                if class_set and name not in class_set:
                    continue

                x1, y1, x2, y2 = map(int, boxes[i])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x2)
                y2 = min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                maskf = masks[i]
                if maskf.dtype != np.uint8:
                    maskf = (maskf > 0.5).astype(np.uint8)
                if maskf.shape != (H, W):
                    maskf = cv2.resize(maskf, (W, H), interpolation=cv2.INTER_NEAREST)

                cands.append(((x1, y1, x2, y2), maskf, name))

        # Assign IDs
        assignments = []
        assigned = [False] * len(cands)
        items = bboxes_by_frame.get(frame_idx, []) if args.json else []

        if args.json and len(items) and len(cands):
            for tid, tbox in items:
                best_iou, best_j = 0.0, -1
                for j, (box, maskf, name) in enumerate(cands):
                    if assigned[j]:
                        continue
                    iou = iou_xyxy(box, tbox)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_j >= 0 and best_iou >= args.iou_match:
                    assignments.append((tid, best_j))
                    assigned[best_j] = True

        if not args.json:
            next_det_id = 0
            for j in range(len(cands)):
                if not assigned[j]:
                    assignments.append((f"det{next_det_id}", j))
                    next_det_id += 1

        # If JSON exists but some detections are unmatched, still show them as detX
        if args.json:
            next_det_id = 0
            for j in range(len(cands)):
                if not assigned[j]:
                    assignments.append((f"det{next_det_id}", j))
                    next_det_id += 1

        # Draw
        for tid, j in assignments:
            (x1, y1, x2, y2), maskf, name = cands[j]
            color = color_for_id(tid)
            footprint_color = (0, 165, 255)  # orange

            if args.draw_masks:
                m = maskf.astype(bool)
                a = float(np.clip(args.mask_alpha, 0.0, 1.0))
                frame[m] = (
                    frame[m] * (1.0 - a) + a * np.array((50, 200, 50), dtype=np.float32)
                ).astype(np.uint8)

            if args.draw_mask_edges:
                contours, _ = cv2.findContours(maskf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (50, 200, 50), 2)

            if args.draw_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if args.draw_labels:
                draw_label(frame, f"id={tid}", (x1, y1 - 8))

            sub = maskf[y1:y2, x1:x2].copy()
            if sub.size == 0:
                continue

            footprint_pts = contact_points_from_mask(sub, smooth_k=args.smooth_k)
            if footprint_pts is not None and len(footprint_pts) >= 2:
                footprint_pts[:, 0] += x1
                footprint_pts[:, 1] += y1
                for k in range(1, len(footprint_pts)):
                    cv2.line(
                        frame,
                        (int(footprint_pts[k - 1, 0]), int(footprint_pts[k - 1, 1])),
                        (int(footprint_pts[k, 0]), int(footprint_pts[k, 1])),
                        footprint_color, 2, cv2.LINE_AA
                    )

        if args.overlay_stats:
            txt = f"frame: {frame_idx + 1}/{total_frames if total_frames else '?'}"
            draw_label(frame, txt, (10, 30))

        writer.write(frame)

        if bar is not None:
            bar.update(1)
        frame_idx += 1

    cap.release()
    writer.release()
    if bar is not None:
        bar.close()

    print("Done.")
    print("Saved video:", out_path)


if __name__ == "__main__":
    main()
