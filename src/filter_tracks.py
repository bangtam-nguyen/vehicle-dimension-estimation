#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter tracking JSON to keep only car tracks/frames.

Ý tưởng:
- Đọc VID_01_tracks.json cũ.
- Chạy YOLO trên video.
- Chỉ giữ detection có class = car.
- Với mỗi bbox trong JSON tại frame tương ứng, tìm car detection có IoU cao nhất.
- Nếu IoU >= ngưỡng, giữ bbox đó.
- Xuất ra JSON mới: VID_01_car_only_tracks.json.

JSON output vẫn giữ format:
[
  {
    "id": ...,
    "timestamps": [...],
    "boxes": [...]
  }
]
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    return inter / (area_a + area_b - inter + 1e-9)


def extract_tracks(data: Any) -> List[Dict[str, Any]]:
    tracks = []

    def walk(obj):
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


def build_json_frame_index(tracks):
    """
    frame_to_items[frame] = [(track_index, track_id, bbox), ...]
    """
    frame_to_items = defaultdict(list)

    for track_idx, trk in enumerate(tracks):
        tid, timestamps, boxes = get_track_fields(trk)

        if tid is None:
            continue

        for ts, box in zip(timestamps, boxes):
            if isinstance(box, list) and len(box) == 4:
                x1, y1, x2, y2 = map(float, box)
                if x2 > x1 and y2 > y1:
                    frame_to_items[int(ts)].append((track_idx, tid, [x1, y1, x2, y2]))

    return frame_to_items


def main():
    ap = argparse.ArgumentParser(
        description="Create car-only tracking JSON by matching original JSON boxes with YOLO car detections."
    )

    ap.add_argument("--video", required=True, help="Input video, e.g. VID_01.mp4")
    ap.add_argument("--json_in", required=True, help="Original tracking JSON")
    ap.add_argument("--json_out", required=True, help="Output car-only JSON")

    ap.add_argument("--model", default="yolov8s-seg.pt", help="YOLO model")
    ap.add_argument("--classes", nargs="+", default=["car"], help="Classes to keep, default: car")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou_th", type=float, default=0.30)

    ap.add_argument("--device", default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--half", action="store_true")

    args = ap.parse_args()

    # -------------------------
    # Load original JSON
    # -------------------------
    with open(args.json_in, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracks = extract_tracks(data)

    if not tracks:
        raise RuntimeError("No tracks found. Expected JSON with timestamps/frames + boxes/bboxes.")

    frame_to_items = build_json_frame_index(tracks)

    print(f"[JSON] Tracks loaded: {len(tracks)}")
    print(f"[JSON] Frames with boxes: {len(frame_to_items)}")

    # -------------------------
    # Open video + YOLO
    # -------------------------
    print(f"[YOLO] Loading model: {args.model}")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    # track_idx -> kept timestamps / boxes
    kept_ts = defaultdict(list)
    kept_boxes = defaultdict(list)

    allowed_classes = set(args.classes)

    frame_idx = 0
    bar = tqdm(total=total_frames, unit="f", dynamic_ncols=True) if tqdm is not None else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        json_items = frame_to_items.get(frame_idx, [])

        # Chỉ chạy YOLO nếu frame này có bbox trong JSON
        if json_items:
            predict_kwargs = {
                "conf": args.conf,
                "verbose": False,
            }

            if args.device is not None:
                predict_kwargs["device"] = args.device

            if args.imgsz is not None:
                predict_kwargs["imgsz"] = args.imgsz

            if args.half:
                predict_kwargs["half"] = True

            res = model.predict(frame, **predict_kwargs)[0]

            car_boxes = []

            if res.boxes is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls_ids = res.boxes.cls.cpu().numpy().astype(int)

                for box, cls_id in zip(xyxy, cls_ids):
                    name = res.names[int(cls_id)]

                    if name not in allowed_classes:
                        continue

                    car_boxes.append([float(v) for v in box.tolist()])

            # Match từng bbox JSON với detection car tốt nhất
            for track_idx, tid, json_box in json_items:
                best_iou = 0.0

                for det_box in car_boxes:
                    score = iou_xyxy(json_box, det_box)
                    if score > best_iou:
                        best_iou = score

                if best_iou >= args.iou_th:
                    kept_ts[track_idx].append(frame_idx)
                    kept_boxes[track_idx].append([round(float(v), 1) for v in json_box])

        if bar is not None:
            bar.update(1)

        frame_idx += 1

    cap.release()

    if bar is not None:
        bar.close()

    # -------------------------
    # Build output JSON
    # -------------------------
    out_tracks = []

    for track_idx, trk in enumerate(tracks):
        if track_idx not in kept_ts:
            continue

        if len(kept_ts[track_idx]) == 0:
            continue

        tid, _, _ = get_track_fields(trk)

        new_trk = {
            "id": tid,
            "timestamps": kept_ts[track_idx],
            "boxes": kept_boxes[track_idx],
            "class": "car",
        }

        out_tracks.append(new_trk)

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(out_tracks, f, ensure_ascii=False, indent=2)

    n_boxes_in = sum(len(get_track_fields(t)[1]) for t in tracks)
    n_boxes_out = sum(len(t["timestamps"]) for t in out_tracks)

    print("Done.")
    print(f"Original tracks: {len(tracks)}")
    print(f"Car-only tracks: {len(out_tracks)}")
    print(f"Original boxes : {n_boxes_in}")
    print(f"Kept car boxes : {n_boxes_out}")
    print(f"Saved: {args.json_out}")


if __name__ == "__main__":
    main()