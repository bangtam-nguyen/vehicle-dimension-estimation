import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


ORDER_LABELS = ["BOTTOM-LEFT", "TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT"]


@dataclass
class DisplayTransform:
    scale: float
    pad_x: int
    pad_y: int
    disp_w: int
    disp_h: int
    orig_w: int
    orig_h: int

    def disp_to_orig(self, x: int, y: int) -> Tuple[float, float]:
        # Convert display coords -> original frame coords
        xo = (x - self.pad_x) / self.scale
        yo = (y - self.pad_y) / self.scale
        xo = float(np.clip(xo, 0, self.orig_w - 1))
        yo = float(np.clip(yo, 0, self.orig_h - 1))
        return xo, yo


def make_display_image(frame_bgr: np.ndarray, max_w: int, max_h: int) -> Tuple[np.ndarray, DisplayTransform]:
    h, w = frame_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    disp_w = int(round(w * scale))
    disp_h = int(round(h * scale))

    resized = cv2.resize(frame_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    # Put resized frame on a black canvas (no pad needed normally, but kept for safety)
    canvas = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
    pad_x, pad_y = 0, 0
    canvas[pad_y:pad_y + disp_h, pad_x:pad_x + disp_w] = resized

    tf = DisplayTransform(
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        disp_w=disp_w,
        disp_h=disp_h,
        orig_w=w,
        orig_h=h,
    )
    return canvas, tf


def draw_ui(img: np.ndarray, points_disp: List[Tuple[int, int]], next_idx: int) -> np.ndarray:
    out = img.copy()

    # Draw selected points
    for i, (x, y) in enumerate(points_disp):
        cv2.circle(out, (x, y), 6, (0, 255, 255), -1)
        cv2.putText(out, str(i + 1), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # Instructions
    lines = [
        "Click 4 points in order: 1) BL  2) TL  3) TR  4) BR",
        f"Next: {ORDER_LABELS[next_idx]}   ({next_idx + 1}/4)" if next_idx < 4 else "Done! Press S to save, or Q to quit.",
        "Keys: U=undo last | R=reset | S=save | Q=quit",
    ]
    y0 = 28
    for line in lines:
        cv2.putText(out, line, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y0 += 26

    return out


def select_4_points(frame_bgr: np.ndarray, max_w: int, max_h: int, window_name: str = "Select 4 points") -> List[Tuple[float, float]]:
    disp, tf = make_display_image(frame_bgr, max_w, max_h)

    points_disp: List[Tuple[int, int]] = []
    points_orig: List[Tuple[float, float]] = []

    state = {"refresh": True}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(points_disp) >= 4:
            return

        xo, yo = tf.disp_to_orig(x, y)
        points_disp.append((x, y))
        points_orig.append((xo, yo))
        state["refresh"] = True

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        if state["refresh"]:
            ui = draw_ui(disp, points_disp, next_idx=len(points_disp))
            cv2.imshow(window_name, ui)
            state["refresh"] = False

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == ord("Q") or key == 27:  # ESC
            cv2.destroyWindow(window_name)
            raise SystemExit("User quit without saving.")
        if key == ord("u") or key == ord("U"):
            if points_disp:
                points_disp.pop()
                points_orig.pop()
                state["refresh"] = True
        if key == ord("r") or key == ord("R"):
            points_disp.clear()
            points_orig.clear()
            state["refresh"] = True
        if key == ord("s") or key == ord("S"):
            if len(points_orig) != 4:
                print(f"[WARN] Need 4 points before saving. Currently: {len(points_orig)}")
                continue
            cv2.destroyWindow(window_name)
            return points_orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--out", default=None, help="Optional output json path")
    ap.add_argument("--max_w", type=int, default=1280, help="Max display width")
    ap.add_argument("--max_h", type=int, default=720, help="Max display height")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit("Failed to read first frame from video.")

    points = select_4_points(frame, args.max_w, args.max_h)

    # Format giống config của bạn (float)
    homography_src_points = [[float(f"{x:.1f}"), float(f"{y:.1f}")] for x, y in points]

    payload = {
        "HOMOGRAPHY_SRC_POINTS": homography_src_points,
        "ROI_POLYGON_SRC": homography_src_points,  # default: same as homography trapezoid
        "ORDER": ["bottom-left", "top-left", "top-right", "bottom-right"],
        "NOTE": "Coordinates are in original frame pixels (x right, y down).",
    }

    print("\n=== Copy-paste into your config ===")
    print("HOMOGRAPHY_SRC_POINTS = [")
    for p in payload["HOMOGRAPHY_SRC_POINTS"]:
        print(f"    [{p[0]}, {p[1]}],")
    print("]\n")
    print("ROI_POLYGON_SRC = HOMOGRAPHY_SRC_POINTS\n")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[OK] Saved: {args.out}")


if __name__ == "__main__":
    main()