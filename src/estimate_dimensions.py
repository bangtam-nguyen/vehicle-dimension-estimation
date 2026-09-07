#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation-based BEV footprint overlay & measurement with per-ID summary.
Modified from the old code:
- focus on car by default
- stabilize footprint with bottom-band instead of only bottom contour
- trim head/tail in image x and BEV eX/eY
- width measured from robust spread along eX (less sensitive to skew)
- reject skewed frames using min_align
- tunable car length fusion and car width/length gates
"""

import argparse
import json
import time
from typing import List, Tuple, Dict, Any, Optional, Deque
from collections import deque, defaultdict
from pathlib import Path

import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO

try:
    import torch
except Exception:
    torch = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# -------------------------- Geometry utils --------------------------
def parse_point(s: str) -> Tuple[float, float]:
    x, y = s.split(",")
    return float(x), float(y)


def estimate_homography(src_pts: List[Tuple[float, float]], dst_pts: List[Tuple[float, float]]) -> np.ndarray:
    if len(src_pts) != 4 or len(dst_pts) != 4:
        raise ValueError("Need exactly 4 source and 4 destination points")
    A = []
    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        A.append([x, y, 1, 0, 0, 0, -xp * x, -xp * y, -xp])
        A.append([0, 0, 0, x, y, 1, -yp * x, -yp * y, -yp])
    A = np.asarray(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3, 3)
    if H[2, 2] != 0:
        H = H / H[2, 2]
    return H


def warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
    warped_h = (H @ pts_h.T).T
    warped = warped_h[:, :2] / warped_h[:, 2:3]
    return warped


def invert_homography(H: np.ndarray) -> np.ndarray:
    Hi = np.linalg.inv(H)
    Hi = Hi / Hi[2, 2]
    return Hi


def min_area_rect(points_xy: np.ndarray):
    pts = points_xy.astype(np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    (w, h) = rect[1]
    width, length = (w, h) if w <= h else (h, w)
    return rect, box, width, length


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + 1e-9)


def get_axes_from_dst(dst_pts: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Derive BEV axes: eY = road direction, eX = perpendicular."""
    dst = np.array(dst_pts, dtype=np.float64)
    xs = dst[:, 0]

    left_idx = np.argsort(xs)[:2]
    right_idx = np.argsort(xs)[-2:]

    left = dst[left_idx][np.argsort(dst[left_idx][:, 1])]
    right = dst[right_idx][np.argsort(dst[right_idx][:, 1])]

    v_left = left[1] - left[0]
    v_right = right[1] - right[0]

    eY = _unit((v_left + v_right) / 2.0)
    if eY[1] < 0:
        eY = -eY
    eX = np.array([eY[1], -eY[0]], dtype=np.float64)
    return eX, eY


def oriented_box_from_axes(x_center: float, y_center: float, width_px: float, length_px: float,
                           eX: np.ndarray, eY: np.ndarray) -> np.ndarray:
    """Build a BEV box aligned with eX/eY from center and dimensions."""
    hw = 0.5 * float(width_px)
    hl = 0.5 * float(length_px)
    coords = [
        (x_center - hw, y_center - hl),
        (x_center + hw, y_center - hl),
        (x_center + hw, y_center + hl),
        (x_center - hw, y_center + hl),
    ]
    pts = [eX * x + eY * y for x, y in coords]
    return np.asarray(pts, dtype=np.float32)


def robust_span_from_projection(proj: np.ndarray, low_q: float, high_q: float) -> Tuple[float, float, float, float]:
    """Return low, high, median center, span from percentiles on a 1D projection."""
    proj = np.asarray(proj, dtype=np.float64)
    proj = proj[np.isfinite(proj)]
    if proj.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    if proj.size == 1:
        v = float(proj[0])
        return v, v, v, 0.0

    low_q = float(np.clip(low_q, 0.0, 100.0))
    high_q = float(np.clip(high_q, 0.0, 100.0))
    if high_q <= low_q:
        low_q, high_q = 5.0, 95.0

    lo = float(np.percentile(proj, low_q))
    hi = float(np.percentile(proj, high_q))
    ctr = float(np.median(proj))
    span = max(0.0, hi - lo)
    return lo, hi, ctr, span



def pca_axes_from_points(points_xy: np.ndarray, ref_long_axis: Optional[np.ndarray] = None):
    """
    PCA on 2D BEV points.
    Returns centroid, lateral axis, longitudinal axis, eigenvalues, eig_ratio.
    The longitudinal axis is aligned with the dominant eigenvector and optionally
    flipped to be consistent with ref_long_axis.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < 2:
        return None, None, None, None, 0.0

    ctr = np.mean(pts, axis=0)
    centered = pts - ctr
    cov = np.cov(centered.T, bias=True)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    e_long = _unit(eigvecs[:, 0])
    if ref_long_axis is not None and float(np.dot(e_long, ref_long_axis)) < 0:
        e_long = -e_long
    e_lat = np.array([-e_long[1], e_long[0]], dtype=np.float64)

    eig_ratio = float(eigvals[0] / (eigvals[1] + 1e-9)) if eigvals.shape[0] >= 2 else 0.0
    return ctr, e_lat, e_long, eigvals, eig_ratio


# -------------------------- Assoc & smoothing --------------------------
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


def online_median_push(deq: Deque[float], x: float, window: int) -> float:
    deq.append(x)
    while len(deq) > window:
        deq.popleft()
    arr = np.fromiter(deq, dtype=np.float64)
    return float(np.median(arr))


# -------------------------- Drawing helpers --------------------------
def draw_label(img, text: str, org: Tuple[int, int], color=(255, 255, 255), bg=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    x = max(0, min(img.shape[1] - tw - 8, x))
    y = max(th + 8, min(img.shape[0] - 4, y))
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), bg, -1)
    cv2.putText(img, text, (x + 3, y - 3), font, scale, color, thick, cv2.LINE_AA)


def draw_quad(img, pts: List[Tuple[int, int]], color=(0, 255, 255), thick=2):
    ptsi = [(int(px), int(py)) for px, py in pts]
    for i in range(4):
        p1 = ptsi[i]
        p2 = ptsi[(i + 1) % 4]
        cv2.line(img, p1, p2, color, thick, cv2.LINE_AA)


def color_for_id(tid: Any) -> Tuple[int, int, int]:
    h = hash(str(tid))
    return (50 + (h & 0x7F), 50 + ((h >> 7) & 0x7F), 50 + ((h >> 14) & 0x7F))


# -------------------------- Contour / footprint extraction --------------------------
def contact_points_from_mask(mask: np.ndarray, smooth_k: int = 5) -> Optional[np.ndarray]:
    """Old bottom contour, kept for visualization."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    pts = []
    for x in np.unique(xs):
        ys_x = ys[xs == x]
        yb = int(np.max(ys_x))
        pts.append((float(x), float(yb)))

    if len(pts) >= 5:
        arr = np.array(pts, dtype=np.float32)
        ys_sm = arr[:, 1].copy()
        k = max(3, int(smooth_k))
        if k % 2 == 0:
            k += 1
        if len(ys_sm) >= k:
            pad = k // 2
            padded = np.r_[np.repeat(ys_sm[0], pad), ys_sm, np.repeat(ys_sm[-1], pad)]
            ys_s = np.convolve(padded, np.ones(k) / k, mode='valid')
            arr[:, 1] = ys_s
        return arr

    return np.array(pts, dtype=np.float32)


def bottom_band_points_from_mask(mask: np.ndarray,
                                 band_ratio: float = 0.20,
                                 trim_x_ratio: float = 0.04,
                                 smooth_k: int = 7) -> Optional[np.ndarray]:
    """
    NEW:
    Build a more stable footprint from the lower band of the mask.
    - trim a bit at left/right ends in image x
    - smooth the bottom envelope
    - keep a vertical band above the smoothed bottom
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    x_unique = np.unique(xs)
    if len(x_unique) < 4:
        return None

    trim_x_ratio = float(np.clip(trim_x_ratio, 0.0, 0.20))
    x_lo = np.percentile(x_unique, trim_x_ratio * 100.0)
    x_hi = np.percentile(x_unique, 100.0 - trim_x_ratio * 100.0)

    keep_trim = (xs >= x_lo) & (xs <= x_hi)
    xs_t = xs[keep_trim]
    ys_t = ys[keep_trim]
    if len(xs_t) == 0:
        return None

    y_min = int(np.min(ys_t))
    y_max = int(np.max(ys_t))
    h = max(1, y_max - y_min + 1)

    x_vals = np.unique(xs_t)
    bottom = np.array([np.max(ys_t[xs_t == x]) for x in x_vals], dtype=np.float32)

    k = max(3, int(smooth_k))
    if k % 2 == 0:
        k += 1
    if len(bottom) >= k:
        pad = k // 2
        padded = np.r_[np.repeat(bottom[0], pad), bottom, np.repeat(bottom[-1], pad)]
        bottom_s = np.convolve(padded, np.ones(k) / k, mode='valid')
    else:
        bottom_s = bottom

    band_px = max(2, int(round(float(np.clip(band_ratio, 0.05, 0.40)) * h)))

    pts = []
    for x, yb in zip(x_vals, bottom_s):
        ys_x = ys_t[xs_t == x]
        y_low = int(round(yb)) - band_px + 1
        ys_keep = ys_x[ys_x >= y_low]
        for y in ys_keep:
            pts.append((float(x), float(y)))

    if len(pts) < 10:
        return None
    return np.asarray(pts, dtype=np.float32)


def lower_body_points_from_mask(mask: np.ndarray,
                                top_ratio: float = 0.45,
                                trim_x_ratio: float = 0.02) -> Optional[np.ndarray]:
    """
    Approximate a body region for length estimation.
    Keep only the lower body part of the mask instead of the very thin bottom band.
    This avoids treating the lower-band thickness as vehicle length.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    x_unique = np.unique(xs)
    if len(x_unique) < 4:
        return None

    trim_x_ratio = float(np.clip(trim_x_ratio, 0.0, 0.20))
    x_lo = np.percentile(x_unique, trim_x_ratio * 100.0)
    x_hi = np.percentile(x_unique, 100.0 - trim_x_ratio * 100.0)

    keep_x = (xs >= x_lo) & (xs <= x_hi)
    xs_t = xs[keep_x]
    ys_t = ys[keep_x]
    if len(xs_t) == 0:
        return None

    y_min = int(np.min(ys_t))
    y_max = int(np.max(ys_t))
    h = max(1, y_max - y_min + 1)
    y_cut = y_min + float(np.clip(top_ratio, 0.10, 0.80)) * h

    keep_y = ys_t >= y_cut
    xs_b = xs_t[keep_y]
    ys_b = ys_t[keep_y]
    if len(xs_b) < 10:
        return None

    pts = np.stack([xs_b.astype(np.float32), ys_b.astype(np.float32)], axis=1)
    return pts


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


# -------------------------- Robust summary helpers --------------------------
def freedman_diaconis_bin_width(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return None
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr <= 0:
        return None
    h = 2 * iqr * (len(x) ** (-1 / 3))
    return h


def hist_mode(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    h = freedman_diaconis_bin_width(x)
    if not h or h <= 0:
        bins = int(np.sqrt(len(x)))
    else:
        rng = (x.min(), x.max())
        bins = max(1, int(np.ceil((rng[1] - rng[0]) / h)))
    cnts, edges = np.histogram(x, bins=bins)
    if len(cnts) == 0:
        return np.nan
    k = int(np.argmax(cnts))
    return (edges[k] + edges[k + 1]) / 2.0


def robust_stats(vals):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return dict(n=0, median=np.nan, mad=np.nan, iqr=np.nan,
                    trimmed_mean=np.nan, mode=np.nan)

    med = np.median(v)
    mad = np.median(np.abs(v - med))
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    sigma = 1.4826 * mad if mad > 0 else (iqr / 1.349 if iqr > 0 else 0.0)

    if sigma > 0:
        inlier = np.abs(v - med) <= 2.5 * sigma
        v_in = v[inlier]
    else:
        v_in = v

    if len(v_in):
        lo, hi = np.percentile(v_in, [20, 80])
        v_trim = v_in[(v_in >= lo) & (v_in <= hi)]
        tmean = np.mean(v_trim) if len(v_trim) else np.mean(v_in)
    else:
        tmean = np.nan

    md = hist_mode(v_in) if len(v_in) >= 10 else hist_mode(v)

    return dict(
        n=len(v),
        median=float(med),
        mad=float(mad),
        iqr=float(iqr),
        trimmed_mean=float(tmean) if not np.isnan(tmean) else np.nan,
        mode=float(md) if md is not None else np.nan
    )


def choose_final(stat):
    med = stat["median"]
    tmean = stat["trimmed_mean"]
    md = stat["mode"]
    iqr = stat["iqr"]
    n = stat["n"]

    if n >= 20 and not np.isnan(md) and iqr > 0:
        if abs(md - med) <= max(0.1 * iqr, 0.03):
            return med, "median"
        return md, "mode"

    if not np.isnan(tmean) and abs(tmean - med) <= max(0.1 * iqr, 0.03):
        return med, "median"

    if not np.isnan(tmean):
        return tmean, "trimmed_mean"

    return med, "median"


# -------------------------- Main --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", nargs=4, required=True, help="Four image points x,y")
    ap.add_argument("--dst", nargs=4, required=True, help="Four BEV points x,y")
    ap.add_argument("--model", default="yolov8s-seg.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--classes", nargs="*", default=["car"])
    ap.add_argument("--json", default=None, help="Optional tracking JSON for IDs")

    # smoothing / scale / output
    ap.add_argument("--smooth_window", type=int, default=11)
    ap.add_argument("--scale_m_per_px", type=float, default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--show_bev_inset", action="store_true")
    ap.add_argument("--bev_inset_width", type=int, default=420)
    ap.add_argument("--draw_src_quad", action="store_true")
    ap.add_argument("--draw_image_ground_rect", action="store_true")

    # performance & status
    ap.add_argument("--device", default=None, help="e.g., 'cpu', 'cuda:0'")
    ap.add_argument("--imgsz", type=int, default=None, help="inference resolution, e.g., 640")
    ap.add_argument("--half", action="store_true", help="use FP16 (if device supports)")
    ap.add_argument("--overlay_stats", action="store_true", help="draw fps/frame/eta text on the frame")

    # auto scale
    ap.add_argument("--scale_from_src_pair", nargs=2, metavar=("x1,y1", "x2,y2"),
                    help="image points on the ground whose real-world distance is known")
    ap.add_argument("--scale_pair_m", type=float, help="real distance (meters) of the above pair")

    # per-ID summary
    ap.add_argument("--summary_out", default="width_summary_by_id.xlsx",
                help="output Excel for per-ID width/length summary (set empty to disable)")
    ap.add_argument("--summary_min_frames", type=int, default=6)
    ap.add_argument("--class_range", default=None,
                    help="optional class-specific plausible WIDTH (m), e.g. car:1.4-2.2")

    # segmentation visualization
    ap.add_argument("--draw_masks", action="store_true", help="overlay instance segmentation masks")
    ap.add_argument("--mask_alpha", type=float, default=0.35, help="mask overlay opacity (0..1)")
    ap.add_argument("--draw_mask_edges", action="store_true", help="draw mask contours/edges")

    # ROI filter
    ap.add_argument("--roi", nargs="+", help="Polygon ROI in image coordinates, format: x1,y1 x2,y2 ...")
    ap.add_argument("--roi_bev", nargs="+", help="Polygon ROI in BEV coordinates, format: x1,y1 x2,y2 ...")

    # output organization
    ap.add_argument("--out_dir", default=None, help="root directory to place all outputs in a run folder")
    ap.add_argument("--run_label", default=None, help="optional label appended to run folder name")
    ap.add_argument("--save_last10_per_id", action="store_true",
                    help="save the last 10 annotated frames for each track ID as separate videos")

    # ---------------- NEW tuning params ----------------
    ap.add_argument("--band_ratio", type=float, default=0.20,
                    help="lower-band height ratio of mask used for footprint (car)")
    ap.add_argument("--trim_x_ratio", type=float, default=0.04,
                    help="trim left/right ends in image x before building footprint band")
    ap.add_argument("--contour_smooth_k", type=int, default=7,
                    help="smoothing kernel for bottom contour / band envelope")
    ap.add_argument("--trim_ex_low", type=float, default=2.0,
                    help="lower percentile on BEV eX projection for width")
    ap.add_argument("--trim_ex_high", type=float, default=98.0,
                    help="upper percentile on BEV eX projection for width")
    ap.add_argument("--trim_ey_low", type=float, default=8.0,
                    help="lower percentile on BEV eY projection for length")
    ap.add_argument("--trim_ey_high", type=float, default=92.0,
                    help="upper percentile on BEV eY projection for length")
    ap.add_argument("--min_align", type=float, default=0.92,
                    help="reject car frames if minAreaRect long side is too skewed vs road axis")
    ap.add_argument("--car_len_mask_w", type=float, default=1.35,
                    help="weight of robust eY length for car")
    ap.add_argument("--car_len_mar_w", type=float, default=0.55,
                    help="weight of minAreaRect long side for car")
    ap.add_argument("--car_width_gate_min", type=float, default=None,
                    help="optional reject car frames narrower than this (m); omit to disable")
    ap.add_argument("--car_width_gate_max", type=float, default=None,
                    help="optional reject car frames wider than this (m); omit to disable")
    ap.add_argument("--car_length_clip_min", type=float, default=None,
                    help="optional clip min car length after estimation (m); omit to disable")
    ap.add_argument("--car_length_clip_max", type=float, default=None,
                    help="optional clip max car length after estimation (m); omit to disable")
    ap.add_argument("--car_length_mode", default="lower_body_span", choices=["lower_body_span", "legacy_fusion"],
                    help="car length estimation mode")
    ap.add_argument("--car_length_body_top_ratio", type=float, default=0.45,
                    help="for lower_body_span: discard this top fraction of the car mask, keep the lower body region")
    ap.add_argument("--car_length_body_trim_x_ratio", type=float, default=0.02,
                    help="for lower_body_span: trim left/right ends before estimating car length")
    ap.add_argument("--car_anchor_mode", default="front", choices=["center", "front"],
                    help="how to anchor car BDB on BEV: center uses y_ctr, front anchors box backward from the front edge")
    ap.add_argument("--car_front_anchor_bias", type=float, default=0.10,
                    help="extra backward shift from the front anchor as a fraction of box length")
    ap.add_argument("--car_vis_length_mode", default="min_of_both", choices=["measured", "footprint", "min_of_both"],
                    help="length used only for drawing the BDB: measured=use measured car length, footprint=use bottom-footprint span, min_of_both=prevent BDB from extending past footprint")
    ap.add_argument("--measurement_mode", default="pca_hybrid", choices=["road_axes", "pca_only", "pca_hybrid"],
                    help="road_axes=old behavior, pca_only=measure on PCA axes, pca_hybrid=PCA primary with road-axis fallback")
    ap.add_argument("--pca_width_blend", type=float, default=0.70,
                    help="for pca_hybrid: blend weight for PCA width versus road-axis width")
    ap.add_argument("--pca_length_blend", type=float, default=0.85,
                    help="for pca_hybrid: blend weight for PCA length versus road-axis length")
    ap.add_argument("--pca_min_eig_ratio", type=float, default=1.05,
                    help="reject or fallback when the footprint is too isotropic for stable PCA")
    # ---------------------------------------------------

    args = ap.parse_args()

    roi_poly = None
    if args.roi:
        roi_poly = np.array([parse_point(s) for s in args.roi], dtype=np.int32)
        print(f"[ROI] Polygon: {roi_poly.tolist()}")

    roi_bev_poly = None
    if args.roi_bev:
        roi_bev_poly = np.array([parse_point(s) for s in args.roi_bev], dtype=np.int32)
        print(f"[ROI_BEV] Polygon (BEV): {roi_bev_poly.tolist()}")

    # ----- Prepare run folder & resolve paths -----
    run_dir = None
    if args.out_dir:
        ts = time.strftime("%Y%m%d-%H%M%S")
        label = args.run_label or Path(args.video).stem
        run_dir = Path(args.out_dir) / f"{ts}_{label}"
        run_dir.mkdir(parents=True, exist_ok=True)

        def in_run_dir(p: Optional[str], default_name: str) -> Optional[str]:
            if p is None or p == "":
                return str(run_dir / default_name)
            return str(run_dir / Path(p).name)

        video_out_path = in_run_dir(args.out, "annotated.mp4")
        csv_path = in_run_dir(args.csv, "bev_dims.xlsx")
        summary_path = in_run_dir(args.summary_out, "width_summary_by_id.xlsx") if args.summary_out else None
    else:
        video_out_path = args.out
        csv_path = args.csv or "bev_dims.xlsx"
        summary_path = args.summary_out

    # ----- Load tracker JSON if provided -----
    bboxes_by_frame: Dict[int, List[Tuple[Any, Tuple[float, float, float, float]]]] = {}
    if args.json:
        with open(args.json, "r", encoding="utf-8") as f:
            data = json.load(f)
        tracks = extract_tracks(data)
        if not tracks:
            raise RuntimeError("No tracks found in JSON. Expected keys like 'timestamps'+'boxes'.")

        for trk in tracks:
            tid, timestamps, boxes = get_track_fields(trk)
            if tid is None:
                tid = trk.get("id", f"t{len(bboxes_by_frame)}")
            for ts_f, bbox in zip(timestamps, boxes):
                if isinstance(bbox, list) and len(bbox) == 4:
                    bboxes_by_frame.setdefault(int(ts_f), []).append((tid, tuple(map(float, bbox))))

    # ----- Homography & axes -----
    src_pts = [parse_point(s) for s in args.src]
    dst_pts = [parse_point(s) for s in args.dst]
    Hmat = estimate_homography(src_pts, dst_pts)
    Hinv = invert_homography(Hmat)
    eX, eY = get_axes_from_dst(dst_pts)

    # ----- Auto scale from known ground pair -----
    def _pp(s):
        x, y = s.split(",")
        return float(x), float(y)

    if args.scale_m_per_px is None and args.scale_from_src_pair and args.scale_pair_m:
        p1 = _pp(args.scale_from_src_pair[0])
        p2 = _pp(args.scale_from_src_pair[1])
        bev = warp_points(Hmat, np.array([[p1[0], p1[1]], [p2[0], p2[1]]], dtype=np.float64))
        d = bev[1] - bev[0]
        dpx = float((d ** 2).sum() ** 0.5)
        args.scale_m_per_px = args.scale_pair_m / max(dpx, 1e-9)
        print(f"[scale] auto m/px = {args.scale_m_per_px:.6f}  (pair span {dpx:.3f} px)")

    # ----- BEV inset geometry -----
    if args.show_bev_inset:
        dst_arr = np.array(dst_pts, dtype=np.float64)
        minx, miny = dst_arr[:, 0].min(), dst_arr[:, 1].min()
        maxx, maxy = dst_arr[:, 0].max(), dst_arr[:, 1].max()
        bev_w = maxx - minx
        bev_h = maxy - miny
        scale_inset = args.bev_inset_width / max(bev_w, 1e-6)
        inset_w = int(args.bev_inset_width)
        inset_h = int(round(bev_h * scale_inset))

    # ----- Model -----
    print("Loading YOLO model:", args.model)
    model = YOLO(args.model)

    # ----- Video IO -----
    print("Opening video:", args.video)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out_path, fourcc, fps_src, (W, Hh))
    print(f"Output video: {video_out_path}  ({W}x{Hh}@{fps_src:.2f}fps)")

    # ----- Per-track smoothing -----
    deq_w: Dict[Any, Deque[float]] = defaultdict(lambda: deque(maxlen=max(1, args.smooth_window)))
    deq_l: Dict[Any, Deque[float]] = defaultdict(lambda: deque(maxlen=max(1, args.smooth_window)))
    last_video_frames: Deque[np.ndarray] = deque(maxlen=10)

    # ----- CSV rows -----
    csv_rows = []

    # ----- Per-ID last 10 annotated frames -----
    track_last_frames: Dict[Any, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=10))

    # ----- Source quad for reference -----
    src_quad = [src_pts[0], src_pts[1], src_pts[3], src_pts[2]]

    last_video_frames = deque(maxlen=10)

    # ----- Status -----
    frame_idx = 0
    t0 = time.perf_counter()
    fps_ema = None
    bar = None
    if tqdm is not None:
        total = total_frames if total_frames is not None else None
        bar = tqdm(total=total, unit="f", dynamic_ncols=True)
        bar.set_description("Processing")

    # ----- Save meta -----
    if run_dir:
        meta = {
            "video": str(Path(args.video).resolve()),
            "model": args.model,
            "classes": args.classes,
            "conf": args.conf,
            "src": args.src,
            "dst": args.dst,
            "scale_m_per_px": args.scale_m_per_px,
            "scale_from_src_pair": args.scale_from_src_pair,
            "scale_pair_m": args.scale_pair_m,
            "device": args.device,
            "imgsz": args.imgsz,
            "half": args.half,
            "draw_masks": args.draw_masks,
            "mask_alpha": args.mask_alpha,
            "draw_mask_edges": args.draw_mask_edges,
            "json": args.json,
            "csv": str(csv_path) if csv_path else None,
            "summary_out": str(summary_path) if summary_path else None,
            "output_video": str(video_out_path),
            "band_ratio": args.band_ratio,
            "trim_x_ratio": args.trim_x_ratio,
            "trim_ex": [args.trim_ex_low, args.trim_ex_high],
            "trim_ey": [args.trim_ey_low, args.trim_ey_high],
            "min_align": args.min_align,
            "car_len_weights": [args.car_len_mask_w, args.car_len_mar_w],
            "car_length_mode": args.car_length_mode,
            "car_length_body_top_ratio": args.car_length_body_top_ratio,
            "car_length_body_trim_x_ratio": args.car_length_body_trim_x_ratio,
            "car_anchor_mode": args.car_anchor_mode,
            "car_front_anchor_bias": args.car_front_anchor_bias,
            "car_vis_length_mode": args.car_vis_length_mode,
            "measurement_mode": args.measurement_mode,
            "pca_width_blend": args.pca_width_blend,
            "pca_length_blend": args.pca_length_blend,
            "pca_min_eig_ratio": args.pca_min_eig_ratio,
        }
        with open(Path(run_dir) / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # ===================== Main loop =====================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.draw_src_quad:
            draw_quad(frame, src_quad, color=(0, 255, 255), thick=2)

        predict_kwargs = {"conf": args.conf, "verbose": False}
        if args.device is not None:
            predict_kwargs["device"] = args.device
        if args.imgsz is not None:
            predict_kwargs["imgsz"] = args.imgsz
        if args.half:
            predict_kwargs["half"] = True

        res = model.predict(frame, **predict_kwargs)[0]

        # ----- Candidates from detection/segmentation -----
        cands = []
        if res.boxes is not None:
            bxyxy = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            names = [res.names[int(c)] for c in clss]
            masks = res.masks.data.cpu().numpy() if res.masks is not None else None

            for i in range(len(bxyxy)):
                name = names[i]
                if args.classes and name not in set(args.classes):
                    continue
                if masks is None:
                    continue

                maskf = masks[i]
                if maskf.dtype != np.uint8:
                    maskf = (maskf > 0.5).astype(np.uint8)
                if maskf.shape != (Hh, W):
                    maskf = cv2.resize(maskf, (W, Hh), interpolation=cv2.INTER_NEAREST)

                cands.append((bxyxy[i].tolist(), maskf, name))

        # ----- Assign IDs by IoU with tracking JSON -----
        items = bboxes_by_frame.get(frame_idx, []) if args.json else []
        assigned = [False] * len(cands)
        assignments = []

        if len(items) and len(cands):
            for tid, tbox in items:
                best_iou, best_j = 0.0, -1
                for j, (box, mask, name) in enumerate(cands):
                    if assigned[j]:
                        continue
                    iou = iou_xyxy(box, tbox)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_j >= 0 and best_iou >= 0.3:
                    assignments.append((tid, best_j))
                    assigned[best_j] = True

        # Unassigned -> detX when no JSON
        next_det_id = 0
        if not args.json:
            for j, (box, mask, name) in enumerate(cands):
                if not assigned[j]:
                    assignments.append((f"det{next_det_id}", j))
                    next_det_id += 1

        # ----- BEV inset canvas -----
        if args.show_bev_inset:
            bev_inset = np.zeros((inset_h, inset_w, 3), dtype=np.uint8)
            cv2.rectangle(bev_inset, (0, 0), (inset_w - 1, inset_h - 1), (100, 100, 100), 1)
            cv2.putText(bev_inset, "BEV footprint", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # ----- Process instances -----
        tids_in_frame = set()
        for tid, j in assignments:
            box, mask, name = cands[j]
            x1, y1, x2, y2 = map(int, box)
            col = color_for_id(tid)

            inside_roi = True
            if roi_poly is not None:
                cx = (x1 + x2) / 2.0
                cy = y2
                inside_roi = cv2.pointPolygonTest(roi_poly, (cx, cy), False) >= 0
            if not inside_roi:
                continue


            if args.draw_masks:
                m = mask.astype(bool)
                a = float(np.clip(args.mask_alpha, 0.0, 1.0))
                frame[m] = (frame[m] * (1.0 - a) + a * np.array(col, dtype=np.float32)).astype(np.uint8)

            if args.draw_mask_edges:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, col, 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

            sub = mask[y1:y2, x1:x2].copy()
            if sub.size == 0:
                continue

            # old contour for visual debug
            vis_pts = contact_points_from_mask(sub, smooth_k=args.contour_smooth_k)
            if vis_pts is not None and len(vis_pts) >= 2:
                vis_pts[:, 0] += x1
                vis_pts[:, 1] += y1
                for k in range(1, len(vis_pts)):
                    cv2.line(
                        frame,
                        (int(vis_pts[k - 1, 0]), int(vis_pts[k - 1, 1])),
                        (int(vis_pts[k, 0]), int(vis_pts[k, 1])),
                        (0, 255, 255), 2, cv2.LINE_AA
                    )

            # NEW: more stable measurement footprint
            if name == "car":
                measure_pts = bottom_band_points_from_mask(
                    sub,
                    band_ratio=args.band_ratio,
                    trim_x_ratio=args.trim_x_ratio,
                    smooth_k=args.contour_smooth_k
                )
                if measure_pts is None:
                    # fallback
                    measure_pts = contact_points_from_mask(sub, smooth_k=args.contour_smooth_k)
            else:
                measure_pts = contact_points_from_mask(sub, smooth_k=args.contour_smooth_k)

            if measure_pts is None or len(measure_pts) < 2:
                continue

            measure_pts = measure_pts.astype(np.float32)
            measure_pts[:, 0] += x1
            measure_pts[:, 1] += y1

            bev_pts = warp_points(Hmat, measure_pts.astype(np.float64))

            length_pts_img = None
            bev_len_pts = None
            if name == "car" and args.car_length_mode == "lower_body_span":
                length_pts_img = lower_body_points_from_mask(
                    sub,
                    top_ratio=args.car_length_body_top_ratio,
                    trim_x_ratio=args.car_length_body_trim_x_ratio
                )
                if length_pts_img is not None and len(length_pts_img) >= 10:
                    length_pts_img = length_pts_img.astype(np.float32)
                    length_pts_img[:, 0] += x1
                    length_pts_img[:, 1] += y1
                    bev_len_pts = warp_points(Hmat, length_pts_img.astype(np.float64))

            inside_roi_bev = True
            if roi_bev_poly is not None:
                cx_bev = float(np.mean(bev_pts[:, 0]))
                cy_bev = float(np.mean(bev_pts[:, 1]))
                inside_roi_bev = cv2.pointPolygonTest(roi_bev_poly, (cx_bev, cy_bev), False) >= 0
            if not inside_roi_bev:
                continue

            # mAR used mainly for skew check and optional fusion
            rect_mar, box_bev_mar, width_px_mar, length_px_mar = min_area_rect(bev_pts.astype(np.float32))
            L_mAR_px = float(max(width_px_mar, length_px_mar))

            v1 = box_bev_mar[1] - box_bev_mar[0]
            v2 = box_bev_mar[2] - box_bev_mar[1]
            long_vec = v1 if np.linalg.norm(v1) > np.linalg.norm(v2) else v2
            align = abs(float(np.dot(long_vec, eY)) / (np.linalg.norm(long_vec) + 1e-9))

            # NEW: reject skewed car frames
            if name == "car" and align < args.min_align:
                continue

            # robust width / length from fixed road axes (legacy reference)
            projX = bev_pts @ eX
            projY = bev_pts @ eY

            x_lo, x_hi, x_ctr_road, width_px_road = robust_span_from_projection(
                projX, args.trim_ex_low, args.trim_ex_high
            )
            y_lo, y_hi, y_ctr_road, L_mask_px_road = robust_span_from_projection(
                projY, args.trim_ey_low, args.trim_ey_high
            )

            if not np.isfinite(width_px_road) or width_px_road <= 0:
                continue
            if not np.isfinite(L_mask_px_road) or L_mask_px_road <= 0:
                continue

            # PCA axes computed from the BEV footprint.
            pca_ctr, pca_eX, pca_eY, pca_eigvals, pca_eig_ratio = pca_axes_from_points(bev_pts, ref_long_axis=eY)
            use_pca = (
                pca_ctr is not None and pca_eX is not None and pca_eY is not None
                and np.isfinite(pca_eig_ratio) and pca_eig_ratio >= args.pca_min_eig_ratio
            )

            if use_pca:
                projXP = bev_pts @ pca_eX
                projYP = bev_pts @ pca_eY
                px_lo, px_hi, x_ctr_pca, width_px_pca = robust_span_from_projection(
                    projXP, args.trim_ex_low, args.trim_ex_high
                )
                py_lo, py_hi, y_ctr_pca, L_mask_px_pca = robust_span_from_projection(
                    projYP, args.trim_ey_low, args.trim_ey_high
                )
            else:
                projXP, projYP = projX, projY
                px_lo, px_hi, x_ctr_pca, width_px_pca = x_lo, x_hi, x_ctr_road, width_px_road
                py_lo, py_hi, y_ctr_pca, L_mask_px_pca = y_lo, y_hi, y_ctr_road, L_mask_px_road

            # IQR diagnostics on the active longitudinal axis
            iqrY = 0.0
            if projYP.size >= 5:
                q1, q3 = np.percentile(projYP, [25, 75])
                iqrY = float(q3 - q1)

            if args.measurement_mode == "road_axes":
                meas_eX, meas_eY = eX, eY
                width_px = width_px_road
                x_ctr = x_ctr_road
                y_ctr = y_ctr_road
                y_hi_use = y_hi
                L_mask_px = L_mask_px_road
                measurement_mode_used = "road_axes"
            else:
                meas_eX, meas_eY = pca_eX, pca_eY
                x_ctr = x_ctr_pca
                y_ctr = y_ctr_pca
                y_hi_use = py_hi
                if not use_pca:
                    meas_eX, meas_eY = eX, eY
                    x_ctr = x_ctr_road
                    y_ctr = y_ctr_road
                    y_hi_use = y_hi
                    width_px = width_px_road
                    L_mask_px = L_mask_px_road
                    measurement_mode_used = "road_axes_fallback"
                elif args.measurement_mode == "pca_only":
                    width_px = width_px_pca
                    L_mask_px = L_mask_px_pca
                    measurement_mode_used = "pca_only"
                else:
                    w_blend = float(np.clip(args.pca_width_blend, 0.0, 1.0))
                    l_blend = float(np.clip(args.pca_length_blend, 0.0, 1.0))
                    width_px = w_blend * width_px_pca + (1.0 - w_blend) * width_px_road
                    L_mask_px = l_blend * L_mask_px_pca + (1.0 - l_blend) * L_mask_px_road
                    measurement_mode_used = "pca_hybrid"

            if not np.isfinite(width_px) or width_px <= 0:
                continue
            if not np.isfinite(L_mask_px) or L_mask_px <= 0:
                continue

            # smooth width
            w_s = online_median_push(deq_w[tid], width_px, args.smooth_window)

            scale = args.scale_m_per_px if args.scale_m_per_px is not None else None

            # length estimation on the active longitudinal axis
            if scale is None:
                if name == "car":
                    if args.car_length_mode == "lower_body_span" and bev_len_pts is not None and len(bev_len_pts) >= 10:
                        projY_len = bev_len_pts @ meas_eY
                        _, _, _, L_body_px = robust_span_from_projection(
                            projY_len, args.trim_ey_low, args.trim_ey_high
                        )
                        L_fused_px = float(L_body_px)
                    else:
                        L_fused_px = (
                            args.car_len_mask_w * L_mask_px + args.car_len_mar_w * L_mAR_px
                        ) / (args.car_len_mask_w + args.car_len_mar_w + 1e-9)
                else:
                    bbox_area = max(1.0, float((x2 - x1) * (y2 - y1)))
                    occ = float(mask[y1:y2, x1:x2].sum()) / bbox_area
                    w_mask_len = (0.6 + 0.8 * align) * min(1.0, 0.5 + 0.8 * occ)
                    w_mar_len = (1.6 - 0.8 * align)
                    if iqrY > 12:
                        w_mask_len *= 0.6
                    L_fused_px = (w_mask_len * L_mask_px + w_mar_len * L_mAR_px) / (w_mask_len + w_mar_len + 1e-9)

                L_show_px = online_median_push(deq_l[tid], L_fused_px, args.smooth_window)
                L_show_m = None
            else:
                width_m_raw = float(width_px) * scale

                if name == "car":
                    if args.car_width_gate_min is not None and width_m_raw < args.car_width_gate_min:
                        continue
                    if args.car_width_gate_max is not None and width_m_raw > args.car_width_gate_max:
                        continue

                L_mask_m = L_mask_px * scale
                L_mar_m = L_mAR_px * scale

                if name == "car":
                    if args.car_length_mode == "lower_body_span" and bev_len_pts is not None and len(bev_len_pts) >= 10:
                        projY_len = bev_len_pts @ meas_eY
                        _, _, _, L_body_px = robust_span_from_projection(
                            projY_len, args.trim_ey_low, args.trim_ey_high
                        )
                        L_fused_m = float(L_body_px) * scale
                    else:
                        L_fused_m = (
                            args.car_len_mask_w * L_mask_m + args.car_len_mar_w * L_mar_m
                        ) / (args.car_len_mask_w + args.car_len_mar_w + 1e-9)

                    if args.car_length_clip_min is not None:
                        L_fused_m = max(float(args.car_length_clip_min), float(L_fused_m))
                    if args.car_length_clip_max is not None:
                        L_fused_m = min(float(args.car_length_clip_max), float(L_fused_m))
                else:
                    if abs(L_mask_m - L_mar_m) <= 0.5:
                        L_fused_m = 0.5 * (L_mask_m + L_mar_m)
                    else:
                        bbox_area = max(1.0, float((x2 - x1) * (y2 - y1)))
                        occ = float(mask[y1:y2, x1:x2].sum()) / bbox_area
                        w_mask_len = (0.6 + 0.8 * align) * min(1.0, 0.5 + 0.8 * occ)
                        w_mar_len = (1.6 - 0.8 * align)
                        if iqrY > 12:
                            w_mask_len *= 0.6
                        L_fused_m = (w_mask_len * L_mask_m + w_mar_len * L_mar_m) / (w_mask_len + w_mar_len + 1e-9)

                L_show_m = online_median_push(deq_l[tid], L_fused_m, args.smooth_window)
                L_show_px = None

            # build a stable BEV box aligned to the active measurement axes for visualization
            if scale is None:
                vis_w_px = w_s
                L_measured_vis_px = L_show_px if L_show_px is not None else L_mask_px
            else:
                vis_w_px = w_s
                L_measured_vis_px = (L_show_m / scale) if (L_show_m is not None and scale > 0) else L_mask_px

            # Footprint-based visual span along road axis (from the same points used for the yellow footprint)
            L_footprint_vis_px = max(1.0, float((py_hi - py_lo) if args.measurement_mode != "road_axes" and use_pca else (y_hi - y_lo)))

            if name == "car":
                if args.car_vis_length_mode == "footprint":
                    vis_l_px = L_footprint_vis_px
                elif args.car_vis_length_mode == "min_of_both":
                    vis_l_px = min(float(L_measured_vis_px), L_footprint_vis_px)
                else:
                    vis_l_px = float(L_measured_vis_px)
            else:
                vis_l_px = float(L_measured_vis_px)

            # For frontal car views, the lower-band points usually come from the front bumper area.
            # Using y_ctr centers the box too far toward the vehicle front. So for cars we can anchor
            # the box from the front edge (y_hi) and extend it backward along the road axis.
            if name == "car" and args.car_anchor_mode == "front":
                extra_back = float(np.clip(args.car_front_anchor_bias, 0.0, 0.25))
                y_center_vis = y_hi_use - (0.5 + extra_back) * vis_l_px
            else:
                y_center_vis = y_ctr

            box_bev_stable = oriented_box_from_axes(
                x_center=x_ctr,
                y_center=y_center_vis,
                width_px=vis_w_px,
                length_px=vis_l_px,
                eX=meas_eX,
                eY=meas_eY
            )

            if args.draw_image_ground_rect:
                box_img = warp_points(Hinv, box_bev_stable.astype(np.float64))
                box_img_i = box_img.astype(int)
                cv2.polylines(frame, [box_img_i], isClosed=True, color=(0, 128, 255),
                              thickness=2, lineType=cv2.LINE_AA)

            # label
            if args.scale_m_per_px is not None and L_show_m is not None:
                wm = w_s * args.scale_m_per_px
                text = f"id={tid} w={wm:.2f}m x l={L_show_m:.2f}m ({measurement_mode_used})"
            else:
                text = f"id={tid} w={w_s:.1f}px x l={L_show_px:.1f}px ({measurement_mode_used})"
            draw_label(frame, text, (x1, y1 - 8), color=(255, 255, 255), bg=(0, 0, 0))

            # CSV row
            row = {
                "frame": frame_idx,
                "track_id": tid,
                "cls": name,
                "align": float(align),
                "width_bev_px_raw": float(width_px),
                "width_bev_px_mar": float(width_px_mar),      # old mAR short side
                "length_bev_px_raw": float(L_mask_px),
                "length_bev_px_mar": float(L_mAR_px),         # mAR long side
                "width_bev_px_s": float(w_s),
                "length_mode": str(args.car_length_mode) if name == "car" else "default",
                "measurement_mode": measurement_mode_used,
                "pca_eig_ratio": float(pca_eig_ratio) if np.isfinite(pca_eig_ratio) else float("nan"),
                "width_bev_px_pca": float(width_px_pca) if np.isfinite(width_px_pca) else float("nan"),
                "length_bev_px_pca": float(L_mask_px_pca) if np.isfinite(L_mask_px_pca) else float("nan"),
            }

            if args.scale_m_per_px is not None:
                row.update({
                    "width_m_raw": float(width_px) * args.scale_m_per_px,
                    "width_m_mar": float(width_px_mar) * args.scale_m_per_px,
                    "width_m_s": float(w_s) * args.scale_m_per_px,
                    "length_mask_m": float(L_mask_px) * args.scale_m_per_px,
                    "length_mar_m": float(L_mAR_px) * args.scale_m_per_px,
                    "length_fused_m": float(L_show_m) if L_show_m is not None else float("nan"),
                    "length_m_s": float(L_show_m) if L_show_m is not None else float("nan"),
                    "length_body_m": float(L_show_m) if (name == "car" and args.car_length_mode == "lower_body_span" and L_show_m is not None) else float("nan"),
                })
            else:
                row.update({
                    "length_bev_px_s": float(L_show_px) if L_show_px is not None else float("nan"),
                    "length_body_px": float(L_show_px) if (name == "car" and args.car_length_mode == "lower_body_span" and L_show_px is not None) else float("nan")
                })

            csv_rows.append(row)

            # draw on BEV inset
            if args.show_bev_inset:
                bx = ((box_bev_stable[:, 0] - minx) * scale_inset).astype(int)
                by = ((box_bev_stable[:, 1] - miny) * scale_inset).astype(int)
                poly = np.stack([bx, by], axis=1)
                cv2.polylines(bev_inset, [poly], isClosed=True, color=(0, 255, 255),
                              thickness=2, lineType=cv2.LINE_AA)

        # ----- Paste BEV inset -----
        if args.show_bev_inset:
            h, w = bev_inset.shape[:2]
            x0, y0 = 10, 10
            x1p, y1p = x0 + w, y0 + h
            if x1p <= frame.shape[1] and y1p <= frame.shape[0]:
                roi = frame[y0:y1p, x0:x1p].copy()
                frame[y0:y1p, x0:x1p] = cv2.addWeighted(roi, 0.6, bev_inset, 0.4, 0)

        # ----- Overlay fps/ETA -----
        if args.overlay_stats and tqdm is not None and bar is not None and hasattr(bar, 'format_dict') and bar.format_dict.get('rate'):
            rate = bar.format_dict['rate']
            if rate:
                fps_ema = rate if fps_ema is None else 0.9 * fps_ema + 0.1 * rate
            eta_s = bar.format_dict.get('remaining') if bar.format_dict.get('remaining') is not None else None
            txt = f"avg:{(fps_ema or 0):5.1f} fps  frame:{frame_idx + 1}/{total_frames if total_frames else '?'}"
            if eta_s is not None:
                m, s = divmod(int(eta_s), 60)
                txt += f"  ETA:{m:02d}:{s:02d}"
            draw_label(frame, txt, (10, 30), color=(255, 255, 255), bg=(0, 0, 0))

        # ----- Always draw frame index -----
        frame_text = f"Frame: {frame_idx + 1}"
        if total_frames:
            frame_text += f"/{total_frames}"
        cv2.putText(frame, frame_text, (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 215, 255), 4, cv2.LINE_AA)

        if args.save_last10_per_id:
            frame_copy = frame.copy()
            for tid in tids_in_frame:
                track_last_frames[tid].append(frame_copy.copy())
        else:
            last_video_frames.append(frame.copy())

        if tqdm is not None and bar is not None:
            bar.update(1)

        frame_idx += 1

    # ===================== End loop =====================
    cap.release()

    if args.save_last10_per_id:
        writer.release()
        try:
            Path(video_out_path).unlink(missing_ok=True)
        except Exception:
            pass

        if run_dir:
            per_id_dir = Path(run_dir) / "per_id_last10"
        else:
            per_id_dir = Path(Path(video_out_path).parent) / f"{Path(video_out_path).stem}_per_id_last10"
        per_id_dir.mkdir(parents=True, exist_ok=True)

        for tid, frames in track_last_frames.items():
            if len(frames) == 0:
                continue
            safe_tid = str(tid).replace("/", "_").replace(" ", "_")
            per_id_path = per_id_dir / f"id_{safe_tid}_last10.mp4"
            per_writer = cv2.VideoWriter(str(per_id_path), fourcc, fps_src, (W, Hh))
            for f in frames:
                per_writer.write(f)
            per_writer.release()
        print(f"Saved per-ID last-10-frame videos in: {per_id_dir}")
    else:
        for f in last_video_frames:
            writer.write(f)
        writer.release()

    if tqdm is not None and bar is not None:
        bar.close()
    # ----- Save per-frame Excel -----

    df = pd.DataFrame(csv_rows)

    if not df.empty and "frame" in df.columns:
        last_10_frame_ids = sorted(df["frame"].unique())[-10:]
        df = df[df["frame"].isin(last_10_frame_ids)].copy()

    if csv_path:
        xlsx_path = str(Path(csv_path).with_suffix(".xlsx"))
        Path(xlsx_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(xlsx_path, index=False)
        print(f"Saved Excel (last 10 frames only): {xlsx_path}")

    # ----- Per-ID summary -----
    if summary_path:
        if df.empty:
            print("No per-frame rows; summary skipped.")
        else:
            col_w_m = None
            for c in ("width_m_s", "width_m_raw"):
                if c in df.columns:
                    col_w_m = c
                    break

            col_l_m = None
            for c in ("length_fused_m", "length_m_s", "length_m_raw"):
                if c in df.columns:
                    col_l_m = c
                    break

            if col_w_m is None and "width_bev_px_s" in df.columns and args.scale_m_per_px is not None:
                df["_width_m"] = df["width_bev_px_s"] * args.scale_m_per_px
                col_w_m = "_width_m"

            if col_l_m is None and "length_bev_px_s" in df.columns and args.scale_m_per_px is not None:
                df["_length_m"] = df["length_bev_px_s"] * args.scale_m_per_px
                col_l_m = "_length_m"

            ranges = {}
            if args.class_range:
                for token in args.class_range.split(","):
                    token = token.strip()
                    if not token:
                        continue
                    k, rng = token.split(":")
                    lo, hi = rng.split("-")
                    ranges[k.strip()] = (float(lo), float(hi))

            def summarize_metric(vals, prefix):
                v = np.asarray(vals, dtype=float)
                v = v[~np.isnan(v)]
                if v.size == 0:
                    return {
                        f"{prefix}_final": np.nan,
                        f"{prefix}_how": "",
                        f"{prefix}_median": np.nan,
                        f"{prefix}_trimmed_mean": np.nan,
                        f"{prefix}_mode": np.nan,
                        f"{prefix}_MAD": np.nan,
                        f"{prefix}_IQR": np.nan,
                        f"{prefix}_CI68_lo": np.nan,
                        f"{prefix}_CI68_hi": np.nan
                    }, 0

                stat = robust_stats(v)
                final, how = choose_final(stat)
                sigma68 = 1.4826 * stat["mad"] if stat["mad"] == stat["mad"] else np.nan

                row = {
                    f"{prefix}_final": final,
                    f"{prefix}_how": how,
                    f"{prefix}_median": stat["median"],
                    f"{prefix}_trimmed_mean": stat["trimmed_mean"],
                    f"{prefix}_mode": stat["mode"],
                    f"{prefix}_MAD": stat["mad"],
                    f"{prefix}_IQR": stat["iqr"],
                    f"{prefix}_CI68_lo": final - sigma68 if sigma68 == sigma68 else np.nan,
                    f"{prefix}_CI68_hi": final + sigma68 if sigma68 == sigma68 else np.nan,
                }
                return row, stat["n"]

            rows = []
            for tid, g in df.groupby("track_id", sort=True):
                g = g.copy()

                if len(g) < args.summary_min_frames:
                    rows.append({"track_id": tid, "n_frames": len(g)})
                    continue

                if col_w_m is not None and ranges and "cls" in g.columns and not g["cls"].empty:
                    cls = str(g["cls"].mode().iloc[0])
                    if cls in ranges:
                        lo, hi = ranges[cls]
                        g = g[(g[col_w_m] >= lo) & (g[col_w_m] <= hi)]

                if g.empty:
                    rows.append({"track_id": tid, "n_frames": 0})
                    continue

                row_out = {"track_id": tid}

                if col_w_m is not None:
                    w_vals = g[col_w_m].astype(float).values
                    w_row, n_use = summarize_metric(w_vals, "width_m")
                    row_out.update({"n_frames": n_use})
                    row_out.update(w_row)
                else:
                    col_w_px = "width_bev_px_s" if "width_bev_px_s" in g.columns else "width_bev_px_raw"
                    w_vals = g[col_w_px].astype(float).values
                    w_row, n_use = summarize_metric(w_vals, "width_px")
                    row_out.update({"n_frames": n_use})
                    row_out.update(w_row)

                if col_l_m is not None:
                    l_vals = g[col_l_m].astype(float).values
                    l_row, _ = summarize_metric(l_vals, "length_m")
                    row_out.update(l_row)
                else:
                    col_l_px = "length_bev_px_s" if "length_bev_px_s" in g.columns else "length_bev_px_raw"
                    if col_l_px in g.columns:
                        l_vals = g[col_l_px].astype(float).values
                        l_row, _ = summarize_metric(l_vals, "length_px")
                        row_out.update(l_row)

                rows.append(row_out)

            out = pd.DataFrame(rows).sort_values("track_id")
            summary_xlsx_path = str(Path(summary_path).with_suffix(".xlsx"))
            Path(summary_xlsx_path).parent.mkdir(parents=True, exist_ok=True)
            out.to_excel(summary_xlsx_path, index=False)
            print(f"Saved per-ID summary Excel: {summary_xlsx_path}")
            try:
                print(out.head(10).to_string(index=False))
            except Exception:
                pass

    # ----- Throughput summary -----
    t1 = time.perf_counter()
    elapsed = t1 - t0
    total = frame_idx
    overall_fps = (total / elapsed) if elapsed > 1e-6 else 0.0
    print(f"Done. Frames: {total}, Elapsed: {elapsed:.2f}s, Throughput: {overall_fps:.2f} FPS")

    if torch is not None:
        try:
            if torch.cuda.is_available():
                print("CUDA device:", torch.cuda.get_device_name(0))
        except Exception:
            pass


if __name__ == "__main__":
    main()