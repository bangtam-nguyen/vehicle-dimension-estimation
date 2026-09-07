#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Đối chiếu kích thước ước tính (summary theo track_id) với kích thước tham chiếu (GT).

Thay đổi so với bản gốc: bỏ 3 đường dẫn hardcode, chuyển sang argparse.
Logic tính sai số giữ nguyên hoàn toàn.

Ví dụ:
    python src/compare_est_vs_gt.py \
        --est output/20260513-163327_car_result_pca_tune5/summary_by_id.xlsx \
        --gt  data/gt_vehicle_size.xlsx \
        --out output/20260513-163327_car_result_pca_tune5/compare_est_vs_gt.xlsx
"""

import argparse

import numpy as np
import pandas as pd


# Ưu tiên từ trên xuống khi dò tên cột trong file summary
EST_WIDTH_CANDIDATES = [
    "width_m_final",
    "width_final_m",
    "width_median_m",
    "width_m_s",
    "width_m_raw",
]

EST_LENGTH_CANDIDATES = [
    "length_m_final",
    "length_final_m",
    "length_median_m",
    "length_fused_m",
    "length_m_s",
    "length_body_m",
]


def pick_column(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Không tìm thấy cột {what} trong file summary. "
        f"Đã thử: {candidates}. Cột hiện có: {df.columns.tolist()}"
    )


def main():
    ap = argparse.ArgumentParser(
        description="So sánh kích thước ước tính với kích thước tham chiếu và tính MAE/MAPE."
    )
    ap.add_argument("--est", required=True,
                    help="File summary theo track_id do car_size_pca_test_perid.py xuất ra (.xlsx)")
    ap.add_argument("--gt", required=True,
                    help="File kích thước tham chiếu (.xlsx), cần có cột track_id, gt_width_m, gt_length_m")
    ap.add_argument("--out", required=True,
                    help="File Excel kết quả đối chiếu")
    ap.add_argument("--gt_unit", default="mm", choices=["mm", "m"],
                    help="Đơn vị trong file GT. Mặc định mm (giống file GT gốc, ví dụ 1750, 4475).")
    args = ap.parse_args()

    df_est = pd.read_excel(args.est)
    df_gt = pd.read_excel(args.gt)

    print("=== Cột trong file kết quả đo ===")
    print(df_est.columns.tolist())
    print("\n=== Cột trong file GT ===")
    print(df_gt.columns.tolist())

    # ---------- 1) Chọn cột width/length từ file summary ----------
    est_width_col = pick_column(df_est, EST_WIDTH_CANDIDATES, "chiều rộng estimated")
    est_length_col = pick_column(df_est, EST_LENGTH_CANDIDATES, "chiều dài estimated")

    print(f"\nDùng cột width estimated : {est_width_col}")
    print(f"Dùng cột length estimated: {est_length_col}")

    df_est = df_est.rename(columns={
        est_width_col: "est_width_m",
        est_length_col: "est_length_m",
    })

    # ---------- 2) Chuẩn hóa GT về mét ----------
    divisor = 1000.0 if args.gt_unit == "mm" else 1.0
    df_gt = df_gt.copy()
    df_gt["gt_width_m_fixed"] = df_gt["gt_width_m"] / divisor
    df_gt["gt_length_m_fixed"] = df_gt["gt_length_m"] / divisor

    # ---------- 3) Ghép theo track_id ----------
    gt_cols = ["track_id", "gt_width_m_fixed", "gt_length_m_fixed"]
    if "note" in df_gt.columns:
        gt_cols.append("note")

    df_cmp = df_est.merge(df_gt[gt_cols], on="track_id", how="left")
    df_cmp = df_cmp.rename(columns={
        "gt_width_m_fixed": "gt_width_m",
        "gt_length_m_fixed": "gt_length_m",
    })

    # ---------- 4) Tính sai số ----------
    df_cmp["error_width_m"] = df_cmp["est_width_m"] - df_cmp["gt_width_m"]
    df_cmp["error_length_m"] = df_cmp["est_length_m"] - df_cmp["gt_length_m"]

    df_cmp["abs_error_width_m"] = df_cmp["error_width_m"].abs()
    df_cmp["abs_error_length_m"] = df_cmp["error_length_m"].abs()

    df_cmp["ape_width_percent"] = np.where(
        df_cmp["gt_width_m"] > 0,
        df_cmp["abs_error_width_m"] / df_cmp["gt_width_m"] * 100,
        np.nan,
    )
    df_cmp["ape_length_percent"] = np.where(
        df_cmp["gt_length_m"] > 0,
        df_cmp["abs_error_length_m"] / df_cmp["gt_length_m"] * 100,
        np.nan,
    )

    # ---------- 5) Sắp xếp cột ----------
    preferred = [
        "track_id", "note", "n_frames",
        "est_width_m", "gt_width_m", "error_width_m",
        "abs_error_width_m", "ape_width_percent",
        "est_length_m", "gt_length_m", "error_length_m",
        "abs_error_length_m", "ape_length_percent",
    ]
    existing = [c for c in preferred if c in df_cmp.columns]
    others = [c for c in df_cmp.columns if c not in existing]
    df_cmp = df_cmp[existing + others]

    # ---------- 6) Lưu ----------
    df_cmp.to_excel(args.out, index=False)
    print(f"\nĐã tạo file: {args.out}")

    # ---------- 7) Tóm tắt ----------
    mae_w = df_cmp["abs_error_width_m"].mean()
    mae_l = df_cmp["abs_error_length_m"].mean()
    mse_w = (df_cmp["error_width_m"] ** 2).mean()
    mse_l = (df_cmp["error_length_m"] ** 2).mean()
    mape_w = df_cmp["ape_width_percent"].mean()
    mape_l = df_cmp["ape_length_percent"].mean()
    n_matched = int(df_cmp["gt_width_m"].notna().sum())

    print("\n=== TÓM TẮT ===")
    print(f"Số xe ghép được GT : {n_matched}")
    print(f"MAE  length = {mae_l:.4f} m   |  MAE  width = {mae_w:.4f} m")
    print(f"RMSE length = {np.sqrt(mse_l):.4f} m   |  RMSE width = {np.sqrt(mse_w):.4f} m")
    print(f"MAPE length = {mape_l:.2f}%      |  MAPE width = {mape_w:.2f}%")


if __name__ == "__main__":
    main()
