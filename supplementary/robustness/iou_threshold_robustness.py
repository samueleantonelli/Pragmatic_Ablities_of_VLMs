"""IoU-threshold robustness: accuracy at IoU >= 0.50, 0.60 and 0.75 from REC log files.

Usage:
  python iou_threshold_robustness.py path/to/log_dir [--out iou_threshold_robustness.csv]
"""
import argparse
import os
import re

import pandas as pd

HEADER = re.compile(r"^(.*?)\s*\|\s*(.*?)\s*\|\s*Desc:\s*(.*)$")
IOU_LINE = re.compile(r"^Image\s+(\d+):\s*IoU\s*=\s*([0-9.]+)")
THRESHOLDS = (0.50, 0.60, 0.75)


def parse_log_directory(log_dir):
    records = []
    for root, _, files in os.walk(log_dir):
        for fname in sorted(files):
            if not fname.endswith(".log") or fname.startswith("slurm"):
                continue
            path = os.path.join(root, fname)
            speaker_fallback = os.path.splitext(fname)[0]
            speaker, condition = None, None
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    m_hdr = HEADER.match(line)
                    if m_hdr:
                        speaker = m_hdr.group(1).strip()
                        condition = m_hdr.group(2).strip()
                        continue
                    m_iou = IOU_LINE.match(line)
                    if m_iou:
                        iou = float(m_iou.group(2))
                        rec = {
                            "speaker": speaker or speaker_fallback,
                            "condition": condition or "unknown",
                            "image_id": m_iou.group(1),
                            "iou": iou,
                        }
                        for t in THRESHOLDS:
                            rec[f"correct_{t:.2f}"] = int(iou >= t)
                        records.append(rec)
    return pd.DataFrame(records)


def analyze_thresholds(df, out_csv):
    if df.empty:
        raise SystemExit("error: no trials found in the log files")
    print(f"Total trials: {len(df)}")

    by_cond = df.groupby("condition").agg(
        trials=("iou", "count"),
        acc_050=("correct_0.50", "mean"),
        acc_060=("correct_0.60", "mean"),
        acc_075=("correct_0.75", "mean"),
        mean_iou=("iou", "mean"),
    ).reset_index()
    for col in ["acc_050", "acc_060", "acc_075"]:
        by_cond[col] = (by_cond[col] * 100).round(2)
    by_cond["mean_iou"] = by_cond["mean_iou"].round(4)
    print("\n== Accuracy by condition (IoU >= 0.50 / 0.60 / 0.75) ==")
    print(by_cond.to_string(index=False))

    detailed = df.groupby(["speaker", "condition"]).agg(
        trials=("iou", "count"),
        acc_050=("correct_0.50", "mean"),
        acc_075=("correct_0.75", "mean"),
        mean_iou=("iou", "mean"),
    ).reset_index()
    detailed["acc_050"] = (detailed["acc_050"] * 100).round(2)
    detailed["acc_075"] = (detailed["acc_075"] * 100).round(2)
    detailed["mean_iou"] = detailed["mean_iou"].round(4)
    print("\n== Detail by speaker and condition ==")
    print(detailed.to_string(index=False))
    detailed.to_csv(out_csv, index=False)
    print(f"\nTable saved to: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", help="directory containing the REC .log files")
    ap.add_argument("--out", default="iou_threshold_robustness.csv")
    a = ap.parse_args()
    if not os.path.isdir(a.log_dir):
        raise SystemExit(f"error: log directory not found: {a.log_dir}")
    analyze_thresholds(parse_log_directory(a.log_dir), a.out)
