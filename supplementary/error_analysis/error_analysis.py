"""Error analysis of per-trial REC results written by behavioral/eval_rec.py.

Usage:
  python error_analysis.py path/to/results.csv [--outdir DIR]

Expected columns (written by eval_rec.py): iou, success_05, speaker, condition,
description, pred_box, distractor_count. `distractor_count` is the number of
same-category distractors (the raw co-occurrence count minus the target).
Requires: pandas
"""
import argparse
import os
import re

import pandas as pd

SPATIAL_PATTERN = re.compile(
    r"\b(left|right|top|bottom|upper|lower|middle|center|front|back|background|foreground|"
    r"beside|next to|near|adjacent|above|below|under|over|between|behind|across)\b",
    re.IGNORECASE,
)


def categorize_iou(v):
    """Bands use the same boundaries as the main analysis: success means IoU >= 0.50."""
    if v == 0.0:
        return "1. Zero (IoU = 0)"
    if v < 0.10:
        return "2. Catastrophic (0 < IoU < 0.10)"
    if v < 0.30:
        return "3. Poor (0.10 <= IoU < 0.30)"
    if v < 0.50:
        return "4. Near miss (0.30 <= IoU < 0.50)"
    if v < 0.75:
        return "5. Good (0.50 <= IoU < 0.75)"
    return "6. Excellent (IoU >= 0.75)"


def main(csv_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} trials from {csv_path}")
    df["has_box"] = df["pred_box"].notna()
    df["success"] = df["success_05"].astype(int)
    df["distractors"] = df["distractor_count"].astype(int)

    # 1. IoU bands. Missing boxes are reported separately from wrong boxes.
    df["iou_category"] = df["iou"].apply(categorize_iou)
    df.loc[~df["has_box"], "iou_category"] = "0. No box produced"
    print("\n== 1. IoU bands (all trials) ==")
    bands = df["iou_category"].value_counts().sort_index().reset_index()
    bands.columns = ["band", "count"]
    bands["pct"] = (100 * bands["count"] / len(df)).round(2)
    print(bands.to_string(index=False))
    bands.to_csv(os.path.join(outdir, "error_iou_bands.csv"), index=False)

    # 2. Failure profile
    errors = df[df["success"] == 0]
    print(f"\n== 2. Failure profile ({len(errors)} errors out of {len(df)} trials) ==")
    prof = errors["iou_category"].value_counts().sort_index().reset_index()
    prof.columns = ["band", "count"]
    prof["pct_of_errors"] = (100 * prof["count"] / len(errors)).round(2)
    print(prof.to_string(index=False))
    prof.to_csv(os.path.join(outdir, "error_profile.csv"), index=False)

    # 3. Description length by condition and outcome
    df["word_count"] = df["description"].fillna("").astype(str).str.split().str.len()
    lens = df.groupby(["condition", "success"])["word_count"].agg(
        mean_words="mean", median_words="median", std_words="std").round(2).reset_index()
    lens["outcome"] = lens["success"].map({1: "correct", 0: "wrong"})
    lens = lens[["condition", "outcome", "mean_words", "median_words", "std_words"]]
    print("\n== 3. Description length (words) by condition and outcome ==")
    print(lens.to_string(index=False))
    lens.to_csv(os.path.join(outdir, "error_description_length.csv"), index=False)

    # 4. Spatial and relational terms, by number of distractors
    df["has_spatial"] = df["description"].fillna("").astype(str).apply(
        lambda s: bool(SPATIAL_PATTERN.search(s)))
    sp = df.groupby(["distractors", "has_spatial"])["success"].agg(
        trials="count", accuracy_pct=lambda x: round(100 * x.mean(), 2)).reset_index()
    sp["spatial_terms"] = sp["has_spatial"].map({True: "yes", False: "no"})
    sp = sp[["distractors", "spatial_terms", "trials", "accuracy_pct"]]
    print("\n== 4. Accuracy by distractor count and spatial terms ==")
    print(sp.to_string(index=False))
    sp.to_csv(os.path.join(outdir, "error_spatial_terms.csv"), index=False)

    # 5. Catastrophic error share by speaker
    df["is_catastrophic"] = (df["iou"] < 0.10) & df["has_box"]
    spk = df.groupby("speaker").agg(
        trials=("success", "size"),
        accuracy_pct=("success", lambda x: round(100 * x.mean(), 2)),
        errors=("success", lambda x: int((x == 0).sum())),
        catastrophic=("is_catastrophic", "sum"),
    ).reset_index()
    spk["error_rate_pct"] = (100 * spk["errors"] / spk["trials"]).round(2)
    spk["catastrophic_share_of_errors_pct"] = (100 * spk["catastrophic"] / spk["errors"].replace(0, pd.NA)).round(2)
    spk = spk.sort_values("accuracy_pct", ascending=False)
    print("\n== 5. Error profile by speaker ==")
    print(spk.to_string(index=False))
    spk.to_csv(os.path.join(outdir, "error_by_speaker.csv"), index=False)

    print(f"\nDone. CSV files written to {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="per-trial results CSV from behavioral/eval_rec.py")
    ap.add_argument("--outdir", default=None, help="output directory (default: next to the input CSV)")
    a = ap.parse_args()
    if not os.path.isfile(a.csv_path):
        raise SystemExit(f"error: input file not found: {a.csv_path}")
    main(a.csv_path, a.outdir or os.path.dirname(os.path.abspath(a.csv_path)))
