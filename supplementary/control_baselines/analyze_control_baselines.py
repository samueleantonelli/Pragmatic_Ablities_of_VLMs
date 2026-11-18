#!/usr/bin/env python3
"""Analyse the JSONL files written by run_control_baselines.py.

Main analysis: items with >= 1 distractor (where a description is needed to pick the target).
Positive control: items with 0 distractors (a single object of the category: salience suffices).

The runner stores the raw co_occurrence count (it includes the target itself) in
`n_distractors`; this script subtracts --co-offset (default 1) to get the true number
of distractors.

Usage:
  python analyze_control_baselines.py "results/results*.jsonl" --outdir results/analysis
Requires: pandas numpy statsmodels
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import mcnemar

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)

ORDER = ["real", "category_only", "shuffled_same_cat", "shuffled_random", "none"]
HUMAN_STYLES = {"spoken", "written"}


def load(pattern, co_offset=1):
    """Reads records written by run_control_baselines.py. `n_distractors` there is the
    raw co_occurrence count (it includes the target itself): converted here as
    co_occurrence - co_offset to get the true number of distractors."""
    recs, versions = [], {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                v = r.get("parser")
                if v != "v1":
                    continue
                versions[v] = versions.get(v, 0) + 1
                nd = r.get("n_distractors")
                r["n_distractors"] = None if nd is None else int(float(nd)) - co_offset
                r["co_occurrence"] = nd
                r["desc_n_words"] = len((r.get("description") or "").split())
                recs.append(r)
    print(f"[load] records by parser version: {versions}")
    if not recs:
        raise SystemExit(f"no usable records (parser v1) matching {pattern}")
    df = pd.DataFrame(recs)
    n_raw = len(df)
    g = df.groupby(["idx", "condition"])
    n_dup_groups = int((g.size() > 1).sum())
    if n_dup_groups:
        disagree = int((g["success50"].nunique() > 1).sum())
        print(f"[load] {n_raw} records -> {n_raw - len(g)} duplicate records over {n_dup_groups} (item, condition) "
              f"pairs; pairs whose duplicates DISAGREE on success50: {disagree}")
    df = df.drop_duplicates(["idx", "condition"], keep="last")
    df = df[df["n_distractors"].notna()].copy()
    df["n_distractors"] = df["n_distractors"].astype(int)
    df["speaker_type"] = np.where(df["style"].isin(HUMAN_STYLES), "human", "model")
    df["has_box"] = df["status"].eq("ok")
    if "image_id" not in df.columns:
        df["image_id"] = np.nan
    df["image_id"] = df["image_id"].fillna(df["key"])  # fallback for records without an image hash
    df["grp"] = pd.factorize(df["image_id"])[0]

    dist = df.drop_duplicates("idx")["n_distractors"].value_counts().sort_index().to_dict()
    print(f"[load] n_distractors (= co_occurrence - {co_offset}) distribution over items: {dist}")
    if 0 not in dist:
        print("[warn] no items with 0 distractors: check --co-offset or whether the run was filtered")

    # a "shuffled" description identical to the real one is not a mismatch: drop those rows
    real_desc = df[df.condition == "real"].set_index("idx")["description"]
    is_shuf = df.condition.isin(["shuffled_same_cat", "shuffled_random"])
    same = is_shuf & (df["description"] == df["idx"].map(real_desc))
    if same.any():
        print(f"[load] dropped {int(same.sum())} shuffled rows whose text equals the real description")
        df = df[~same]
    return df


def boot_ci(d, col="success50", n=1000, seed=0):
    """95% CI of accuracy resampling IMAGES (an image appears with several speakers and conditions)."""
    rng = np.random.default_rng(seed)
    g = d.groupby("image_id")[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    k = len(g)
    vals = [s[i].sum() / c[i].sum() for i in (rng.integers(0, k, k) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


def acc_table(df, by=None):
    rows = []
    groups = [("all", df)] if by is None else list(df.groupby(by))
    for gval, d in groups:
        for cond in ORDER:
            x = d[d.condition == cond]
            if x.empty:
                continue
            lo, hi = boot_ci(x)
            xb = x[x.has_box]
            rows.append({
                "group": gval, "condition": cond, "n": len(x),
                "acc@0.5": 100 * x.success50.mean(), "ci_lo": 100 * lo, "ci_hi": 100 * hi,
                "acc@0.75": 100 * x.success75.mean(), "mean_iou": x.iou.mean(),
                "box_%": 100 * x.has_box.mean(),
                "acc@0.5|box": 100 * xb.success50.mean() if len(xb) else np.nan,
            })
    return pd.DataFrame(rows).round(2)


def paired_mcnemar(d):
    wide = d.pivot(index="idx", columns="condition", values="success50")
    out = []
    for c in [c for c in ORDER if c != "real" and c in wide]:
        w = wide[["real", c]].dropna()
        b = int(((w.real == 1) & (w[c] == 0)).sum())   # real right, control wrong
        cc = int(((w.real == 0) & (w[c] == 1)).sum())  # real wrong, control right
        p = mcnemar([[0, b], [cc, 0]], exact=True).pvalue if b + cc else np.nan
        out.append({"control": c, "n_pairs": len(w), "real_only": b, "control_only": cc,
                    "p_exact": p, "p_fmt": "<0.001" if p < 1e-3 else f"{p:.3f}"})
    return pd.DataFrame(out)


def logit(d, label, conds):
    d = d[d.condition.isin(conds)].copy()
    try:
        m = smf.logit("success50 ~ C(condition, Treatment('real')) + n_distractors", data=d).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": d["grp"]})
    except Exception as e:  # e.g. perfect separation
        print(f"\n== Logit ({label}) failed: {type(e).__name__}: {e}")
        return None
    ci = m.conf_int()
    t = pd.DataFrame({"beta": m.params, "se": m.bse, "p": m.pvalues, "OR": np.exp(m.params),
                      "OR_lo": np.exp(ci[0]), "OR_hi": np.exp(ci[1])}).round(4)
    t.index = [i.replace("C(condition, Treatment('real'))", "cond") for i in t.index]
    print(f"\n== Logistic regression, SE clustered by image identifier ({label}) ==\n{t}")
    return t


def main(pattern, outdir, co_offset=1):
    os.makedirs(outdir, exist_ok=True)
    df = load(pattern, co_offset)
    main_df = df[df.n_distractors >= 1]
    ctrl_df = df[df.n_distractors == 0]
    print(f"{df.idx.nunique()} items | >=1 distractor: {main_df.idx.nunique()} | "
          f"0 distractors: {ctrl_df.idx.nunique()} | conditions: {sorted(df.condition.unique())}")

    def save(name, t):
        t.to_csv(os.path.join(outdir, name + ".csv"), index=False)

    print("\n== Parse status by condition (>=1 distractor) ==")
    ps = pd.crosstab(main_df.condition, main_df.status)
    print(ps)
    ps.to_csv(os.path.join(outdir, "parse_status.csv"))

    for name, d in [("MAIN: >= 1 distractor", main_df), ("POSITIVE CONTROL: 0 distractors", ctrl_df)]:
        t = acc_table(d)
        print(f"\n== {name}: accuracy by condition (95% CI: cluster bootstrap over images) ==")
        print(t.to_string(index=False))
        save("acc_" + name.split(":")[0].lower().replace(" ", "_"), t)

    t = acc_table(df, by="n_distractors")
    t["chance_%"] = (100.0 / (t["group"].astype(float) + 1)).round(1)  # picking one of the k+1 same-category objects
    print("\n== By number of distractors (all items); chance_% = 100/(distractors+1) ==")
    print(t.to_string(index=False))
    save("acc_by_n_distractors", t)

    t = acc_table(main_df, by="speaker_type")
    print("\n== Human vs model descriptions (>= 1 distractor) ==")
    print(t.to_string(index=False))
    save("acc_by_speaker_type", t)

    t = acc_table(main_df, by="style")
    print("\n== By description style (>= 1 distractor) ==")
    print(t.to_string(index=False))
    save("acc_by_style", t)

    # donors that are very short can legitimately describe the true target as well
    sc = main_df[main_df.condition == "shuffled_same_cat"].copy()
    if len(sc) > 30:
        sc["donor_len"] = pd.qcut(sc["desc_n_words"], 3, duplicates="drop").astype(str)
        t = sc.groupby("donor_len").agg(n=("idx", "size"), acc50=("success50", "mean"),
                                        box_rate=("has_box", "mean")).round(3).reset_index()
        print("\n== shuffled_same_cat by donor description length (>= 1 distractor) ==")
        print(t.to_string(index=False))
        save("shuffled_same_cat_by_donor_len", t)

    rl = main_df[main_df.condition == "real"].groupby("red_leak").agg(
        n=("idx", "size"), acc50=("success50", "mean")).round(3).reset_index()
    print("\n== Red-box mentions in real descriptions (>= 1 distractor) ==")
    print(rl.to_string(index=False))
    save("red_leak", rl)

    for name, d in [("MAIN: >= 1 distractor", main_df), ("POSITIVE CONTROL: 0 distractors", ctrl_df)]:
        t = paired_mcnemar(d)
        print(f"\n== Paired exact McNemar, real vs control ({name}) ==")
        print(t.to_string(index=False))
        save("mcnemar_" + name.split(":")[0].lower().replace(" ", "_"), t)

    # random-description condition is mostly "no box produced": keep it out of the main model
    logit(main_df, ">= 1 distractor, without shuffled_random",
          ["real", "none", "shuffled_same_cat", "category_only"])
    logit(main_df, ">= 1 distractor, all conditions", ORDER)

    # examples of answers without a box, to check what the model actually says
    nb = df[~df.has_box & df.condition.isin(["shuffled_same_cat", "shuffled_random", "real"])]
    parts = [g.sample(min(25, len(g)), random_state=0) for _, g in nb.groupby("condition")]
    ex = pd.concat(parts)[["idx", "condition", "n_distractors", "description", "raw_output"]]
    ex.to_csv(os.path.join(outdir, "no_box_examples.csv"), index=False)
    print(f"\n[saved] CSVs (incl. no_box_examples.csv) in {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="glob of JSONL result files")
    ap.add_argument("--outdir", default="results/analysis")
    ap.add_argument("--co-offset", type=int, default=1,
                    help="n_distractors = co_occurrence - offset")
    a = ap.parse_args()
    main(a.pattern, a.outdir, a.co_offset)