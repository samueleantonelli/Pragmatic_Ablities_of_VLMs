import sys
import re
import os
import pandas as pd
from datasets import load_dataset

def parse_log_file(file_path):
    # Pattern to match trial header: speaker | prompt_type | Desc: text
    header_pattern = re.compile(r"([a-zA-Z0-9_\-]+)\s*\|\s*([a-zA-Z0-9_\-]+)\s*\|\s*Desc:\s*(.*)$")
    # Pattern with "=" to correctly read "IoU = ..."
    eval_pattern = re.compile(r"Image\s+(\d+):\s*IoU\s*=\s*([0-9.]+)\s*(?:→|->)\s*(Correct|Wrong)")

    trials = []
    current_trial = {}

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue

            if "Desc:" in line_str:
                h_match = header_pattern.search(line_str)
                if h_match:
                    current_trial = {
                        "speaker": h_match.group(1).strip(),
                        "prompt_type": h_match.group(2).strip(),
                        "description": h_match.group(3).strip()
                    }
                    continue

            e_match = eval_pattern.search(line_str)
            if e_match and current_trial:
                current_trial["image_idx"] = int(e_match.group(1))
                current_trial["iou"] = float(e_match.group(2))
                current_trial["success"] = 1 if e_match.group(3) == "Correct" else 0
                trials.append(current_trial)
                current_trial = {}

    return pd.DataFrame(trials)

def run_stratification(log_file_or_dir, local_meta_path=None):
    print("1. Parsing log files...")
    all_trials = []
    
    if os.path.isdir(log_file_or_dir):
        log_files = [f for f in os.listdir(log_file_or_dir) if f.endswith(".log")]
        print(f"Found {len(log_files)} .log files in directory.")
        for fname in log_files:
            path = os.path.join(log_file_or_dir, fname)
            df_sub = parse_log_file(path)
            all_trials.append(df_sub)
        df_results = pd.concat(all_trials, ignore_index=True) if all_trials else pd.DataFrame()
    else:
        df_results = parse_log_file(log_file_or_dir)

    print(f"Extracted {len(df_results)} trials from logs.")
    if df_results.empty:
        print("No trials extracted. Check paths or log format.")
        return

    print("\n2. Loading and filtering refOI dataset metadata...")
    if local_meta_path and os.path.exists(local_meta_path):
        print(f"Loading local metadata from: {local_meta_path}")
        if local_meta_path.endswith(".csv"):
            df_meta = pd.read_csv(local_meta_path)
        else:
            df_meta = pd.read_json(local_meta_path)
    else:
        print("Streaming refOI metadata directly (text columns only, fast)...")
        ds = load_dataset("Seed42Lab/refOI-TLHF", split="train", streaming=True)
        meta_rows = []
        for i, item in enumerate(ds):
            meta_rows.append({
                "description": item.get("description", item.get("full_description", item.get("referring_expression"))),
                "co_occurrence": item.get("co_occurrence", item.get("co_occurrences", None)),
                "prompt_type": item.get("prompt_type", None)
            })
            if (i + 1) % 5000 == 0:
                print(f"Processed {i + 1} metadata entries...")
        df_meta = pd.DataFrame(meta_rows)

    print(f"Loaded refOI metadata with {len(df_meta)} rows.")

    co_col = "co_occurrence"
    desc_col = "description"

    # FILTER: keep only co_occurrence > 1 (excludes single-object items)
    df_meta = df_meta[df_meta[co_col] > 1].copy()
    print(f"Filtered metadata to co_occurrence > 1. Remaining rows: {len(df_meta)}")

    print("\n3. Merging predictions with distractor counts...")
    df_results["desc_clean"] = df_results["description"].astype(str).str.strip().str.lower()
    df_meta["desc_clean"] = df_meta[desc_col].astype(str).str.strip().str.lower()

    meta_subset = df_meta[["desc_clean", co_col]].dropna().drop_duplicates("desc_clean")
    merged = pd.merge(df_results, meta_subset, on="desc_clean", how="inner")
    print(f"Successfully matched {len(merged)} trials with ground-truth distractor counts.")

    if merged.empty:
        print("Warning: 0 rows matched. Check description strings.")
        return

    # 1. Overall stratification by distractors
    print("\n=== Overall accuracy stratified (co_occurrence > 1) ===")
    strat_gen = merged.groupby(co_col)["success"].agg(
        total_trials="count",
        accuracy="mean"
    ).reset_index()
    strat_gen["accuracy_percent"] = (strat_gen["accuracy"] * 100).round(2)
    print(strat_gen.to_string(index=False))
    strat_gen.to_csv("qwen_distractor_filtered_overall.csv", index=False)

    # 2. Stratification by prompt
    print("\n=== Accuracy stratified by prompt type ===")
    strat_prompt = merged.groupby([co_col, "prompt_type"])["success"].agg(
        total_trials="count",
        accuracy="mean"
    ).reset_index()
    strat_prompt["accuracy_percent"] = (strat_prompt["accuracy"] * 100).round(2)
    print(strat_prompt.to_string(index=False))
    strat_prompt.to_csv("qwen_distractor_filtered_by_prompt.csv", index=False)

    # 3. Stratification by speaker model
    print("\n=== Accuracy stratified by speaker model ===")
    strat_speaker = merged.groupby(["speaker", co_col])["success"].agg(
        total_trials="count",
        accuracy="mean"
    ).reset_index()
    strat_speaker["accuracy_percent"] = (strat_speaker["accuracy"] * 100).round(2)
    print(strat_speaker.to_string(index=False))
    strat_speaker.to_csv("qwen_distractor_filtered_by_speaker.csv", index=False)

    print("\nAll filtered tables were saved as CSV.")

if __name__ == "__main__":
    log_input = sys.argv[1]
    local_metadata = None 
    run_stratification(log_input, local_meta_path=local_metadata)