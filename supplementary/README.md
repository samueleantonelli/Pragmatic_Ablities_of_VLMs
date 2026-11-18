# Supplementary analyses

Analyses run in response to the reviewers. They are not part of the main REC and probing pipeline in `behavioral/`, `representational/` and `analysis/`.

| Folder | Content |
| :--- | :--- |
| `robustness/` | IoU threshold robustness (0.50, 0.75) from the listener logs |
| `distractor_stratification/` | Accuracy stratified by number of same-category distractors and by speaker |
| `error_analysis/` | IoU distribution of failures, description length, spatial terms, catastrophic errors |
| `control_baselines/` | Control conditions (no description, shuffled descriptions, category only) and their analysis |

## Usage

```bash
# robustness: directory of listener .log files
python supplementary/robustness/iou_threshold_robustness.py path/to/log_dir

# distractor stratification: same directory
python supplementary/distractor_stratification/distractor_stratification.py path/to/log_dir

# error analysis: per-trial CSV
python supplementary/error_analysis/error_analysis.py path/to/trials_level_data.csv

# control baselines: run the listener, then analyse the JSONL output
python supplementary/control_baselines/run_control_baselines.py --n-items 2000 --out results.jsonl
python supplementary/control_baselines/analyze_control_baselines.py "results/results*.jsonl" --outdir results/analysis
```
